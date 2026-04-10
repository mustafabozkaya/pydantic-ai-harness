"""Code mode toolset that runs LLM-generated Python in a Monty sandbox."""

from __future__ import annotations

import keyword
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Annotated, Any

from pydantic import Field, TypeAdapter
from pydantic_ai import AbstractToolset, RunContext, ToolDefinition, WrapperToolset
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, UserError
from pydantic_ai.function_signature import FunctionSignature
from pydantic_ai.messages import ToolCallPart, ToolReturn, ToolReturnContent, ToolReturnPart, is_multi_modal_content
from pydantic_ai.tools import AgentDepsT, ToolSelector, matches_tool_selector
from pydantic_ai.toolsets.abstract import SchemaValidatorProt, ToolsetTool

try:
    from pydantic_ai.toolsets._tool_search import _SEARCH_TOOLS_NAME  # pyright: ignore[reportPrivateUsage]
except ImportError:  # pragma: no cover
    _SEARCH_TOOLS_NAME = 'search_tools'  # pyright: ignore[reportConstantRedefinition]

try:
    from pydantic_monty import (
        MontyRepl,
        MontyRuntimeError,
        MontySyntaxError,
        MontyTypingError,
    )
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'pydantic-monty is required for CodeMode. Install it with: pip install "pydantic-harness[code-mode]"'
    ) from _import_error
from typing_extensions import NotRequired, TypedDict


class _RunCodeArguments(TypedDict):
    code: Annotated[str, Field(description='The Python code to execute in the sandbox.')]
    restart: NotRequired[
        Annotated[
            bool,
            Field(
                description='Set to true to reset REPL state. When false (default), state is preserved between calls.'
            ),
        ]
    ]


_RUN_CODE_TOOL_NAME = 'run_code'
_RUN_CODE_ADAPTER = TypeAdapter(_RunCodeArguments)
_RUN_CODE_JSON_SCHEMA = _RUN_CODE_ADAPTER.json_schema()
_RUN_CODE_ARGS_VALIDATOR: SchemaValidatorProt = _RUN_CODE_ADAPTER.validator  # pyright: ignore[reportAssignmentType]
# Used to serialize tool return values before sending into Monty (dump_python)
# and to reconstruct multimodal types (e.g. BinaryContent) from Monty results (validate_python).
_TOOL_RETURN_CONTENT_TA: TypeAdapter[Any] = TypeAdapter(ToolReturnContent)

_RUN_CODE_BASE_DESCRIPTION = """\
Write and run Python code in a sandboxed environment.

The sandbox uses Monty, a subset of Python. Key restrictions:
- **No classes**: class definitions are not supported
- **No third-party libraries**: only the standard library modules listed below can be used
- **Importable standard library modules**: `sys`, `typing`, `asyncio`, `math`, `json`, `re`, `datetime`, `os`, `pathlib`. These must be imported at the top of your snippet before use, just like in regular Python. For example: `import asyncio` then `results = await asyncio.gather(tool_one(...), tool_two(...))`.
- **No `import *`**: wildcard imports are not supported

State is preserved between calls (REPL-style). Set `restart: true` to reset state.

The last expression's value is automatically captured as the return value — you do **not** need to \
`print()` it. Avoid `print()` for return values as it produces Python string representations, not \
structured data. Use `print()` only for supplementary logging or debug output.

Returns the last expression's value directly. If `print()` was also called, returns \
`{"output": "<printed text>", "result": <last expression>}`.\
"""

_FUNCTIONS_HEADER = """\

The following functions are available inside the sandbox. Call them directly \
(do **not** redefine or import them) and `await` the result. All parameters are keyword-only. \
All tool functions are async: invoke them with `await`, e.g. `result = await tool_name(arg=value)`. \
Calling without `await` returns an unresolved future, not the value.\
"""

_SEARCH_TOOLS_MODIFIER = (
    ' Note: discovered tools become callable as functions inside the run_code sandbox in subsequent invocations.'
)

_TOOL_SEARCH_ADDENDUM = (
    f'\n\nNot all functions may be available initially.'
    f' Use the `{_SEARCH_TOOLS_NAME}` tool to discover additional functions'
    f' that will become callable in subsequent `run_code` invocations.'
)

_INVALID_IDENT_CHARS = re.compile(r'[^a-zA-Z0-9_]')


def _sanitize_tool_name(name: str) -> str:
    """Turn a tool name into a valid Python identifier.

    Replaces hyphens, dots, and other non-identifier characters with underscores,
    prepends `_` if the result starts with a digit or is a Python keyword.
    """
    sanitized = _INVALID_IDENT_CHARS.sub('_', name)
    if sanitized and sanitized[0].isdigit():
        sanitized = f'_{sanitized}'
    if keyword.iskeyword(sanitized):
        sanitized = f'{sanitized}_'
    return sanitized or '_'


@dataclass(kw_only=True)
class _RunCodeTool(ToolsetTool[AgentDepsT]):
    """ToolsetTool subclass that caches data computed during `get_tools`.

    Avoids a redundant `get_tools` call in `call_tool` by storing the
    callable tool definitions and name mapping on the tool instance itself.
    Follows the same pattern as `_SearchTool` in pydantic-ai's
    `ToolSearchToolset`.
    """

    callable_defs: dict[str, ToolDefinition]
    """Tool definitions callable from inside the sandbox, keyed by (possibly sanitized) name."""

    sanitized_to_original: dict[str, str]
    """Maps sanitized Python-safe names back to original tool names (only for renamed tools)."""

    wrapped_tools: dict[str, ToolsetTool[AgentDepsT]]
    """The wrapped toolset's tools, keyed by original name."""


@dataclass
class CodeModeToolset(WrapperToolset[AgentDepsT]):
    """Implementation toolset for the `CodeMode` capability.

    Exposes a single `run_code` tool alongside any native (non-sandboxed) tools.
    Tools selected by `tool_selector` are presented to the model as Python
    function signatures inside the `run_code` tool description and become
    callable from the sandbox at runtime. Non-selected tools remain visible
    to the model as normal tool calls.

    Tools that require deferred execution (kind `external`/`unapproved`) or
    deferred loading (`defer_loading=True`) cannot be called from inside the
    sandbox and are dropped with a one-time `UserWarning`.
    """

    tool_selector: ToolSelector[AgentDepsT] = 'all'
    """Which wrapped tools to sandbox inside `run_code`. Non-matching tools
    are exposed as native tools."""

    max_retries: int = 3
    """Maximum number of retries for the `run_code` tool (syntax errors count as retries)."""

    # init=False so `replace()` in `for_run` produces a fresh instance with _repl=None,
    # giving each agent run isolated REPL state. Lazy-initialized on first call_tool.
    _repl: MontyRepl | None = field(default=None, init=False, repr=False)

    # Tracks deferred-tool names we've already warned about so we don't spam the
    # logs every step. Reset on `for_run` because each run gets a fresh instance.
    _warned_deferred: set[str] = field(default_factory=set[str], init=False, repr=False)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Return a fresh toolset instance with isolated REPL state for this agent run."""
        wrapped = await self.wrapped.for_run(ctx)
        return replace(self, wrapped=wrapped)

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Update the wrapped toolset for this step while preserving REPL state."""
        new_wrapped = await self.wrapped.for_run_step(ctx)
        if new_wrapped is self.wrapped:
            return self
        new_self = replace(self, wrapped=new_wrapped)
        new_self._repl = self._repl
        new_self._warned_deferred = self._warned_deferred
        return new_self

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        """Return the `run_code` tool plus any native (non-sandboxed) tools."""
        wrapped_tools = await self.wrapped.get_tools(ctx)

        # Split tools into sandboxed vs native based on the selector.
        # The search_tools tool (from ToolSearchToolset) is always kept native
        # so the model can discover deferred tools alongside run_code.
        sandboxed_tools: dict[str, ToolsetTool[AgentDepsT]] = {}
        native_tools: dict[str, ToolsetTool[AgentDepsT]] = {}
        for name, tool in wrapped_tools.items():
            if name == _SEARCH_TOOLS_NAME:
                native_tools[name] = tool
            elif await matches_tool_selector(self.tool_selector, ctx, tool.tool_def):
                sandboxed_tools[name] = tool
            else:
                native_tools[name] = tool

        callable_defs, sanitized_to_original, native_fallbacks = self._partition_callable_tools(sandboxed_tools)

        # Tools that matched the selector but can't run in the sandbox (deferred
        # execution, deferred loading) are promoted back to native tools so
        # they remain visible to the model.
        for name in native_fallbacks:
            native_tools[name] = sandboxed_tools[name]

        description = self._build_description(callable_defs)

        if _RUN_CODE_TOOL_NAME in native_tools:
            raise UserError(
                f"Tool name '{_RUN_CODE_TOOL_NAME}' is reserved for code mode. Rename your tool to avoid conflicts."
            )

        # When search_tools is present, append context about run_code to its
        # description and add a discovery note to the run_code description.
        has_search_tools = _SEARCH_TOOLS_NAME in native_tools
        if has_search_tools:
            search_tool = native_tools[_SEARCH_TOOLS_NAME]
            native_tools[_SEARCH_TOOLS_NAME] = replace(
                search_tool,
                tool_def=replace(
                    search_tool.tool_def,
                    description=(search_tool.tool_def.description or '') + _SEARCH_TOOLS_MODIFIER,
                ),
            )
            description += _TOOL_SEARCH_ADDENDUM

        result: dict[str, ToolsetTool[AgentDepsT]] = dict(native_tools)
        result[_RUN_CODE_TOOL_NAME] = _RunCodeTool(
            toolset=self,
            tool_def=ToolDefinition(
                name=_RUN_CODE_TOOL_NAME,
                description=description,
                parameters_json_schema=_RUN_CODE_JSON_SCHEMA,
                metadata={'code_arg_name': 'code', 'code_arg_language': 'python'},
                sequential=True,
            ),
            max_retries=self.max_retries,
            args_validator=_RUN_CODE_ARGS_VALIDATOR,
            callable_defs=callable_defs,
            sanitized_to_original=sanitized_to_original,
            wrapped_tools=wrapped_tools,
        )
        return result

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        """Execute Python code in the sandbox, or pass through to a native tool."""
        if not isinstance(tool, _RunCodeTool):
            # Native (non-sandboxed) tool — pass through to the wrapped toolset.
            return await self.wrapped.call_tool(name, tool_args, ctx, tool)

        code = tool_args['code']
        restart = tool_args.get('restart', False)

        if self._repl is None or restart:
            self._repl = MontyRepl()

        callable_defs = tool.callable_defs
        sanitized_to_original = tool.sanitized_to_original

        # Build a ToolManager for the sandbox's inner tools so that sandboxed
        # tool calls go through the standard validation/execution path. We
        # inherit `root_capability` from the agent's ToolManager (for capability
        # hooks) but use the *wrapped* toolset and its tools.
        from pydantic_ai.tool_manager import ToolManager

        parent_tm = ctx.tool_manager
        assert parent_tm is not None, 'CodeModeToolset requires ctx.tool_manager to be set'
        tool_manager = ToolManager(
            toolset=self.wrapped,
            root_capability=parent_tm.root_capability,
            ctx=ctx,
            tools=tool.wrapped_tools,
        )

        # Collect nested tool calls and returns keyed by tool_call_id so they
        # can be attached as metadata on the run_code ToolReturnPart.
        nested_calls: dict[str, ToolCallPart] = {}
        nested_returns: dict[str, ToolReturnPart] = {}
        call_counter = 0

        async def dispatch_tool_call(original_name: str, kwargs: dict[str, Any]) -> Any:
            """Dispatch a single tool call from inside the sandbox."""
            nonlocal call_counter
            call_counter += 1
            parent_id = ctx.tool_call_id or 'pyd_ai_code_mode'
            tool_call_id = f'{parent_id}__{call_counter}'
            call_part = ToolCallPart(tool_name=original_name, args=kwargs, tool_call_id=tool_call_id)
            nested_calls[tool_call_id] = call_part

            try:
                result = await tool_manager.handle_call(call_part, wrap_validation_errors=False)
            except (CallDeferred, ApprovalRequired) as e:
                # Approval/deferral require a round-trip back to the caller, which
                # the sandbox cannot do. We raise UserError here; because this runs
                # inside Monty's external-function callback, Monty catches it as a
                # RuntimeError and wraps it in MontyRuntimeError, which our caller
                # then translates to ModelRetry. The error message is preserved
                # through the chain so the model (or developer) sees the cause.
                raise UserError(
                    'Tool approval and deferral are not supported in code mode. '
                    f'Tool {original_name!r} raised {type(e).__name__}; ensure wrapped '
                    'tools do not use approval or deferral when used with CodeMode.'
                ) from e

            # Unwrap ToolReturn to get the plain value for the sandbox,
            # preserving the full ToolReturn metadata on the return part.
            return_metadata: Any = None
            if isinstance(result, ToolReturn):
                return_metadata = result.metadata
                result = result.return_value

            nested_returns[tool_call_id] = ToolReturnPart(
                tool_name=original_name,
                content=result,
                tool_call_id=tool_call_id,
                metadata=return_metadata,
            )

            # Serialize to JSON-compatible form so Monty receives only plain data.
            return _TOOL_RETURN_CONTENT_TA.dump_python(result)

        # For sanitized names (e.g. `get_weather` from `get-weather`), Monty sees the
        # safe name but we dispatch using the original name from the wrapped toolset.
        external_functions = {
            safe_name: _make_sandbox_callable(
                sanitized_to_original.get(safe_name, safe_name),
                dispatch_tool_call,
            )
            for safe_name in callable_defs
        } or None

        capture = _PrintCapture()

        try:
            result = await self._repl.feed_run_async(
                code,
                external_functions=external_functions,
                print_callback=capture,
            )
        except MontySyntaxError as e:
            raise ModelRetry(f'Syntax error in code:\n{e.display()}') from e
        except MontyTypingError as e:  # pragma: no cover — MontyRepl doesn't raise this yet
            raise ModelRetry(f'Type error in code:\n{e.display()}') from e
        except MontyRuntimeError as e:
            # Note: exceptions raised inside dispatch_tool_call (e.g. UserError
            # from ApprovalRequired, or ModelRetry from a wrapped tool) get caught
            # by Monty and re-wrapped as MontyRuntimeError. The original exception
            # message is preserved in the display string, so the model sees a
            # useful error. This means ModelRetry from a wrapped tool gets
            # double-wrapped (ModelRetry → MontyRuntimeError → ModelRetry), but
            # the retry semantics are the same — the model gets another chance.
            raise ModelRetry(f'Runtime error:\n{e.display()}') from e

        # TODO: For stdio-based driver runtimes (e.g. Monty's current driver loop),
        # print output goes to stdout which is also used for JSON protocol parsing.
        # This will need a different capture strategy for those runtimes.
        printed = capture.joined

        # Validate result to reconstruct multimodal types (e.g. BinaryContent from
        # serialized dicts) so they flow through to the model natively.
        if result is not None:
            result = _TOOL_RETURN_CONTENT_TA.validate_python(result)

        # Build return value:
        # - No print → return result directly (multimodal content stays top-level
        #   so _split_content can extract it for native model delivery)
        # - Print + multimodal result → list format so _split_content can extract files
        # - Print + plain result → dict with output/result keys
        if not printed:
            return_value: Any = result if result is not None else {}
        elif result is None:
            return_value = {'output': printed}
        elif _contains_multimodal(result):
            # Flatten lists so _split_content can find each multimodal item at top level.
            return_value = [printed, *result] if isinstance(result, list) else [printed, result]
        else:
            return_value = {'output': printed, 'result': result}

        return ToolReturn(
            return_value=return_value,
            metadata={'tool_calls': nested_calls, 'tool_returns': nested_returns},
        )

    def _partition_callable_tools(
        self, wrapped_tools: dict[str, ToolsetTool[AgentDepsT]]
    ) -> tuple[dict[str, ToolDefinition], dict[str, str], set[str]]:
        """Return tool definitions that can be called from inside the sandbox.

        Tool names that are not valid Python identifiers (e.g. MCP tools with
        hyphens or dots like `get-weather`, `api.call`) are sanitized to
        underscored forms and mapped back to their original names for dispatch.

        Tools requiring deferred execution (kind `external`/`unapproved`) or
        deferred loading (`defer_loading=True`) cannot run in the sandbox and
        are excluded from `callable_defs`. Their names are returned in the
        third element so the caller can promote them back to native tools.

        Returns:
            A tuple of `(callable_defs, sanitized_to_original, native_fallbacks)`
            where `native_fallbacks` contains original tool names that should
            be exposed as native tools instead of being sandboxed.
        """
        callable_defs: dict[str, ToolDefinition] = {}
        sanitized_to_original: dict[str, str] = {}
        native_fallbacks: set[str] = set()
        for name, tool in wrapped_tools.items():
            td = tool.tool_def
            if td.defer:
                if name not in self._warned_deferred:
                    self._warned_deferred.add(name)
                    warnings.warn(
                        f'CodeMode: tool {name!r} requires deferred execution '
                        f'(kind={td.kind!r}) and cannot be called from inside the '
                        f'sandbox; it will be exposed as a native tool instead.',
                        UserWarning,
                        stacklevel=2,
                    )
                native_fallbacks.add(name)
                continue
            if td.defer_loading:
                if name not in self._warned_deferred:
                    self._warned_deferred.add(name)
                    warnings.warn(
                        f'CodeMode: tool {name!r} uses deferred loading (tool search) '
                        f'and cannot be pre-registered in the sandbox; it will be '
                        f'exposed as a native tool instead.',
                        UserWarning,
                        stacklevel=2,
                    )
                native_fallbacks.add(name)
                continue

            safe_name = _sanitize_tool_name(name)
            if safe_name == _RUN_CODE_TOOL_NAME:
                raise UserError(
                    f"Tool name '{name}' (sanitized to '{safe_name}') conflicts with the code mode "
                    f'meta-tool. Rename your tool to avoid conflicts.'
                )
            if safe_name in callable_defs:
                existing = sanitized_to_original.get(safe_name, safe_name)
                warnings.warn(
                    f'CodeMode: tool {name!r} (sanitized to {safe_name!r}) collides '
                    f'with {existing!r}; {name!r} will be hidden from the sandbox.',
                    UserWarning,
                    stacklevel=2,
                )
                continue
            # Warn when a sandboxed tool has no return schema — the generated
            # signature will show `-> Any`, giving the model no type information
            # about the return shape, which limits code mode effectiveness.
            if td.return_schema is None and name not in self._warned_deferred:
                self._warned_deferred.add(name)
                warnings.warn(
                    f'CodeMode: tool {name!r} has no return schema; '
                    f'its signature will show `-> Any`, which may reduce code mode effectiveness.',
                    UserWarning,
                    stacklevel=2,
                )

            if safe_name != name:
                sanitized_to_original[safe_name] = name
                td = replace(td, name=safe_name)

            callable_defs[safe_name] = td
        return callable_defs, sanitized_to_original, native_fallbacks

    def _build_description(self, callable_defs: dict[str, ToolDefinition]) -> str:
        """Render the `run_code` description: base prose + TypedDicts + function signatures."""
        if not callable_defs:
            return _RUN_CODE_BASE_DESCRIPTION

        sigs: list[FunctionSignature] = []
        for td in callable_defs.values():
            assert td.function_signature is not None, f'function_signature missing for tool {td.name!r}'
            sigs.append(td.function_signature)
        conflicting = FunctionSignature.get_conflicting_type_names(sigs)

        type_blocks = FunctionSignature.render_type_definitions(sigs, conflicting)
        function_blocks = [
            td.render_signature('...', is_async=True, conflicting_type_names=conflicting)
            for td in callable_defs.values()
        ]

        sections = [_RUN_CODE_BASE_DESCRIPTION, _FUNCTIONS_HEADER]
        if type_blocks:
            sections.append('```python\n' + '\n\n'.join(type_blocks) + '\n```')
        sections.append('```python\n' + '\n\n'.join(function_blocks) + '\n```')
        return '\n\n'.join(sections)


def _contains_multimodal(value: Any) -> bool:
    """Check if a value is or directly contains multimodal content (images, audio, etc.)."""
    if is_multi_modal_content(value):
        return True
    if isinstance(value, list):
        return any(is_multi_modal_content(item) for item in value)  # pyright: ignore[reportUnknownVariableType]
    return False


class _PrintCapture:
    """Accumulates print-callback chunks from `MontyRepl.feed_run_async`.

    Pulled out to module scope (rather than a closure inside `call_tool`) so the
    callback path is testable in isolation. The Rust-side `feed_run_async` invokes
    the callback from a worker that coverage.py's per-thread tracer doesn't see,
    so a closure body would never be marked as executed even though it runs.
    """

    def __init__(self) -> None:
        self._chunks: list[str] = []

    def __call__(self, _stream: str, text: str) -> None:
        self._chunks.append(text)

    @property
    def joined(self) -> str:
        return ''.join(self._chunks)


def _make_sandbox_callable(
    original_name: str,
    dispatch: Callable[..., Any],
) -> Callable[..., Any]:
    """Build the `async def` callable that Monty dispatches to for a sandbox tool.

    Thin wrapper that routes keyword args to the shared `dispatch_tool_call`
    closure defined in `call_tool`, which handles ToolManager dispatch,
    exception handling, result serialization, and nested-parts bookkeeping.
    """

    async def wrapper(**kwargs: Any) -> Any:
        return await dispatch(original_name, kwargs)

    return wrapper
