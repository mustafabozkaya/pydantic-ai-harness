"""Code mode toolset that runs LLM-generated Python in a Monty sandbox."""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, cast

from pydantic import Field, TypeAdapter
from pydantic_ai import AbstractToolset, RunContext, ToolDefinition, WrapperToolset
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, UserError
from pydantic_ai.function_signature import FunctionSignature
from pydantic_ai.messages import ToolCallPart, ToolReturn, ToolReturnPart
from pydantic_ai.tools import AgentDepsT, ToolSelector, matches_tool_selector
from pydantic_ai.toolsets.abstract import SchemaValidatorProt, ToolsetTool
from pydantic_monty import (
    MontyRepl,
    MontyRuntimeError,
    MontySyntaxError,
    MontyTypingError,
)
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
_RUN_CODE_ARGS_VALIDATOR = _RUN_CODE_ADAPTER.validator
_TOOL_RETURN_ADAPTER = TypeAdapter(Any)

_RUN_CODE_BASE_DESCRIPTION = """\
Write and run Python code in a sandboxed environment.

The sandbox uses Monty, a subset of Python. Key restrictions:
- **No classes**: class definitions are not supported
- **No third-party libraries**: only the standard library modules listed below can be used
- **Importable standard library modules**: `sys`, `typing`, `asyncio`, `math`, `json`, `re`, `datetime`, `os`, `pathlib`. These must be imported at the top of your snippet before use, just like in regular Python. For example: `import asyncio` then `results = await asyncio.gather(tool_one(...), tool_two(...))`.
- **No `import *`**: wildcard imports are not supported

State is preserved between calls (REPL-style). Set `restart: true` to reset state.

Returns: {
  "output": "Text printed to stdout via print() calls. Omitted if nothing was printed.",
  "result": "The value of the last expression, like a Python REPL. Omitted if the last statement is not an expression (e.g. assignment, function call with no return value)."
}\
"""

_FUNCTIONS_HEADER = """\

The following functions are available inside the sandbox. Call them directly \
(do **not** redefine or import them) and `await` the result. All parameters are keyword-only. \
All tool functions are async: invoke them with `await`, e.g. `result = await tool_name(arg=value)`. \
Calling without `await` returns an unresolved future, not the value.\
"""

_INVALID_IDENT_CHARS = re.compile(r'[^a-zA-Z0-9_]')


def _sanitize_tool_name(name: str) -> str:
    """Turn a tool name into a valid Python identifier.

    Replaces hyphens, dots, and other non-identifier characters with underscores,
    and prepends `_` if the result starts with a digit.
    """
    sanitized = _INVALID_IDENT_CHARS.sub('_', name)
    if sanitized and sanitized[0].isdigit():
        sanitized = f'_{sanitized}'
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
    are exposed as native tool calls."""

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
        sandboxed_tools: dict[str, ToolsetTool[AgentDepsT]] = {}
        native_tools: dict[str, ToolsetTool[AgentDepsT]] = {}
        for name, tool in wrapped_tools.items():
            if await matches_tool_selector(self.tool_selector, ctx, tool.tool_def):
                sandboxed_tools[name] = tool
            else:
                native_tools[name] = tool

        callable_defs, sanitized_to_original = self._partition_callable_tools(sandboxed_tools)
        description = self._build_description(callable_defs)

        # TODO: When CodeMode becomes a core Pydantic AI feature, ensure that
        # the `search_tool` injected by ToolSearchToolset is excluded from
        # code-mode-ification when `tool_selector='all'`.
        result: dict[str, ToolsetTool[AgentDepsT]] = dict(native_tools)
        result[_RUN_CODE_TOOL_NAME] = _RunCodeTool(
            toolset=self,
            tool_def=ToolDefinition(
                name=_RUN_CODE_TOOL_NAME,
                description=description,
                parameters_json_schema=_RUN_CODE_JSON_SCHEMA,
                metadata={'code_arg_name': 'code', 'code_arg_language': 'python'},
            ),
            max_retries=self.max_retries,
            args_validator=cast(SchemaValidatorProt, _RUN_CODE_ARGS_VALIDATOR),
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
        from pydantic_ai._tool_manager import ToolManager  # pyright: ignore[reportPrivateUsage]

        parent_tm = ctx.tool_manager
        tool_manager: Any = None
        if parent_tm is not None:
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
            tool_call_id = f'pai__{call_counter}'
            call_part = ToolCallPart(tool_name=original_name, args=kwargs, tool_call_id=tool_call_id)
            nested_calls[tool_call_id] = call_part

            try:
                if tool_manager is not None:
                    result = await tool_manager.handle_call(call_part, wrap_validation_errors=False)
                else:
                    # Direct dispatch for tests without an agent (ctx.tool_manager is None).
                    result = await self.wrapped.call_tool(
                        original_name, kwargs, ctx, tool.wrapped_tools[original_name]
                    )
            except (CallDeferred, ApprovalRequired) as e:
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
            return _TOOL_RETURN_ADAPTER.dump_python(result)

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
        except MontyTypingError as e:
            raise ModelRetry(f'Type error in code:\n{e.display()}') from e
        except MontyRuntimeError as e:
            raise ModelRetry(f'Runtime error:\n{e.display()}') from e

        # TODO: For stdio-based driver runtimes (e.g. Monty's current driver loop),
        # print output goes to stdout which is also used for JSON protocol parsing.
        # This will need a different capture strategy for those runtimes.
        response: dict[str, Any] = {}
        printed = capture.joined
        if printed:
            response['output'] = printed
        if result is not None:
            response['result'] = result
        return ToolReturn(
            return_value=response,
            metadata={'tool_calls': nested_calls, 'tool_returns': nested_returns},
        )

    def _partition_callable_tools(
        self, wrapped_tools: dict[str, ToolsetTool[AgentDepsT]]
    ) -> tuple[dict[str, ToolDefinition], dict[str, str]]:
        """Return tool definitions that can be called from inside the sandbox.

        Tool names that are not valid Python identifiers (e.g. MCP tools with
        hyphens or dots like `get-weather`, `api.call`) are sanitized to
        underscored forms and mapped back to their original names for dispatch.

        Tools requiring deferred execution (kind `external`/`unapproved`) are
        dropped because the sandbox cannot pause and resume for approval
        round-trips. Tools with `defer_loading=True` (tool search) are dropped
        because they are only discoverable dynamically. Both emit a one-time
        `UserWarning`.

        Returns:
            A tuple of `(callable_defs, sanitized_to_original)` where
            `sanitized_to_original` maps each sanitized name back to its
            original tool name (only for tools that were actually renamed).
        """
        callable_defs: dict[str, ToolDefinition] = {}
        sanitized_to_original: dict[str, str] = {}
        for name, tool in wrapped_tools.items():
            td = tool.tool_def
            if td.defer:
                if name not in self._warned_deferred:
                    self._warned_deferred.add(name)
                    warnings.warn(
                        f'CodeMode: tool {name!r} requires deferred execution '
                        f'(kind={td.kind!r}) and cannot be called from inside the '
                        f'sandbox; it will be hidden from run_code.',
                        UserWarning,
                        stacklevel=2,
                    )
                continue
            if td.defer_loading:
                if name not in self._warned_deferred:
                    self._warned_deferred.add(name)
                    warnings.warn(
                        f'CodeMode: tool {name!r} uses deferred loading (tool search) '
                        f'and cannot be pre-registered in the sandbox; it will be '
                        f'hidden from run_code.',
                        UserWarning,
                        stacklevel=2,
                    )
                continue

            safe_name = _sanitize_tool_name(name)
            if safe_name in callable_defs:
                existing = sanitized_to_original.get(safe_name, safe_name)
                warnings.warn(
                    f'CodeMode: tool {name!r} (sanitized to {safe_name!r}) collides '
                    f'with {existing!r}; {name!r} will be hidden from the sandbox.',
                    UserWarning,
                    stacklevel=2,
                )
                continue
            if safe_name != name:
                sanitized_to_original[safe_name] = name
                td = replace(td, name=safe_name)

            callable_defs[safe_name] = td
        return callable_defs, sanitized_to_original

    def _build_description(self, callable_defs: dict[str, ToolDefinition]) -> str:
        """Render the `run_code` description: base prose + TypedDicts + function signatures."""
        if not callable_defs:
            return _RUN_CODE_BASE_DESCRIPTION

        sigs = [cast(FunctionSignature, td.function_signature) for td in callable_defs.values()]
        conflicting = FunctionSignature.dedup_referenced_types(sigs)

        type_blocks = self._render_type_definitions(sigs, conflicting)
        function_blocks = [
            td.render_signature('...', is_async=True, conflicting_type_names=conflicting)
            for td in callable_defs.values()
        ]

        sections = [_RUN_CODE_BASE_DESCRIPTION, _FUNCTIONS_HEADER]
        if type_blocks:
            sections.append('```python\n' + '\n\n'.join(type_blocks) + '\n```')
        sections.append('```python\n' + '\n\n'.join(function_blocks) + '\n```')
        return '\n\n'.join(sections)

    @staticmethod
    def _render_type_definitions(
        sigs: list[FunctionSignature],
        conflicting: frozenset[str],
    ) -> list[str]:
        """Render unique TypedDict definitions for the function prelude.

        For types whose names conflict across tools, each is rendered under
        the owning tool's name prefix (e.g. `get_user_Address`).
        """
        unique_types = FunctionSignature.collect_unique_referenced_types(sigs)
        if not unique_types:
            return []

        # Build owner map: for each unique type that needs a prefix, find which
        # signature (tool) owns it so we can render with the right prefix.
        owner_for: dict[int, str] = {}
        for sig in sigs:
            for tsig in sig.referenced_types:
                if tsig.name in conflicting and id(tsig) not in owner_for:
                    owner_for[id(tsig)] = sig.name

        rendered: list[str] = []
        for tsig in unique_types:
            owner = owner_for.get(id(tsig))
            rendered.append(tsig.render_definition(owner_name=owner, conflicting_type_names=conflicting))
        return rendered


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
