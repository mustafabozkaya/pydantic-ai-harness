"""Code mode toolset that runs LLM-generated Python in a Monty sandbox.

Instead of using Monty's convenience `feed_run_async` method (which spawns a
background thread and uses `call_soon_threadsafe` to dispatch async callbacks),
this module drives the Monty REPL via the synchronous `feed_start`/`resume`
snapshot API. This approach:

- **Works with Temporal**: Temporal's workflow sandbox disables threads and
  doesn't implement `call_soon_threadsafe`, so `feed_run_async` hangs.
  The snapshot approach avoids both.
- **Enables parallel tool calls**: `asyncio.gather`-style concurrency from
  sandbox code is supported via `FutureSnapshot`.
- **Improves error propagation**: exceptions from tool calls are passed back
  into the sandbox via `ExternalException`, giving Monty full context for
  error messages.
"""

from __future__ import annotations

import asyncio
import keyword
import re
import warnings
from collections.abc import Callable, Coroutine
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
    from pydantic_monty import (
        ExternalException,
        ExternalReturnValue,
        FunctionSnapshot,
        FutureSnapshot,
        Monty,
        MontyComplete,
        MontyRepl,
        MontyRuntimeError,
        MontySyntaxError,
        MontyTypingError,
        NameLookupSnapshot,
    )
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'pydantic-monty is required for CodeMode. Install it with: pip install "pydantic-harness[code-mode]"'
    ) from _import_error
from typing_extensions import NotRequired, TypedDict

# Type alias for the dispatch callback passed to _execution_loop.
_DispatchFn = Callable[[str, dict[str, Any]], Coroutine[Any, Any, Any]]


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

        if _RUN_CODE_TOOL_NAME in native_tools:
            raise UserError(
                f"Tool name '{_RUN_CODE_TOOL_NAME}' is reserved for code mode. Rename your tool to avoid conflicts."
            )

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

        # Determine freshness *before* creating the REPL so that if type
        # checking fails (raises ModelRetry), the REPL stays None and the
        # next retry still gets type-checked.
        fresh_repl = self._repl is None or restart

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

        # Determine whether sandbox tool calls should be resolved sequentially.
        # We ask the inner ToolManager which checks both the global parallel
        # execution mode context var (set by durable execution engines like
        # DBOS) and per-tool sequential flags. We use original (unsanitized)
        # tool names so ToolManager can look up their definitions.
        probe_calls = [ToolCallPart(tool_name=n, args={}) for n in tool.wrapped_tools]
        sequential = tool_manager.get_parallel_execution_mode(probe_calls) != 'parallel'

        # Collect nested tool calls and returns keyed by tool_call_id so they
        # can be attached as metadata on the run_code ToolReturnPart.
        nested_calls: dict[str, ToolCallPart] = {}
        nested_returns: dict[str, ToolReturnPart] = {}
        call_counter = 0

        async def dispatch_tool_call(original_name: str, kwargs: dict[str, Any]) -> Any:
            """Dispatch a single tool call from inside the sandbox.

            Returns the serialized tool result on success. On failure, the
            exception propagates — the execution loop passes it back into
            Monty via `ExternalException` so the sandbox sees it at the
            `await` site.
            """
            nonlocal call_counter
            call_counter += 1
            parent_id = ctx.tool_call_id or 'pyd_ai_code_mode'
            tool_call_id = f'{parent_id}__{call_counter}'
            call_part = ToolCallPart(tool_name=original_name, args=kwargs, tool_call_id=tool_call_id)
            nested_calls[tool_call_id] = call_part

            try:
                result = await tool_manager.handle_call(call_part, wrap_validation_errors=False)
            except (CallDeferred, ApprovalRequired) as e:
                # Approval/deferral require a round-trip back to the caller,
                # which the sandbox cannot do. Raise UserError so the execution
                # loop passes it into Monty as an ExternalException; Monty
                # re-raises it as MontyRuntimeError, which we catch and convert
                # to ModelRetry. The error message is preserved through the chain.
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

        # Static type checking on fresh REPL sessions (first call or after
        # restart). Skipped on subsequent calls because accumulated REPL state
        # (variables from prior snippets) is invisible to the stateless checker.
        # Runs before REPL creation so that if this raises ModelRetry, the REPL
        # stays None and the next retry still gets type-checked.
        if fresh_repl and callable_defs:
            self._type_check(code, callable_defs)

        # Create the REPL after type checking passes.
        if fresh_repl:
            self._repl = MontyRepl()
        assert self._repl is not None

        capture = _PrintCapture()

        try:
            monty_state = self._repl.feed_start(code, print_callback=capture)
            completed = await _execution_loop(
                monty_state, dispatch_tool_call, callable_defs, sanitized_to_original, sequential
            )
        except MontySyntaxError as e:
            raise ModelRetry(f'Syntax error in code:\n{_prepend_prints(e.display(), capture)}') from e
        except MontyTypingError as e:  # pragma: no cover — MontyRepl.feed_start doesn't raise this
            raise ModelRetry(f'Type error in code:\n{_prepend_prints(e.display(), capture)}') from e
        except MontyRuntimeError as e:
            # Exceptions raised inside dispatch_tool_call (e.g. UserError from
            # ApprovalRequired, or ModelRetry from a wrapped tool) are passed
            # back into Monty via ExternalException. Monty re-raises them at the
            # await site; if the sandbox code doesn't catch them, they bubble up
            # as MontyRuntimeError. The original exception message is preserved
            # in the display string, so the model sees a useful error. This means
            # ModelRetry from a wrapped tool gets double-wrapped
            # (ModelRetry → MontyRuntimeError → ModelRetry), but the retry
            # semantics are the same — the model gets another chance.
            raise ModelRetry(f'Runtime error:\n{_prepend_prints(e.display(), capture)}') from e

        result = completed.output
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
        return callable_defs, sanitized_to_original

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

    @staticmethod
    def _build_type_check_stubs(callable_defs: dict[str, ToolDefinition]) -> str:
        """Build Python stubs for Monty's static type checker.

        Used to type-check the first code snippet (or after restart) before
        execution. The stubs declare the available tool functions so Monty
        can verify argument types and catch errors before runtime.
        """
        sigs: list[FunctionSignature] = []
        for td in callable_defs.values():
            assert td.function_signature is not None, f'function_signature missing for tool {td.name!r}'
            sigs.append(td.function_signature)
        conflicting = FunctionSignature.get_conflicting_type_names(sigs)

        parts = ['import asyncio\nfrom typing import Any, TypedDict, NotRequired, Literal']
        type_blocks = FunctionSignature.render_type_definitions(sigs, conflicting)
        parts.extend(type_blocks)
        parts.extend(
            td.render_signature('raise NotImplementedError()', is_async=True, conflicting_type_names=conflicting)
            for td in callable_defs.values()
        )
        return '\n\n'.join(parts)

    @staticmethod
    def _type_check(code: str, callable_defs: dict[str, ToolDefinition]) -> None:
        """Type-check a code snippet against tool signatures before execution.

        Uses Monty's stateless type checker with function stubs. Only sound
        when the REPL has no accumulated state (first call or after restart).

        Raises:
            ModelRetry: If the code has type errors or syntax errors.
        """
        stubs = CodeModeToolset._build_type_check_stubs(callable_defs)
        try:
            Monty(code, type_check=True, type_check_stubs=stubs)
        except MontyTypingError as e:
            raise ModelRetry(f'Type error in code:\n{e.display()}') from e
        except MontySyntaxError as e:
            raise ModelRetry(f'Syntax error in code:\n{e.display()}') from e


async def _execution_loop(
    monty_state: FunctionSnapshot | FutureSnapshot | NameLookupSnapshot | MontyComplete,
    dispatch: _DispatchFn,
    callable_defs: dict[str, ToolDefinition],
    sanitized_to_original: dict[str, str],
    sequential: bool,
) -> MontyComplete:
    """Drive the Monty REPL via the snapshot API until execution completes.

    This replaces `feed_run_async` and avoids background threads and
    `call_soon_threadsafe`, making it safe to run inside restricted
    event loops (e.g. Temporal workflow sandbox).

    All tool calls are deferred (`resume(future=...)`) so Monty always
    sees a consistent async interface. The sandbox code controls parallelism
    via `await` and `asyncio.gather`.

    When Monty yields a `FutureSnapshot`, pending tasks are resolved
    either in parallel (`asyncio.gather`) or sequentially (one at a time),
    depending on the `sequential` flag. This mirrors the behavior of
    `ToolManager.get_parallel_execution_mode` in the pydantic-ai agent
    graph, where any sequential tool or a durable-execution engine like
    DBOS forces all tool calls to be serialised.
    """
    # In parallel mode, calls are eagerly scheduled as Tasks so they run
    # concurrently. In sequential mode, we store bare coroutines and only
    # await them one-at-a-time at FutureSnapshot resolution to prevent the
    # event loop from interleaving execution.
    pending: dict[int, asyncio.Task[Any] | Coroutine[Any, Any, Any]] = {}
    try:
        while not isinstance(monty_state, MontyComplete):
            if isinstance(monty_state, NameLookupSnapshot):
                # Unresolved variable name — resume without a value to raise
                # NameError in the sandbox (we don't inject names into scope).
                monty_state = monty_state.resume()
            elif isinstance(monty_state, FunctionSnapshot):
                fn_name = monty_state.function_name
                if fn_name in callable_defs:
                    if monty_state.args:
                        # Tool functions are keyword-only; positional args indicate
                        # a sandbox code error.
                        monty_state = monty_state.resume(
                            exception=TypeError(
                                f'{fn_name}() does not accept positional arguments; use keyword arguments'
                            )
                        )
                        continue
                    original_name = sanitized_to_original.get(fn_name, fn_name)
                    coro = dispatch(original_name, monty_state.kwargs)
                    if sequential:
                        # Store the bare coroutine — don't schedule it yet.
                        pending[monty_state.call_id] = coro
                    else:
                        # Eagerly schedule as a Task for concurrent execution.
                        pending[monty_state.call_id] = asyncio.ensure_future(coro)
                    monty_state = monty_state.resume(future=...)
                else:
                    # Unknown function — resume with NameError so the sandbox
                    # sees a clear error at the call site.
                    monty_state = monty_state.resume(exception=NameError(f'Unknown function: {fn_name}'))
            else:
                # FutureSnapshot — Monty is awaiting one or more deferred futures.
                pending_ids = monty_state.pending_call_ids
                if not pending_ids:  # pragma: no cover
                    monty_state = monty_state.resume(results={})
                    continue

                results: dict[int, ExternalReturnValue | ExternalException] = {}
                if sequential:
                    # Sequential mode: await coroutines one at a time to prevent
                    # the event loop from interleaving execution. Required by
                    # durable execution engines (e.g. DBOS).
                    for cid in pending_ids:
                        results[cid] = await _resolve_coro(pending.pop(cid))
                else:
                    # Parallel mode (default): gather all pending tasks.
                    pending_tasks = [pending[cid] for cid in pending_ids]
                    settled = await asyncio.gather(*pending_tasks, return_exceptions=True)
                    for cid in pending_ids:
                        del pending[cid]
                    for cid, outcome in zip(pending_ids, settled):
                        results[cid] = _settle_outcome(outcome)

                monty_state = monty_state.resume(results=results)  # pyright: ignore[reportArgumentType]
    finally:
        # Cancel any orphaned tasks (e.g. if an exception interrupted the loop
        # between deferring a FunctionSnapshot and resolving its FutureSnapshot).
        for item in pending.values():  # pragma: no cover
            if isinstance(item, asyncio.Task):
                item.cancel()
            else:
                item.close()  # Close bare coroutines to avoid RuntimeWarning

    return monty_state


async def _resolve_coro(coro: Coroutine[Any, Any, Any] | asyncio.Task[Any]) -> ExternalReturnValue | ExternalException:
    """Await a single coroutine/task and wrap the result for Monty."""
    try:
        result = await coro
    except Exception as exc:
        return ExternalException(exception=exc)
    else:
        return ExternalReturnValue(return_value=result)


def _settle_outcome(outcome: Any) -> ExternalReturnValue | ExternalException:
    """Wrap an `asyncio.gather(return_exceptions=True)` outcome for Monty."""
    if isinstance(outcome, Exception):
        return ExternalException(exception=outcome)
    if isinstance(outcome, BaseException):  # pragma: no cover
        raise outcome
    return ExternalReturnValue(return_value=outcome)


def _prepend_prints(error_message: str, capture: _PrintCapture) -> str:
    """Prepend any captured print output to an error message.

    When sandbox code prints debug output before crashing, this preserves
    that output in the error so the model can use it for debugging.
    """
    printed = capture.joined.rstrip('\n')
    if not printed:
        return error_message
    return f'[stdout before error]\n{printed}\n[/stdout before error]\n{error_message}'


def _contains_multimodal(value: Any) -> bool:
    """Check if a value is or directly contains multimodal content (images, audio, etc.)."""
    if is_multi_modal_content(value):
        return True
    if isinstance(value, list):
        return any(is_multi_modal_content(item) for item in value)  # pyright: ignore[reportUnknownVariableType]
    return False


class _PrintCapture:
    """Accumulates print-callback chunks from the Monty REPL.

    Pulled out to module scope (rather than a closure inside `call_tool`) so
    the callback path is testable in isolation and visible to coverage.py.
    """

    def __init__(self) -> None:
        self._chunks: list[str] = []

    def __call__(self, _stream: str, text: str) -> None:
        self._chunks.append(text)

    @property
    def joined(self) -> str:
        return ''.join(self._chunks)
