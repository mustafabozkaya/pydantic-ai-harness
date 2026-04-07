"""Code execution toolset that runs LLM-generated Python in a Monty sandbox."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, TypedDict, cast

from pydantic import Field, TypeAdapter
from pydantic_ai import AbstractToolset, RunContext, ToolDefinition, WrapperToolset
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.function_signature import (
    FunctionSignature,
    TypeSignature,
    _render_tool_name,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets.abstract import SchemaValidatorProt, ToolsetTool
from pydantic_monty import (
    MontyRepl,
    MontyRuntimeError,
    MontySyntaxError,
    MontyTypingError,
)
from typing_extensions import NotRequired


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

_RUN_CODE_BASE_DESCRIPTION = """\
Write and run Python code in a sandboxed environment.

The sandbox uses Monty, a subset of Python. Key restrictions:
- **No classes**: class definitions are not supported
- **No third-party libraries**: only the standard library modules listed below are available
- **Available modules**: `sys`, `typing`, `asyncio`, `math`, `json`, `re`, `datetime`, `os`, `pathlib`
- **No `import *`**: wildcard imports are not supported
- **All available tool functions are async**: invoke them with `await`, e.g. `result = await tool_name(arg=value)`. Calling without `await` returns an unresolved future, not the value.

State is preserved between calls (REPL-style). Set `restart: true` to reset state.

Returns: {
  "output": "Text printed to stdout via print() calls. Omitted if nothing was printed.",
  "result": "The value of the last expression, like a Python REPL. Omitted if the last statement is not an expression (e.g. assignment, function call with no return value)."
}\
"""

_FUNCTIONS_HEADER = """\

The following functions are available inside the sandbox. Call them directly \
(do **not** redefine or import them) and `await` the result. All parameters are keyword-only.\
"""

# TODO: Sanitize tool names that aren't valid Python identifiers (e.g. MCP tools with
# hyphens/dots like `get-weather`, `api.call`) and map them back on dispatch.


@dataclass
class CodeExecutionToolset(WrapperToolset[AgentDepsT]):
    """Executes LLM-generated Python code in a Monty sandbox.

    Exposes a single `run_code` tool. Tools from the wrapped toolset are
    presented to the model as Python function signatures inside the `run_code`
    tool description and become callable from the sandbox at runtime.

    Tools that are deferred (`kind` of `external`/`unapproved`, or
    `defer_loading=True`) cannot be called from inside the sandbox and are
    dropped from the available functions; a `UserWarning` is emitted the first
    time each such tool is encountered per run.
    """

    # init=False so `replace()` in `for_run` produces a fresh instance with _repl=None,
    # giving each agent run isolated REPL state. Lazy-initialized on first call_tool.
    _repl: MontyRepl | None = field(default=None, init=False, repr=False)

    # Tracks deferred-tool names we've already warned about so we don't spam the
    # logs every step. Reset on `for_run` because each run gets a fresh instance.
    _warned_deferred: set[str] = field(default_factory=set[str], init=False, repr=False)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Return a fresh toolset instance with isolated REPL state for this agent run."""
        # `replace()` creates a new instance — _repl resets to None since it's init=False,
        # so concurrent agents sharing the same toolset don't leak state between runs.
        wrapped = await self.wrapped.for_run(ctx)
        return replace(self, wrapped=wrapped)

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Update the wrapped toolset for this step while preserving REPL state."""
        new_wrapped = await self.wrapped.for_run_step(ctx)
        if new_wrapped is self.wrapped:
            return self
        # replace() resets _repl to None since it's init=False. Without this,
        # the LLM could set x=1 in step 1, then get a NameError for x in step 2
        # just because the wrapped toolset changed between turns.
        new_self = replace(self, wrapped=new_wrapped)
        new_self._repl = self._repl
        new_self._warned_deferred = self._warned_deferred
        return new_self

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        """Return the `run_code` tool with a description that lists current callable signatures."""
        # Fetch wrapped tools here (not just at call time) so the tool description
        # the model sees lists exactly the functions that will be available when it
        # writes code. The wrapped toolset can change between steps, so we rebuild
        # the description per step rather than caching.
        wrapped_tools = await self.wrapped.get_tools(ctx)
        callable_defs = self._partition_callable_tools(wrapped_tools)
        description = self._build_description(callable_defs)

        return {
            _RUN_CODE_TOOL_NAME: ToolsetTool(
                toolset=self,
                tool_def=ToolDefinition(
                    name=_RUN_CODE_TOOL_NAME,
                    description=description,
                    parameters_json_schema=_RUN_CODE_JSON_SCHEMA,
                ),
                max_retries=3,
                args_validator=cast(SchemaValidatorProt, _RUN_CODE_ARGS_VALIDATOR),
            ),
        }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        """Execute Python code in the sandbox, dispatching any external function calls to wrapped tools."""
        code = tool_args['code']
        restart = tool_args.get('restart', False)

        if self._repl is None or restart:
            self._repl = MontyRepl()

        wrapped_tools = await self.wrapped.get_tools(ctx)
        callable_defs = self._partition_callable_tools(wrapped_tools)

        # Pass None instead of {} when there are no tools — Monty treats them differently.
        # Dispatch through `self.wrapped`, NOT through `tool.toolset`: when the wrapped
        # toolset is itself a wrapper (e.g. `CombinedToolset`), `tool.toolset` is the
        # innermost owning toolset and will assert-fail on the wrapper-typed `tool`.
        external_functions = {
            t_name: _make_async_tool_wrapper(t_name, self.wrapped, wrapped_tools[t_name], ctx)
            for t_name in callable_defs
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
        return response

    def _partition_callable_tools(self, wrapped_tools: dict[str, ToolsetTool[AgentDepsT]]) -> dict[str, ToolDefinition]:
        """Return tool definitions that can be called from inside the sandbox.

        Deferred tools (`kind` of `external`/`unapproved`, or `defer_loading=True`)
        cannot be invoked synchronously inside the sandbox so they are dropped here
        with a one-time warning per tool. Output tools are excluded by
        `get_wrapper_toolset`'s contract and never reach this method, so we trust
        the contract and don't defensively check for them.
        """
        callable_defs: dict[str, ToolDefinition] = {}
        for name, tool in wrapped_tools.items():
            td = tool.tool_def
            if td.defer or td.defer_loading:
                if name not in self._warned_deferred:
                    self._warned_deferred.add(name)
                    warnings.warn(
                        f'CodeMode does not currently support deferred tool {name!r} '
                        f'(kind={td.kind!r}, defer_loading={td.defer_loading}); it will '
                        f'be hidden from the sandbox.',
                        UserWarning,
                        stacklevel=2,
                    )
                continue
            callable_defs[name] = td
        return callable_defs

    def _build_description(self, callable_defs: dict[str, ToolDefinition]) -> str:
        """Render the `run_code` description: base prose + TypedDicts + function signatures."""
        if not callable_defs:
            return _RUN_CODE_BASE_DESCRIPTION

        # `function_signature` is guaranteed non-None for non-output tools by
        # `ToolDefinition.__post_init__`, and output tools are filtered out by the
        # `get_wrapper_toolset` contract before they reach us — see `_partition_callable_tools`.
        sigs = [cast(FunctionSignature, td.function_signature) for td in callable_defs.values()]

        # `dedup_referenced_types` mutates the signature objects in place. The signature
        # objects are owned by the (cached) `ToolDefinition.function_signature` field, so
        # successive calls to `_build_description` see types already deduped — that's
        # idempotent and safe.
        FunctionSignature.dedup_referenced_types(sigs)

        type_blocks = _render_type_definitions(callable_defs)
        # The wrappers we hand to Monty are always `async def`, so the rendered signature
        # must say `async def` too — otherwise the LLM writes `tool_name(...)` without
        # `await` and gets back an unresolved future instead of the value.
        function_blocks = [td.render_signature('...', is_async=True) for td in callable_defs.values()]

        sections = [_RUN_CODE_BASE_DESCRIPTION, _FUNCTIONS_HEADER]
        if type_blocks:
            sections.append('```python\n' + '\n\n'.join(type_blocks) + '\n```')
        sections.append('```python\n' + '\n\n'.join(function_blocks) + '\n```')
        return '\n\n'.join(sections)


def _render_type_definitions(callable_defs: dict[str, ToolDefinition]) -> list[str]:
    """Render each unique referenced `TypedDict` definition for the function prelude.

    Why this lives here and not in `pydantic_ai.function_signature`:

    `FunctionSignature.collect_unique_referenced_types(...)` upstream gives us the
    deduped list of types but doesn't know which tool to attribute each type to,
    which we need for `needs_prefix=True` types — those resolve their final class
    name (e.g. `get_company_Address`) from the `_render_tool_name` ContextVar at
    render time. Walking the per-tool list ourselves lets us:

    1. Use the upstream helper for dedup-by-id (single source of truth).
    2. Build an owner map so each prefixed type renders with the correct prefix.

    A "render the whole prelude" helper would fit naturally on `FunctionSignature`
    upstream — worth pushing back into PR #4964 as a follow-up — but for now this
    keeps the demo unblocked without depending on more API surface from pydantic-ai.
    """
    sigs_by_owner = [(name, cast(FunctionSignature, td.function_signature)) for name, td in callable_defs.items()]

    # Build owner map for prefixed types: first owner wins, matching the order
    # `dedup_referenced_types` walks signatures in.
    owner_for: dict[int, str] = {}
    for tool_name, sig in sigs_by_owner:
        for tsig in sig.referenced_types:
            if tsig.needs_prefix and id(tsig) not in owner_for:
                owner_for[id(tsig)] = tool_name

    unique_types = FunctionSignature.collect_unique_referenced_types([sig for _, sig in sigs_by_owner])

    rendered: list[str] = []
    for tsig in unique_types:
        if tsig.needs_prefix:
            owner = owner_for[id(tsig)]
            rendered.append(_render_with_owner(tsig, owner))
        else:
            rendered.append(tsig.render_definition())
    return rendered


def _render_with_owner(tsig: TypeSignature, owner: str) -> str:
    """Render a `TypeSignature` definition under a specific owning tool name.

    Sets the `_render_tool_name` ContextVar so `display_name` resolves the
    prefixed form (e.g. `get_company_Address`). The ContextVar is the only public
    handle for prefix resolution upstream, even though it's underscore-prefixed.
    """
    token = _render_tool_name.set(owner)
    try:
        return tsig.render_definition()
    finally:
        _render_tool_name.reset(token)


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


def _make_async_tool_wrapper(
    tool_name: str,
    outer_toolset: AbstractToolset[Any],
    outer_tool: ToolsetTool[Any],
    ctx: RunContext[Any],
) -> Callable[..., Any]:
    """Build the `async def` wrapper that Monty will dispatch to for `tool_name`.

    The wrapper is always async because `AbstractToolset.call_tool` is async, even
    when the underlying tool is a plain sync function. Monty's dispatch loop sees
    the returned coroutine, schedules it as a future, and the LLM-side code awaits
    it — which is why the rendered signatures use `async def` and the description
    tells the LLM to use `await`.

    We dispatch through `outer_toolset.call_tool(...)` rather than
    `outer_tool.toolset.call_tool(...)` because `outer_tool.toolset` may be the
    innermost owning toolset, while `outer_tool` itself is a wrapper-typed tool
    minted by an outer wrapper (e.g. `_CombinedToolsetTool` from `CombinedToolset`).
    Calling the inner toolset with a wrapper-typed tool trips an isinstance assert
    inside upstream toolset implementations like `_AgentFunctionToolset.call_tool`.
    """

    async def wrapper(**kwargs: Any) -> Any:
        return await outer_toolset.call_tool(tool_name, kwargs, ctx, outer_tool)

    return wrapper
