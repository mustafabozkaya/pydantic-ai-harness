"""Tests for the `CodeMode` capability and `CodeExecutionToolset`.

Style follows `pydantic_ai/tests/test_toolsets.py`: module-level
`pytestmark = pytest.mark.anyio`, an `anyio_backend` fixture, async tests, and a
`build_run_context` factory. The `anyio` package's pytest plugin is already
loaded by the project (no extra dev dependency needed).
"""

from __future__ import annotations

from typing import Any, TypedDict, TypeVar

import pytest
from pydantic_ai import (
    AbstractToolset,
    Agent,
    CombinedToolset,
    RunContext,
    Tool,
    ToolDefinition,
)
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.usage import RunUsage
from pydantic_core import SchemaValidator, core_schema
from pydantic_monty import Monty, MontyRepl, MontyTypingError

from pydantic_harness.capabilities import CodeMode
from pydantic_harness.toolsets import CodeExecutionToolset
from pydantic_harness.toolsets.code_execution.run_code import (
    _PrintCapture,  # pyright: ignore[reportPrivateUsage]
)

pytestmark = pytest.mark.anyio

T = TypeVar('T')


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend (matching upstream pydantic-ai)."""
    return 'asyncio'


def build_run_context(deps: T, run_step: int = 0) -> RunContext[T]:
    """Build a `RunContext` for invoking toolsets directly in tests.

    Mirrors the helper at `pydantic_ai/tests/test_toolsets.py`.
    """
    return RunContext[T](
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=run_step,
    )


# ---------------------------------------------------------------------------
# Sample tool functions used by tests
# ---------------------------------------------------------------------------


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def greet(name: str, greeting: str = 'Hello') -> str:
    """Greet someone."""
    return f'{greeting}, {name}!'


class Address(TypedDict):
    """A simple postal address."""

    street: str
    city: str


class Person(TypedDict):
    """A person with a home address."""

    name: str
    home: Address


def lookup_person(person: Person, count: int = 1) -> str:
    """Look up details for a person."""
    return f'{count}x {person["name"]} @ {person["home"]["street"]}'


# Hand-built `ToolDefinition` objects + a tiny stub toolset are used by
# `test_conflicting_typed_dicts_get_tool_name_prefix` to exercise the
# `needs_prefix=True` rendering path. Going through Pydantic's JSON schema generator
# would not produce a true `$def`-key collision (Pydantic disambiguates `$def` keys
# by Python class identity even when `__name__` matches), so we build the schemas by
# hand and feed them through a fake toolset.


def _make_address_tool_def(name: str, description: str, addr_field: str) -> ToolDefinition:
    """Build a `ToolDefinition` whose `$defs` contains an `Address` type with one field."""
    return ToolDefinition(
        name=name,
        description=description,
        parameters_json_schema={
            'type': 'object',
            '$defs': {
                'Address': {
                    'type': 'object',
                    'title': 'Address',
                    'properties': {addr_field: {'type': 'string'}},
                    'required': [addr_field],
                },
            },
            'properties': {
                'addr': {'$ref': '#/$defs/Address'},
                'label': {'type': 'string'},
            },
            'required': ['addr', 'label'],
        },
    )


class _StaticToolset(AbstractToolset[None]):
    """A minimal `AbstractToolset` that returns a fixed set of `ToolDefinition`s.

    Mirrors the `MockToolsetWithInstructions` pattern from `pydantic_ai/tests/test_toolsets.py`.
    Used by tests that need to construct hand-crafted `ToolDefinition`s without going
    through the function-introspection pipeline.
    """

    def __init__(self, tool_defs: list[ToolDefinition], results: dict[str, Any] | None = None) -> None:
        self._tool_defs = tool_defs
        self._results = results or {}

    @property
    def id(self) -> str | None:
        return None  # pragma: no cover - required by AbstractToolset, never read in tests

    async def get_tools(self, ctx: RunContext[None]) -> dict[str, ToolsetTool[None]]:
        return {
            td.name: ToolsetTool(
                toolset=self,
                tool_def=td,
                max_retries=1,
                args_validator=_ANY_VALIDATOR,
            )
            for td in self._tool_defs
        }

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[None],
        tool: ToolsetTool[None],
    ) -> Any:
        # Tests always set up `_results` for every tool name they invoke; the
        # fallback exists only to keep the abstract contract satisfied.
        return self._results[name]


_ANY_VALIDATOR = SchemaValidator(schema=core_schema.any_schema())


def _build_function_toolset(*tools: Any) -> FunctionToolset[None]:
    return FunctionToolset[None](tools=[Tool(t) for t in tools])


# ---------------------------------------------------------------------------
# `tools='all'` (default) behaviour
# ---------------------------------------------------------------------------


async def test_default_wraps_all_tools_behind_run_code() -> None:
    """`CodeMode()` exposes only `run_code` and renders every tool as an `async def`."""
    toolset = _build_function_toolset(add, greet)
    wrapper = CodeMode[None]().get_wrapper_toolset(toolset)
    assert isinstance(wrapper, CodeExecutionToolset)

    tools = await wrapper.get_tools(build_run_context(None))
    assert list(tools.keys()) == ['run_code']

    description = tools['run_code'].tool_def.description
    assert description is not None
    assert 'async def add(*, a: int, b: int) -> int' in description
    assert 'async def greet(*, name: str, greeting: str' in description
    assert '"""Add two numbers."""' in description
    # The base description must tell the model to await tool calls.
    assert 'await' in description


async def test_run_code_executes_call_through_monty() -> None:
    """End-to-end: `run_code` runs Python in Monty and dispatches to a sync wrapped tool."""
    toolset = _build_function_toolset(add)
    wrapper = CodeMode[None]().get_wrapper_toolset(toolset)
    assert isinstance(wrapper, CodeExecutionToolset)

    ctx = build_run_context(None)
    tools = await wrapper.get_tools(ctx)
    result = await wrapper.call_tool(
        'run_code',
        {'code': 'print(await add(a=2, b=3))'},
        ctx,
        tools['run_code'],
    )
    assert result == {'output': '5\n'}


async def test_run_code_executes_string_returning_tool_with_default_arg() -> None:
    """End-to-end: a string-returning tool with a default arg is callable from the sandbox.

    Exercises (a) string return values flowing back through the await/dispatch loop,
    (b) default-argument handling — the LLM-side code only passes `name`, not `greeting`.
    """
    wrapper = CodeMode[None]().get_wrapper_toolset(_build_function_toolset(greet))
    assert isinstance(wrapper, CodeExecutionToolset)
    ctx = build_run_context(None)
    tools = await wrapper.get_tools(ctx)
    result = await wrapper.call_tool(
        'run_code',
        {'code': "print(await greet(name='Alice'))"},
        ctx,
        tools['run_code'],
    )
    assert result == {'output': 'Hello, Alice!\n'}


async def test_run_code_can_chain_multiple_tool_calls_in_one_snippet() -> None:
    """A realistic LLM snippet that calls two tools in one `run_code` invocation."""
    wrapper = CodeMode[None]().get_wrapper_toolset(_build_function_toolset(add, greet))
    assert isinstance(wrapper, CodeExecutionToolset)
    ctx = build_run_context(None)
    tools = await wrapper.get_tools(ctx)
    code = "total = await add(a=2, b=3)\nmsg = await greet(name=str(total), greeting='Result is')\nprint(msg)"
    result = await wrapper.call_tool('run_code', {'code': code}, ctx, tools['run_code'])
    assert result == {'output': 'Result is, 5!\n'}


async def test_run_code_renders_no_arg_tool_signature() -> None:
    """A no-argument tool renders as `async def name() -> ...` (without `(*, ...)`).

    Covers the empty-params branch of `FunctionSignature._render` and verifies the
    no-args path through Monty round-trips correctly.
    """

    def now_iso() -> str:
        """Return a fake fixed timestamp."""
        return '2026-04-08T12:00:00Z'

    wrapper = CodeMode[None]().get_wrapper_toolset(_build_function_toolset(now_iso))
    assert isinstance(wrapper, CodeExecutionToolset)
    ctx = build_run_context(None)
    tools = await wrapper.get_tools(ctx)

    description = tools['run_code'].tool_def.description
    assert description is not None
    # Note the lack of `(*, ...)` — empty params render as `()`.
    assert 'async def now_iso() -> str' in description
    assert 'async def now_iso(*' not in description

    result = await wrapper.call_tool(
        'run_code',
        {'code': 'print(await now_iso())'},
        ctx,
        tools['run_code'],
    )
    assert result == {'output': '2026-04-08T12:00:00Z\n'}


async def test_run_code_state_persists_between_calls() -> None:
    """REPL state must survive across consecutive `run_code` calls within a run."""
    wrapper = CodeMode[None]().get_wrapper_toolset(_build_function_toolset(add))
    assert isinstance(wrapper, CodeExecutionToolset)

    ctx = build_run_context(None)
    tools = await wrapper.get_tools(ctx)
    run_code = tools['run_code']

    first = await wrapper.call_tool('run_code', {'code': 'x = await add(a=1, b=2)'}, ctx, run_code)
    assert first == {}  # assignment, no output, no expression result
    second = await wrapper.call_tool('run_code', {'code': 'print(x * 10)'}, ctx, run_code)
    assert second == {'output': '30\n'}


async def test_run_code_restart_resets_repl_state() -> None:
    """Passing `restart=True` clears any previously-set names in the sandbox."""
    wrapper = CodeMode[None]().get_wrapper_toolset(_build_function_toolset(add))
    assert isinstance(wrapper, CodeExecutionToolset)

    ctx = build_run_context(None)
    tools = await wrapper.get_tools(ctx)
    run_code = tools['run_code']

    await wrapper.call_tool('run_code', {'code': 'x = 99'}, ctx, run_code)
    # After restart, `x` should no longer exist — Monty surfaces this as a NameError
    # which the toolset translates into a `ModelRetry`.
    with pytest.raises(ModelRetry, match=r"name 'x' is not defined"):
        await wrapper.call_tool('run_code', {'code': 'print(x)', 'restart': True}, ctx, run_code)


async def test_run_code_returns_last_expression_value() -> None:
    """When the last statement is an expression, its value is returned in `result`."""
    wrapper = CodeMode[None]().get_wrapper_toolset(_build_function_toolset(add))
    assert isinstance(wrapper, CodeExecutionToolset)
    ctx = build_run_context(None)
    tools = await wrapper.get_tools(ctx)
    result = await wrapper.call_tool('run_code', {'code': '1 + 2'}, ctx, tools['run_code'])
    assert result == {'result': 3}


async def test_run_code_syntax_error_becomes_model_retry() -> None:
    """A Python syntax error is surfaced as `ModelRetry` so the model can fix it."""
    wrapper = CodeMode[None]().get_wrapper_toolset(_build_function_toolset(add))
    assert isinstance(wrapper, CodeExecutionToolset)
    ctx = build_run_context(None)
    tools = await wrapper.get_tools(ctx)
    with pytest.raises(ModelRetry, match=r'Syntax error in code'):
        await wrapper.call_tool('run_code', {'code': 'def ('}, ctx, tools['run_code'])


async def test_run_code_typing_error_becomes_model_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `MontyTypingError` raised by the REPL is translated into a `ModelRetry`.

    `MontyRepl.feed_run_async` does not currently raise `MontyTypingError` itself —
    type checking lives on `Monty.type_check()` — so we mint a real
    `MontyTypingError` instance via `Monty(...).type_check()` and monkey-patch
    `MontyRepl.feed_run_async` to re-raise it. This protects the harness's error
    translation logic against future regressions if upstream Monty starts raising
    typing errors from the REPL itself, or if we add type checking on top.
    """
    # Mint a real `MontyTypingError` from upstream — the class can't be constructed
    # directly from Python because it's a Rust-side exception type.
    real_typing_error: MontyTypingError | None = None
    try:
        Monty('"hello" + 1').type_check()
    except MontyTypingError as e:
        real_typing_error = e
    assert real_typing_error is not None, 'failed to elicit a real MontyTypingError to inject'

    async def _raise_typing_error(self: MontyRepl, code: str, **kwargs: Any) -> Any:
        del self, code, kwargs  # Unused — we always raise.
        raise real_typing_error

    monkeypatch.setattr(MontyRepl, 'feed_run_async', _raise_typing_error)

    wrapper = CodeMode[None]().get_wrapper_toolset(_build_function_toolset(add))
    assert isinstance(wrapper, CodeExecutionToolset)
    ctx = build_run_context(None)
    tools = await wrapper.get_tools(ctx)

    with pytest.raises(ModelRetry, match=r'Type error in code') as exc_info:
        await wrapper.call_tool(
            'run_code',
            {'code': '"hello" + 1'},
            ctx,
            tools['run_code'],
        )
    # The retry message should embed Monty's own diagnostic so the model sees the
    # exact line/column information.
    assert 'unsupported-operator' in str(exc_info.value)


# ---------------------------------------------------------------------------
# `for_run` / `for_run_step` lifecycle
# ---------------------------------------------------------------------------


async def test_for_run_returns_fresh_instance_with_cleared_repl() -> None:
    """`for_run` must hand back a new toolset instance — concurrent runs cannot share REPL state."""
    wrapper = CodeMode[None]().get_wrapper_toolset(_build_function_toolset(add))
    assert isinstance(wrapper, CodeExecutionToolset)
    ctx = build_run_context(None)

    # Force lazy REPL creation on the *original* instance.
    tools = await wrapper.get_tools(ctx)
    await wrapper.call_tool('run_code', {'code': 'x = 1'}, ctx, tools['run_code'])
    assert wrapper._repl is not None  # pyright: ignore[reportPrivateUsage]

    fresh = await wrapper.for_run(ctx)
    assert isinstance(fresh, CodeExecutionToolset)
    assert fresh is not wrapper
    assert fresh._repl is None  # pyright: ignore[reportPrivateUsage]


async def test_for_run_step_short_circuits_when_wrapped_unchanged() -> None:
    """If the inner toolset doesn't change between steps, `for_run_step` returns `self` unchanged."""
    wrapper = CodeMode[None]().get_wrapper_toolset(_build_function_toolset(add))
    assert isinstance(wrapper, CodeExecutionToolset)
    ctx = build_run_context(None)
    same = await wrapper.for_run_step(ctx)
    assert same is wrapper


async def test_for_run_step_preserves_repl_when_wrapped_changes() -> None:
    """When the wrapped toolset changes between steps, REPL state must carry over to the new instance."""

    class _SwappingToolset(AbstractToolset[None]):
        """Returns a *different* underlying toolset on each `for_run_step` call."""

        def __init__(self) -> None:
            self._inner = _build_function_toolset(add)
            self._step = 0

        @property
        def id(self) -> str | None:
            return None  # pragma: no cover - required by AbstractToolset, never read

        async def get_tools(self, ctx: RunContext[None]) -> dict[str, ToolsetTool[None]]:
            return await self._inner.get_tools(ctx)

        async def call_tool(  # pragma: no cover - test only exercises lifecycle methods, not call_tool
            self,
            name: str,
            tool_args: dict[str, Any],
            ctx: RunContext[None],
            tool: ToolsetTool[None],
        ) -> Any:
            return await self._inner.call_tool(name, tool_args, ctx, tool)

        async def for_run_step(self, ctx: RunContext[None]) -> AbstractToolset[None]:
            # Return a brand-new toolset on every step so `is` comparison fails in
            # `CodeExecutionToolset.for_run_step`, forcing the rebuild branch.
            self._step += 1
            new_self = _SwappingToolset()
            new_self._step = self._step
            return new_self

    wrapper = CodeMode[None]().get_wrapper_toolset(_SwappingToolset())
    assert isinstance(wrapper, CodeExecutionToolset)
    ctx = build_run_context(None)

    # Lazily create the REPL on the original instance.
    tools = await wrapper.get_tools(ctx)
    await wrapper.call_tool('run_code', {'code': 'x = 7'}, ctx, tools['run_code'])
    original_repl = wrapper._repl  # pyright: ignore[reportPrivateUsage]
    assert original_repl is not None

    next_step = await wrapper.for_run_step(ctx)
    assert isinstance(next_step, CodeExecutionToolset)
    assert next_step is not wrapper
    # State carries over so the LLM doesn't lose its variables between steps.
    assert next_step._repl is original_repl  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# Filter behaviour
# ---------------------------------------------------------------------------


async def test_filter_keeps_rejected_tools_native() -> None:
    """A callable filter sandboxes accepted tools and leaves the rest visible to the model."""
    capability = CodeMode[None](tools=lambda ctx, td: td.name == 'add')
    wrapper = capability.get_wrapper_toolset(_build_function_toolset(add, greet))
    assert isinstance(wrapper, CombinedToolset)

    tools = await wrapper.get_tools(build_run_context(None))
    assert sorted(tools.keys()) == ['greet', 'run_code']

    description = tools['run_code'].tool_def.description
    assert description is not None
    assert 'async def add(*, a: int, b: int)' in description
    # `greet` is exposed natively, so it must NOT appear inside the run_code description
    assert 'async def greet' not in description


async def test_filter_excluding_everything_yields_run_code_with_no_functions() -> None:
    """A filter that rejects every tool produces a `run_code` with no functions block."""
    capability = CodeMode[None](tools=lambda ctx, td: False)
    wrapper = capability.get_wrapper_toolset(_build_function_toolset(add, greet))
    assert isinstance(wrapper, CombinedToolset)

    tools = await wrapper.get_tools(build_run_context(None))
    assert sorted(tools.keys()) == ['add', 'greet', 'run_code']

    description = tools['run_code'].tool_def.description
    assert description is not None
    assert 'functions are available inside the sandbox' not in description


async def test_filter_uses_run_context_for_dynamic_decisions() -> None:
    """The filter receives the live `RunContext` so it can vary per run/step."""
    seen_steps: list[int] = []

    def filter_func(ctx: RunContext[None], td: Any) -> bool:
        seen_steps.append(ctx.run_step)
        return td.name == 'add'

    wrapper = CodeMode[None](tools=filter_func).get_wrapper_toolset(_build_function_toolset(add, greet))
    assert isinstance(wrapper, CombinedToolset)
    await wrapper.get_tools(build_run_context(None, run_step=7))
    assert 7 in seen_steps


# ---------------------------------------------------------------------------
# TypedDict prelude rendering
# ---------------------------------------------------------------------------


async def test_typed_dict_arguments_render_as_prelude() -> None:
    """Tools with structured (TypedDict) parameters render their types in the prelude."""
    wrapper = CodeMode[None]().get_wrapper_toolset(_build_function_toolset(lookup_person))
    assert isinstance(wrapper, CodeExecutionToolset)

    description = (await wrapper.get_tools(build_run_context(None)))['run_code'].tool_def.description
    assert description is not None
    # Type prelude
    assert 'class Address(TypedDict):' in description
    assert 'street: str' in description
    assert 'class Person(TypedDict):' in description
    assert 'home: Address' in description
    # Function signature references the TypedDict
    assert 'async def lookup_person(*, person: Person, count: int = 1) -> str' in description


async def test_typed_dict_argument_round_trips_through_monty() -> None:
    """End-to-end with a structured argument: dict literal flows through Monty into the tool."""
    wrapper = CodeMode[None]().get_wrapper_toolset(_build_function_toolset(lookup_person))
    assert isinstance(wrapper, CodeExecutionToolset)

    ctx = build_run_context(None)
    tools = await wrapper.get_tools(ctx)
    code = (
        "addr = {'street': '1 Main St', 'city': 'NYC'}\n"
        "p = {'name': 'Alice', 'home': addr}\n"
        'print(await lookup_person(person=p, count=3))'
    )
    result = await wrapper.call_tool('run_code', {'code': code}, ctx, tools['run_code'])
    assert result == {'output': '3x Alice @ 1 Main St\n'}


async def test_conflicting_typed_dicts_get_tool_name_prefix() -> None:
    """Two tools whose `$defs` collide on `Address` get tool-name prefixes in the prelude."""
    user_td = _make_address_tool_def('get_user', 'Get a user.', 'street')
    company_td = _make_address_tool_def('get_company', 'Get a company.', 'country')
    static = _StaticToolset(
        [user_td, company_td],
        results={'get_user': 'user-result', 'get_company': 'company-result'},
    )

    wrapper = CodeMode[None]().get_wrapper_toolset(static)
    assert isinstance(wrapper, CodeExecutionToolset)

    ctx = build_run_context(None)
    tools = await wrapper.get_tools(ctx)
    description = tools['run_code'].tool_def.description
    assert description is not None
    # First-encountered `Address` keeps its bare name, the second is prefixed by its owning tool.
    assert 'class Address(TypedDict):' in description
    assert 'class get_company_Address(TypedDict):' in description
    assert 'addr: Address' in description
    assert 'addr: get_company_Address' in description

    # End-to-end through Monty: both tools are callable from inside the sandbox.
    result = await wrapper.call_tool(
        'run_code',
        {
            'code': (
                "u = await get_user(addr={'street': 'main'}, label='u')\n"
                "c = await get_company(addr={'country': 'usa'}, label='c')\n"
                'print(u, c)'
            ),
        },
        ctx,
        tools['run_code'],
    )
    assert result == {'output': 'user-result company-result\n'}


# ---------------------------------------------------------------------------
# Deferred tools
# ---------------------------------------------------------------------------


async def test_deferred_tools_are_dropped_with_one_time_warning() -> None:
    """Tools with `defer_loading=True` are excluded from the sandbox; warning fires once per run."""

    def later() -> str:
        """A deferred tool."""
        return 'later'  # pragma: no cover - deferred tools are filtered out and never invoked

    toolset = FunctionToolset[None](tools=[Tool(add), Tool(later, defer_loading=True)])
    wrapper = CodeMode[None]().get_wrapper_toolset(toolset)
    assert isinstance(wrapper, CodeExecutionToolset)

    ctx = build_run_context(None)
    with pytest.warns(UserWarning, match=r"deferred tool 'later'"):
        tools = await wrapper.get_tools(ctx)

    description = tools['run_code'].tool_def.description
    assert description is not None
    assert 'async def add' in description
    assert 'async def later' not in description

    # Second `get_tools` call must not warn again — the set is preserved across calls
    # within the same toolset instance.
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter('error')
        await wrapper.get_tools(ctx)


# ---------------------------------------------------------------------------
# Agent.run end-to-end (with FunctionModel hand-driving the model output)
# ---------------------------------------------------------------------------


async def test_code_mode_via_agent_run_executes_run_code_and_returns_result() -> None:
    """End-to-end through `Agent.run`: a `FunctionModel` issues a `run_code` call, the
    sandbox dispatches to a wrapped tool, and the second model turn observes the
    tool's return value before producing the final text output.
    """
    from pydantic_ai.messages import (
        ModelMessage,
        ModelRequest,
        ModelResponse,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
    )
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    observed_tool_calls: list[str] = []
    observed_tool_returns: list[Any] = []
    seen_tool_definitions: list[list[str]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Snapshot what tool definitions the model is being shown each turn —
        # if `CodeMode` is wired correctly the model only ever sees `run_code`.
        seen_tool_definitions.append([td.name for td in info.function_tools])

        # First turn: issue a `run_code` call that calls the wrapped `add` tool
        # through the sandbox.
        if not observed_tool_calls:
            code = 'result = await add(a=4, b=6)\nprint(f"add returned {result}")\nresult'
            observed_tool_calls.append(code)
            return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': code})])

        # Second turn: pull the `run_code` return value out of the most recent
        # ModelRequest (which is the one Pydantic AI just appended after dispatch).
        last_request = messages[-1]
        assert isinstance(last_request, ModelRequest)
        run_code_return = next(
            p for p in last_request.parts if isinstance(p, ToolReturnPart) and p.tool_name == 'run_code'
        )
        observed_tool_returns.append(run_code_return.content)
        return ModelResponse(parts=[TextPart(f'sum is {observed_tool_returns[-1]["result"]}')])

    agent: Agent[None, str] = Agent(FunctionModel(model_fn), capabilities=[CodeMode[None]()])

    @agent.tool_plain
    def add(a: int, b: int) -> int:  # pyright: ignore[reportUnusedFunction]
        """Add two numbers."""
        return a + b

    result = await agent.run('please add 4 and 6')

    # The model was shown only `run_code` — the wrapped `add` tool is hidden behind it.
    assert seen_tool_definitions[0] == ['run_code']
    assert seen_tool_definitions[1] == ['run_code']

    # The first turn issued exactly the code we expected and the sandbox returned
    # both the printed output and the value of the trailing expression.
    assert len(observed_tool_calls) == 1
    assert len(observed_tool_returns) == 1
    assert observed_tool_returns[0] == {'output': 'add returned 10\n', 'result': 10}

    # The agent's final output reflects the value flowing through the sandbox.
    assert result.output == 'sum is 10'


# ---------------------------------------------------------------------------
# Capability registration
# ---------------------------------------------------------------------------


async def test_code_mode_can_be_registered_as_agent_capability() -> None:
    """`CodeMode` can be passed via `Agent(capabilities=[...])` without raising."""
    Agent(TestModel(), capabilities=[CodeMode[None]()])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def test_print_capture_concatenates_chunks_in_order() -> None:
    """`_PrintCapture` accumulates print-callback chunks and joins them on read.

    Lives in the production module rather than as a closure inside `call_tool` so
    coverage.py sees it execute even when Monty's Rust-side worker thread bypasses
    the per-thread tracer hooks. This unit test exercises it directly.
    """
    capture = _PrintCapture()
    assert capture.joined == ''
    capture('stdout', 'hello')
    capture('stdout', ' ')
    capture('stdout', 'world\n')
    assert capture.joined == 'hello world\n'
