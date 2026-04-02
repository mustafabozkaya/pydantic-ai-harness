"""Tests for the ToolBudget capability."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai.agent import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage

from pydantic_harness import ToolBudget, ToolBudgetExceeded


@pytest.fixture(params=['asyncio'])
def anyio_backend(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[no-any-return]


def _text(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def _call_tool(name: str, call_id: str, args: str = '{}') -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name=name, args=args, tool_call_id=call_id)])


def _count_tool_returns(messages: list[ModelMessage]) -> int:
    return sum(1 for msg in messages for part in msg.parts if isinstance(part, ToolReturnPart))


# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------


def _repeated_tool_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Keeps calling the first available tool until it disappears, then returns text."""
    call_num = _count_tool_returns(messages) + 1
    if info.function_tools:
        return _call_tool(info.function_tools[0].name, f'c{call_num}')
    return _text('done')


def _two_tool_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Calls tool_a if available, else tool_b (up to 3 total), else done."""
    call_num = _count_tool_returns(messages) + 1
    if call_num > 5:
        return _text('done')
    tool_names = {t.name for t in info.function_tools}
    if 'tool_a' in tool_names:
        return _call_tool('tool_a', f'c{call_num}')
    if 'tool_b' in tool_names:
        return _call_tool('tool_b', f'c{call_num}')
    return _text('done')  # pragma: no cover


def _always_call_tool_a(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Always tries to call tool_a regardless of available tools."""
    n = _count_tool_returns(messages) + 1
    return _call_tool('tool_a', f'c{n}')


# ---------------------------------------------------------------------------
# Tests: total budget
# ---------------------------------------------------------------------------


async def test_total_budget_removes_tools_when_exhausted():
    cap = ToolBudget(max_total_calls=2)
    agent = Agent(FunctionModel(_repeated_tool_model), capabilities=[cap])

    @agent.tool_plain
    def my_tool() -> str:
        return 'ok'

    result = await agent.run('go')
    assert result.output == 'done'
    tool_returns = [p for msg in result.all_messages() for p in msg.parts if isinstance(p, ToolReturnPart)]
    assert len(tool_returns) == 2


async def test_total_budget_inform_returns_message():
    """When total budget is hit and model still calls a tool, it gets a message back."""
    cap = ToolBudget(max_total_calls=1, action='inform')
    agent = Agent(FunctionModel(_repeated_tool_model), capabilities=[cap])

    @agent.tool_plain
    def my_tool() -> str:
        return 'ok'

    result = await agent.run('go')
    assert result.output == 'done'
    # Only 1 real tool return (the budget allows 1 call).
    tool_returns = [p for msg in result.all_messages() for p in msg.parts if isinstance(p, ToolReturnPart)]
    assert len(tool_returns) == 1


async def test_total_budget_error_action():
    """When action='error' and budget is exceeded, ToolBudgetExceeded is raised."""
    cap = ToolBudget(max_total_calls=1, action='error')
    agent = Agent(FunctionModel(_always_call_tool_a), capabilities=[cap])

    @agent.tool_plain
    def tool_a() -> str:
        return 'ok'

    with pytest.raises(ToolBudgetExceeded, match='Total tool call budget'):
        await agent.run('go')


# ---------------------------------------------------------------------------
# Tests: per-tool budget
# ---------------------------------------------------------------------------


async def test_per_tool_limit_removes_exhausted_tool():
    cap = ToolBudget(max_per_tool={'tool_a': 1})
    agent = Agent(FunctionModel(_two_tool_model), capabilities=[cap])

    @agent.tool_plain
    def tool_a() -> str:
        return 'a'

    @agent.tool_plain
    def tool_b() -> str:
        return 'b'

    result = await agent.run('go')
    assert result.output == 'done'
    tool_a_returns = [
        p
        for msg in result.all_messages()
        for p in msg.parts
        if isinstance(p, ToolReturnPart) and p.tool_name == 'tool_a'
    ]
    assert len(tool_a_returns) == 1


async def test_per_tool_limit_does_not_affect_other_tools():
    cap = ToolBudget(max_per_tool={'tool_a': 1})
    call_seq: list[str] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_names = {t.name for t in info.function_tools}
        n = _count_tool_returns(messages) + 1
        if n <= 3:
            name = 'tool_a' if 'tool_a' in tool_names else 'tool_b'
            return _call_tool(name, f'c{n}')
        return _text('done')

    agent = Agent(FunctionModel(model_fn), capabilities=[cap])

    @agent.tool_plain
    def tool_a() -> str:
        call_seq.append('a')
        return 'a'

    @agent.tool_plain
    def tool_b() -> str:
        call_seq.append('b')
        return 'b'

    result = await agent.run('go')
    assert result.output == 'done'
    assert call_seq[0] == 'a'
    assert all(c == 'b' for c in call_seq[1:])


async def test_per_tool_error_action():
    cap = ToolBudget(max_per_tool={'tool_a': 1}, action='error')
    agent = Agent(FunctionModel(_always_call_tool_a), capabilities=[cap])

    @agent.tool_plain
    def tool_a() -> str:
        return 'ok'

    with pytest.raises(ToolBudgetExceeded, match="Tool 'tool_a'"):
        await agent.run('go')


# ---------------------------------------------------------------------------
# Tests: combined total + per-tool
# ---------------------------------------------------------------------------


async def test_per_tool_exhausted_before_total():
    cap = ToolBudget(max_total_calls=10, max_per_tool={'tool_a': 1})
    agent = Agent(FunctionModel(_two_tool_model), capabilities=[cap])

    @agent.tool_plain
    def tool_a() -> str:
        return 'a'

    @agent.tool_plain
    def tool_b() -> str:
        return 'b'

    result = await agent.run('go')
    assert result.output == 'done'
    tool_a_returns = [
        p
        for msg in result.all_messages()
        for p in msg.parts
        if isinstance(p, ToolReturnPart) and p.tool_name == 'tool_a'
    ]
    assert len(tool_a_returns) == 1


async def test_total_exhausted_removes_all_function_tools():
    cap = ToolBudget(max_total_calls=2)

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        """Alternates tool_a and tool_b so both get called."""
        n = _count_tool_returns(messages) + 1
        tool_names = {t.name for t in info.function_tools}
        if n % 2 == 1 and 'tool_a' in tool_names:
            return _call_tool('tool_a', f'c{n}')
        if 'tool_b' in tool_names:
            return _call_tool('tool_b', f'c{n}')
        return _text('done')

    agent = Agent(FunctionModel(model_fn), capabilities=[cap])

    @agent.tool_plain
    def tool_a() -> str:
        return 'a'

    @agent.tool_plain
    def tool_b() -> str:
        return 'b'

    result = await agent.run('go')
    assert result.output == 'done'
    all_returns = [p for msg in result.all_messages() for p in msg.parts if isinstance(p, ToolReturnPart)]
    assert len(all_returns) == 2


# ---------------------------------------------------------------------------
# Tests: for_run isolation
# ---------------------------------------------------------------------------


async def test_separate_runs_have_independent_counts():
    cap = ToolBudget(max_total_calls=2)
    agent = Agent(FunctionModel(_repeated_tool_model), capabilities=[cap])

    @agent.tool_plain
    def my_tool() -> str:
        return 'ok'

    r1 = await agent.run('go')
    returns_1 = [p for msg in r1.all_messages() for p in msg.parts if isinstance(p, ToolReturnPart)]
    assert len(returns_1) == 2

    r2 = await agent.run('go')
    returns_2 = [p for msg in r2.all_messages() for p in msg.parts if isinstance(p, ToolReturnPart)]
    assert len(returns_2) == 2


# ---------------------------------------------------------------------------
# Tests: prepare_tools preserves non-function tools
# ---------------------------------------------------------------------------


async def test_output_tools_not_removed_when_total_exhausted():
    cap = ToolBudget[Any](max_total_calls=0)
    ctx = RunContext[Any](deps=None, model=FunctionModel(_repeated_tool_model), usage=RunUsage())
    run_cap = await cap.for_run(ctx)
    tool_defs = [
        ToolDefinition(name='func_tool', kind='function'),
        ToolDefinition(name='out_tool', kind='output'),
    ]
    result = await run_cap.prepare_tools(ctx, tool_defs)
    names = [td.name for td in result]
    assert 'func_tool' not in names
    assert 'out_tool' in names


# ---------------------------------------------------------------------------
# Tests: defaults and edge cases
# ---------------------------------------------------------------------------


async def test_no_limits_means_unlimited():
    def bounded_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        n = _count_tool_returns(messages)
        if n >= 5:
            return _text('done')
        if info.function_tools:
            return _call_tool(info.function_tools[0].name, f'c{n + 1}')
        return _text('done')  # pragma: no cover

    agent = Agent(FunctionModel(bounded_model), capabilities=[ToolBudget()])

    @agent.tool_plain
    def tool() -> str:
        return 'ok'

    result = await agent.run('go')
    assert result.output == 'done'
    all_returns = [p for msg in result.all_messages() for p in msg.parts if isinstance(p, ToolReturnPart)]
    assert len(all_returns) == 5


def test_exception_has_tool_name():
    exc = ToolBudgetExceeded('my_tool', 'budget exceeded')
    assert exc.tool_name == 'my_tool'
    assert str(exc) == 'budget exceeded'


async def test_max_total_calls_zero_hides_all_function_tools():
    cap = ToolBudget(max_total_calls=0)

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if info.function_tools:
            return _call_tool(info.function_tools[0].name, 'c1')  # pragma: no cover
        return _text('no tools')

    agent = Agent(FunctionModel(model_fn), capabilities=[cap])

    @agent.tool_plain
    def my_tool() -> str:
        return 'ok'  # pragma: no cover

    result = await agent.run('go')
    assert result.output == 'no tools'


async def test_inform_per_tool_via_batch_tool_calls():
    """When the model calls a sequential tool multiple times in one response,
    the second call hits the inform path in wrap_tool_execute."""
    cap = ToolBudget(max_per_tool={'tool_a': 1}, action='inform')

    def batch_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for msg in messages:
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    return _text('done')
        return ModelResponse(
            parts=[
                ToolCallPart(tool_name='tool_a', args='{}', tool_call_id='c1'),
                ToolCallPart(tool_name='tool_a', args='{}', tool_call_id='c2'),
            ]
        )

    agent = Agent(FunctionModel(batch_model), capabilities=[cap])

    @agent.tool_plain(sequential=True)
    def tool_a() -> str:
        return 'ok'

    result = await agent.run('go')
    assert result.output == 'done'
    tool_returns = [p for msg in result.all_messages() for p in msg.parts if isinstance(p, ToolReturnPart)]
    assert len(tool_returns) == 2
    contents = [str(p.content) for p in tool_returns]
    assert any('Tool call budget exceeded' in c for c in contents)


async def test_inform_total_via_batch_tool_calls():
    """Total budget exceeded mid-batch with sequential tool should inform the model."""
    cap = ToolBudget(max_total_calls=1, action='inform')

    def batch_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for msg in messages:
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    return _text('done')
        return ModelResponse(
            parts=[
                ToolCallPart(tool_name='tool_a', args='{}', tool_call_id='c1'),
                ToolCallPart(tool_name='tool_a', args='{}', tool_call_id='c2'),
            ]
        )

    agent = Agent(FunctionModel(batch_model), capabilities=[cap])

    @agent.tool_plain(sequential=True)
    def tool_a() -> str:
        return 'ok'

    result = await agent.run('go')
    assert result.output == 'done'
    tool_returns = [p for msg in result.all_messages() for p in msg.parts if isinstance(p, ToolReturnPart)]
    assert len(tool_returns) == 2
    contents = [str(p.content) for p in tool_returns]
    assert any('Tool call budget exceeded' in c for c in contents)


async def test_for_run_returns_fresh_instance():
    cap = ToolBudget[Any](max_total_calls=5, max_per_tool={'a': 2}, action='error')
    ctx = RunContext[Any](deps=None, model=FunctionModel(_repeated_tool_model), usage=RunUsage())
    run_cap = await cap.for_run(ctx)
    assert run_cap is not cap
    assert run_cap.max_total_calls == 5
    assert run_cap.max_per_tool == {'a': 2}
    assert run_cap.action == 'error'
