"""Tests for the SubAgent capability."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_harness.subagent import SubAgent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_delegate_call(agent_name: str, task: str, tool_call_id: str = 'call-1') -> ModelResponse:
    """Build a ModelResponse that calls the delegate_task tool."""
    return ModelResponse(
        parts=[
            ToolCallPart(
                tool_name='delegate_task',
                args=json.dumps({'agent_name': agent_name, 'task': task}),
                tool_call_id=tool_call_id,
            )
        ]
    )


def _parent_model_that_delegates(agent_name: str, task: str) -> FunctionModel:
    """Create a FunctionModel that delegates once, then returns a text answer."""
    call_count = 0

    def handle(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_delegate_call(agent_name, task)
        return ModelResponse(parts=[TextPart('Parent final answer')])

    return FunctionModel(handle)


def _simple_sub_model(output: str = 'sub done') -> FunctionModel:
    """A sub-agent model that always returns the given text."""
    return FunctionModel(lambda msgs, info: ModelResponse(parts=[TextPart(output)]))


def _sub_model_with_tool(tool_name: str) -> FunctionModel:
    """A sub-agent model that calls a tool on first request, then returns text."""
    call_count = 0

    def handle(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1 and info.function_tools:
            return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args='{}', tool_call_id='sub-call-1')])
        return ModelResponse(parts=[TextPart('sub done')])

    return FunctionModel(handle)


# ---------------------------------------------------------------------------
# Tools defined at module level to avoid annotation resolution issues
# ---------------------------------------------------------------------------


_captured_deps: list[Any] = []


async def _capture_deps_tool(ctx: RunContext[Any]) -> str:
    """Capture deps via RunContext."""
    _captured_deps.append(ctx.deps)
    return 'captured'


async def _check_deps_tool(ctx: RunContext[Any]) -> str:
    """Check that deps is None."""
    _captured_deps.append(ctx.deps)
    return 'checked'


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestSubAgentConstruction:
    """Test SubAgent construction and configuration."""

    def test_is_capability(self) -> None:
        agent: Agent[None] = Agent(TestModel(), description='Helper')
        cap = SubAgent[None](agents={'helper': agent})
        assert isinstance(cap, AbstractCapability)

    def test_descriptions_from_agent_description(self) -> None:
        agent: Agent[None] = Agent(TestModel(), description='Researches topics')
        cap = SubAgent[None](agents={'researcher': agent})
        assert cap._resolved_descriptions['researcher'] == 'Researches topics'  # pyright: ignore[reportPrivateUsage]

    def test_descriptions_from_agent_name(self) -> None:
        agent: Agent[None] = Agent(TestModel(), name='my-helper')
        cap = SubAgent[None](agents={'helper': agent})
        assert cap._resolved_descriptions['helper'] == 'my-helper'  # pyright: ignore[reportPrivateUsage]

    def test_descriptions_fallback(self) -> None:
        agent: Agent[None] = Agent(TestModel())
        cap = SubAgent[None](agents={'helper': agent})
        assert cap._resolved_descriptions['helper'] == 'Sub-agent: helper'  # pyright: ignore[reportPrivateUsage]

    def test_explicit_descriptions_override(self) -> None:
        agent: Agent[None] = Agent(TestModel(), description='From agent')
        cap = SubAgent[None](
            agents={'helper': agent},
            descriptions={'helper': 'Custom description'},
        )
        assert cap._resolved_descriptions['helper'] == 'Custom description'  # pyright: ignore[reportPrivateUsage]

    def test_get_serialization_name_is_none(self) -> None:
        assert SubAgent.get_serialization_name() is None

    def test_empty_agents_instructions_none(self) -> None:
        cap: SubAgent[None] = SubAgent(agents={})
        assert cap.get_instructions() is None

    def test_empty_agents_toolset_none(self) -> None:
        cap: SubAgent[None] = SubAgent(agents={})
        assert cap.get_toolset() is None


# ---------------------------------------------------------------------------
# Instructions tests
# ---------------------------------------------------------------------------


class TestSubAgentInstructions:
    """Test system prompt injection."""

    def test_instructions_list_agents(self) -> None:
        a: Agent[None] = Agent(TestModel(), description='Researches topics')
        b: Agent[None] = Agent(TestModel(), description='Writes code')
        cap = SubAgent[None](agents={'researcher': a, 'coder': b})
        instructions = cap.get_instructions()
        assert instructions is not None
        assert 'researcher' in instructions
        assert 'coder' in instructions
        assert 'delegate_task' in instructions

    def test_instructions_include_descriptions(self) -> None:
        agent: Agent[None] = Agent(TestModel(), description='Does math')
        cap = SubAgent[None](agents={'calculator': agent})
        instructions = cap.get_instructions()
        assert instructions is not None
        assert 'Does math' in instructions


# ---------------------------------------------------------------------------
# Toolset tests
# ---------------------------------------------------------------------------


class TestSubAgentToolset:
    """Test that the delegate_task tool is registered correctly."""

    def test_get_toolset_not_none(self) -> None:
        agent: Agent[None] = Agent(TestModel(), description='Helper')
        cap = SubAgent[None](agents={'helper': agent})
        toolset = cap.get_toolset()
        assert toolset is not None

    def test_tool_description_lists_agents(self) -> None:
        a: Agent[None] = Agent(TestModel(), description='Researches')
        b: Agent[None] = Agent(TestModel(), description='Codes')
        cap = SubAgent[None](agents={'researcher': a, 'coder': b})
        desc = cap._delegate_task_description()  # pyright: ignore[reportPrivateUsage]
        assert 'researcher' in desc
        assert 'coder' in desc
        assert 'Researches' in desc
        assert 'Codes' in desc


# ---------------------------------------------------------------------------
# End-to-end delegation tests
# ---------------------------------------------------------------------------


class TestSubAgentDelegation:
    """Test end-to-end delegation via agent.run."""

    @pytest.mark.anyio
    async def test_delegate_returns_sub_output(self) -> None:
        """The parent delegates to a sub-agent and gets its output as a tool result."""
        sub: Agent[None] = Agent(_simple_sub_model('hello from sub'), description='Echoes input')

        parent_model = _parent_model_that_delegates('echo', 'say hello')
        parent: Agent[None] = Agent(
            parent_model,
            capabilities=[SubAgent[None](agents={'echo': sub})],
        )
        result = await parent.run('Please delegate')
        assert result.output == 'Parent final answer'

    @pytest.mark.anyio
    async def test_unknown_agent_triggers_model_retry(self) -> None:
        """Calling delegate_task with an unknown name raises ModelRetry, then the model corrects itself."""
        sub: Agent[None] = Agent(_simple_sub_model(), description='Echoes')

        call_count = 0

        def handle(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_delegate_call('nonexistent', 'test')
            if call_count == 2:
                return _make_delegate_call('echo', 'test')
            return ModelResponse(parts=[TextPart('Done')])

        parent: Agent[None] = Agent(
            FunctionModel(handle),
            capabilities=[SubAgent[None](agents={'echo': sub})],
        )
        result = await parent.run('Go')
        assert result.output == 'Done'

    @pytest.mark.anyio
    async def test_deps_passed_to_subagent(self) -> None:
        """When pass_deps=True, the sub-agent receives the parent's deps."""
        _captured_deps.clear()

        sub: Agent[Any] = Agent(
            _sub_model_with_tool('capture_deps'),
            description='Captures deps',
            tools=[Tool(_capture_deps_tool, name='capture_deps')],
        )

        parent_model = _parent_model_that_delegates('sub', 'do it')
        parent: Agent[Any] = Agent(
            parent_model,
            capabilities=[SubAgent[Any](agents={'sub': sub}, pass_deps=True)],
        )
        result = await parent.run('Go', deps='my-dep-value')
        assert result.output == 'Parent final answer'
        assert _captured_deps == ['my-dep-value']

    @pytest.mark.anyio
    async def test_pass_deps_false(self) -> None:
        """When pass_deps=False, sub-agents receive None for deps."""
        _captured_deps.clear()

        sub: Agent[Any] = Agent(
            _sub_model_with_tool('check_deps'),
            description='Checks deps',
            tools=[Tool(_check_deps_tool, name='check_deps')],
        )

        parent_model = _parent_model_that_delegates('sub', 'check')
        parent: Agent[Any] = Agent(
            parent_model,
            capabilities=[SubAgent[Any](agents={'sub': sub}, pass_deps=False)],
        )
        result = await parent.run('Go', deps='should-not-be-passed')
        assert result.output == 'Parent final answer'
        assert _captured_deps == [None]

    @pytest.mark.anyio
    async def test_multiple_agents(self) -> None:
        """Multiple sub-agents can be registered and called sequentially."""
        sub_a: Agent[None] = Agent(_simple_sub_model('alpha'), description='Agent A')
        sub_b: Agent[None] = Agent(_simple_sub_model('beta'), description='Agent B')

        call_count = 0

        def parent_handle(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_delegate_call('a', 'task for a')
            if call_count == 2:
                return _make_delegate_call('b', 'task for b')
            return ModelResponse(parts=[TextPart('All done')])

        parent: Agent[None] = Agent(
            FunctionModel(parent_handle),
            capabilities=[SubAgent[None](agents={'a': sub_a, 'b': sub_b})],
        )
        result = await parent.run('Use both')
        assert result.output == 'All done'


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestSubAgentImport:
    """Test that SubAgent is importable from the top-level package."""

    def test_import_from_package(self) -> None:
        from pydantic_harness import SubAgent as SubAgentFromPkg

        assert SubAgentFromPkg is SubAgent

    def test_in_all(self) -> None:
        import pydantic_harness

        assert 'SubAgent' in pydantic_harness.__all__
