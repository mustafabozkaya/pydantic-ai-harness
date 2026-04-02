"""Tests for the SubAgent capability."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_harness.subagent import SubAgent, _format_output, _shareable_history

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


def _make_delegate_tasks_call(tasks: list[dict[str, str]], tool_call_id: str = 'call-1') -> ModelResponse:
    """Build a ModelResponse that calls the delegate_tasks (plural) tool."""
    return ModelResponse(
        parts=[
            ToolCallPart(
                tool_name='delegate_tasks',
                args=json.dumps({'tasks': tasks}),
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
        assert 'delegate_tasks' in instructions

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
# Share history tests
# ---------------------------------------------------------------------------


_captured_history: list[list[ModelMessage] | None] = []


async def _capture_history_tool(ctx: RunContext[Any]) -> str:
    """Capture the message history on the sub-agent's RunContext."""
    _captured_history.append(list(ctx.messages))
    return 'captured'


class TestShareableHistory:
    """Test _shareable_history helper."""

    def test_strips_trailing_tool_call_response(self) -> None:
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='hi')]),
            ModelResponse(parts=[ToolCallPart(tool_name='t', args='{}', tool_call_id='c1')]),
        ]
        result = _shareable_history(messages)
        assert len(result) == 1
        assert isinstance(result[0], ModelRequest)

    def test_preserves_trailing_text_response(self) -> None:
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='hi')]),
            ModelResponse(parts=[TextPart('reply')]),
        ]
        result = _shareable_history(messages)
        assert len(result) == 2

    def test_empty_list(self) -> None:
        assert _shareable_history([]) == []


class TestSubAgentShareHistory:
    """Test share_history parameter."""

    @pytest.mark.anyio
    async def test_share_history_false_by_default(self) -> None:
        """With default share_history=False, sub-agent gets no parent history."""
        _captured_history.clear()

        sub: Agent[Any] = Agent(
            _sub_model_with_tool('capture_history'),
            description='Captures history',
            tools=[Tool(_capture_history_tool, name='capture_history')],
        )

        parent_model = _parent_model_that_delegates('sub', 'do it')
        parent: Agent[Any] = Agent(
            parent_model,
            capabilities=[SubAgent[Any](agents={'sub': sub})],
        )
        result = await parent.run('Hello parent')
        assert result.output == 'Parent final answer'
        # Sub-agent should have only its own messages (not the parent's)
        assert len(_captured_history) == 1
        sub_messages = _captured_history[0]
        assert sub_messages is not None
        # The sub-agent messages should not contain the parent's user prompt
        all_text = str(sub_messages)
        assert 'Hello parent' not in all_text

    @pytest.mark.anyio
    async def test_share_history_true(self) -> None:
        """With share_history=True, sub-agent receives parent's message history."""
        _captured_history.clear()

        sub: Agent[Any] = Agent(
            _sub_model_with_tool('capture_history'),
            description='Captures history',
            tools=[Tool(_capture_history_tool, name='capture_history')],
        )

        parent_model = _parent_model_that_delegates('sub', 'do it')
        parent: Agent[Any] = Agent(
            parent_model,
            capabilities=[SubAgent[Any](agents={'sub': sub}, share_history=True)],
        )
        result = await parent.run('Hello parent')
        assert result.output == 'Parent final answer'
        assert len(_captured_history) == 1
        sub_messages = _captured_history[0]
        assert sub_messages is not None
        # The sub-agent's history should contain the parent's original user prompt
        all_text = str(sub_messages)
        assert 'Hello parent' in all_text


# ---------------------------------------------------------------------------
# Parallel delegation tests
# ---------------------------------------------------------------------------


class TestSubAgentDelegateTasks:
    """Test the delegate_tasks (parallel) tool."""

    @pytest.mark.anyio
    async def test_delegate_tasks_parallel(self) -> None:
        """delegate_tasks runs multiple sub-agents and returns all outputs."""
        sub_a: Agent[None] = Agent(_simple_sub_model('alpha'), description='Agent A')
        sub_b: Agent[None] = Agent(_simple_sub_model('beta'), description='Agent B')

        call_count = 0

        def parent_handle(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_delegate_tasks_call(
                    [
                        {'agent': 'a', 'task': 'task for a'},
                        {'agent': 'b', 'task': 'task for b'},
                    ]
                )
            return ModelResponse(parts=[TextPart('All done parallel')])

        parent: Agent[None] = Agent(
            FunctionModel(parent_handle),
            capabilities=[SubAgent[None](agents={'a': sub_a, 'b': sub_b})],
        )
        result = await parent.run('Use both in parallel')
        assert result.output == 'All done parallel'

    @pytest.mark.anyio
    async def test_delegate_tasks_returns_results_in_order(self) -> None:
        """Results from delegate_tasks are in the same order as the input tasks."""
        sub_a: Agent[None] = Agent(_simple_sub_model('first'), description='A')
        sub_b: Agent[None] = Agent(_simple_sub_model('second'), description='B')

        call_count = 0
        captured_tool_return: list[Any] = []

        def parent_handle(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_delegate_tasks_call(
                    [
                        {'agent': 'a', 'task': 'go a'},
                        {'agent': 'b', 'task': 'go b'},
                    ]
                )
            # Capture the tool return from the messages
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart) and part.tool_name == 'delegate_tasks':
                        captured_tool_return.append(part.content)
            return ModelResponse(parts=[TextPart('Done')])

        parent: Agent[None] = Agent(
            FunctionModel(parent_handle),
            capabilities=[SubAgent[None](agents={'a': sub_a, 'b': sub_b})],
        )
        await parent.run('Go')
        assert len(captured_tool_return) == 1
        assert captured_tool_return[0] == ['first', 'second']

    @pytest.mark.anyio
    async def test_delegate_tasks_unknown_agent_triggers_retry(self) -> None:
        """If any task references an unknown agent, ModelRetry is raised."""
        sub: Agent[None] = Agent(_simple_sub_model('ok'), description='Sub')

        call_count = 0

        def parent_handle(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_delegate_tasks_call(
                    [
                        {'agent': 'nonexistent', 'task': 'bad'},
                    ]
                )
            if call_count == 2:
                return _make_delegate_tasks_call(
                    [
                        {'agent': 'sub', 'task': 'good'},
                    ]
                )
            return ModelResponse(parts=[TextPart('Recovered')])

        parent: Agent[None] = Agent(
            FunctionModel(parent_handle),
            capabilities=[SubAgent[None](agents={'sub': sub})],
        )
        result = await parent.run('Go')
        assert result.output == 'Recovered'


# ---------------------------------------------------------------------------
# Structured output tests
# ---------------------------------------------------------------------------


class _SampleModel(BaseModel):
    name: str
    value: int


class TestFormatOutput:
    """Test _format_output for structured output preservation."""

    def test_str_passthrough(self) -> None:
        assert _format_output('hello') == 'hello'

    def test_pydantic_model_json(self) -> None:
        model = _SampleModel(name='test', value=42)
        result = _format_output(model)
        parsed = json.loads(result)
        assert parsed == {'name': 'test', 'value': 42}

    def test_dict_json(self) -> None:
        result = _format_output({'key': 'val', 'num': 1})
        parsed = json.loads(result)
        assert parsed == {'key': 'val', 'num': 1}

    def test_list_json(self) -> None:
        result = _format_output([1, 2, 3])
        parsed = json.loads(result)
        assert parsed == [1, 2, 3]

    def test_other_type_repr(self) -> None:
        result = _format_output(42)
        assert result == '42'

    def test_bool_repr(self) -> None:
        result = _format_output(True)
        assert result == 'True'


class TestStructuredOutputEndToEnd:
    """Test that structured sub-agent outputs are preserved in delegation."""

    @pytest.mark.anyio
    async def test_pydantic_model_output(self) -> None:
        """A sub-agent returning a Pydantic model gets JSON-serialized."""
        sub: Agent[None, _SampleModel] = Agent(
            TestModel(custom_output_args={'name': 'sub', 'value': 99}),
            output_type=_SampleModel,
            description='Returns structured',
        )

        call_count = 0
        captured_tool_return: list[str] = []

        def parent_handle(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_delegate_call('sub', 'get data')
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart) and part.tool_name == 'delegate_task':
                        captured_tool_return.append(str(part.content))
            return ModelResponse(parts=[TextPart('Done')])

        parent: Agent[None] = Agent(
            FunctionModel(parent_handle),
            capabilities=[SubAgent[None](agents={'sub': sub})],
        )
        await parent.run('Get structured data')
        assert len(captured_tool_return) == 1
        parsed = json.loads(captured_tool_return[0])
        assert parsed == {'name': 'sub', 'value': 99}

    @pytest.mark.anyio
    async def test_dict_output(self) -> None:
        """A sub-agent returning a dict gets JSON-serialized."""
        sub: Agent[None, dict[str, int]] = Agent(
            TestModel(custom_output_args={'a': 1}),
            output_type=dict[str, int],
            description='Returns dict',
        )

        call_count = 0
        captured_tool_return: list[str] = []

        def parent_handle(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_delegate_call('sub', 'get dict')
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart) and part.tool_name == 'delegate_task':
                        captured_tool_return.append(str(part.content))
            return ModelResponse(parts=[TextPart('Done')])

        parent: Agent[None] = Agent(
            FunctionModel(parent_handle),
            capabilities=[SubAgent[None](agents={'sub': sub})],
        )
        await parent.run('Get dict data')
        assert len(captured_tool_return) == 1
        parsed = json.loads(captured_tool_return[0])
        assert parsed == {'a': 1}


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
