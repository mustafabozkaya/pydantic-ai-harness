"""Tests for the `OutputGuard` capability."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext

from pydantic_ai_harness import GuardResult, OutputBlocked, OutputGuard
from pydantic_ai_harness.guardrails import GuardrailError

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


class TestOutputGuard:
    """Integration tests for the `OutputGuard` capability driven through `Agent.run`."""

    async def test_allows_safe_output(self):
        agent = Agent(
            TestModel(custom_output_text='harmless reply'),
            capabilities=[OutputGuard[None](guard=lambda out: 'SSN' not in str(out))],
        )
        result = await agent.run('hello')
        assert result.output == 'harmless reply'

    async def test_blocks_unsafe_output_with_default_message(self):
        agent = Agent(
            TestModel(custom_output_text='leaks SSN 123-45-6789'),
            capabilities=[OutputGuard[None](guard=lambda out: 'SSN' not in str(out))],
        )
        with pytest.raises(OutputBlocked, match='Output blocked by output guardrail.'):
            await agent.run('hello')

    async def test_guard_result_message_reflects_output(self):
        def guard(output: object) -> GuardResult:
            return GuardResult(safe=False, message=f'blocked output: {output}')

        agent = Agent(
            TestModel(custom_output_text='leaks SSN'),
            capabilities=[OutputGuard[None](guard=guard)],
        )
        with pytest.raises(OutputBlocked, match='blocked output: leaks SSN'):
            await agent.run('hello')

    async def test_guard_receives_run_context(self):
        seen: list[object] = []

        def guard(ctx: RunContext[None], output: object) -> bool:
            seen.append(ctx.prompt)
            return 'SSN' not in str(output)

        agent = Agent(
            TestModel(custom_output_text='harmless reply'),
            capabilities=[OutputGuard[None](guard=guard)],
        )
        result = await agent.run('hello')
        assert result.output == 'harmless reply'
        assert seen == ['hello']

    async def test_async_guard_awaited(self):
        async def guard(output: object) -> bool:
            await asyncio.sleep(0)
            return 'bad' not in str(output)

        agent = Agent(
            TestModel(custom_output_text='ok reply'),
            capabilities=[OutputGuard[None](guard=guard)],
        )
        assert (await agent.run('prompt')).output == 'ok reply'

        agent_bad = Agent(
            TestModel(custom_output_text='bad reply'),
            capabilities=[OutputGuard[None](guard=guard)],
        )
        with pytest.raises(OutputBlocked):
            await agent_bad.run('prompt')

    async def test_raising_propagates(self):
        def guard(_: object) -> bool:
            raise RuntimeError('guard exploded')

        agent = Agent(
            TestModel(custom_output_text='anything'),
            capabilities=[OutputGuard[None](guard=guard)],
        )
        with pytest.raises(RuntimeError, match='guard exploded'):
            await agent.run('hello')

    async def test_receives_structured_output_unchanged(self):
        """For typed outputs the guard gets the model instance, not a stringified form."""

        class Answer(BaseModel):
            reply: str
            internal_url: str

        seen: list[object] = []

        def guard(output: object) -> bool:
            seen.append(output)
            assert isinstance(output, Answer)
            return 'internal.example.com' not in output.internal_url

        agent = Agent(
            TestModel(custom_output_args={'reply': 'hi', 'internal_url': 'https://public.example.com/x'}),
            output_type=Answer,
            capabilities=[OutputGuard[None](guard=guard)],
        )
        result = await agent.run('hello')
        assert isinstance(result.output, Answer)
        assert seen == [result.output]

        agent_bad = Agent(
            TestModel(custom_output_args={'reply': 'hi', 'internal_url': 'https://internal.example.com/x'}),
            output_type=Answer,
            capabilities=[OutputGuard[None](guard=guard)],
        )
        with pytest.raises(OutputBlocked):
            await agent_bad.run('hello')

    def test_output_blocked_is_guardrail_error(self):
        assert issubclass(OutputBlocked, GuardrailError)
