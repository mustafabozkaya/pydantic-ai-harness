"""Tests for the `BackgroundTools` capability."""

from __future__ import annotations

import asyncio

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from pydantic_ai_harness import BackgroundTools

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _ack_seen(messages: list[ModelMessage]) -> bool:
    """True if any tool return in the history is a background-execution ack."""
    return any(
        isinstance(part, ToolReturnPart) and 'running in background' in str(part.content)
        for msg in messages
        if isinstance(msg, ModelRequest)
        for part in msg.parts
    )


def _follow_up_seen(messages: list[ModelMessage], needle: str) -> bool:
    """True if any system prompt in the history contains *needle* (e.g. 'completed' / 'failed')."""
    return any(
        isinstance(part, SystemPromptPart) and needle in part.content
        for msg in messages
        if isinstance(msg, ModelRequest)
        for part in msg.parts
    )


class TestBackgroundTools:
    """Cover the metadata-default selector path: spawn, ack, deliver, error, cancel."""

    async def test_metadata_marked_tool_runs_in_background(self) -> None:
        """A tool with `metadata={'background': True}` returns an ack and delivers result as follow-up."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name='slow_research', args='{"query": "topic"}')],
                    usage=RequestUsage(input_tokens=10, output_tokens=5),
                )
            if call_count == 2:
                # Agent saw the ack; produce a placeholder, drain holds it back.
                return ModelResponse(
                    parts=[TextPart(content='waiting')],
                    usage=RequestUsage(input_tokens=10, output_tokens=5),
                )
            # Third call: the follow-up has been delivered.
            assert _follow_up_seen(messages, 'completed')
            return ModelResponse(
                parts=[TextPart(content='got result')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )

        agent = Agent(FunctionModel(model_fn), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def slow_research(query: str) -> str:  # pyright: ignore[reportUnusedFunction]
            await asyncio.sleep(0.01)
            return f'researched {query}'

        result = await agent.run('research X')
        assert result.output == 'got result'
        assert _ack_seen(result.all_messages())

    async def test_failure_delivered_as_follow_up(self) -> None:
        """A background tool that raises produces a 'failed' follow-up message."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name='broken', args='{}')],
                    usage=RequestUsage(input_tokens=10, output_tokens=5),
                )
            if call_count == 2:
                return ModelResponse(
                    parts=[TextPart(content='waiting')],
                    usage=RequestUsage(input_tokens=10, output_tokens=5),
                )
            assert _follow_up_seen(messages, 'failed')
            return ModelResponse(
                parts=[TextPart(content='handled error')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )

        agent = Agent(FunctionModel(model_fn), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def broken() -> str:  # pyright: ignore[reportUnusedFunction]
            await asyncio.sleep(0.01)
            raise RuntimeError('boom')

        result = await agent.run('go')
        assert result.output == 'handled error'

    async def test_unmarked_tool_runs_synchronously(self) -> None:
        """A tool without the metadata flag is executed normally; no ack, no follow-up."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                if isinstance(msg, ModelRequest):
                    for part in msg.parts:
                        if isinstance(part, ToolReturnPart) and part.content == 'sync result':
                            return ModelResponse(
                                parts=[TextPart(content='done')],
                                usage=RequestUsage(input_tokens=10, output_tokens=5),
                            )
            return ModelResponse(
                parts=[ToolCallPart(tool_name='plain', args='{}')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )

        agent = Agent(FunctionModel(model_fn), capabilities=[BackgroundTools()])

        @agent.tool_plain
        def plain() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'sync result'

        result = await agent.run('go')
        assert result.output == 'done'
        assert not _ack_seen(result.all_messages())

    async def test_run_abort_cancels_live_tasks(self) -> None:
        """When the surrounding run is cancelled (e.g. timeout), live background tasks are cancelled too."""
        cancel_seen = asyncio.Event()

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if _ack_seen(messages):
                return ModelResponse(
                    parts=[TextPart(content='done')],
                    usage=RequestUsage(input_tokens=10, output_tokens=5),
                )
            return ModelResponse(
                parts=[ToolCallPart(tool_name='slow', args='{}')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )

        agent = Agent(FunctionModel(model_fn), capabilities=[BackgroundTools()])

        @agent.tool_plain(metadata={'background': True})
        async def slow() -> str:  # pyright: ignore[reportUnusedFunction]
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancel_seen.set()
                raise
            return 'never'  # pragma: no cover -- task is cancelled before completing

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(agent.run('go'), timeout=0.5)

        await asyncio.wait_for(cancel_seen.wait(), timeout=1)


class TestSelectors:
    """Cover the non-default `tools=...` selectors: name list, predicate, custom dict."""

    async def test_name_list_selector(self) -> None:
        """`tools=['name']` selects without needing metadata."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name='by_name', args='{}')],
                    usage=RequestUsage(input_tokens=10, output_tokens=5),
                )
            if call_count == 2:
                return ModelResponse(
                    parts=[TextPart(content='waiting')],
                    usage=RequestUsage(input_tokens=10, output_tokens=5),
                )
            return ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )

        agent = Agent(FunctionModel(model_fn), capabilities=[BackgroundTools(tools=['by_name'])])

        @agent.tool_plain
        async def by_name() -> str:  # pyright: ignore[reportUnusedFunction]
            await asyncio.sleep(0.01)
            return 'value'

        result = await agent.run('go')
        assert result.output == 'done'
        assert _ack_seen(result.all_messages())

    async def test_custom_metadata_key_selector(self) -> None:
        """`tools={'async': True}` matches any other metadata key."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name='custom', args='{}')],
                    usage=RequestUsage(input_tokens=10, output_tokens=5),
                )
            if call_count == 2:
                return ModelResponse(
                    parts=[TextPart(content='waiting')],
                    usage=RequestUsage(input_tokens=10, output_tokens=5),
                )
            return ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )

        agent = Agent(
            FunctionModel(model_fn),
            capabilities=[BackgroundTools(tools={'async': True})],
        )

        @agent.tool_plain(metadata={'async': True})
        async def custom() -> str:  # pyright: ignore[reportUnusedFunction]
            await asyncio.sleep(0.01)
            return 'value'

        result = await agent.run('go')
        assert result.output == 'done'
        assert _ack_seen(result.all_messages())
