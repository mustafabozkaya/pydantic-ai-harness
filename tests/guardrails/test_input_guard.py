"""Tests for the `InputGuard` capability."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import SkipModelRequest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness import GuardResult, InputBlocked, InputGuard
from pydantic_ai_harness.guardrails._capability import _extract_prompt  # pyright: ignore[reportPrivateUsage]

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _build_ctx_and_req(
    run_step: int = 1,
    prompt: str | None = 'hello world',
) -> tuple[RunContext[None], ModelRequestContext]:
    model = TestModel()
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content=prompt)])] if prompt is not None else []
    req_ctx = ModelRequestContext(
        model=model,
        messages=messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )
    run_ctx: RunContext[None] = RunContext(
        deps=None,
        model=model,
        usage=RunUsage(),
        prompt=prompt,
        messages=messages,
        run_step=run_step,
    )
    return run_ctx, req_ctx


class TestInputGuard:
    """Integration tests for the `InputGuard` capability driven through `Agent.run`."""

    async def test_allows_when_safe(self):
        calls: list[str] = []

        def guard(prompt: str) -> bool:
            calls.append(prompt)
            return True

        agent = Agent(TestModel(custom_output_text='ok'), capabilities=[InputGuard[None](guard=guard)])
        result = await agent.run('hello')

        assert result.output == 'ok'
        assert calls == ['hello']

    async def test_block_uses_default_message(self):
        agent = Agent(
            TestModel(custom_output_text='would be model output'),
            capabilities=[InputGuard[None](guard=lambda _: False)],
        )
        result = await agent.run('hello')

        assert result.output == 'Request blocked by input guardrail.'

    async def test_guard_result_without_message_uses_default(self):
        agent = Agent(
            TestModel(custom_output_text='would be model output'),
            capabilities=[InputGuard[None](guard=lambda _: GuardResult(safe=False))],
        )
        result = await agent.run('hello')

        assert result.output == 'Request blocked by input guardrail.'

    async def test_async_guard_awaited(self):
        async def guard(prompt: str) -> bool:
            await asyncio.sleep(0)
            return 'safe' in prompt

        agent = Agent(TestModel(custom_output_text='ok'), capabilities=[InputGuard[None](guard=guard)])

        assert (await agent.run('safe message')).output == 'ok'
        assert (await agent.run('bad message')).output == 'Request blocked by input guardrail.'

    async def test_raising_propagates(self):
        def guard(_: str) -> bool:
            raise InputBlocked('policy violation')

        agent = Agent(TestModel(custom_output_text='ok'), capabilities=[InputGuard[None](guard=guard)])
        with pytest.raises(InputBlocked, match='policy violation'):
            await agent.run('anything')

    async def test_guard_result_message_reflects_prompt(self):
        def guard(prompt: str) -> GuardResult:
            return GuardResult(safe=False, message=f'blocked: {prompt}')

        agent = Agent(
            TestModel(custom_output_text='would be model output'),
            capabilities=[InputGuard[None](guard=guard)],
        )
        result = await agent.run('secret stuff')

        assert result.output == 'blocked: secret stuff'

    async def test_guard_receives_run_context(self):
        seen: list[object] = []

        def guard(ctx: RunContext[None], prompt: str) -> bool:
            seen.append(ctx.prompt)
            return True

        agent = Agent(TestModel(custom_output_text='ok'), capabilities=[InputGuard[None](guard=guard)])
        result = await agent.run('hello')

        assert result.output == 'ok'
        assert seen == ['hello']

    async def test_sequential_runs_guard_then_handler(self):
        run_ctx, req_ctx = _build_ctx_and_req()
        sentinel = ModelResponse(parts=[TextPart(content='direct')])
        calls: list[str] = []

        def guard(prompt: str) -> bool:
            calls.append(prompt)
            return True

        async def handler(_: Any) -> ModelResponse:
            return sentinel

        ig = InputGuard[None](guard=guard, parallel=False)
        out = await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=handler)
        assert out is sentinel
        assert calls == ['hello world']

    async def test_sequential_skipped_when_prompt_missing(self):
        run_ctx, req_ctx = _build_ctx_and_req(prompt=None)
        sentinel = ModelResponse(parts=[TextPart(content='direct')])

        called: list[str] = []

        def guard(prompt: str) -> bool:  # pragma: no cover — should not be called
            called.append(prompt)
            return True

        async def handler(_: Any) -> ModelResponse:
            return sentinel

        ig = InputGuard[None](guard=guard, parallel=False)
        out = await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=handler)
        assert out is sentinel
        assert called == []

    async def test_runs_once_across_tool_loop(self):
        """End-to-end: guard fires once even when the model makes multiple tool calls."""
        calls: list[str] = []

        def guard(prompt: str) -> bool:
            calls.append(prompt)
            return True

        # TestModel(call_tools='all') calls each tool once, then returns text — two model
        # requests total.
        model = TestModel(call_tools='all', custom_output_text='done')
        agent = Agent(model, capabilities=[InputGuard[None](guard=guard)])

        @agent.tool_plain
        def ping() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'pong'

        result = await agent.run('hello')
        assert result.output == 'done'
        assert calls == ['hello']

    async def test_sequential_skips_guard_on_subsequent_steps(self):
        """After the first model request, the guard must not re-run."""
        run_ctx, req_ctx = _build_ctx_and_req(run_step=2)
        sentinel = ModelResponse(parts=[TextPart(content='direct')])

        called: list[str] = []

        def guard(prompt: str) -> bool:  # pragma: no cover — should not be called after step 1
            called.append(prompt)
            return False

        async def handler(_: Any) -> ModelResponse:
            return sentinel

        ig = InputGuard[None](guard=guard, parallel=False)
        out = await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=handler)
        assert out is sentinel
        assert called == []


class TestInputGuardParallel:
    """Tests for `InputGuard(parallel=True)` exercising the race between guard and handler."""

    async def test_allows_handler_to_return(self):
        run_ctx, req_ctx = _build_ctx_and_req()
        sentinel = ModelResponse(parts=[TextPart(content='from handler')])

        async def handler(_: Any) -> ModelResponse:
            return sentinel

        guard = InputGuard[None](guard=lambda _: True, parallel=True)
        out = await guard.wrap_model_request(run_ctx, request_context=req_ctx, handler=handler)
        assert out is sentinel

    async def test_trips_and_cancels_handler(self):
        run_ctx, req_ctx = _build_ctx_and_req()
        handler_cancelled = asyncio.Event()
        handler_started = asyncio.Event()

        async def slow_handler(_: Any) -> ModelResponse:
            handler_started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                handler_cancelled.set()
                raise
            return ModelResponse(parts=[TextPart(content='should never')])  # pragma: no cover

        async def guard(_: str) -> GuardResult:
            await handler_started.wait()
            return GuardResult(safe=False, message='blocked!')

        ig = InputGuard[None](guard=guard, parallel=True)
        with pytest.raises(SkipModelRequest) as exc_info:
            await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=slow_handler)

        assert exc_info.value.response.parts[0] == TextPart(content='blocked!')
        await asyncio.sleep(0)
        assert handler_cancelled.is_set()

    async def test_guard_raises_propagates(self):
        run_ctx, req_ctx = _build_ctx_and_req()

        async def slow_handler(_: Any) -> ModelResponse:
            await asyncio.sleep(10)
            return ModelResponse(parts=[TextPart(content='never')])  # pragma: no cover

        async def guard(_: str) -> bool:
            raise InputBlocked('hard policy failure')

        ig = InputGuard[None](guard=guard, parallel=True)
        with pytest.raises(InputBlocked, match='hard policy failure'):
            await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=slow_handler)

    async def test_handler_finishes_before_guard(self):
        """Handler completes first; guard still has to be awaited for a verdict."""
        run_ctx, req_ctx = _build_ctx_and_req()
        sentinel = ModelResponse(parts=[TextPart(content='from handler')])
        release_guard = asyncio.Event()

        async def fast_handler(_: Any) -> ModelResponse:
            return sentinel

        async def slow_guard(_: str) -> bool:
            await release_guard.wait()
            return True

        async def runner() -> ModelResponse:
            ig = InputGuard[None](guard=slow_guard, parallel=True)
            return await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=fast_handler)

        task = asyncio.create_task(runner())
        for _ in range(3):
            await asyncio.sleep(0)
        release_guard.set()
        assert await task is sentinel

    async def test_handler_finishes_then_guard_trips(self):
        """Handler returns first, then the guard trips — `SkipModelRequest` still wins."""
        run_ctx, req_ctx = _build_ctx_and_req()
        release_guard = asyncio.Event()

        async def fast_handler(_: Any) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='from handler')])

        async def slow_guard(_: str) -> GuardResult:
            await release_guard.wait()
            return GuardResult(safe=False, message='late trip')

        async def runner() -> ModelResponse:
            ig = InputGuard[None](guard=slow_guard, parallel=True)
            return await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=fast_handler)

        task = asyncio.create_task(runner())
        for _ in range(3):
            await asyncio.sleep(0)
        release_guard.set()
        with pytest.raises(SkipModelRequest) as exc_info:
            await task
        assert exc_info.value.response.parts[0] == TextPart(content='late trip')

    async def test_handler_raises_while_guard_runs(self):
        """When the handler raises, `finally` cancels the still-running guard."""
        run_ctx, req_ctx = _build_ctx_and_req()
        guard_cancelled = asyncio.Event()

        async def failing_handler(_: Any) -> ModelResponse:
            raise RuntimeError('model boom')

        async def slow_guard(_: str) -> bool:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                guard_cancelled.set()
                raise
            return True  # pragma: no cover

        ig = InputGuard[None](guard=slow_guard, parallel=True)
        with pytest.raises(RuntimeError, match='model boom'):
            await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=failing_handler)
        await asyncio.sleep(0)
        assert guard_cancelled.is_set()

    async def test_skipped_when_prompt_missing(self):
        run_ctx, req_ctx = _build_ctx_and_req(prompt=None)
        sentinel = ModelResponse(parts=[TextPart(content='direct')])

        called: list[str] = []

        def guard(prompt: str) -> bool:  # pragma: no cover — should never be called
            called.append(prompt)
            return False

        async def handler(_: Any) -> ModelResponse:
            return sentinel

        ig = InputGuard[None](guard=guard, parallel=True)
        out = await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=handler)
        assert out is sentinel
        assert called == []

    async def test_skips_guard_on_subsequent_steps(self):
        """`wrap_model_request` must pass the handler through without running the guard past step 1."""
        run_ctx, req_ctx = _build_ctx_and_req(run_step=2)
        sentinel = ModelResponse(parts=[TextPart(content='direct')])
        called: list[str] = []

        def guard(prompt: str) -> bool:  # pragma: no cover — should not be called after step 1
            called.append(prompt)
            return False

        async def handler(_: Any) -> ModelResponse:
            return sentinel

        ig = InputGuard[None](guard=guard, parallel=True)
        out = await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=handler)
        assert out is sentinel
        assert called == []

    async def test_no_dangling_tasks_when_handler_raises(self):
        """`finally` must drain cancelled tasks so they don't outlive the call.

        `task.cancel()` only schedules a `CancelledError` for the next loop tick — without
        awaiting, the task stays in `asyncio.all_tasks()` after `wrap_model_request`
        returns, leaks into the surrounding scope, and produces "Task exception was never
        retrieved" warnings if it later raises.
        """
        run_ctx, req_ctx = _build_ctx_and_req()

        async def failing_handler(_: Any) -> ModelResponse:
            raise RuntimeError('handler boom')

        async def slow_guard(_: str) -> bool:
            await asyncio.sleep(10)
            return True  # pragma: no cover

        current = asyncio.current_task()
        before = {t for t in asyncio.all_tasks() if t is not current}

        ig = InputGuard[None](guard=slow_guard, parallel=True)
        with pytest.raises(RuntimeError, match='handler boom'):
            await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=failing_handler)

        leftover = {t for t in asyncio.all_tasks() if t is not current} - before
        assert leftover == set(), f'guard/handler tasks must be drained, got dangling: {leftover}'


class TestExtractPrompt:
    """Unit tests for the `_extract_prompt` helper."""

    def test_from_messages(self):
        """Extraction falls back to the most recent `UserPromptPart`."""

        class _Ctx:
            prompt = None

        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='first')]),
            ModelResponse(parts=[TextPart(content='assistant')]),
            ModelRequest(parts=[UserPromptPart(content='second')]),
        ]
        assert _extract_prompt(_Ctx(), messages) == 'second'  # pyright: ignore[reportArgumentType]

    def test_stringifies_non_str_prompt(self):
        class _Ctx:
            prompt = ['multimodal', 'content']

        assert _extract_prompt(_Ctx(), []) == str(['multimodal', 'content'])  # pyright: ignore[reportArgumentType]

    def test_stringifies_non_str_message_part(self):
        class _Ctx:
            prompt = None

        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content=['multi'])])]
        assert _extract_prompt(_Ctx(), messages) == str(['multi'])  # pyright: ignore[reportArgumentType]

    def test_returns_none_when_no_user_prompt_part(self):
        """A history containing only model responses yields `None`."""

        class _Ctx:
            prompt = None

        messages: list[ModelMessage] = [ModelResponse(parts=[TextPart(content='only model parts here')])]
        assert _extract_prompt(_Ctx(), messages) is None  # pyright: ignore[reportArgumentType]
