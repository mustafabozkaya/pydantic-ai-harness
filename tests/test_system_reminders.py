"""Tests for the SystemReminders capability."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, SystemPromptPart, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters

from pydantic_harness import Reminder, SystemReminders
from pydantic_harness.system_reminders import AsyncDynamicReminder, DynamicReminder


def _make_run_context(*, run_step: int = 1) -> Any:
    """Create a minimal RunContext-like object for testing."""
    ctx = MagicMock()
    ctx.run_step = run_step
    return ctx


def _make_request_context(
    messages: list[Any] | None = None,
) -> ModelRequestContext:
    """Create a ModelRequestContext with the given messages."""
    if messages is None:
        messages = [ModelRequest.user_text_prompt('hello')]
    return ModelRequestContext(
        model=MagicMock(),
        messages=messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )


def _dynamic_content(ctx: Any) -> str | None:
    return 'dynamic content'


def _returns_none(ctx: Any) -> str | None:
    return None


def _returns_dynamic(ctx: Any) -> str | None:
    return 'dynamic'


# --- Reminder validation ---


class TestReminderValidation:
    def test_valid_reminder(self) -> None:
        r = Reminder('test', interval=3)
        assert r.content == 'test'
        assert r.interval == 3

    def test_default_interval(self) -> None:
        r = Reminder('test')
        assert r.interval == 1

    def test_zero_interval_raises(self) -> None:
        with pytest.raises(ValueError, match='interval must be >= 1'):
            Reminder('test', interval=0)

    def test_negative_interval_raises(self) -> None:
        with pytest.raises(ValueError, match='interval must be >= 1'):
            Reminder('test', interval=-1)


# --- SystemReminders validation ---


class TestSystemRemindersValidation:
    def test_requires_at_least_one_reminder(self) -> None:
        with pytest.raises(ValueError, match='At least one'):
            SystemReminders()

    def test_static_reminders_only(self) -> None:
        sr = SystemReminders(reminders=[Reminder('test')])
        assert len(sr.reminders) == 1
        assert len(sr.dynamic_reminders) == 0

    def test_dynamic_reminders_only(self) -> None:
        sr = SystemReminders(dynamic_reminders=[_returns_dynamic])
        assert len(sr.reminders) == 0
        assert len(sr.dynamic_reminders) == 1

    def test_both_kinds(self) -> None:
        sr = SystemReminders(
            reminders=[Reminder('static')],
            dynamic_reminders=[_returns_dynamic],
        )
        assert len(sr.reminders) == 1
        assert len(sr.dynamic_reminders) == 1


# --- for_run isolation ---


class TestForRun:
    @pytest.mark.anyio
    async def test_for_run_returns_fresh_instance(self) -> None:
        sr = SystemReminders(reminders=[Reminder('test')])
        ctx = _make_run_context()
        fresh = await sr.for_run(ctx)
        assert fresh is not sr

    @pytest.mark.anyio
    async def test_for_run_preserves_config(self) -> None:
        reminders = [Reminder('a', interval=2), Reminder('b', interval=5)]
        dynamic: list[DynamicReminder | AsyncDynamicReminder] = [_returns_dynamic]
        sr = SystemReminders(reminders=reminders, dynamic_reminders=dynamic)
        ctx = _make_run_context()
        fresh = await sr.for_run(ctx)
        assert fresh.reminders is reminders
        assert fresh.dynamic_reminders is dynamic

    @pytest.mark.anyio
    async def test_for_run_resets_counter(self) -> None:
        sr = SystemReminders(reminders=[Reminder('test')])
        # Simulate some requests to increment counter.
        sr._request_count = 5  # pyright: ignore[reportPrivateUsage]
        ctx = _make_run_context()
        fresh = await sr.for_run(ctx)
        assert fresh._request_count == 0  # pyright: ignore[reportPrivateUsage]


# --- Static reminder injection ---


class TestStaticReminders:
    @pytest.mark.anyio
    async def test_interval_1_fires_every_request(self) -> None:
        sr = SystemReminders(reminders=[Reminder('always')])
        ctx = _make_run_context()

        for _ in range(3):
            req_ctx = _make_request_context()
            await sr.before_model_request(ctx, req_ctx)

            last_msg = req_ctx.messages[-1]
            assert isinstance(last_msg, ModelRequest)
            system_parts = [p for p in last_msg.parts if isinstance(p, SystemPromptPart)]
            assert len(system_parts) == 1
            assert system_parts[0].content == 'always'

    @pytest.mark.anyio
    async def test_interval_3_fires_on_3rd_request(self) -> None:
        sr = SystemReminders(reminders=[Reminder('every third', interval=3)])
        ctx = _make_run_context()

        # Requests 1 and 2: no injection.
        for _ in range(2):
            req_ctx = _make_request_context()
            await sr.before_model_request(ctx, req_ctx)
            last_msg = req_ctx.messages[-1]
            assert isinstance(last_msg, ModelRequest)
            system_parts = [p for p in last_msg.parts if isinstance(p, SystemPromptPart)]
            assert len(system_parts) == 0

        # Request 3: injection.
        req_ctx = _make_request_context()
        await sr.before_model_request(ctx, req_ctx)
        last_msg = req_ctx.messages[-1]
        assert isinstance(last_msg, ModelRequest)
        system_parts = [p for p in last_msg.parts if isinstance(p, SystemPromptPart)]
        assert len(system_parts) == 1
        assert system_parts[0].content == 'every third'

    @pytest.mark.anyio
    async def test_multiple_reminders_different_intervals(self) -> None:
        sr = SystemReminders(
            reminders=[
                Reminder('every 2', interval=2),
                Reminder('every 3', interval=3),
            ],
        )
        ctx = _make_run_context()

        # Request 1: none.
        req_ctx = _make_request_context()
        await sr.before_model_request(ctx, req_ctx)
        assert _system_contents(req_ctx) == []

        # Request 2: "every 2" only.
        req_ctx = _make_request_context()
        await sr.before_model_request(ctx, req_ctx)
        assert _system_contents(req_ctx) == ['every 2']

        # Request 3: "every 3" only.
        req_ctx = _make_request_context()
        await sr.before_model_request(ctx, req_ctx)
        assert _system_contents(req_ctx) == ['every 3']

        # Request 4: "every 2" only.
        req_ctx = _make_request_context()
        await sr.before_model_request(ctx, req_ctx)
        assert _system_contents(req_ctx) == ['every 2']

        # Request 5: none.
        req_ctx = _make_request_context()
        await sr.before_model_request(ctx, req_ctx)
        assert _system_contents(req_ctx) == []

        # Request 6: both.
        req_ctx = _make_request_context()
        await sr.before_model_request(ctx, req_ctx)
        assert _system_contents(req_ctx) == ['every 2', 'every 3']


# --- Dynamic reminder injection ---


class TestDynamicReminders:
    @pytest.mark.anyio
    async def test_sync_dynamic_returning_string(self) -> None:
        sr = SystemReminders(dynamic_reminders=[_dynamic_content])
        ctx = _make_run_context()

        req_ctx = _make_request_context()
        await sr.before_model_request(ctx, req_ctx)
        assert _system_contents(req_ctx) == ['dynamic content']

    @pytest.mark.anyio
    async def test_sync_dynamic_returning_none_skips(self) -> None:
        sr = SystemReminders(dynamic_reminders=[_returns_none])
        ctx = _make_run_context()

        req_ctx = _make_request_context()
        await sr.before_model_request(ctx, req_ctx)
        assert _system_contents(req_ctx) == []

    @pytest.mark.anyio
    async def test_async_dynamic_reminder(self) -> None:
        async def async_reminder(ctx: Any) -> str | None:
            return 'async content'

        sr = SystemReminders(dynamic_reminders=[async_reminder])
        ctx = _make_run_context()

        req_ctx = _make_request_context()
        await sr.before_model_request(ctx, req_ctx)
        assert _system_contents(req_ctx) == ['async content']

    @pytest.mark.anyio
    async def test_async_dynamic_returning_none(self) -> None:
        async def async_none(ctx: Any) -> str | None:
            return None

        sr = SystemReminders(dynamic_reminders=[async_none])
        ctx = _make_run_context()

        req_ctx = _make_request_context()
        await sr.before_model_request(ctx, req_ctx)
        assert _system_contents(req_ctx) == []

    @pytest.mark.anyio
    async def test_dynamic_receives_run_context(self) -> None:
        def step_check(ctx: Any) -> str | None:
            if ctx.run_step > 10:
                return 'wrap up'
            return None

        sr = SystemReminders(dynamic_reminders=[step_check])

        # Low step: no reminder.
        ctx_low = _make_run_context(run_step=5)
        req_ctx = _make_request_context()
        await sr.before_model_request(ctx_low, req_ctx)
        assert _system_contents(req_ctx) == []

        # High step: reminder fires.
        ctx_high = _make_run_context(run_step=15)
        req_ctx = _make_request_context()
        await sr.before_model_request(ctx_high, req_ctx)
        assert _system_contents(req_ctx) == ['wrap up']


# --- Mixed static and dynamic ---


class TestMixedReminders:
    @pytest.mark.anyio
    async def test_static_and_dynamic_combined(self) -> None:
        sr = SystemReminders(
            reminders=[Reminder('static', interval=1)],
            dynamic_reminders=[_returns_dynamic],
        )
        ctx = _make_run_context()

        req_ctx = _make_request_context()
        await sr.before_model_request(ctx, req_ctx)
        assert _system_contents(req_ctx) == ['static', 'dynamic']

    @pytest.mark.anyio
    async def test_static_fires_dynamic_skips(self) -> None:
        sr = SystemReminders(
            reminders=[Reminder('static', interval=1)],
            dynamic_reminders=[_returns_none],
        )
        ctx = _make_run_context()

        req_ctx = _make_request_context()
        await sr.before_model_request(ctx, req_ctx)
        assert _system_contents(req_ctx) == ['static']


# --- Message injection behavior ---


class TestMessageInjection:
    @pytest.mark.anyio
    async def test_appends_to_last_model_request(self) -> None:
        """Reminder parts are appended to the last ModelRequest."""
        sr = SystemReminders(reminders=[Reminder('reminder')])
        ctx = _make_run_context()

        messages: list[Any] = [
            ModelRequest(parts=[UserPromptPart('first')]),
            ModelRequest(parts=[UserPromptPart('second')]),
        ]
        req_ctx = _make_request_context(messages)
        await sr.before_model_request(ctx, req_ctx)

        # First request unchanged.
        first = req_ctx.messages[0]
        assert isinstance(first, ModelRequest)
        assert len(first.parts) == 1

        # Second request has reminder appended.
        second = req_ctx.messages[1]
        assert isinstance(second, ModelRequest)
        assert len(second.parts) == 2
        assert isinstance(second.parts[1], SystemPromptPart)
        assert second.parts[1].content == 'reminder'

    @pytest.mark.anyio
    async def test_preserves_existing_parts(self) -> None:
        """Existing parts on the ModelRequest are preserved."""
        sr = SystemReminders(reminders=[Reminder('reminder')])
        ctx = _make_run_context()

        req_ctx = _make_request_context(
            [
                ModelRequest(parts=[UserPromptPart('user msg'), SystemPromptPart(content='existing')]),
            ]
        )
        await sr.before_model_request(ctx, req_ctx)

        msg = req_ctx.messages[0]
        assert isinstance(msg, ModelRequest)
        assert len(msg.parts) == 3
        assert isinstance(msg.parts[0], UserPromptPart)
        assert isinstance(msg.parts[1], SystemPromptPart)
        assert msg.parts[1].content == 'existing'
        assert isinstance(msg.parts[2], SystemPromptPart)
        assert msg.parts[2].content == 'reminder'

    @pytest.mark.anyio
    async def test_creates_request_when_none_exists(self) -> None:
        """If no ModelRequest exists, a new one is created."""
        sr = SystemReminders(reminders=[Reminder('orphan')])
        ctx = _make_run_context()

        req_ctx = _make_request_context(messages=[])
        await sr.before_model_request(ctx, req_ctx)

        assert len(req_ctx.messages) == 1
        msg = req_ctx.messages[0]
        assert isinstance(msg, ModelRequest)
        assert len(msg.parts) == 1
        assert isinstance(msg.parts[0], SystemPromptPart)
        assert msg.parts[0].content == 'orphan'

    @pytest.mark.anyio
    async def test_skips_model_response_to_find_last_request(self) -> None:
        """When the last message is a ModelResponse, skip it to find the ModelRequest."""
        sr = SystemReminders(reminders=[Reminder('reminder')])
        ctx = _make_run_context()

        messages: list[Any] = [
            ModelRequest(parts=[UserPromptPart('hello')]),
            ModelResponse(parts=[TextPart(content='hi')]),
        ]
        req_ctx = _make_request_context(messages)
        await sr.before_model_request(ctx, req_ctx)

        # The ModelRequest should have the reminder appended.
        first = req_ctx.messages[0]
        assert isinstance(first, ModelRequest)
        assert len(first.parts) == 2
        assert isinstance(first.parts[1], SystemPromptPart)
        assert first.parts[1].content == 'reminder'

        # The ModelResponse should be unchanged.
        second = req_ctx.messages[1]
        assert isinstance(second, ModelResponse)

    @pytest.mark.anyio
    async def test_no_injection_when_nothing_fires(self) -> None:
        """Messages are untouched when no reminders fire."""
        sr = SystemReminders(reminders=[Reminder('skip', interval=3)])
        ctx = _make_run_context()

        original_msg = ModelRequest(parts=[UserPromptPart('hello')])
        req_ctx = _make_request_context([original_msg])
        await sr.before_model_request(ctx, req_ctx)

        # Request 1: interval=3 doesn't fire.
        msg = req_ctx.messages[0]
        assert isinstance(msg, ModelRequest)
        assert len(msg.parts) == 1


# --- Serialization ---


class TestSerialization:
    def test_not_serializable(self) -> None:
        assert SystemReminders.get_serialization_name() is None


# --- Helpers ---


def _system_contents(req_ctx: ModelRequestContext) -> list[str]:
    """Extract system prompt contents from the last ModelRequest in a request context."""
    if not req_ctx.messages:  # pragma: no cover
        return []
    last = req_ctx.messages[-1]
    if not isinstance(last, ModelRequest):  # pragma: no cover
        return []
    return [p.content for p in last.parts if isinstance(p, SystemPromptPart)]
