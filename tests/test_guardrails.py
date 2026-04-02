"""Tests for guardrail capabilities."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage

from pydantic_harness.guardrails import (
    AsyncGuardrail,
    BudgetExceededError,
    CostGuard,
    GuardrailError,
    GuardrailFailed,
    GuardrailResult,
    InputBlocked,
    InputGuardrail,
    OutputBlocked,
    OutputGuardrail,
    ToolBlocked,
    ToolGuard,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Restrict to asyncio — pydantic-ai internals use asyncio.gather."""
    return 'asyncio'


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    def test_guardrail_error_is_base(self) -> None:
        assert issubclass(InputBlocked, GuardrailError)
        assert issubclass(OutputBlocked, GuardrailError)
        assert issubclass(BudgetExceededError, GuardrailError)
        assert issubclass(ToolBlocked, GuardrailError)

    def test_tool_blocked_attributes(self) -> None:
        err = ToolBlocked('my_tool', reason='denied')
        assert err.tool_name == 'my_tool'
        assert err.reason == 'denied'
        assert "Tool 'my_tool' blocked: denied" in str(err)

    def test_tool_blocked_no_reason(self) -> None:
        err = ToolBlocked('my_tool')
        assert err.reason == ''
        assert str(err) == "Tool 'my_tool' blocked"

    def test_budget_exceeded_detail(self) -> None:
        err = BudgetExceededError('Token budget exceeded: 200/100')
        assert err.detail == 'Token budget exceeded: 200/100'
        assert 'Token budget exceeded' in str(err)


# ---------------------------------------------------------------------------
# InputGuardrail
# ---------------------------------------------------------------------------


class TestInputGuardrail:
    async def test_sync_guard_allows(self) -> None:
        """Sync guard returning True should allow the run to proceed."""
        agent = Agent(TestModel(), capabilities=[InputGuardrail(guard=lambda text: True)])
        result = await agent.run('Hello')
        assert result.output is not None

    async def test_sync_guard_blocks(self) -> None:
        """Sync guard returning False should raise InputBlocked."""
        agent = Agent(TestModel(), capabilities=[InputGuardrail(guard=lambda text: False)])
        with pytest.raises(InputBlocked, match='Input blocked by guardrail'):
            await agent.run('Hello')

    async def test_async_guard_allows(self) -> None:
        """Async guard returning True should allow the run to proceed."""

        async def safe_check(text: str) -> bool:
            return True

        agent = Agent(TestModel(), capabilities=[InputGuardrail(guard=safe_check)])
        result = await agent.run('Hello')
        assert result.output is not None

    async def test_async_guard_blocks(self) -> None:
        """Async guard returning False should raise InputBlocked."""

        async def unsafe_check(text: str) -> bool:
            return False

        agent = Agent(TestModel(), capabilities=[InputGuardrail(guard=unsafe_check)])
        with pytest.raises(InputBlocked):
            await agent.run('Hello')

    async def test_guard_receives_prompt_text(self) -> None:
        """Guard function should receive the actual prompt text."""
        received: list[str] = []

        def capture(text: str) -> bool:
            received.append(text)
            return True

        agent = Agent(TestModel(), capabilities=[InputGuardrail(guard=capture)])
        await agent.run('test prompt 123')
        assert received == ['test prompt 123']

    async def test_guard_blocks_with_content_in_message(self) -> None:
        """The error message should include a truncated version of the input."""

        def block_sql(text: str) -> bool:
            return 'DROP TABLE' not in text

        agent = Agent(TestModel(), capabilities=[InputGuardrail(guard=block_sql)])
        with pytest.raises(InputBlocked, match='Input blocked by guardrail'):
            await agent.run('DROP TABLE users')

    async def test_none_prompt_skips_guard(self) -> None:
        """When prompt is None, the guard function should not be called."""
        called = False

        def guard(text: str) -> bool:  # pragma: no cover
            nonlocal called
            called = True
            return False

        guardrail = InputGuardrail(guard=guard)
        ctx = _make_run_context()
        await guardrail.before_run(ctx)
        assert not called

    async def test_empty_string_input(self) -> None:
        received: list[str] = []

        def capture(text: str) -> bool:
            received.append(text)
            return True

        agent = Agent(TestModel(), capabilities=[InputGuardrail(guard=capture)])
        result = await agent.run('')
        assert received == ['']
        assert result.output is not None

    async def test_non_string_prompt_converted(self) -> None:
        received: list[str] = []

        def capture(text: str) -> bool:
            received.append(text)
            return True

        guardrail = InputGuardrail(guard=capture)
        ctx = _make_run_context(prompt=['hello'])
        await guardrail.before_run(ctx)
        assert received == ["['hello']"]

    async def test_multiple_input_guardrails(self) -> None:
        received_a: list[str] = []
        received_b: list[str] = []

        agent = Agent(
            TestModel(),
            capabilities=[
                InputGuardrail(guard=lambda text: received_a.append(text) or True),  # type: ignore[func-returns-value]
                InputGuardrail(guard=lambda text: received_b.append(text) or True),  # type: ignore[func-returns-value]
            ],
        )
        await agent.run('multi guard test')
        assert received_a == ['multi guard test']
        assert received_b == ['multi guard test']

    def test_not_serializable(self) -> None:
        """InputGuardrail should not be spec-serializable (takes a callable)."""
        assert InputGuardrail.get_serialization_name() is None


# ---------------------------------------------------------------------------
# OutputGuardrail
# ---------------------------------------------------------------------------


class TestOutputGuardrail:
    async def test_sync_guard_allows(self) -> None:
        """Sync guard returning True should pass the result through."""
        agent = Agent(
            TestModel(custom_output_text='safe output'),
            capabilities=[OutputGuardrail(guard=lambda text: True)],
        )
        result = await agent.run('Hello')
        assert result.output == 'safe output'

    async def test_sync_guard_blocks(self) -> None:
        """Sync guard returning False should raise OutputBlocked."""
        agent = Agent(
            TestModel(custom_output_text='bad output'),
            capabilities=[OutputGuardrail(guard=lambda text: False)],
        )
        with pytest.raises(OutputBlocked, match='Output blocked by guardrail'):
            await agent.run('Hello')

    async def test_async_guard_allows(self) -> None:
        """Async guard returning True should pass the result through."""

        async def safe_check(text: str) -> bool:
            return True

        agent = Agent(
            TestModel(custom_output_text='good output'),
            capabilities=[OutputGuardrail(guard=safe_check)],
        )
        result = await agent.run('Hello')
        assert result.output == 'good output'

    async def test_async_guard_blocks(self) -> None:
        """Async guard returning False should raise OutputBlocked."""

        async def unsafe_check(text: str) -> bool:
            return False

        agent = Agent(
            TestModel(custom_output_text='bad output'),
            capabilities=[OutputGuardrail(guard=unsafe_check)],
        )
        with pytest.raises(OutputBlocked):
            await agent.run('Hello')

    async def test_guard_receives_output_text(self) -> None:
        """Guard function should receive the stringified output."""
        received: list[str] = []

        def capture(text: str) -> bool:
            received.append(text)
            return True

        agent = Agent(
            TestModel(custom_output_text='hello world'),
            capabilities=[OutputGuardrail(guard=capture)],
        )
        await agent.run('test')
        assert received == ['hello world']

    async def test_guard_content_check(self) -> None:
        """Guard should be able to check output content."""

        def no_secrets(text: str) -> bool:
            return 'sk-' not in text

        agent = Agent(
            TestModel(custom_output_text='Your key is sk-abc123'),
            capabilities=[OutputGuardrail(guard=no_secrets)],
        )
        with pytest.raises(OutputBlocked):
            await agent.run('What is my API key?')

    def test_not_serializable(self) -> None:
        """OutputGuardrail should not be spec-serializable (takes a callable)."""
        assert OutputGuardrail.get_serialization_name() is None


# ---------------------------------------------------------------------------
# CostGuard
# ---------------------------------------------------------------------------


class TestCostGuard:
    async def test_no_limits_set(self) -> None:
        """With all limits None, runs should proceed normally."""
        agent = Agent(TestModel(), capabilities=[CostGuard()])
        result = await agent.run('Hello')
        assert result.output is not None

    async def test_high_limits_allow(self) -> None:
        """Limits well above usage should not trigger."""
        agent = Agent(
            TestModel(),
            capabilities=[CostGuard(max_total_tokens=1_000_000)],
        )
        result = await agent.run('Hello')
        assert result.output is not None

    async def test_input_token_limit_exceeded(self) -> None:
        """Exceeding input token limit should raise BudgetExceededError."""
        guard = CostGuard(max_input_tokens=10)
        ctx = _make_run_context(input_tokens=100)
        with pytest.raises(BudgetExceededError, match='Input token budget exceeded'):
            await guard.before_model_request(ctx, _mock_request_context())

    async def test_output_token_limit_exceeded(self) -> None:
        """Exceeding output token limit should raise BudgetExceededError."""
        guard = CostGuard(max_output_tokens=10)
        ctx = _make_run_context(output_tokens=100)
        with pytest.raises(BudgetExceededError, match='Output token budget exceeded'):
            await guard.before_model_request(ctx, _mock_request_context())

    async def test_total_token_limit_exceeded(self) -> None:
        """Exceeding total token limit should raise BudgetExceededError."""
        guard = CostGuard(max_total_tokens=50)
        ctx = _make_run_context(input_tokens=30, output_tokens=30)
        with pytest.raises(BudgetExceededError, match='Total token budget exceeded'):
            await guard.before_model_request(ctx, _mock_request_context())

    async def test_within_limits_passes(self) -> None:
        """Usage within all limits should pass through."""
        guard = CostGuard(max_input_tokens=100, max_output_tokens=100, max_total_tokens=200)
        ctx = _make_run_context(input_tokens=10, output_tokens=10)
        result = await guard.before_model_request(ctx, _mock_request_context())
        assert result is not None

    def test_serialization_name(self) -> None:
        """CostGuard should be spec-serializable."""
        assert CostGuard.get_serialization_name() == 'CostGuard'


# ---------------------------------------------------------------------------
# ToolGuard
# ---------------------------------------------------------------------------


class TestToolGuard:
    async def test_blocked_tools_hidden(self) -> None:
        """Blocked tools should be removed from the tool definitions list."""
        guard = ToolGuard(blocked=['dangerous_tool'])
        ctx = _make_run_context()

        tool_defs = [
            _make_tool_def('safe_tool'),
            _make_tool_def('dangerous_tool'),
            _make_tool_def('another_tool'),
        ]

        result = await guard.prepare_tools(ctx, tool_defs)
        names = [td.name for td in result]
        assert 'dangerous_tool' not in names
        assert 'safe_tool' in names
        assert 'another_tool' in names

    async def test_no_blocked_tools_passes_through(self) -> None:
        """With empty blocked list, all tools should pass through."""
        guard = ToolGuard()
        ctx = _make_run_context()

        tool_defs = [_make_tool_def('tool_a'), _make_tool_def('tool_b')]

        result = await guard.prepare_tools(ctx, tool_defs)
        assert len(result) == 2

    async def test_approval_denied_raises(self) -> None:
        """When approval callback returns False, ToolBlocked should be raised."""
        guard = ToolGuard(
            require_approval=['send_email'],
            approval_callback=lambda name, args: False,
        )
        ctx = _make_run_context()
        call = ToolCallPart(tool_name='send_email', args='{}')
        tool_def = _make_tool_def('send_email')

        with pytest.raises(ToolBlocked, match='approval denied'):
            await guard.before_tool_execute(ctx, call=call, tool_def=tool_def, args={'to': 'user@example.com'})

    async def test_approval_granted_passes(self) -> None:
        """When approval callback returns True, args should pass through."""
        guard = ToolGuard(
            require_approval=['send_email'],
            approval_callback=lambda name, args: True,
        )
        ctx = _make_run_context()
        call = ToolCallPart(tool_name='send_email', args='{}')
        tool_def = _make_tool_def('send_email')

        result = await guard.before_tool_execute(ctx, call=call, tool_def=tool_def, args={'to': 'user@example.com'})
        assert result == {'to': 'user@example.com'}

    async def test_async_approval_callback(self) -> None:
        """Async approval callbacks should work."""

        async def approve(name: str, args: dict[str, Any]) -> bool:
            return True

        guard = ToolGuard(require_approval=['my_tool'], approval_callback=approve)
        ctx = _make_run_context()
        call = ToolCallPart(tool_name='my_tool', args='{}')
        tool_def = _make_tool_def('my_tool')

        result = await guard.before_tool_execute(ctx, call=call, tool_def=tool_def, args={'x': 1})
        assert result == {'x': 1}

    async def test_no_callback_raises(self) -> None:
        """When require_approval is set but no callback provided, ToolBlocked should be raised."""
        guard = ToolGuard(require_approval=['my_tool'])
        ctx = _make_run_context()
        call = ToolCallPart(tool_name='my_tool', args='{}')
        tool_def = _make_tool_def('my_tool')

        with pytest.raises(ToolBlocked, match='no callback configured'):
            await guard.before_tool_execute(ctx, call=call, tool_def=tool_def, args={})

    async def test_unrestricted_tool_passes(self) -> None:
        """Tools not in require_approval should pass through without checking."""
        guard = ToolGuard(require_approval=['restricted'])
        ctx = _make_run_context()
        call = ToolCallPart(tool_name='unrestricted', args='{}')
        tool_def = _make_tool_def('unrestricted')

        result = await guard.before_tool_execute(ctx, call=call, tool_def=tool_def, args={'a': 'b'})
        assert result == {'a': 'b'}

    def test_not_serializable(self) -> None:
        """ToolGuard should not be spec-serializable (takes a callable)."""
        assert ToolGuard.get_serialization_name() is None


# ---------------------------------------------------------------------------
# Integration: multiple guardrails on one agent
# ---------------------------------------------------------------------------


class TestComposition:
    async def test_input_and_output_guardrails_together(self) -> None:
        """Both input and output guardrails should work when combined."""
        agent = Agent(
            TestModel(custom_output_text='safe'),
            capabilities=[
                InputGuardrail(guard=lambda text: True),
                OutputGuardrail(guard=lambda text: True),
            ],
        )
        result = await agent.run('Hello')
        assert result.output == 'safe'

    async def test_input_guardrail_blocks_before_output(self) -> None:
        """If input guardrail blocks, output guardrail should never run."""
        output_called = False

        def output_guard(text: str) -> bool:  # pragma: no cover
            nonlocal output_called
            output_called = True
            return True

        agent = Agent(
            TestModel(custom_output_text='safe'),
            capabilities=[
                InputGuardrail(guard=lambda text: False),
                OutputGuardrail(guard=output_guard),
            ],
        )
        with pytest.raises(InputBlocked):
            await agent.run('Hello')

        assert not output_called


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# InputGuardrail — warn mode and context_guard
# ---------------------------------------------------------------------------


class TestInputGuardrailWarn:
    async def test_warn_mode_logs_instead_of_raising(self, caplog: pytest.LogCaptureFixture) -> None:
        """In warn mode, a failing guard should log a warning and allow the run."""
        agent = Agent(
            TestModel(),
            capabilities=[InputGuardrail(guard=lambda text: False, on_fail='warn')],
        )
        with caplog.at_level(logging.WARNING, logger='pydantic_harness.guardrails'):
            result = await agent.run('Hello')
        assert result.output is not None
        assert 'Input blocked by guardrail' in caplog.text

    async def test_warn_mode_no_log_when_passing(self, caplog: pytest.LogCaptureFixture) -> None:
        """In warn mode, a passing guard should not log anything."""
        agent = Agent(
            TestModel(),
            capabilities=[InputGuardrail(guard=lambda text: True, on_fail='warn')],
        )
        with caplog.at_level(logging.WARNING, logger='pydantic_harness.guardrails'):
            result = await agent.run('Hello')
        assert result.output is not None
        assert 'Input blocked' not in caplog.text

    async def test_context_guard_receives_ctx_and_text(self) -> None:
        """context_guard should receive RunContext and the prompt text."""
        received_ctx: list[RunContext[Any]] = []
        received_text: list[str] = []

        def ctx_guard(ctx: RunContext[Any], text: str) -> bool:
            received_ctx.append(ctx)
            received_text.append(text)
            return True

        agent = Agent(TestModel(), capabilities=[InputGuardrail(context_guard=ctx_guard)])
        await agent.run('test with context')
        assert len(received_ctx) == 1
        assert received_text == ['test with context']

    async def test_context_guard_blocks(self) -> None:
        """context_guard returning False should raise InputBlocked."""

        def ctx_guard(ctx: RunContext[Any], text: str) -> bool:
            return False

        agent = Agent(TestModel(), capabilities=[InputGuardrail(context_guard=ctx_guard)])
        with pytest.raises(InputBlocked):
            await agent.run('Hello')

    async def test_async_context_guard_warn_mode(self, caplog: pytest.LogCaptureFixture) -> None:
        async def ctx_guard(ctx: RunContext[Any], text: str) -> bool:
            return False

        agent = Agent(TestModel(), capabilities=[InputGuardrail(context_guard=ctx_guard, on_fail='warn')])
        with caplog.at_level(logging.WARNING, logger='pydantic_harness.guardrails'):
            result = await agent.run('Hello')
        assert result.output is not None
        assert 'Input blocked by guardrail' in caplog.text

    def test_no_guard_raises_value_error(self) -> None:
        """Neither guard nor context_guard should raise ValueError."""
        with pytest.raises(ValueError, match='Either guard or context_guard must be provided'):
            InputGuardrail()

    def test_both_guards_raises_value_error(self) -> None:
        """Both guard and context_guard should raise ValueError."""
        with pytest.raises(ValueError, match='Only one of guard or context_guard'):
            InputGuardrail(guard=lambda text: True, context_guard=lambda ctx, text: True)


# ---------------------------------------------------------------------------
# OutputGuardrail — warn mode and context_guard
# ---------------------------------------------------------------------------


class TestOutputGuardrailWarn:
    async def test_warn_mode_logs_instead_of_raising(self, caplog: pytest.LogCaptureFixture) -> None:
        """In warn mode, a failing guard should log and pass the result through."""
        agent = Agent(
            TestModel(custom_output_text='bad output'),
            capabilities=[OutputGuardrail(guard=lambda text: False, on_fail='warn')],
        )
        with caplog.at_level(logging.WARNING, logger='pydantic_harness.guardrails'):
            result = await agent.run('Hello')
        assert result.output == 'bad output'
        assert 'Output blocked by guardrail' in caplog.text

    async def test_warn_mode_no_log_when_passing(self, caplog: pytest.LogCaptureFixture) -> None:
        """In warn mode, a passing guard should not log."""
        agent = Agent(
            TestModel(custom_output_text='good output'),
            capabilities=[OutputGuardrail(guard=lambda text: True, on_fail='warn')],
        )
        with caplog.at_level(logging.WARNING, logger='pydantic_harness.guardrails'):
            result = await agent.run('Hello')
        assert result.output == 'good output'
        assert 'Output blocked' not in caplog.text

    async def test_context_guard_receives_ctx_and_text(self) -> None:
        """context_guard should receive RunContext and the output text."""
        received_text: list[str] = []

        def ctx_guard(ctx: RunContext[Any], text: str) -> bool:
            received_text.append(text)
            return True

        agent = Agent(
            TestModel(custom_output_text='hello world'),
            capabilities=[OutputGuardrail(context_guard=ctx_guard)],
        )
        await agent.run('test')
        assert received_text == ['hello world']

    async def test_context_guard_blocks(self) -> None:
        """context_guard returning False should raise OutputBlocked."""

        def ctx_guard(ctx: RunContext[Any], text: str) -> bool:
            return False

        agent = Agent(
            TestModel(custom_output_text='bad'),
            capabilities=[OutputGuardrail(context_guard=ctx_guard)],
        )
        with pytest.raises(OutputBlocked):
            await agent.run('Hello')

    async def test_async_context_guard_allows(self) -> None:
        """Async context_guard returning True should allow output."""

        async def ctx_guard(ctx: RunContext[Any], text: str) -> bool:
            return True

        agent = Agent(
            TestModel(custom_output_text='ok'),
            capabilities=[OutputGuardrail(context_guard=ctx_guard)],
        )
        result = await agent.run('test')
        assert result.output is not None

    async def test_async_context_guard_blocks(self) -> None:
        """Async context_guard returning False should raise OutputBlocked."""

        async def ctx_guard(ctx: RunContext[Any], text: str) -> bool:
            return False

        agent = Agent(
            TestModel(custom_output_text='bad'),
            capabilities=[OutputGuardrail(context_guard=ctx_guard)],
        )
        with pytest.raises(OutputBlocked):
            await agent.run('Hello')

    def test_no_guard_raises_value_error(self) -> None:
        """Neither guard nor context_guard should raise ValueError."""
        with pytest.raises(ValueError, match='Either guard or context_guard must be provided'):
            OutputGuardrail()

    def test_both_guards_raises_value_error(self) -> None:
        """Both guard and context_guard should raise ValueError."""
        with pytest.raises(ValueError, match='Only one of guard or context_guard'):
            OutputGuardrail(guard=lambda text: True, context_guard=lambda ctx, text: True)


# ---------------------------------------------------------------------------
# GuardrailResult
# ---------------------------------------------------------------------------


class TestGuardrailResult:
    def test_passed(self) -> None:
        r = GuardrailResult(passed=True, reason='all good')
        assert r.passed is True
        assert r.reason == 'all good'

    def test_failed(self) -> None:
        r = GuardrailResult(passed=False, reason='prompt injection detected')
        assert r.passed is False
        assert r.reason == 'prompt injection detected'

    def test_default_reason(self) -> None:
        r = GuardrailResult(passed=True)
        assert r.reason == ''

    def test_frozen(self) -> None:
        r = GuardrailResult(passed=True)
        with pytest.raises(AttributeError):
            r.passed = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GuardrailFailed exception
# ---------------------------------------------------------------------------


class TestGuardrailFailedException:
    def test_hierarchy(self) -> None:
        assert issubclass(GuardrailFailed, GuardrailError)

    def test_attributes(self) -> None:
        result = GuardrailResult(passed=False, reason='bad input')
        err = GuardrailFailed(result)
        assert err.result is result
        assert 'Guardrail failed: bad input' in str(err)


# ---------------------------------------------------------------------------
# AsyncGuardrail — blocking mode
# ---------------------------------------------------------------------------


class TestAsyncGuardrailBlocking:
    async def test_blocking_passes(self) -> None:
        """In blocking mode, a passing guard should allow the model call."""

        async def guard(messages: list[ModelMessage]) -> GuardrailResult:
            return GuardrailResult(passed=True)

        agent = Agent(TestModel(), capabilities=[AsyncGuardrail(guard=guard, mode='blocking')])
        result = await agent.run('Hello')
        assert result.output is not None

    async def test_blocking_fails(self) -> None:
        """In blocking mode, a failing guard should raise GuardrailFailed."""

        async def guard(messages: list[ModelMessage]) -> GuardrailResult:
            return GuardrailResult(passed=False, reason='blocked')

        agent = Agent(TestModel(), capabilities=[AsyncGuardrail(guard=guard, mode='blocking')])
        with pytest.raises(GuardrailFailed, match='blocked'):
            await agent.run('Hello')

    async def test_blocking_model_not_called_on_failure(self) -> None:
        """In blocking mode, the model should never be called if the guard fails."""
        model_called = False

        async def guard(messages: list[ModelMessage]) -> GuardrailResult:
            return GuardrailResult(passed=False, reason='nope')

        guardrail = AsyncGuardrail(guard=guard, mode='blocking')
        ctx = _make_run_context()
        req_ctx = _make_model_request_context()

        async def mock_handler(rc: Any) -> Any:
            nonlocal model_called
            model_called = True  # pragma: no cover

        with pytest.raises(GuardrailFailed):
            await guardrail.wrap_model_request(ctx, request_context=req_ctx, handler=mock_handler)
        assert not model_called


# ---------------------------------------------------------------------------
# AsyncGuardrail — monitoring mode
# ---------------------------------------------------------------------------


class TestAsyncGuardrailMonitoring:
    async def test_monitoring_passes(self) -> None:
        """In monitoring mode, a passing guard should return the model result."""

        async def guard(messages: list[ModelMessage]) -> GuardrailResult:
            return GuardrailResult(passed=True)

        agent = Agent(TestModel(), capabilities=[AsyncGuardrail(guard=guard, mode='monitoring')])
        result = await agent.run('Hello')
        assert result.output is not None

    async def test_monitoring_logs_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        """In monitoring mode, a failing guard should log but not raise."""

        async def guard(messages: list[ModelMessage]) -> GuardrailResult:
            return GuardrailResult(passed=False, reason='suspicious content')

        agent = Agent(
            TestModel(custom_output_text='model output'),
            capabilities=[AsyncGuardrail(guard=guard, mode='monitoring')],
        )
        with caplog.at_level(logging.WARNING, logger='pydantic_harness.guardrails'):
            result = await agent.run('Hello')
        assert result.output == 'model output'
        assert 'suspicious content' in caplog.text


# ---------------------------------------------------------------------------
# AsyncGuardrail — concurrent mode
# ---------------------------------------------------------------------------


class TestAsyncGuardrailConcurrent:
    async def test_concurrent_both_pass(self) -> None:
        """In concurrent mode, when guard passes, model result should be returned."""

        async def guard(messages: list[ModelMessage]) -> GuardrailResult:
            return GuardrailResult(passed=True)

        agent = Agent(TestModel(), capabilities=[AsyncGuardrail(guard=guard, mode='concurrent')])
        result = await agent.run('Hello')
        assert result.output is not None

    async def test_concurrent_guard_fails(self) -> None:
        """In concurrent mode, a failing guard should cancel model and raise."""

        async def guard(messages: list[ModelMessage]) -> GuardrailResult:
            return GuardrailResult(passed=False, reason='injection detected')

        agent = Agent(TestModel(), capabilities=[AsyncGuardrail(guard=guard, mode='concurrent')])
        with pytest.raises(GuardrailFailed, match='injection detected'):
            await agent.run('Hello')

    async def test_concurrent_guard_fails_first_cancels_model(self) -> None:
        """When the guard finishes before the model and fails, model should be cancelled."""
        model_completed = False

        async def slow_guard(messages: list[ModelMessage]) -> GuardrailResult:
            # Finishes immediately with failure
            return GuardrailResult(passed=False, reason='fast fail')

        guardrail = AsyncGuardrail(guard=slow_guard, mode='concurrent')
        ctx = _make_run_context()
        req_ctx = _make_model_request_context()

        async def slow_handler(rc: Any) -> Any:
            nonlocal model_completed
            await asyncio.sleep(10)  # pragma: no cover
            model_completed = True  # pragma: no cover

        with pytest.raises(GuardrailFailed, match='fast fail'):
            await guardrail.wrap_model_request(ctx, request_context=req_ctx, handler=slow_handler)
        assert not model_completed

    async def test_concurrent_model_finishes_first_guard_fails(self) -> None:
        """When the model finishes first but guard eventually fails, should still raise."""

        async def delayed_guard(messages: list[ModelMessage]) -> GuardrailResult:
            # Sleep long enough that asyncio.wait returns model as the first-completed
            await asyncio.sleep(0.05)
            return GuardrailResult(passed=False, reason='late failure')

        guardrail = AsyncGuardrail(guard=delayed_guard, mode='concurrent')
        ctx = _make_run_context()
        req_ctx = _make_model_request_context()

        async def fast_handler(rc: Any) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='done')])

        with pytest.raises(GuardrailFailed, match='late failure'):
            await guardrail.wrap_model_request(ctx, request_context=req_ctx, handler=fast_handler)

    async def test_concurrent_model_finishes_first_guard_passes(self) -> None:
        """When the model finishes first and guard passes, model response is returned."""

        async def delayed_guard(messages: list[ModelMessage]) -> GuardrailResult:
            await asyncio.sleep(0.05)
            return GuardrailResult(passed=True)

        guardrail = AsyncGuardrail(guard=delayed_guard, mode='concurrent')
        ctx = _make_run_context()
        req_ctx = _make_model_request_context()

        async def fast_handler(rc: Any) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='done')])

        response = await guardrail.wrap_model_request(ctx, request_context=req_ctx, handler=fast_handler)
        assert response.parts[0].content == 'done'  # type: ignore[union-attr]

    async def test_concurrent_default_mode(self) -> None:
        """The default mode should be 'concurrent'."""
        guard = AsyncMock(return_value=GuardrailResult(passed=True))
        guardrail = AsyncGuardrail(guard=guard)
        assert guardrail.mode == 'concurrent'


# ---------------------------------------------------------------------------
# AsyncGuardrail — context-aware guard (2-arg)
# ---------------------------------------------------------------------------


class TestAsyncGuardrailWithContext:
    async def test_guard_receives_ctx_and_messages(self) -> None:
        """A 2-arg guard should receive RunContext and messages."""
        received_ctx: list[RunContext[Any]] = []
        received_msgs: list[list[ModelMessage]] = []

        async def ctx_guard(ctx: RunContext[Any], messages: list[ModelMessage]) -> GuardrailResult:
            received_ctx.append(ctx)
            received_msgs.append(messages)
            return GuardrailResult(passed=True)

        agent = Agent(TestModel(), capabilities=[AsyncGuardrail(guard=ctx_guard, mode='blocking')])
        await agent.run('Hello')
        assert len(received_ctx) == 1
        assert len(received_msgs) == 1

    async def test_1arg_guard_receives_only_messages(self) -> None:
        """A 1-arg guard should receive only messages."""
        received_msgs: list[list[ModelMessage]] = []

        async def msg_guard(messages: list[ModelMessage]) -> GuardrailResult:
            received_msgs.append(messages)
            return GuardrailResult(passed=True)

        agent = Agent(TestModel(), capabilities=[AsyncGuardrail(guard=msg_guard, mode='blocking')])
        await agent.run('Hello')
        assert len(received_msgs) == 1


# ---------------------------------------------------------------------------
# AsyncGuardrail — misc
# ---------------------------------------------------------------------------


class TestAsyncGuardrailMisc:
    def test_not_serializable(self) -> None:
        """AsyncGuardrail should not be spec-serializable."""
        assert AsyncGuardrail.get_serialization_name() is None


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestImports:
    def test_import_from_package(self) -> None:
        """All public symbols should be importable from the package root."""
        from pydantic_harness import (
            AsyncGuardrail,
            BudgetExceededError,
            CostGuard,
            GuardrailError,
            GuardrailFailed,
            GuardrailResult,
            InputBlocked,
            InputGuardrail,
            OutputBlocked,
            OutputGuardrail,
            ToolBlocked,
            ToolGuard,
        )

        assert AsyncGuardrail is not None
        assert InputGuardrail is not None
        assert OutputGuardrail is not None
        assert CostGuard is not None
        assert ToolGuard is not None
        assert GuardrailError is not None
        assert InputBlocked is not None
        assert OutputBlocked is not None
        assert BudgetExceededError is not None
        assert ToolBlocked is not None
        assert GuardrailResult is not None
        assert GuardrailFailed is not None

    def test_import_from_guardrails_module(self) -> None:
        """All public symbols should be importable from the guardrails module."""
        from pydantic_harness.guardrails import (
            AsyncGuardrail,
            BudgetExceededError,
            CostGuard,
            GuardrailError,
            GuardrailFailed,
            GuardrailResult,
            InputBlocked,
            InputGuardrail,
            OutputBlocked,
            OutputGuardrail,
            ToolBlocked,
            ToolGuard,
        )

        assert AsyncGuardrail is not None
        assert InputGuardrail is not None
        assert OutputGuardrail is not None
        assert CostGuard is not None
        assert ToolGuard is not None
        assert GuardrailError is not None
        assert InputBlocked is not None
        assert OutputBlocked is not None
        assert BudgetExceededError is not None
        assert ToolBlocked is not None
        assert GuardrailResult is not None
        assert GuardrailFailed is not None


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_run_context(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    prompt: str | list[str] | None = None,
) -> RunContext[None]:
    """Create a minimal RunContext for unit testing hooks directly."""
    return RunContext[None](
        deps=None,
        model=TestModel(),
        usage=RunUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        prompt=prompt,
    )


def _make_tool_def(name: str) -> ToolDefinition:
    """Create a minimal ToolDefinition for testing."""
    return ToolDefinition(name=name, description=f'Tool {name}')


class _MockRequestContext:
    """Minimal stand-in for ModelRequestContext in unit tests."""


def _mock_request_context() -> Any:
    """Create a mock request context for CostGuard tests."""
    return _MockRequestContext()


def _make_model_request_context() -> Any:
    """Create a mock ModelRequestContext with a messages list for AsyncGuardrail tests."""

    class _Ctx:
        messages: list[ModelMessage] = []

    return _Ctx()
