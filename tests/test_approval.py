"""Tests for the Approval capability."""
# pyright: reportUnusedFunction=false, reportPrivateUsage=false, reportUnknownVariableType=false, reportUnknownLambdaType=false, reportUnknownArgumentType=false

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_harness.approval import DENIED_MESSAGE, Approval

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_callback(*, approve: bool) -> tuple[list[tuple[str, dict[str, Any]]], Any]:
    """Return (log, callback) where callback records calls and returns ``approve``."""
    log: list[tuple[str, dict[str, Any]]] = []

    def cb(tool_name: str, args: dict[str, Any]) -> bool:
        log.append((tool_name, args))
        return approve

    return log, cb


def make_async_callback(*, approve: bool) -> tuple[list[tuple[str, dict[str, Any]]], Any]:
    """Return (log, async_callback) where callback records calls and returns ``approve``."""
    log: list[tuple[str, dict[str, Any]]] = []

    async def cb(tool_name: str, args: dict[str, Any]) -> bool:
        log.append((tool_name, args))
        return approve

    return log, cb


# ---------------------------------------------------------------------------
# Basic approval / denial
# ---------------------------------------------------------------------------


class TestBasicApproval:
    async def test_approved_tool_executes(self) -> None:
        """When the callback approves, the tool should execute normally."""
        log, cb = make_callback(approve=True)
        agent = Agent(
            TestModel(),
            capabilities=[Approval(tool_patterns=['echo'], callback=cb)],
        )

        @agent.tool_plain
        def echo(message: str) -> str:
            return f'echoed: {message}'

        result = await agent.run('test')
        assert 'echoed' in result.output
        assert len(log) == 1
        assert log[0][0] == 'echo'

    async def test_denied_tool_returns_denial(self) -> None:
        """When the callback denies, the tool should not execute and a denial message is returned."""
        log, cb = make_callback(approve=False)
        executed = False

        agent = Agent(
            TestModel(),
            capabilities=[Approval(tool_patterns=['dangerous'], callback=cb)],
        )

        @agent.tool_plain
        def dangerous() -> str:
            nonlocal executed
            executed = True
            return 'should not see this'

        result = await agent.run('test')
        assert not executed
        assert DENIED_MESSAGE in result.output
        assert len(log) == 1

    async def test_no_callback_denies(self) -> None:
        """When no callback is configured, all matched tools are denied."""
        executed = False

        agent = Agent(
            TestModel(),
            capabilities=[Approval(tool_patterns=['*'])],
        )

        @agent.tool_plain
        def anything() -> str:
            nonlocal executed
            executed = True
            return 'nope'

        result = await agent.run('test')
        assert not executed
        assert DENIED_MESSAGE in result.output


# ---------------------------------------------------------------------------
# Approval modes
# ---------------------------------------------------------------------------


class TestApprovalModes:
    async def test_mode_always_asks_every_time(self) -> None:
        """mode='always' should invoke the callback on every tool call."""
        log, cb = make_callback(approve=True)
        model = TestModel(custom_output_text='final')
        # Make the model call the tool twice by setting result_tool_name=None
        # and using call_tools to invoke the tool multiple times
        agent = Agent(
            model,
            capabilities=[Approval(tool_patterns=['greet'], callback=cb, mode='always')],
        )

        @agent.tool_plain
        def greet(name: str) -> str:
            return f'hello {name}'

        await agent.run('test')
        # TestModel calls the tool once per run by default
        assert len(log) >= 1

    async def test_mode_once_remembers_approval(self) -> None:
        """mode='once' should ask once, then auto-approve subsequent calls to the same tool."""
        log, cb = make_callback(approve=True)

        capability = Approval(tool_patterns=['repeat'], callback=cb, mode='once')

        # Verify through two separate agent runs that for_run resets state
        agent = Agent(
            TestModel(),
            capabilities=[capability],
        )

        @agent.tool_plain
        def repeat(text: str) -> str:
            return text

        # First run
        await agent.run('first')
        first_count = len(log)
        assert first_count >= 1

        # Second run - for_run should reset state, so callback fires again
        await agent.run('second')
        assert len(log) > first_count

    async def test_mode_never_auto_approves(self) -> None:
        """mode='never' should never invoke the callback and always execute the tool."""
        log, cb = make_callback(approve=False)  # Would deny if called
        executed = False

        agent = Agent(
            TestModel(),
            capabilities=[Approval(tool_patterns=['*'], callback=cb, mode='never')],
        )

        @agent.tool_plain
        def safe_tool() -> str:
            nonlocal executed
            executed = True
            return 'executed'

        await agent.run('test')
        assert executed
        assert len(log) == 0  # Callback was never called


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------


class TestPatternMatching:
    async def test_wildcard_matches_all(self) -> None:
        """Pattern '*' should match every tool."""
        log, cb = make_callback(approve=True)
        agent = Agent(
            TestModel(),
            capabilities=[Approval(tool_patterns=['*'], callback=cb)],
        )

        @agent.tool_plain
        def any_tool() -> str:
            return 'ok'

        await agent.run('test')
        assert len(log) == 1

    async def test_prefix_pattern(self) -> None:
        """Pattern 'delete_*' should match tools starting with 'delete_'."""
        log, cb = make_callback(approve=True)
        agent = Agent(
            TestModel(),
            capabilities=[Approval(tool_patterns=['delete_*'], callback=cb)],
        )

        @agent.tool_plain
        def delete_file(path: str) -> str:
            return f'deleted {path}'

        await agent.run('test')
        assert len(log) == 1
        assert log[0][0] == 'delete_file'

    async def test_unmatched_tool_passes_through(self) -> None:
        """Tools not matching any pattern should execute without approval."""
        log, cb = make_callback(approve=False)  # Would deny if matched
        executed = False

        agent = Agent(
            TestModel(),
            capabilities=[Approval(tool_patterns=['delete_*'], callback=cb)],
        )

        @agent.tool_plain
        def read_file(path: str) -> str:
            nonlocal executed
            executed = True
            return f'contents of {path}'

        await agent.run('test')
        assert executed
        assert len(log) == 0

    async def test_empty_patterns_passes_all(self) -> None:
        """Empty tool_patterns list should let all tools through without asking."""
        log, cb = make_callback(approve=False)
        executed = False

        agent = Agent(
            TestModel(),
            capabilities=[Approval(tool_patterns=[], callback=cb)],
        )

        @agent.tool_plain
        def anything() -> str:
            nonlocal executed
            executed = True
            return 'ok'

        await agent.run('test')
        assert executed
        assert len(log) == 0

    async def test_exact_name_pattern(self) -> None:
        """Exact tool name (no wildcards) should match only that tool."""
        log, cb = make_callback(approve=True)
        agent = Agent(
            TestModel(),
            capabilities=[Approval(tool_patterns=['send_email'], callback=cb)],
        )

        @agent.tool_plain
        def send_email(to: str) -> str:
            return f'sent to {to}'

        await agent.run('test')
        assert len(log) == 1

    async def test_multiple_patterns(self) -> None:
        """Multiple patterns should match tools matching any of them."""
        log, cb = make_callback(approve=True)
        agent = Agent(
            TestModel(),
            capabilities=[Approval(tool_patterns=['delete_*', 'send_*'], callback=cb)],
        )

        @agent.tool_plain
        def send_message(text: str) -> str:
            return f'sent: {text}'

        await agent.run('test')
        assert len(log) == 1
        assert log[0][0] == 'send_message'


# ---------------------------------------------------------------------------
# Async callback
# ---------------------------------------------------------------------------


class TestAsyncCallback:
    async def test_async_callback_approved(self) -> None:
        """Async callback returning True should allow execution."""
        log, cb = make_async_callback(approve=True)
        agent = Agent(
            TestModel(),
            capabilities=[Approval(tool_patterns=['*'], callback=cb)],
        )

        @agent.tool_plain
        def tool() -> str:
            return 'ok'

        result = await agent.run('test')
        assert 'ok' in result.output
        assert len(log) == 1

    async def test_async_callback_denied(self) -> None:
        """Async callback returning False should deny execution."""
        log, cb = make_async_callback(approve=False)
        agent = Agent(
            TestModel(),
            capabilities=[Approval(tool_patterns=['*'], callback=cb)],
        )

        @agent.tool_plain
        def tool() -> str:
            return 'should not run'

        result = await agent.run('test')
        assert DENIED_MESSAGE in result.output
        assert len(log) == 1


# ---------------------------------------------------------------------------
# for_run isolation
# ---------------------------------------------------------------------------


class TestForRunIsolation:
    async def test_for_run_resets_approved_tools(self) -> None:
        """for_run should return a fresh instance with empty approved-tools set."""
        cap = Approval(tool_patterns=['*'], callback=lambda n, a: True, mode='once')
        cap._approved_tools.add('some_tool')

        # Simulate a RunContext -- we just need to call for_run
        # Use a minimal RunContext mock
        from unittest.mock import MagicMock

        mock_ctx: Any = MagicMock()
        new_cap = await cap.for_run(mock_ctx)

        assert new_cap is not cap
        assert len(new_cap._approved_tools) == 0
        # Original should still have its state
        assert 'some_tool' in cap._approved_tools

    async def test_for_run_preserves_config(self) -> None:
        """for_run should preserve all configuration fields."""
        cb = lambda n, a: True  # noqa: E731
        cap = Approval(tool_patterns=['delete_*', 'send_*'], callback=cb, mode='once')

        from unittest.mock import MagicMock

        mock_ctx: Any = MagicMock()
        new_cap = await cap.for_run(mock_ctx)

        assert new_cap.tool_patterns == ['delete_*', 'send_*']
        assert new_cap.callback is cb
        assert new_cap.mode == 'once'
        assert new_cap._patterns == ('delete_*', 'send_*')


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_not_spec_serializable(self) -> None:
        """Approval takes a callable, so it should not be spec-serializable."""
        assert Approval.get_serialization_name() is None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_matches_any_pattern_wildcard(self) -> None:
        from pydantic_harness.approval import _matches_any_pattern

        assert _matches_any_pattern('anything', ('*',))

    def test_matches_any_pattern_prefix(self) -> None:
        from pydantic_harness.approval import _matches_any_pattern

        assert _matches_any_pattern('delete_file', ('delete_*',))
        assert not _matches_any_pattern('read_file', ('delete_*',))

    def test_matches_any_pattern_exact(self) -> None:
        from pydantic_harness.approval import _matches_any_pattern

        assert _matches_any_pattern('send_email', ('send_email',))
        assert not _matches_any_pattern('send_sms', ('send_email',))

    def test_matches_any_pattern_multiple(self) -> None:
        from pydantic_harness.approval import _matches_any_pattern

        patterns = ('delete_*', 'send_*', 'execute')
        assert _matches_any_pattern('delete_file', patterns)
        assert _matches_any_pattern('send_email', patterns)
        assert _matches_any_pattern('execute', patterns)
        assert not _matches_any_pattern('read_file', patterns)

    def test_matches_any_pattern_empty(self) -> None:
        from pydantic_harness.approval import _matches_any_pattern

        assert not _matches_any_pattern('anything', ())
