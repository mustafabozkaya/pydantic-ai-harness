"""Tests for ToolOrphanRepair capability."""

from __future__ import annotations

import logging
import warnings

import pytest
from pydantic_ai.messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from pydantic_harness.tool_orphan_repair import ToolOrphanRepair
from pydantic_harness.tool_orphan_repair import _repair_messages  # pyright: ignore[reportPrivateUsage]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ModelRequestPart = SystemPromptPart | UserPromptPart | ToolReturnPart | RetryPromptPart


def _user_request(text: str = 'hello') -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _tool_call_response(*calls: tuple[str, str]) -> ModelResponse:
    """Create a response with ToolCallParts: (tool_name, tool_call_id)."""
    return ModelResponse(parts=[ToolCallPart(tool_name=n, args='{}', tool_call_id=cid) for n, cid in calls])


def _tool_return_request(*returns: tuple[str, str], extra_parts: list[ModelRequestPart] | None = None) -> ModelRequest:
    """Create a request with ToolReturnParts: (tool_name, tool_call_id)."""
    parts: list[ModelRequestPart] = [ToolReturnPart(tool_name=n, content='ok', tool_call_id=cid) for n, cid in returns]
    if extra_parts:  # pragma: no cover – convenience parameter unused so far
        parts.extend(extra_parts)
    return ModelRequest(parts=parts)


# ---------------------------------------------------------------------------
# No-op / passthrough
# ---------------------------------------------------------------------------


class TestNoRepairsNeeded:
    def test_empty_messages(self) -> None:
        result: list[ModelMessage] = _repair_messages([], warn=False)
        assert result == []

    def test_single_user_request(self) -> None:
        msgs: list[ModelMessage] = [_user_request()]
        assert _repair_messages(msgs, warn=False) == msgs

    def test_clean_tool_round_trip(self) -> None:
        """Response with tool call followed by request with matching return -- no repair."""
        msgs: list[ModelMessage] = [
            _user_request(),
            _tool_call_response(('get_weather', 'call_1')),
            _tool_return_request(('get_weather', 'call_1')),
        ]
        result = _repair_messages(msgs, warn=False)
        assert len(result) == 3
        # All messages pass through unchanged.
        assert result[0] is msgs[0]
        assert result[1] is msgs[1]

    def test_response_without_tool_calls(self) -> None:
        """A plain text response needs no repair."""
        msgs: list[ModelMessage] = [
            _user_request(),
            ModelResponse(parts=[TextPart(content='Sure!')]),
        ]
        result = _repair_messages(msgs, warn=False)
        assert len(result) == 2

    def test_clean_builtin_tool_round_trip(self) -> None:
        """BuiltinToolCallPart with matching BuiltinToolReturnPart in same response."""
        msgs: list[ModelMessage] = [
            _user_request(),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(tool_name='code_exec', args='{}', tool_call_id='bc_1'),
                    BuiltinToolReturnPart(tool_name='code_exec', content='result', tool_call_id='bc_1'),
                ]
            ),
        ]
        result = _repair_messages(msgs, warn=False)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Orphaned tool calls (call without matching return)
# ---------------------------------------------------------------------------


class TestOrphanedToolCalls:
    def test_injects_synthetic_return(self) -> None:
        """Orphaned ToolCallPart gets a synthetic ToolReturnPart in the next request."""
        msgs: list[ModelMessage] = [
            _user_request(),
            _tool_call_response(('get_weather', 'call_1')),
            # Request has no return for call_1.
            _user_request('what now?'),
        ]
        result = _repair_messages(msgs, warn=False)
        assert len(result) == 3

        repaired_request = result[2]
        assert isinstance(repaired_request, ModelRequest)
        return_parts = [p for p in repaired_request.parts if isinstance(p, ToolReturnPart)]
        assert len(return_parts) == 1
        assert return_parts[0].tool_call_id == 'call_1'
        assert return_parts[0].tool_name == 'get_weather'
        assert return_parts[0].content == 'Tool call was not completed.'

    def test_injects_return_for_multiple_orphaned_calls(self) -> None:
        msgs: list[ModelMessage] = [
            _user_request(),
            _tool_call_response(('tool_a', 'ca'), ('tool_b', 'cb')),
            _user_request('continue'),
        ]
        result = _repair_messages(msgs, warn=False)
        repaired_request = result[2]
        assert isinstance(repaired_request, ModelRequest)
        return_parts = [p for p in repaired_request.parts if isinstance(p, ToolReturnPart)]
        assert len(return_parts) == 2
        assert {p.tool_call_id for p in return_parts} == {'ca', 'cb'}

    def test_partial_match_injects_only_missing(self) -> None:
        """One call matched, one orphaned -- only the orphan gets a synthetic return."""
        msgs: list[ModelMessage] = [
            _user_request(),
            _tool_call_response(('tool_a', 'ca'), ('tool_b', 'cb')),
            _tool_return_request(('tool_a', 'ca')),
        ]
        result = _repair_messages(msgs, warn=False)
        repaired_request = result[2]
        assert isinstance(repaired_request, ModelRequest)
        return_parts = [p for p in repaired_request.parts if isinstance(p, ToolReturnPart)]
        assert len(return_parts) == 2
        ids = {p.tool_call_id for p in return_parts}
        assert ids == {'ca', 'cb'}

    def test_retry_prompt_counts_as_match(self) -> None:
        """A RetryPromptPart with matching tool_call_id counts as a match."""
        msgs: list[ModelMessage] = [
            _user_request(),
            _tool_call_response(('get_weather', 'call_1')),
            ModelRequest(
                parts=[
                    RetryPromptPart(content='bad args', tool_name='get_weather', tool_call_id='call_1'),
                ]
            ),
        ]
        result = _repair_messages(msgs, warn=False)
        assert len(result) == 3
        repaired_request = result[2]
        assert isinstance(repaired_request, ModelRequest)
        # No synthetic return injected -- RetryPromptPart matched the call.
        return_parts = [p for p in repaired_request.parts if isinstance(p, ToolReturnPart)]
        assert len(return_parts) == 0

    def test_custom_orphan_content(self) -> None:
        msgs: list[ModelMessage] = [
            _user_request(),
            _tool_call_response(('fetch', 'c1')),
            _user_request('next'),
        ]
        result = _repair_messages(msgs, orphan_call_content='timed out', warn=False)
        repaired_request = result[2]
        assert isinstance(repaired_request, ModelRequest)
        return_parts = [p for p in repaired_request.parts if isinstance(p, ToolReturnPart)]
        assert return_parts[0].content == 'timed out'


# ---------------------------------------------------------------------------
# Orphaned builtin tool calls
# ---------------------------------------------------------------------------


class TestOrphanedBuiltinToolCalls:
    def test_injects_builtin_return_in_same_response(self) -> None:
        """Orphaned BuiltinToolCallPart gets a BuiltinToolReturnPart in the same response."""
        msgs: list[ModelMessage] = [
            _user_request(),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(tool_name='code_exec', args='print(1)', tool_call_id='bc_1'),
                ]
            ),
        ]
        result = _repair_messages(msgs, warn=False)
        assert len(result) == 2
        repaired_response = result[1]
        assert isinstance(repaired_response, ModelResponse)
        builtin_returns = [p for p in repaired_response.parts if isinstance(p, BuiltinToolReturnPart)]
        assert len(builtin_returns) == 1
        assert builtin_returns[0].tool_call_id == 'bc_1'
        assert builtin_returns[0].tool_name == 'code_exec'
        assert builtin_returns[0].content == 'Tool call was not completed.'


# ---------------------------------------------------------------------------
# Orphaned tool returns (return without matching call)
# ---------------------------------------------------------------------------


class TestOrphanedToolReturns:
    def test_strips_return_with_no_matching_call(self) -> None:
        """ToolReturnPart whose call_id doesn't match any call is stripped."""
        msgs: list[ModelMessage] = [
            _user_request(),
            _tool_call_response(('get_weather', 'call_1')),
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name='get_weather', content='ok', tool_call_id='call_1'),
                    ToolReturnPart(tool_name='ghost', content='orphaned', tool_call_id='no_match'),
                ]
            ),
        ]
        result = _repair_messages(msgs, warn=False)
        repaired_request = result[2]
        assert isinstance(repaired_request, ModelRequest)
        return_parts = [p for p in repaired_request.parts if isinstance(p, ToolReturnPart)]
        assert len(return_parts) == 1
        assert return_parts[0].tool_call_id == 'call_1'

    def test_strips_retry_prompt_with_no_matching_call(self) -> None:
        """RetryPromptPart whose call_id doesn't match any call is stripped."""
        msgs: list[ModelMessage] = [
            _user_request(),
            _tool_call_response(('get_weather', 'call_1')),
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name='get_weather', content='ok', tool_call_id='call_1'),
                    RetryPromptPart(content='retry me', tool_name='phantom', tool_call_id='no_match'),
                ]
            ),
        ]
        result = _repair_messages(msgs, warn=False)
        repaired_request = result[2]
        assert isinstance(repaired_request, ModelRequest)
        retry_parts = [p for p in repaired_request.parts if isinstance(p, RetryPromptPart)]
        assert len(retry_parts) == 0


# ---------------------------------------------------------------------------
# Trailing response with unmatched tool calls
# ---------------------------------------------------------------------------


class TestTrailingResponse:
    def test_drops_trailing_response_with_only_tool_calls(self) -> None:
        """A trailing response containing only unmatched tool calls is dropped entirely."""
        msgs: list[ModelMessage] = [
            _user_request(),
            _tool_call_response(('fetch', 'c1')),
        ]
        result = _repair_messages(msgs, warn=False)
        assert len(result) == 1
        assert isinstance(result[0], ModelRequest)

    def test_keeps_trailing_response_with_text_strips_calls(self) -> None:
        """Trailing response with text + tool calls: keep text, strip calls."""
        msgs: list[ModelMessage] = [
            _user_request(),
            ModelResponse(
                parts=[
                    TextPart(content='Let me check...'),
                    ToolCallPart(tool_name='fetch', args='{}', tool_call_id='c1'),
                ]
            ),
        ]
        result = _repair_messages(msgs, warn=False)
        assert len(result) == 2
        repaired_response = result[1]
        assert isinstance(repaired_response, ModelResponse)
        assert len(repaired_response.parts) == 1
        assert isinstance(repaired_response.parts[0], TextPart)


# ---------------------------------------------------------------------------
# Empty request after stripping
# ---------------------------------------------------------------------------


class TestEmptyRequestPlaceholder:
    def test_orphaned_return_replaced_by_synthetic(self) -> None:
        """Request with only an orphaned return: the orphan is stripped and a synthetic return injected."""
        msgs: list[ModelMessage] = [
            _user_request(),
            _tool_call_response(('get_weather', 'call_1')),
            # Request has a return for the wrong id -- orphaned.
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name='ghost', content='orphaned', tool_call_id='wrong_id'),
                ]
            ),
        ]
        result = _repair_messages(msgs, warn=False)
        repaired_request = result[2]
        assert isinstance(repaired_request, ModelRequest)
        # The orphaned return was stripped, but a synthetic return for call_1 was injected.
        return_parts = [p for p in repaired_request.parts if isinstance(p, ToolReturnPart)]
        assert len(return_parts) == 1
        assert return_parts[0].tool_call_id == 'call_1'
        assert return_parts[0].content == 'Tool call was not completed.'

    def test_system_prompt_only_request_gets_synthetic_return(self) -> None:
        """A request with SystemPromptPart + orphaned return: synthetic return injected, system prompt kept."""
        msgs: list[ModelMessage] = [
            _user_request(),
            _tool_call_response(('fetch', 'c1')),
            ModelRequest(
                parts=[
                    SystemPromptPart(content='You are helpful.'),
                    ToolReturnPart(tool_name='ghost', content='orphaned', tool_call_id='wrong_id'),
                ]
            ),
        ]
        result = _repair_messages(msgs, warn=False)
        repaired_request = result[2]
        assert isinstance(repaired_request, ModelRequest)
        # System prompt kept, orphaned return stripped, synthetic return injected.
        system_parts = [p for p in repaired_request.parts if isinstance(p, SystemPromptPart)]
        assert len(system_parts) == 1
        return_parts = [p for p in repaired_request.parts if isinstance(p, ToolReturnPart)]
        assert len(return_parts) == 1
        assert return_parts[0].tool_call_id == 'c1'


# ---------------------------------------------------------------------------
# Warning behavior
# ---------------------------------------------------------------------------


class TestWarnings:
    def test_emits_warning_when_repairs_made(self) -> None:
        msgs: list[ModelMessage] = [
            _user_request(),
            _tool_call_response(('fetch', 'c1')),
            _user_request('next'),
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            _repair_messages(msgs, warn=True)

        assert len(w) == 1
        assert 'ToolOrphanRepair' in str(w[0].message)
        assert '1 orphaned' in str(w[0].message)

    def test_no_warning_when_clean(self) -> None:
        msgs: list[ModelMessage] = [
            _user_request(),
            ModelResponse(parts=[TextPart(content='hi')]),
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            _repair_messages(msgs, warn=True)

        assert len(w) == 0

    def test_no_warning_when_disabled(self) -> None:
        msgs: list[ModelMessage] = [
            _user_request(),
            _tool_call_response(('fetch', 'c1')),
            _user_request('next'),
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            _repair_messages(msgs, warn=False)

        assert len(w) == 0


# ---------------------------------------------------------------------------
# Complex / multi-turn scenarios
# ---------------------------------------------------------------------------


class TestMultiTurnScenarios:
    def test_multiple_response_request_pairs(self) -> None:
        """Two consecutive tool round-trips, second one orphaned."""
        msgs: list[ModelMessage] = [
            _user_request(),
            # First round-trip: clean.
            _tool_call_response(('tool_a', 'ca')),
            _tool_return_request(('tool_a', 'ca')),
            # Second round-trip: orphaned.
            _tool_call_response(('tool_b', 'cb')),
            _user_request('done'),
        ]
        result = _repair_messages(msgs, warn=False)
        assert len(result) == 5

        # The second request should have a synthetic return.
        repaired = result[4]
        assert isinstance(repaired, ModelRequest)
        return_parts = [p for p in repaired.parts if isinstance(p, ToolReturnPart)]
        assert len(return_parts) == 1
        assert return_parts[0].tool_call_id == 'cb'

    def test_mixed_orphaned_and_clean_in_same_request(self) -> None:
        """Request has one valid return and one orphaned return."""
        msgs: list[ModelMessage] = [
            _user_request(),
            _tool_call_response(('tool_a', 'ca')),
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name='tool_a', content='ok', tool_call_id='ca'),
                    ToolReturnPart(tool_name='phantom', content='bad', tool_call_id='no_match'),
                    UserPromptPart(content='next step'),
                ]
            ),
        ]
        result = _repair_messages(msgs, warn=False)
        repaired = result[2]
        assert isinstance(repaired, ModelRequest)
        return_parts = [p for p in repaired.parts if isinstance(p, ToolReturnPart)]
        assert len(return_parts) == 1
        assert return_parts[0].tool_call_id == 'ca'
        user_parts = [p for p in repaired.parts if isinstance(p, UserPromptPart)]
        assert len(user_parts) == 1

    def test_request_not_following_response_passes_through(self) -> None:
        """A ModelRequest at position 0 (not preceded by a response) passes through."""
        msgs: list[ModelMessage] = [
            _user_request('first'),
            ModelResponse(parts=[TextPart(content='hi')]),
            _user_request('second'),
        ]
        result = _repair_messages(msgs, warn=False)
        assert len(result) == 3

    def test_builtin_and_regular_orphans_in_same_response(self) -> None:
        """Response has both an orphaned builtin call and an orphaned regular call."""
        msgs: list[ModelMessage] = [
            _user_request(),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(tool_name='code_exec', args='x=1', tool_call_id='bc_1'),
                    ToolCallPart(tool_name='get_weather', args='{}', tool_call_id='tc_1'),
                ]
            ),
            _user_request('continue'),
        ]
        result = _repair_messages(msgs, warn=False)
        assert len(result) == 3

        # Response should have the builtin call + synthetic builtin return.
        repaired_response = result[1]
        assert isinstance(repaired_response, ModelResponse)
        builtin_returns = [p for p in repaired_response.parts if isinstance(p, BuiltinToolReturnPart)]
        assert len(builtin_returns) == 1
        assert builtin_returns[0].tool_call_id == 'bc_1'

        # Request should have a synthetic regular return.
        repaired_request = result[2]
        assert isinstance(repaired_request, ModelRequest)
        return_parts = [p for p in repaired_request.parts if isinstance(p, ToolReturnPart)]
        assert len(return_parts) == 1
        assert return_parts[0].tool_call_id == 'tc_1'

    def test_preserves_existing_user_prompt_parts(self) -> None:
        """Existing UserPromptPart in a request is preserved alongside injected returns."""
        msgs: list[ModelMessage] = [
            _user_request(),
            _tool_call_response(('fetch', 'c1')),
            ModelRequest(
                parts=[
                    UserPromptPart(content='user text'),
                ]
            ),
        ]
        result = _repair_messages(msgs, warn=False)
        repaired = result[2]
        assert isinstance(repaired, ModelRequest)
        user_parts = [p for p in repaired.parts if isinstance(p, UserPromptPart)]
        assert len(user_parts) == 1
        assert user_parts[0].content == 'user text'
        return_parts = [p for p in repaired.parts if isinstance(p, ToolReturnPart)]
        assert len(return_parts) == 1


# ---------------------------------------------------------------------------
# Debug logging
# ---------------------------------------------------------------------------


class TestDebugLogging:
    def test_logs_injected_synthetic_return(self, caplog: pytest.LogCaptureFixture) -> None:
        msgs: list[ModelMessage] = [
            _user_request(),
            _tool_call_response(('get_weather', 'call_1')),
            _user_request('next'),
        ]
        with caplog.at_level(logging.DEBUG, logger='pydantic_harness.tool_orphan_repair'):
            _repair_messages(msgs, warn=False)
        assert any('Injected synthetic ToolReturnPart' in r.message and 'call_1' in r.message for r in caplog.records)

    def test_logs_stripped_orphaned_return(self, caplog: pytest.LogCaptureFixture) -> None:
        msgs: list[ModelMessage] = [
            _user_request(),
            _tool_call_response(('get_weather', 'call_1')),
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name='get_weather', content='ok', tool_call_id='call_1'),
                    ToolReturnPart(tool_name='ghost', content='orphaned', tool_call_id='no_match'),
                ]
            ),
        ]
        with caplog.at_level(logging.DEBUG, logger='pydantic_harness.tool_orphan_repair'):
            _repair_messages(msgs, warn=False)
        assert any('Stripped orphaned ToolReturnPart' in r.message and 'no_match' in r.message for r in caplog.records)

    def test_logs_stripped_orphaned_retry_prompt(self, caplog: pytest.LogCaptureFixture) -> None:
        msgs: list[ModelMessage] = [
            _user_request(),
            _tool_call_response(('get_weather', 'call_1')),
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name='get_weather', content='ok', tool_call_id='call_1'),
                    RetryPromptPart(content='retry', tool_name='phantom', tool_call_id='no_match'),
                ]
            ),
        ]
        with caplog.at_level(logging.DEBUG, logger='pydantic_harness.tool_orphan_repair'):
            _repair_messages(msgs, warn=False)
        assert any('Stripped orphaned RetryPromptPart' in r.message and 'no_match' in r.message for r in caplog.records)

    def test_logs_dropped_trailing_response(self, caplog: pytest.LogCaptureFixture) -> None:
        msgs: list[ModelMessage] = [
            _user_request(),
            _tool_call_response(('fetch', 'c1')),
        ]
        with caplog.at_level(logging.DEBUG, logger='pydantic_harness.tool_orphan_repair'):
            _repair_messages(msgs, warn=False)
        assert any('Dropped trailing response' in r.message and 'c1' in r.message for r in caplog.records)

    def test_logs_stripped_trailing_tool_calls(self, caplog: pytest.LogCaptureFixture) -> None:
        msgs: list[ModelMessage] = [
            _user_request(),
            ModelResponse(
                parts=[
                    TextPart(content='Let me check...'),
                    ToolCallPart(tool_name='fetch', args='{}', tool_call_id='c1'),
                ]
            ),
        ]
        with caplog.at_level(logging.DEBUG, logger='pydantic_harness.tool_orphan_repair'):
            _repair_messages(msgs, warn=False)
        assert any('Stripped orphaned tool call' in r.message and 'c1' in r.message for r in caplog.records)

    def test_logs_builtin_tool_call_repair(self, caplog: pytest.LogCaptureFixture) -> None:
        msgs: list[ModelMessage] = [
            _user_request(),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(tool_name='code_exec', args='print(1)', tool_call_id='bc_1'),
                ]
            ),
        ]
        with caplog.at_level(logging.DEBUG, logger='pydantic_harness.tool_orphan_repair'):
            _repair_messages(msgs, warn=False)
        assert any(
            'Injected synthetic BuiltinToolReturnPart' in r.message and 'bc_1' in r.message for r in caplog.records
        )

    def test_logs_placeholder_insertion(self, caplog: pytest.LogCaptureFixture) -> None:
        """When all parts are stripped and only system prompt remains, a placeholder is logged."""
        msgs: list[ModelMessage] = [
            _user_request(),
            _tool_call_response(('fetch', 'c1')),
            ModelRequest(
                parts=[
                    SystemPromptPart(content='You are helpful.'),
                    ToolReturnPart(tool_name='ghost', content='orphaned', tool_call_id='wrong_id'),
                ]
            ),
        ]
        with caplog.at_level(logging.DEBUG, logger='pydantic_harness.tool_orphan_repair'):
            _repair_messages(msgs, warn=False)
        # The synthetic return for c1 provides a non-system part, so the placeholder
        # is NOT needed here. Instead, verify the orphaned return stripping was logged.
        assert any('Stripped orphaned ToolReturnPart' in r.message for r in caplog.records)
        assert any('Injected synthetic ToolReturnPart' in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# before_model_request integration
# ---------------------------------------------------------------------------


class TestBeforeModelRequest:
    @pytest.mark.anyio
    async def test_before_model_request_repairs_messages(self) -> None:
        """The capability's ``before_model_request`` hook delegates to ``_repair_messages``."""
        from unittest.mock import MagicMock

        from pydantic_ai.models import ModelRequestContext

        cap: ToolOrphanRepair = ToolOrphanRepair(warn=False)

        msgs: list[ModelMessage] = [
            _user_request(),
            _tool_call_response(('fetch', 'c1')),
            # Request is missing the return for c1 — should be injected.
            _user_request('follow-up'),
        ]

        mock_ctx = MagicMock()
        request_context = MagicMock(spec=ModelRequestContext)
        request_context.messages = list(msgs)

        result = await cap.before_model_request(mock_ctx, request_context)
        assert result is request_context
        # A synthetic ToolReturnPart for c1 should have been injected.
        repaired = request_context.messages
        assert any(
            isinstance(p, ToolReturnPart) and p.tool_call_id == 'c1'
            for msg in repaired
            if isinstance(msg, ModelRequest)
            for p in msg.parts
        )


# ---------------------------------------------------------------------------
# Consecutive ModelResponse messages (branch coverage for line 135->138)
# ---------------------------------------------------------------------------


class TestConsecutiveResponses:
    def test_consecutive_model_responses(self) -> None:
        """Two consecutive ModelResponses (no interleaved request) are handled."""
        msgs: list[ModelMessage] = [
            _user_request(),
            # First response with an orphaned tool call.
            _tool_call_response(('alpha', 'a1')),
            # Second response immediately follows (no request in between).
            ModelResponse(parts=[TextPart(content='some text')]),
            _user_request('next'),
        ]
        repaired = _repair_messages(msgs, warn=False)
        # The first response's orphaned call should be dropped since
        # the next message is a ModelResponse, not a ModelRequest.
        assert len(repaired) == 3  # user, text-response, user
