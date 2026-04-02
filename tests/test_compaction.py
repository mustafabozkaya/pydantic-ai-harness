"""Tests for pydantic_harness.compaction capabilities."""

from __future__ import annotations

import dataclasses
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.usage import RunUsage

from pydantic_harness.compaction import (
    _SUMMARY_PREFIX,
    Compaction,
    LimitWarner,
    SlidingWindow,
    _extract_previous_summary,
    _extract_system_prompts,
    _find_first_user_message,
    _find_safe_cutoff,
    _find_token_cutoff,
    _format_messages,
    _is_safe_cutoff,
    estimate_token_count,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(
    *,
    requests: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> Any:
    """Build a minimal RunContext-like object for testing hooks."""

    @dataclasses.dataclass
    class _FakeModel:
        model_id: str = 'test-model'

    usage = RunUsage(requests=requests, input_tokens=input_tokens, output_tokens=output_tokens)

    @dataclasses.dataclass
    class _FakeCtx:
        usage: RunUsage
        model: Any = dataclasses.field(default_factory=_FakeModel)
        deps: None = None

    return _FakeCtx(usage=usage)


def _make_request_context(messages: list[ModelMessage]) -> ModelRequestContext:
    """Build a ModelRequestContext wrapping the given messages."""

    @dataclasses.dataclass
    class _FakeModel:
        model_id: str = 'test-model'

    return ModelRequestContext(
        model=_FakeModel(),  # type: ignore[arg-type]
        messages=messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )


def _user(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _assistant(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def _tool_call(tool_name: str, call_id: str) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args='{}', tool_call_id=call_id)])


def _tool_return(tool_name: str, call_id: str, content: str = 'ok') -> ModelRequest:
    return ModelRequest(parts=[ToolReturnPart(tool_name=tool_name, content=content, tool_call_id=call_id)])


# ---------------------------------------------------------------------------
# estimate_token_count
# ---------------------------------------------------------------------------


class TestEstimateTokenCount:
    def test_empty(self):
        assert estimate_token_count([]) == 0

    def test_user_message(self):
        msgs: list[ModelMessage] = [_user('hello world')]  # 11 chars => 2 tokens
        assert estimate_token_count(msgs) == 11 // 4

    def test_system_prompt(self):
        msgs: list[ModelMessage] = [ModelRequest(parts=[SystemPromptPart(content='x' * 100)])]
        assert estimate_token_count(msgs) == 25

    def test_assistant_text(self):
        msgs: list[ModelMessage] = [_assistant('y' * 80)]
        assert estimate_token_count(msgs) == 20

    def test_tool_call_and_return(self):
        msgs: list[ModelMessage] = [
            _tool_call('search', 'tc1'),
            _tool_return('search', 'tc1', 'result text here'),
        ]
        assert estimate_token_count(msgs) > 0


# ---------------------------------------------------------------------------
# _is_safe_cutoff
# ---------------------------------------------------------------------------


class TestIsSafeCutoff:
    def test_cutoff_beyond_end(self):
        msgs: list[ModelMessage] = [_user('a'), _assistant('b')]
        assert _is_safe_cutoff(msgs, 10) is True

    def test_no_tool_pairs(self):
        msgs: list[ModelMessage] = [_user('a'), _assistant('b'), _user('c')]
        assert _is_safe_cutoff(msgs, 1) is True

    def test_safe_when_both_sides_kept(self):
        msgs: list[ModelMessage] = [
            _user('a'),
            _tool_call('fn', 'tc1'),
            _tool_return('fn', 'tc1'),
            _user('b'),
        ]
        # Cutting before the tool pair (index 0) is safe: both call and return are kept.
        assert _is_safe_cutoff(msgs, 0) is True

    def test_unsafe_when_splitting_pair(self):
        msgs: list[ModelMessage] = [
            _user('a'),
            _tool_call('fn', 'tc1'),
            _tool_return('fn', 'tc1'),
            _user('b'),
        ]
        # Cutting at index 2: call (idx 1) is before cutoff, return (idx 2) is at cutoff (after).
        assert _is_safe_cutoff(msgs, 2) is False

    def test_safe_when_pair_entirely_discarded(self):
        msgs: list[ModelMessage] = [
            _tool_call('fn', 'tc1'),
            _tool_return('fn', 'tc1'),
            _user('a'),
            _assistant('b'),
        ]
        # Cutting at 2: both call and return are before cutoff (discarded together).
        assert _is_safe_cutoff(msgs, 2) is True


# ---------------------------------------------------------------------------
# _find_safe_cutoff
# ---------------------------------------------------------------------------


class TestFindSafeCutoff:
    def test_keep_zero_returns_length(self):
        msgs: list[ModelMessage] = [_user('a'), _assistant('b')]
        assert _find_safe_cutoff(msgs, 0) == 2

    def test_fewer_messages_than_keep(self):
        msgs: list[ModelMessage] = [_user('a')]
        assert _find_safe_cutoff(msgs, 5) == 0

    def test_normal_cutoff(self):
        msgs: list[ModelMessage] = [_user('a'), _assistant('b'), _user('c'), _assistant('d')]
        # Keep 2 => target cutoff is 2.
        assert _find_safe_cutoff(msgs, 2) == 2

    def test_adjusts_for_tool_pair(self):
        msgs: list[ModelMessage] = [
            _user('a'),
            _tool_call('fn', 'tc1'),
            _tool_return('fn', 'tc1'),
            _user('b'),
            _assistant('c'),
        ]
        # Keep 3 => target cutoff is 2, but that splits the tool pair.
        # Should adjust to 1 (keep tool call and return together).
        cutoff = _find_safe_cutoff(msgs, 3)
        assert cutoff == 1


# ---------------------------------------------------------------------------
# _find_token_cutoff
# ---------------------------------------------------------------------------


class TestFindTokenCutoff:
    def test_already_within_budget(self):
        msgs: list[ModelMessage] = [_user('hi')]
        assert _find_token_cutoff(msgs, 999999) == 0

    def test_empty(self):
        assert _find_token_cutoff([], 100) == 0

    def test_trims_to_budget(self):
        # Each message contributes ~3 tokens (12 chars / 4).
        msgs: list[ModelMessage] = [_user('x' * 12) for _ in range(20)]
        cutoff = _find_token_cutoff(msgs, 30)  # Budget for ~10 messages.
        assert cutoff > 0
        remaining = msgs[cutoff:]
        assert estimate_token_count(remaining) <= 30


# ---------------------------------------------------------------------------
# SlidingWindow
# ---------------------------------------------------------------------------


class TestSlidingWindow:
    def test_validation_no_trigger(self):
        with pytest.raises(ValueError, match='At least one of max_messages or max_tokens must be set'):
            SlidingWindow()

    def test_validation_negative_max_messages(self):
        with pytest.raises(ValueError, match='max_messages must be positive'):
            SlidingWindow(max_messages=0)

    def test_validation_negative_max_tokens(self):
        with pytest.raises(ValueError, match='max_tokens must be positive'):
            SlidingWindow(max_tokens=-1)

    def test_validation_negative_keep_messages(self):
        with pytest.raises(ValueError, match='keep_messages must be non-negative'):
            SlidingWindow(max_messages=10, keep_messages=-1)

    def test_validation_negative_keep_tokens(self):
        with pytest.raises(ValueError, match='keep_tokens must be non-negative'):
            SlidingWindow(max_messages=10, keep_tokens=-1)

    @pytest.mark.anyio
    async def test_no_trim_below_threshold(self):
        sw = SlidingWindow(max_messages=10, keep_messages=5)
        messages: list[ModelMessage] = [_user('a'), _assistant('b')]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await sw.before_model_request(ctx, rc)
        assert len(result.messages) == 2

    @pytest.mark.anyio
    async def test_trims_when_above_message_threshold(self):
        sw = SlidingWindow(max_messages=5, keep_messages=3, preserve_first_user_message=False)
        messages: list[ModelMessage] = [_user(f'msg-{i}') for i in range(8)]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await sw.before_model_request(ctx, rc)
        assert len(result.messages) <= 3

    @pytest.mark.anyio
    async def test_trims_by_token_threshold(self):
        sw = SlidingWindow(max_tokens=10, keep_messages=2)
        messages: list[ModelMessage] = [_user('x' * 40) for _ in range(5)]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await sw.before_model_request(ctx, rc)
        assert len(result.messages) < 5

    @pytest.mark.anyio
    async def test_preserves_tool_pairs(self):
        sw = SlidingWindow(max_messages=4, keep_messages=2)
        messages: list[ModelMessage] = [
            _user('start'),
            _tool_call('fn', 'tc1'),
            _tool_return('fn', 'tc1'),
            _user('end'),
            _assistant('done'),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await sw.before_model_request(ctx, rc)
        # Should not split the tool pair.
        remaining = result.messages
        call_ids: set[str] = set()
        return_ids: set[str] = set()
        for msg in remaining:
            if isinstance(msg, ModelResponse):
                for part in msg.parts:
                    if isinstance(part, ToolCallPart) and part.tool_call_id:
                        call_ids.add(part.tool_call_id)
            else:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return_ids.add(part.tool_call_id)
        # Every call ID in remaining must have its return.
        assert call_ids <= return_ids

    @pytest.mark.anyio
    async def test_keep_tokens_mode(self):
        sw = SlidingWindow(max_messages=3, keep_tokens=10, preserve_first_user_message=False)
        # Each message = 20 chars = 5 tokens.  Total = 50 tokens.
        messages: list[ModelMessage] = [_user('x' * 20) for _ in range(10)]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await sw.before_model_request(ctx, rc)
        assert estimate_token_count(result.messages) <= 10
        assert len(result.messages) < 10


# ---------------------------------------------------------------------------
# LimitWarner
# ---------------------------------------------------------------------------


class TestLimitWarner:
    def test_validation_no_limits(self):
        with pytest.raises(ValueError, match='At least one of'):
            LimitWarner()

    def test_validation_negative_max_iterations(self):
        with pytest.raises(ValueError, match='max_iterations must be positive'):
            LimitWarner(max_iterations=-1)

    def test_validation_negative_max_context_tokens(self):
        with pytest.raises(ValueError, match='max_context_tokens must be positive'):
            LimitWarner(max_context_tokens=0)

    def test_validation_negative_max_total_tokens(self):
        with pytest.raises(ValueError, match='max_total_tokens must be positive'):
            LimitWarner(max_total_tokens=-5)

    def test_validation_bad_threshold(self):
        with pytest.raises(ValueError, match='warning_threshold'):
            LimitWarner(max_iterations=10, warning_threshold=0)

    def test_validation_negative_critical_remaining(self):
        with pytest.raises(ValueError, match='critical_remaining_iterations'):
            LimitWarner(max_iterations=10, critical_remaining_iterations=-1)

    def test_validation_empty_warn_on(self):
        with pytest.raises(ValueError, match='warn_on must not be empty'):
            LimitWarner(max_iterations=10, warn_on=[])

    def test_validation_warn_on_without_limit(self):
        with pytest.raises(ValueError, match="'total_tokens' requires"):
            LimitWarner(max_iterations=10, warn_on=['total_tokens'])

    @pytest.mark.anyio
    async def test_no_warning_below_threshold(self):
        lw = LimitWarner(max_iterations=100)
        messages: list[ModelMessage] = [_user('hi')]
        rc = _make_request_context(messages)
        ctx = _make_ctx(requests=10)
        result = await lw.before_model_request(ctx, rc)
        # No warning appended.
        assert len(result.messages) == 1

    @pytest.mark.anyio
    async def test_iteration_warning_urgent(self):
        lw = LimitWarner(max_iterations=20, warning_threshold=0.7, critical_remaining_iterations=3)
        messages: list[ModelMessage] = [_user('hi')]
        rc = _make_request_context(messages)
        # 15/20 = 75% usage, 5 remaining > critical_remaining_iterations=3 => URGENT.
        ctx = _make_ctx(requests=15)
        result = await lw.before_model_request(ctx, rc)
        assert len(result.messages) == 2
        last = result.messages[-1]
        assert isinstance(last, ModelRequest)
        text = last.parts[0]
        assert isinstance(text, UserPromptPart)
        assert isinstance(text.content, str)
        assert 'URGENT' in text.content
        assert '[LimitWarner]' in text.content

    @pytest.mark.anyio
    async def test_iteration_warning_critical(self):
        lw = LimitWarner(max_iterations=10, warning_threshold=0.7, critical_remaining_iterations=3)
        messages: list[ModelMessage] = [_user('hi')]
        rc = _make_request_context(messages)
        ctx = _make_ctx(requests=9)  # 1 remaining.
        result = await lw.before_model_request(ctx, rc)
        last = result.messages[-1]
        assert isinstance(last, ModelRequest)
        text = last.parts[0]
        assert isinstance(text, UserPromptPart)
        assert isinstance(text.content, str)
        assert 'CRITICAL' in text.content

    @pytest.mark.anyio
    async def test_context_window_warning(self):
        lw = LimitWarner(max_context_tokens=10)
        # Create a message that exceeds 70% of 10 tokens.
        messages: list[ModelMessage] = [_user('x' * 40)]  # ~10 tokens.
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await lw.before_model_request(ctx, rc)
        assert len(result.messages) == 2

    @pytest.mark.anyio
    async def test_total_tokens_warning(self):
        lw = LimitWarner(max_total_tokens=100)
        messages: list[ModelMessage] = [_user('hi')]
        rc = _make_request_context(messages)
        ctx = _make_ctx(input_tokens=50, output_tokens=30)  # 80 total.
        result = await lw.before_model_request(ctx, rc)
        assert len(result.messages) == 2

    @pytest.mark.anyio
    async def test_strips_old_warnings(self):
        lw = LimitWarner(max_iterations=10, warning_threshold=0.7)
        old_warning = ModelRequest(parts=[UserPromptPart(content='[LimitWarner]\nOld warning')])
        messages: list[ModelMessage] = [_user('hi'), old_warning]
        rc = _make_request_context(messages)
        ctx = _make_ctx(requests=5)  # Below threshold.
        result = await lw.before_model_request(ctx, rc)
        # Old warning removed, no new warning added (below threshold).
        assert len(result.messages) == 1

    @pytest.mark.anyio
    async def test_multiple_warnings_ordered(self):
        lw = LimitWarner(max_iterations=10, max_total_tokens=100)
        messages: list[ModelMessage] = [_user('hi')]
        rc = _make_request_context(messages)
        ctx = _make_ctx(requests=8, input_tokens=50, output_tokens=30)
        result = await lw.before_model_request(ctx, rc)
        last = result.messages[-1]
        assert isinstance(last, ModelRequest)
        text = last.parts[0]
        assert isinstance(text, UserPromptPart)
        assert isinstance(text.content, str)
        # Iterations should come before total_tokens.
        assert text.content.index('Iterations') < text.content.index('Total tokens')


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------


class TestCompaction:
    def test_validation_no_trigger(self):
        with pytest.raises(ValueError, match='At least one of max_messages or max_tokens must be set'):
            Compaction(model='test', max_messages=None, max_tokens=None)

    def test_validation_negative_max_messages(self):
        with pytest.raises(ValueError, match='max_messages must be positive'):
            Compaction(model='test', max_messages=0)

    def test_validation_negative_max_tokens(self):
        with pytest.raises(ValueError, match='max_tokens must be positive'):
            Compaction(model='test', max_tokens=-1)

    def test_validation_negative_keep_messages(self):
        with pytest.raises(ValueError, match='keep_messages must be non-negative'):
            Compaction(model='test', max_messages=10, keep_messages=-1)

    def test_validation_negative_keep_tokens(self):
        with pytest.raises(ValueError, match='keep_tokens must be non-negative'):
            Compaction(model='test', max_messages=10, keep_tokens=-1)

    @pytest.mark.anyio
    async def test_no_compaction_below_threshold(self):
        comp = Compaction(model='test', max_messages=100)
        messages: list[ModelMessage] = [_user('hi')]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await comp.before_model_request(ctx, rc)
        assert result.messages == messages

    @pytest.mark.anyio
    async def test_compaction_replaces_old_messages(self):
        comp = Compaction(model='test:m', max_messages=3, keep_messages=1, preserve_first_user_message=False)
        messages: list[ModelMessage] = [
            _user('first'),
            _assistant('response 1'),
            _user('second'),
            _assistant('response 2'),
            _user('third'),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        mock_result = AsyncMock()
        mock_result.output = 'Summary of conversation.'

        with patch('pydantic_ai.Agent') as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.return_value = mock_agent_instance

            result = await comp.before_model_request(ctx, rc)

        # Should have summary message + 1 kept message.
        assert len(result.messages) == 2
        first_msg = result.messages[0]
        assert isinstance(first_msg, ModelRequest)
        # The summary should be in a SystemPromptPart.
        sys_parts = [p for p in first_msg.parts if isinstance(p, SystemPromptPart)]
        assert len(sys_parts) >= 1
        assert 'Summary of conversation.' in sys_parts[-1].content

    @pytest.mark.anyio
    async def test_compaction_preserves_system_prompts(self):
        comp = Compaction(model='test:m', max_messages=3, keep_messages=1)
        messages: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content='You are a helpful assistant.')]),
            _user('first'),
            _assistant('response 1'),
            _user('second'),
            _assistant('response 2'),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        mock_result = AsyncMock()
        mock_result.output = 'A summary.'

        with patch('pydantic_ai.Agent') as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.return_value = mock_agent_instance

            result = await comp.before_model_request(ctx, rc)

        first_msg = result.messages[0]
        assert isinstance(first_msg, ModelRequest)
        # Should have the original system prompt preserved.
        sys_contents = [p.content for p in first_msg.parts if isinstance(p, SystemPromptPart)]
        assert 'You are a helpful assistant.' in sys_contents

    @pytest.mark.anyio
    async def test_compaction_preserves_tool_pairs(self):
        comp = Compaction(model='test:m', max_messages=4, keep_messages=2)
        messages: list[ModelMessage] = [
            _user('start'),
            _tool_call('fn', 'tc1'),
            _tool_return('fn', 'tc1'),
            _user('middle'),
            _assistant('response'),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        mock_result = AsyncMock()
        mock_result.output = 'Summary.'

        with patch('pydantic_ai.Agent') as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.return_value = mock_agent_instance

            result = await comp.before_model_request(ctx, rc)

        # Tool pairs in remaining messages should be intact.
        remaining = result.messages
        call_ids: set[str] = set()
        return_ids: set[str] = set()
        for msg in remaining:
            if isinstance(msg, ModelResponse):
                for part in msg.parts:
                    if isinstance(part, ToolCallPart) and part.tool_call_id:
                        call_ids.add(part.tool_call_id)
            else:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return_ids.add(part.tool_call_id)
        assert call_ids <= return_ids

    @pytest.mark.anyio
    async def test_compaction_token_trigger(self):
        comp = Compaction(model='test:m', max_tokens=5, keep_messages=1)
        messages: list[ModelMessage] = [_user('x' * 40) for _ in range(5)]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        mock_result = AsyncMock()
        mock_result.output = 'Token-based summary.'

        with patch('pydantic_ai.Agent') as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.return_value = mock_agent_instance

            result = await comp.before_model_request(ctx, rc)

        assert len(result.messages) >= 1
        # Summary message should exist.
        first_msg = result.messages[0]
        assert isinstance(first_msg, ModelRequest)

    @pytest.mark.anyio
    async def test_compaction_keep_tokens_mode(self):
        comp = Compaction(model='test:m', max_messages=3, keep_tokens=5)
        messages: list[ModelMessage] = [_user('x' * 40) for _ in range(5)]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        mock_result = AsyncMock()
        mock_result.output = 'Token-keep summary.'

        with patch('pydantic_ai.Agent') as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.return_value = mock_agent_instance

            result = await comp.before_model_request(ctx, rc)

        assert len(result.messages) >= 1


# ---------------------------------------------------------------------------
# _format_messages
# ---------------------------------------------------------------------------


class TestFormatMessages:
    def test_user_and_assistant(self):
        msgs: list[ModelMessage] = [_user('hi'), _assistant('hello')]
        text = _format_messages(msgs)
        assert 'User: hi' in text
        assert 'Assistant: hello' in text

    def test_system_prompt(self):
        msgs: list[ModelMessage] = [ModelRequest(parts=[SystemPromptPart(content='be helpful')])]
        text = _format_messages(msgs)
        assert 'System: be helpful' in text

    def test_tool_call_and_return(self):
        msgs: list[ModelMessage] = [
            _tool_call('search', 'tc1'),
            _tool_return('search', 'tc1', 'found it'),
        ]
        text = _format_messages(msgs)
        assert 'Tool Call [search]' in text
        assert 'Tool [search]: found it' in text

    def test_long_tool_return_truncated(self):
        msgs: list[ModelMessage] = [_tool_return('fn', 'tc1', 'x' * 600)]
        text = _format_messages(msgs)
        assert '...' in text


# ---------------------------------------------------------------------------
# _extract_system_prompts
# ---------------------------------------------------------------------------


class TestExtractSystemPrompts:
    def test_extracts_leading_system_parts(self):
        msgs: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content='sys1')]),
            _user('hi'),
        ]
        parts = _extract_system_prompts(msgs)
        assert len(parts) == 1
        assert parts[0].content == 'sys1'

    def test_stops_at_non_system(self):
        msgs: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content='sys1'), UserPromptPart(content='hi')]),
        ]
        parts = _extract_system_prompts(msgs)
        assert len(parts) == 1

    def test_empty_when_no_system(self):
        msgs: list[ModelMessage] = [_user('hi')]
        parts = _extract_system_prompts(msgs)
        assert parts == []

    def test_stops_at_non_request(self):
        msgs: list[ModelMessage] = [_assistant('hello'), _user('hi')]
        parts = _extract_system_prompts(msgs)
        assert parts == []


# ---------------------------------------------------------------------------
# Package-level exports
# ---------------------------------------------------------------------------


class TestExports:
    def test_package_exports(self):
        import pydantic_harness

        assert hasattr(pydantic_harness, 'SlidingWindow')
        assert hasattr(pydantic_harness, 'LimitWarner')
        assert hasattr(pydantic_harness, 'Compaction')


# ---------------------------------------------------------------------------
# Additional coverage — multi-modal content, edge cases
# ---------------------------------------------------------------------------


class TestUserPromptMultiModal:
    """Cover _user_prompt_text_for_counting and _user_prompt_text for non-string UserContent."""

    def test_estimate_with_text_content_parts(self):
        from pydantic_ai.messages import TextContent

        part = UserPromptPart(content=[TextContent(content='hello')])
        msgs: list[ModelMessage] = [ModelRequest(parts=[part])]
        # 5 chars / 4 = 1 token.
        assert estimate_token_count(msgs) == 1

    def test_estimate_with_str_content_parts(self):
        """UserContent can also be plain str items in a sequence."""
        part = UserPromptPart(content=['hello', 'world'])
        msgs: list[ModelMessage] = [ModelRequest(parts=[part])]
        # 10 chars / 4 = 2 tokens.
        assert estimate_token_count(msgs) == 2

    def test_format_with_text_content(self):
        from pydantic_ai.messages import TextContent

        part = UserPromptPart(content=[TextContent(content='multi-part')])
        msgs: list[ModelMessage] = [ModelRequest(parts=[part])]
        text = _format_messages(msgs)
        assert 'User: multi-part' in text

    def test_format_with_str_content(self):
        part = UserPromptPart(content=['one', 'two'])
        msgs: list[ModelMessage] = [ModelRequest(parts=[part])]
        text = _format_messages(msgs)
        assert 'User: one two' in text

    def test_format_empty_sequence(self):
        part = UserPromptPart(content=[])
        msgs: list[ModelMessage] = [ModelRequest(parts=[part])]
        text = _format_messages(msgs)
        assert 'User: ' in text


class TestLimitWarnerEdgeCases:
    """Cover LimitWarner edge cases for marker detection and stripping."""

    @pytest.mark.anyio
    async def test_strip_warning_with_only_marker_message(self):
        """A message composed entirely of a marker part should be removed."""
        lw = LimitWarner(max_iterations=100)
        marker_msg = ModelRequest(parts=[UserPromptPart(content='[LimitWarner]\nold')])
        messages: list[ModelMessage] = [_user('real'), marker_msg]
        rc = _make_request_context(messages)
        ctx = _make_ctx(requests=5)
        result = await lw.before_model_request(ctx, rc)
        # Marker message should be stripped; only the real message remains.
        assert len(result.messages) == 1

    @pytest.mark.anyio
    async def test_strip_warning_system_prompt_marker(self):
        """Marker in a SystemPromptPart should also be detected."""
        lw = LimitWarner(max_iterations=100)
        marker_msg = ModelRequest(parts=[SystemPromptPart(content='[LimitWarner]\nold')])
        messages: list[ModelMessage] = [_user('real'), marker_msg]
        rc = _make_request_context(messages)
        ctx = _make_ctx(requests=5)
        result = await lw.before_model_request(ctx, rc)
        assert len(result.messages) == 1

    @pytest.mark.anyio
    async def test_strip_mixed_parts_keeps_non_marker(self):
        """A message with both marker and non-marker parts should keep the non-marker parts."""
        lw = LimitWarner(max_iterations=100)
        mixed = ModelRequest(
            parts=[
                UserPromptPart(content='keep this'),
                UserPromptPart(content='[LimitWarner]\nremove this'),
            ]
        )
        messages: list[ModelMessage] = [mixed]
        rc = _make_request_context(messages)
        ctx = _make_ctx(requests=5)
        result = await lw.before_model_request(ctx, rc)
        assert len(result.messages) == 1
        first = result.messages[0]
        assert isinstance(first, ModelRequest)
        assert len(first.parts) == 1

    @pytest.mark.anyio
    async def test_context_warning_below_threshold(self):
        """Context window should not warn when below threshold."""
        lw = LimitWarner(max_context_tokens=1000)
        messages: list[ModelMessage] = [_user('hi')]  # ~0.5 tokens, well below 70%.
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await lw.before_model_request(ctx, rc)
        assert len(result.messages) == 1

    @pytest.mark.anyio
    async def test_total_tokens_warning_critical(self):
        """Total tokens at or above limit should produce CRITICAL."""
        lw = LimitWarner(max_total_tokens=100)
        messages: list[ModelMessage] = [_user('hi')]
        rc = _make_request_context(messages)
        ctx = _make_ctx(input_tokens=60, output_tokens=50)  # 110 total, above limit.
        result = await lw.before_model_request(ctx, rc)
        last = result.messages[-1]
        assert isinstance(last, ModelRequest)
        text = last.parts[0]
        assert isinstance(text, UserPromptPart)
        assert isinstance(text.content, str)
        assert 'CRITICAL' in text.content

    @pytest.mark.anyio
    async def test_context_window_critical(self):
        """Context window at or above limit should produce CRITICAL."""
        lw = LimitWarner(max_context_tokens=5)
        messages: list[ModelMessage] = [_user('x' * 40)]  # ~10 tokens, well above 5.
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await lw.before_model_request(ctx, rc)
        last = result.messages[-1]
        assert isinstance(last, ModelRequest)
        text = last.parts[0]
        assert isinstance(text, UserPromptPart)
        assert isinstance(text.content, str)
        assert 'CRITICAL' in text.content

    def test_warn_on_subset(self):
        """Can configure warn_on to only include specific limits."""
        lw = LimitWarner(max_iterations=10, max_total_tokens=100, warn_on=['iterations'])
        assert lw._active_kinds == ('iterations',)


class TestCompactionEdgeCases:
    """Cover Compaction edge cases."""

    @pytest.mark.anyio
    async def test_compaction_cutoff_zero_no_change(self):
        """When cutoff is 0, no compaction should occur (messages all kept)."""
        comp = Compaction(model='test:m', max_messages=2, keep_messages=10)
        # Only 3 messages, keep_messages=10 means cutoff=0.
        messages: list[ModelMessage] = [_user('a'), _assistant('b'), _user('c')]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await comp.before_model_request(ctx, rc)
        assert len(result.messages) == 3


class TestSlidingWindowEdgeCases:
    """Cover SlidingWindow edge cases."""

    @pytest.mark.anyio
    async def test_cutoff_zero_no_trim(self):
        """When the cutoff resolves to 0, messages should not be trimmed."""
        sw = SlidingWindow(max_messages=2, keep_messages=10)
        # 3 messages, but keep_messages=10 => cutoff=0.
        messages: list[ModelMessage] = [_user('a'), _assistant('b'), _user('c')]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await sw.before_model_request(ctx, rc)
        assert len(result.messages) == 3

    @pytest.mark.anyio
    async def test_token_not_triggered_when_below(self):
        """Token trigger should not fire below threshold."""
        sw = SlidingWindow(max_tokens=999999, keep_messages=2)
        messages: list[ModelMessage] = [_user('hi')]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await sw.before_model_request(ctx, rc)
        assert len(result.messages) == 1


class TestLimitWarnerMarkerDetection:
    """Cover _is_marker_part return False for non-text parts."""

    @pytest.mark.anyio
    async def test_non_string_user_prompt_not_detected_as_marker(self):
        """UserPromptPart with non-string content should not match marker."""
        lw = LimitWarner(max_iterations=100)
        # Create a ModelRequest with a ToolReturnPart (not a marker).
        messages: list[ModelMessage] = [
            _user('real'),
            _tool_return('fn', 'tc1', 'some result'),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx(requests=5)
        result = await lw.before_model_request(ctx, rc)
        assert len(result.messages) == 2

    @pytest.mark.anyio
    async def test_strip_preserves_model_responses(self):
        """ModelResponse messages pass through strip unchanged."""
        lw = LimitWarner(max_iterations=100)
        messages: list[ModelMessage] = [
            _user('hi'),
            _assistant('response'),
            ModelRequest(parts=[UserPromptPart(content='[LimitWarner]\nold')]),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx(requests=5)
        result = await lw.before_model_request(ctx, rc)
        # Marker message removed; user and assistant remain.
        assert len(result.messages) == 2
        assert isinstance(result.messages[1], ModelResponse)


class TestLimitWarnerTotalTokensBelowThreshold:
    """Cover _build_total_tokens_warning returning None when below threshold."""

    @pytest.mark.anyio
    async def test_total_tokens_below_threshold(self):
        lw = LimitWarner(max_total_tokens=1000)
        messages: list[ModelMessage] = [_user('hi')]
        rc = _make_request_context(messages)
        ctx = _make_ctx(input_tokens=10, output_tokens=10)  # 20 total, 2% of 1000.
        result = await lw.before_model_request(ctx, rc)
        assert len(result.messages) == 1  # No warning.


# ---------------------------------------------------------------------------
# Tokenizer parameter
# ---------------------------------------------------------------------------


class TestTokenizerParameter:
    """Tests for the optional tokenizer parameter on estimate_token_count,
    SlidingWindow, and Compaction."""

    def test_estimate_token_count_with_tokenizer(self):
        """Custom tokenizer should override the heuristic."""
        msgs: list[ModelMessage] = [_user('hello world')]
        # Heuristic: 11 chars / 4 = 2 tokens.
        assert estimate_token_count(msgs) == 2
        # Custom tokenizer: count words instead.
        assert estimate_token_count(msgs, tokenizer=lambda s: len(s.split())) == 2

    def test_estimate_token_count_tokenizer_called_per_segment(self):
        """Tokenizer is called once per text segment, results are summed."""
        calls: list[str] = []

        def tracking_tokenizer(s: str) -> int:
            calls.append(s)
            return 10

        msgs: list[ModelMessage] = [_user('a'), _assistant('b')]
        result = estimate_token_count(msgs, tokenizer=tracking_tokenizer)
        assert result == 20
        assert len(calls) == 2

    @pytest.mark.anyio
    async def test_sliding_window_with_tokenizer(self):
        """SlidingWindow should use the tokenizer for token-based triggers."""
        # Custom tokenizer: 1 token per character.
        sw = SlidingWindow(
            max_tokens=10,
            keep_tokens=5,
            tokenizer=lambda s: len(s),
            preserve_first_user_message=False,
        )
        # Each message has 4 chars = 4 tokens with this tokenizer. 5 messages = 20 tokens.
        messages: list[ModelMessage] = [_user('abcd') for _ in range(5)]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await sw.before_model_request(ctx, rc)
        # With keep_tokens=5 and 4 tokens per message, should keep 1 message.
        remaining_tokens = estimate_token_count(result.messages, tokenizer=lambda s: len(s))
        assert remaining_tokens <= 5

    @pytest.mark.anyio
    async def test_sliding_window_tokenizer_threshold_check(self):
        """SlidingWindow tokenizer should be used for the trigger check."""
        # Tokenizer that inflates counts: 100 tokens per char.
        sw = SlidingWindow(
            max_tokens=50,
            keep_messages=1,
            tokenizer=lambda s: len(s) * 100,
            preserve_first_user_message=False,
        )
        # 2 chars * 100 = 200 tokens per message. Only 1 message but still > 50.
        messages: list[ModelMessage] = [_user('ab'), _user('cd')]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await sw.before_model_request(ctx, rc)
        assert len(result.messages) == 1

    @pytest.mark.anyio
    async def test_compaction_with_tokenizer(self):
        """Compaction should use the tokenizer for token-based triggers."""
        # Tokenizer: 1 token per char.
        comp = Compaction(
            model='test:m',
            max_tokens=10,
            keep_messages=1,
            tokenizer=lambda s: len(s),
            preserve_first_user_message=False,
            incremental=False,
        )
        # Each message: 'abcde' = 5 chars = 5 tokens. 4 messages = 20 tokens > 10.
        messages: list[ModelMessage] = [_user('abcde') for _ in range(4)]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        mock_result = AsyncMock()
        mock_result.output = 'Token summary.'

        with patch('pydantic_ai.Agent') as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.return_value = mock_agent_instance

            result = await comp.before_model_request(ctx, rc)

        # Should have triggered compaction.
        assert len(result.messages) >= 1
        first_msg = result.messages[0]
        assert isinstance(first_msg, ModelRequest)
        sys_parts = [p for p in first_msg.parts if isinstance(p, SystemPromptPart)]
        assert any('Token summary.' in p.content for p in sys_parts)

    def test_find_token_cutoff_with_tokenizer(self):
        """_find_token_cutoff should use the tokenizer."""
        messages: list[ModelMessage] = [_user('abcde') for _ in range(10)]
        # Tokenizer: 1 token per char. Each message = 5 tokens.
        cutoff = _find_token_cutoff(messages, 15, tokenizer=lambda s: len(s))
        remaining = messages[cutoff:]
        assert estimate_token_count(remaining, tokenizer=lambda s: len(s)) <= 15


# ---------------------------------------------------------------------------
# Preserve first user message
# ---------------------------------------------------------------------------


class TestPreserveFirstUserMessage:
    """Tests for the preserve_first_user_message parameter."""

    def test_find_first_user_message_found(self):
        msgs: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content='sys')]),
            _user('first'),
            _user('second'),
        ]
        result = _find_first_user_message(msgs)
        assert result is not None
        assert isinstance(result.parts[0], UserPromptPart)
        assert result.parts[0].content == 'first'

    def test_find_first_user_message_none(self):
        msgs: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content='sys')]),
            _assistant('hello'),
        ]
        assert _find_first_user_message(msgs) is None

    @pytest.mark.anyio
    async def test_sliding_window_preserves_first_user(self):
        sw = SlidingWindow(max_messages=3, keep_messages=2, preserve_first_user_message=True)
        messages: list[ModelMessage] = [
            _user('original task'),
            _assistant('got it'),
            _user('follow-up 1'),
            _assistant('done'),
            _user('follow-up 2'),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await sw.before_model_request(ctx, rc)
        # The first user message ('original task') should be preserved even though
        # it was outside the keep window.
        user_contents: list[str] = []
        for msg in result.messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                        user_contents.append(part.content)
        assert 'original task' in user_contents

    @pytest.mark.anyio
    async def test_sliding_window_no_duplicate_when_in_window(self):
        """First user message should not be duplicated if already in the kept window."""
        sw = SlidingWindow(max_messages=3, keep_messages=5, preserve_first_user_message=True)
        messages: list[ModelMessage] = [
            _user('task'),
            _assistant('ok'),
            _user('more'),
            _assistant('done'),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await sw.before_model_request(ctx, rc)
        assert len(result.messages) == 4  # Not triggered since 4 < 5 keep.

    @pytest.mark.anyio
    async def test_sliding_window_disabled_preserve(self):
        """When preserve_first_user_message=False, first user message is not kept."""
        sw = SlidingWindow(max_messages=3, keep_messages=1, preserve_first_user_message=False)
        messages: list[ModelMessage] = [
            _user('original'),
            _assistant('a'),
            _user('b'),
            _assistant('c'),
            _user('last'),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await sw.before_model_request(ctx, rc)
        assert len(result.messages) == 1
        user_contents: list[str] = []
        for msg in result.messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                        user_contents.append(part.content)
        assert 'original' not in user_contents

    @pytest.mark.anyio
    async def test_compaction_preserves_first_user(self):
        comp = Compaction(model='test:m', max_messages=3, keep_messages=1, preserve_first_user_message=True)
        messages: list[ModelMessage] = [
            _user('build a web app'),
            _assistant('response 1'),
            _user('second'),
            _assistant('response 2'),
            _user('third'),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        mock_result = AsyncMock()
        mock_result.output = 'Summary.'

        with patch('pydantic_ai.Agent') as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.return_value = mock_agent_instance

            result = await comp.before_model_request(ctx, rc)

        # Summary message + first user message + 1 kept = 3.
        assert len(result.messages) == 3
        # First message is the summary (with system prompts).
        assert isinstance(result.messages[0], ModelRequest)
        sys_parts = [p for p in result.messages[0].parts if isinstance(p, SystemPromptPart)]
        assert any('Summary.' in p.content for p in sys_parts)
        # Second message is the preserved first user message.
        assert isinstance(result.messages[1], ModelRequest)
        user_parts = [p for p in result.messages[1].parts if isinstance(p, UserPromptPart)]
        assert len(user_parts) == 1
        assert user_parts[0].content == 'build a web app'

    @pytest.mark.anyio
    async def test_compaction_no_duplicate_first_user_when_in_window(self):
        """First user message already in kept window should not be duplicated."""
        comp = Compaction(model='test:m', max_messages=3, keep_messages=5, preserve_first_user_message=True)
        messages: list[ModelMessage] = [
            _user('task'),
            _assistant('ok'),
            _user('more'),
            _assistant('done'),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await comp.before_model_request(ctx, rc)
        # Not triggered since keep_messages > len(messages).
        assert len(result.messages) == 4

    @pytest.mark.anyio
    async def test_sliding_window_no_user_messages(self):
        """When there are no user messages, preservation is a no-op."""
        sw = SlidingWindow(max_messages=2, keep_messages=1, preserve_first_user_message=True)
        messages: list[ModelMessage] = [
            _assistant('a'),
            _assistant('b'),
            _assistant('c'),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()
        result = await sw.before_model_request(ctx, rc)
        assert len(result.messages) == 1


# ---------------------------------------------------------------------------
# Incremental summarization
# ---------------------------------------------------------------------------


class TestIncrementalSummarization:
    """Tests for the incremental parameter on Compaction."""

    def test_extract_previous_summary_found(self):
        msgs: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content=f'{_SUMMARY_PREFIX}Old summary text.')]),
            _user('hi'),
        ]
        assert _extract_previous_summary(msgs) == 'Old summary text.'

    def test_extract_previous_summary_not_found(self):
        msgs: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content='Regular system prompt.')]),
            _user('hi'),
        ]
        assert _extract_previous_summary(msgs) is None

    def test_extract_previous_summary_empty_messages(self):
        assert _extract_previous_summary([]) is None

    def test_extract_previous_summary_skips_non_requests(self):
        msgs: list[ModelMessage] = [
            _assistant('hi'),
            _user('hello'),
        ]
        assert _extract_previous_summary(msgs) is None

    @pytest.mark.anyio
    async def test_incremental_includes_previous_summary(self):
        """When incremental=True and a prior summary exists, it should be included in the prompt."""
        comp = Compaction(
            model='test:m',
            max_messages=3,
            keep_messages=1,
            incremental=True,
            preserve_first_user_message=False,
        )
        # Simulate a conversation that already has a summary from prior compaction.
        messages: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content=f'{_SUMMARY_PREFIX}Previous context here.')]),
            _user('new input 1'),
            _assistant('response 1'),
            _user('new input 2'),
            _assistant('response 2'),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        mock_result = AsyncMock()
        mock_result.output = 'Extended summary.'

        with patch('pydantic_ai.Agent') as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.return_value = mock_agent_instance

            await comp.before_model_request(ctx, rc)

        # Verify the summarization prompt included the previous summary.
        call_args = mock_agent_instance.run.call_args
        prompt_text = call_args[0][0]
        assert '<previous_summary>' in prompt_text
        assert 'Previous context here.' in prompt_text

    @pytest.mark.anyio
    async def test_incremental_no_previous_summary(self):
        """When incremental=True but no prior summary exists, prompt should be plain."""
        comp = Compaction(
            model='test:m',
            max_messages=3,
            keep_messages=1,
            incremental=True,
            preserve_first_user_message=False,
        )
        messages: list[ModelMessage] = [
            _user('first'),
            _assistant('response 1'),
            _user('second'),
            _assistant('response 2'),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        mock_result = AsyncMock()
        mock_result.output = 'Fresh summary.'

        with patch('pydantic_ai.Agent') as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.return_value = mock_agent_instance

            await comp.before_model_request(ctx, rc)

        call_args = mock_agent_instance.run.call_args
        prompt_text = call_args[0][0]
        assert '<previous_summary>' not in prompt_text

    @pytest.mark.anyio
    async def test_incremental_disabled(self):
        """When incremental=False, the previous summary should not be included."""
        comp = Compaction(
            model='test:m',
            max_messages=3,
            keep_messages=1,
            incremental=False,
            preserve_first_user_message=False,
        )
        messages: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content=f'{_SUMMARY_PREFIX}Old summary.')]),
            _user('new input'),
            _assistant('response'),
            _user('another'),
            _assistant('another response'),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        mock_result = AsyncMock()
        mock_result.output = 'Regenerated summary.'

        with patch('pydantic_ai.Agent') as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.return_value = mock_agent_instance

            await comp.before_model_request(ctx, rc)

        call_args = mock_agent_instance.run.call_args
        prompt_text = call_args[0][0]
        assert '<previous_summary>' not in prompt_text

    @pytest.mark.anyio
    async def test_incremental_output_contains_summary(self):
        """The output after incremental compaction should contain the new summary."""
        comp = Compaction(
            model='test:m',
            max_messages=3,
            keep_messages=1,
            incremental=True,
            preserve_first_user_message=False,
        )
        messages: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content=f'{_SUMMARY_PREFIX}Old context.')]),
            _user('a'),
            _assistant('b'),
            _user('c'),
            _assistant('d'),
        ]
        rc = _make_request_context(messages)
        ctx = _make_ctx()

        mock_result = AsyncMock()
        mock_result.output = 'Extended context summary.'

        with patch('pydantic_ai.Agent') as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.return_value = mock_agent_instance

            result = await comp.before_model_request(ctx, rc)

        first_msg = result.messages[0]
        assert isinstance(first_msg, ModelRequest)
        sys_parts = [p for p in first_msg.parts if isinstance(p, SystemPromptPart)]
        assert any('Extended context summary.' in p.content for p in sys_parts)
