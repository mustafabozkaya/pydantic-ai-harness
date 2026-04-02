"""Compaction capabilities for managing conversation context.

Provides three capabilities for controlling conversation history size:

- `SlidingWindow` — zero-cost message trimming via a sliding window
- `LimitWarner` — injects warnings when approaching iteration/token limits
- `Compaction` — LLM-powered summarization of older messages
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.tools import RunContext

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN = 4
"""Rough approximation: ~4 characters per token on average."""


def _collect_text(messages: Sequence[ModelMessage]) -> list[str]:
    """Collect all text segments from a sequence of messages."""
    segments: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    segments.append(_user_prompt_text_for_counting(part))
                elif isinstance(part, SystemPromptPart):
                    segments.append(part.content)
                elif isinstance(part, ToolReturnPart):
                    segments.append(str(part.content))
        else:
            for part in msg.parts:
                if isinstance(part, TextPart):
                    segments.append(part.content)
                elif isinstance(part, ToolCallPart):
                    segments.append(part.tool_name)
                    segments.append(str(part.args))
    return segments


def _user_prompt_text_for_counting(part: UserPromptPart) -> str:
    """Extract text content from a user prompt part for counting."""
    if isinstance(part.content, str):
        return part.content
    texts: list[str] = []
    for item in part.content:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, TextContent):
            texts.append(item.content)
    return ''.join(texts)


def estimate_token_count(
    messages: Sequence[ModelMessage],
    tokenizer: Callable[[str], int] | None = None,
) -> int:
    """Approximate token count for a sequence of messages.

    Args:
        messages: Messages to count tokens for.
        tokenizer: Optional callable that returns the token count for a string.
            When ``None``, falls back to a ~4 characters-per-token heuristic.
    """
    segments = _collect_text(messages)
    if tokenizer is not None:
        return sum(tokenizer(s) for s in segments)
    return sum(len(s) for s in segments) // _CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Safe cutoff logic — preserves tool-call / tool-return pairs
# ---------------------------------------------------------------------------

_TOOL_PAIR_SEARCH_RANGE = 5
"""Number of messages to search around a cutoff point for tool-call pairs."""


def _is_safe_cutoff(
    messages: list[ModelMessage],
    cutoff: int,
    search_range: int = _TOOL_PAIR_SEARCH_RANGE,
) -> bool:
    """Return True if cutting at *cutoff* does not orphan any tool-call pair.

    A tool-call pair is a ``ToolCallPart`` in a ``ModelResponse`` together with
    the corresponding ``ToolReturnPart`` in a subsequent ``ModelRequest``.  Both
    sides must end up on the same side of the cut.
    """
    if cutoff >= len(messages):
        return True

    start = max(0, cutoff - search_range)
    end = min(len(messages), cutoff + search_range)

    for i in range(start, end):
        msg = messages[i]
        if not isinstance(msg, ModelResponse):
            continue

        call_ids: set[str] = set()
        for part in msg.parts:
            if isinstance(part, ToolCallPart) and part.tool_call_id:
                call_ids.add(part.tool_call_id)

        if not call_ids:
            continue

        for j in range(i + 1, len(messages)):
            later = messages[j]
            if not isinstance(later, ModelRequest):
                continue
            for rpart in later.parts:
                if isinstance(rpart, ToolReturnPart) and rpart.tool_call_id in call_ids:
                    call_before = i < cutoff
                    return_before = j < cutoff
                    if call_before != return_before:
                        return False

    return True


def _find_safe_cutoff(messages: list[ModelMessage], keep: int) -> int:
    """Find a cutoff index that keeps *keep* tail messages without splitting tool pairs.

    Returns 0 if trimming is unnecessary (fewer messages than *keep*).
    """
    if keep == 0:
        return len(messages)
    if len(messages) <= keep:
        return 0

    target = len(messages) - keep
    for idx in range(target, -1, -1):
        if _is_safe_cutoff(messages, idx):
            return idx
    return 0  # pragma: no cover


def _find_token_cutoff(
    messages: list[ModelMessage],
    target_tokens: int,
    tokenizer: Callable[[str], int] | None = None,
) -> int:
    """Binary-search for a cutoff such that ``messages[cutoff:]`` fits in *target_tokens*.

    Adjusts the result so that no tool-call pairs are orphaned.
    """
    if not messages or estimate_token_count(messages, tokenizer) <= target_tokens:
        return 0

    lo, hi = 0, len(messages)
    candidate = len(messages)

    while lo < hi:
        mid = (lo + hi) // 2
        if estimate_token_count(messages[mid:], tokenizer) <= target_tokens:
            candidate = mid
            hi = mid
        else:
            lo = mid + 1

    if candidate >= len(messages):
        candidate = max(0, len(messages) - 1)  # pragma: no cover

    # Walk backward to a safe point.
    for idx in range(candidate, -1, -1):
        if _is_safe_cutoff(messages, idx):
            return idx
    return 0  # pragma: no cover


# ---------------------------------------------------------------------------
# First user message preservation
# ---------------------------------------------------------------------------


def _find_first_user_message(messages: list[ModelMessage]) -> ModelRequest | None:
    """Return the first ``ModelRequest`` that contains a ``UserPromptPart``, or ``None``."""
    for msg in messages:
        if isinstance(msg, ModelRequest) and any(isinstance(p, UserPromptPart) for p in msg.parts):
            return msg
    return None


def _prepend_first_user_message(
    original: list[ModelMessage],
    cutoff: int,
    trimmed: list[ModelMessage],
) -> list[ModelMessage]:
    """Ensure the first user message from *original* appears in *trimmed*.

    If the first ``ModelRequest`` containing a ``UserPromptPart`` in *original*
    was discarded (its index is before *cutoff*) and is not already in *trimmed*,
    prepend it.
    """
    first = _find_first_user_message(original)
    if first is None:
        return trimmed
    idx = original.index(first)
    if idx < cutoff and first not in trimmed:
        return [first, *trimmed]
    return trimmed


# ---------------------------------------------------------------------------
# SlidingWindow
# ---------------------------------------------------------------------------


@dataclass
class SlidingWindow(AbstractCapability[AgentDepsT]):
    """Zero-cost sliding-window trimmer.

    When the conversation exceeds a configurable threshold (message count or
    estimated token count), the oldest messages are discarded while preserving
    tool-call / tool-return pairs.  No LLM calls are made.

    Trimming happens in ``before_model_request`` so it is transparent to the
    rest of the agent run.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_harness.compaction import SlidingWindow

        agent = Agent(
            'openai:gpt-4o',
            capabilities=[SlidingWindow(max_messages=80, keep_messages=40)],
        )
        ```
    """

    max_messages: int | None = None
    """Trigger trimming when message count reaches this value. ``None`` disables."""

    max_tokens: int | None = None
    """Trigger trimming when estimated token count reaches this value. ``None`` disables."""

    keep_messages: int = 40
    """Number of tail messages to retain after trimming (message-count trigger)."""

    keep_tokens: int | None = None
    """Target token budget after trimming (token-count trigger).

    When ``None``, falls back to ``keep_messages``.
    """

    tokenizer: Callable[[str], int] | None = None
    """Optional tokenizer for accurate token counting.

    A callable that returns the token count for a given string.
    When ``None``, uses a ~4 characters-per-token heuristic.
    """

    preserve_first_user_message: bool = True
    """When ``True``, the first ``ModelRequest`` containing a ``UserPromptPart``
    is always kept after trimming, in addition to system prompts.
    """

    def __post_init__(self) -> None:  # noqa: D105
        if self.max_messages is None and self.max_tokens is None:
            raise ValueError('At least one of max_messages or max_tokens must be set.')
        if self.max_messages is not None and self.max_messages < 1:
            raise ValueError('max_messages must be positive.')
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError('max_tokens must be positive.')
        if self.keep_messages < 0:
            raise ValueError('keep_messages must be non-negative.')
        if self.keep_tokens is not None and self.keep_tokens < 0:
            raise ValueError('keep_tokens must be non-negative.')

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Trim the message list if it exceeds the configured threshold."""
        messages: list[ModelMessage] = list(request_context.messages)
        triggered = False

        if self.max_messages is not None and len(messages) > self.max_messages:
            triggered = True
        if not triggered and self.max_tokens is not None:
            if estimate_token_count(messages, self.tokenizer) > self.max_tokens:
                triggered = True

        if not triggered:
            return request_context

        if self.keep_tokens is not None:
            cutoff = _find_token_cutoff(messages, self.keep_tokens, self.tokenizer)
        else:
            cutoff = _find_safe_cutoff(messages, self.keep_messages)

        if cutoff > 0:
            trimmed = messages[cutoff:]
            if self.preserve_first_user_message:
                trimmed = _prepend_first_user_message(messages, cutoff, trimmed)
            request_context.messages = trimmed

        return request_context


# ---------------------------------------------------------------------------
# LimitWarner
# ---------------------------------------------------------------------------

WarningKind = Literal['iterations', 'context_window', 'total_tokens']
"""Categories of limits that can trigger warnings."""

_WARNING_ORDER: tuple[WarningKind, ...] = ('iterations', 'context_window', 'total_tokens')
_MARKER = '[LimitWarner]'


@dataclass(frozen=True)
class _Warning:
    kind: WarningKind
    severity: Literal['URGENT', 'CRITICAL']
    details: str


@dataclass
class LimitWarner(AbstractCapability[AgentDepsT]):
    """Injects a warning message when the agent approaches configured limits.

    The warning is appended as a trailing ``ModelRequest`` with a
    ``UserPromptPart`` so that the model treats it as a distinct user turn
    (models tend to pay more attention to user messages than system messages).

    Previous warnings injected by this capability are stripped before deciding
    whether to inject a new one.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_harness.compaction import LimitWarner

        agent = Agent(
            'openai:gpt-4o',
            capabilities=[LimitWarner(
                max_iterations=40,
                max_context_tokens=100_000,
            )],
        )
        ```
    """

    max_iterations: int | None = None
    """Maximum allowed requests for the run."""

    max_context_tokens: int | None = None
    """Maximum context-window size to warn against."""

    max_total_tokens: int | None = None
    """Maximum cumulative run token budget to warn against."""

    warn_on: list[WarningKind] | None = None
    """Which limits should emit warnings.  Defaults to all configured limits."""

    warning_threshold: float = 0.7
    """Fraction of a limit at which warnings begin (between 0 and 1)."""

    critical_remaining_iterations: int = 3
    """Remaining request count at which iteration warnings become CRITICAL."""

    _active_kinds: tuple[WarningKind, ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:  # noqa: D105
        if self.max_iterations is not None and self.max_iterations <= 0:
            raise ValueError('max_iterations must be positive.')
        if self.max_context_tokens is not None and self.max_context_tokens <= 0:
            raise ValueError('max_context_tokens must be positive.')
        if self.max_total_tokens is not None and self.max_total_tokens <= 0:
            raise ValueError('max_total_tokens must be positive.')
        if not 0 < self.warning_threshold <= 1:
            raise ValueError('warning_threshold must be between 0 (exclusive) and 1 (inclusive).')
        if self.critical_remaining_iterations < 0:
            raise ValueError('critical_remaining_iterations must be non-negative.')

        configured: dict[WarningKind, int | None] = {
            'iterations': self.max_iterations,
            'context_window': self.max_context_tokens,
            'total_tokens': self.max_total_tokens,
        }
        if all(v is None for v in configured.values()):
            raise ValueError('At least one of max_iterations, max_context_tokens, or max_total_tokens must be set.')

        if self.warn_on is None:
            self._active_kinds = tuple(k for k in _WARNING_ORDER if configured[k] is not None)
        else:
            if not self.warn_on:
                raise ValueError('warn_on must not be empty.')
            for kind in self.warn_on:
                if configured[kind] is None:
                    raise ValueError(f'{kind!r} requires its corresponding max_* limit to be configured.')
            self._active_kinds = tuple(dict.fromkeys(self.warn_on))

    # -- internal helpers --

    @staticmethod
    def _is_marker_part(part: Any) -> bool:
        if isinstance(part, SystemPromptPart):
            return _MARKER in part.content
        if isinstance(part, UserPromptPart) and isinstance(part.content, str):
            return _MARKER in part.content
        return False

    def _strip_old_warnings(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        cleaned: list[ModelMessage] = []
        for msg in messages:
            if not isinstance(msg, ModelRequest):
                cleaned.append(msg)
                continue
            parts = [p for p in msg.parts if not self._is_marker_part(p)]
            if not parts:
                continue
            if len(parts) == len(msg.parts):
                cleaned.append(msg)
            else:
                cleaned.append(ModelRequest(parts=parts))
        return cleaned

    def _build_iteration_warning(self, ctx: RunContext[AgentDepsT]) -> _Warning | None:
        if self.max_iterations is None or 'iterations' not in self._active_kinds:
            return None
        usage_frac = ctx.usage.requests / self.max_iterations
        if usage_frac < self.warning_threshold:
            return None
        remaining = max(0, self.max_iterations - ctx.usage.requests)
        severity: Literal['URGENT', 'CRITICAL'] = (
            'CRITICAL' if remaining <= self.critical_remaining_iterations else 'URGENT'
        )
        details = f'Iterations: {ctx.usage.requests}/{self.max_iterations} requests used ({usage_frac:.0%}); {remaining} remaining.'
        return _Warning(kind='iterations', severity=severity, details=details)

    def _build_context_warning(self, context_tokens: int) -> _Warning | None:
        if self.max_context_tokens is None or 'context_window' not in self._active_kinds:
            return None  # pragma: no cover
        usage_frac = context_tokens / self.max_context_tokens
        if usage_frac < self.warning_threshold:
            return None
        remaining = max(0, self.max_context_tokens - context_tokens)
        severity: Literal['URGENT', 'CRITICAL'] = 'CRITICAL' if usage_frac >= 1 else 'URGENT'
        details = f'Context window: {context_tokens}/{self.max_context_tokens} tokens used ({usage_frac:.0%}); {remaining} remaining.'
        return _Warning(kind='context_window', severity=severity, details=details)

    def _build_total_tokens_warning(self, ctx: RunContext[AgentDepsT]) -> _Warning | None:
        if self.max_total_tokens is None or 'total_tokens' not in self._active_kinds:
            return None
        total = ctx.usage.total_tokens
        usage_frac = total / self.max_total_tokens
        if usage_frac < self.warning_threshold:
            return None
        remaining = max(0, self.max_total_tokens - total)
        severity: Literal['URGENT', 'CRITICAL'] = 'CRITICAL' if usage_frac >= 1 else 'URGENT'
        details = f'Total tokens: {total}/{self.max_total_tokens} used ({usage_frac:.0%}); {remaining} remaining.'
        return _Warning(kind='total_tokens', severity=severity, details=details)

    @staticmethod
    def _format_warning(warnings: list[_Warning]) -> str:
        severity: Literal['URGENT', 'CRITICAL'] = (
            'URGENT' if all(w.severity == 'URGENT' for w in warnings) else 'CRITICAL'
        )
        guidance = (
            'Complete the current task efficiently and avoid unnecessary tool calls.'
            if severity == 'URGENT'
            else 'Complete the current task immediately and avoid unnecessary tool calls.'
        )
        lines = [_MARKER, f'{severity}: Configured run limits are approaching.']
        lines.extend(f'- {w.details}' for w in warnings)
        lines.append(guidance)
        return '\n'.join(lines)

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Strip old warnings, then inject a new one if thresholds are exceeded."""
        messages = self._strip_old_warnings(list(request_context.messages))

        active: list[_Warning] = []

        w = self._build_iteration_warning(ctx)
        if w is not None:
            active.append(w)

        if self.max_context_tokens is not None and 'context_window' in self._active_kinds:
            context_tokens = estimate_token_count(messages)
            w = self._build_context_warning(context_tokens)
            if w is not None:
                active.append(w)

        w = self._build_total_tokens_warning(ctx)
        if w is not None:
            active.append(w)

        if not active:
            request_context.messages = messages
            return request_context

        order = {k: i for i, k in enumerate(_WARNING_ORDER)}
        active.sort(key=lambda w: order[w.kind])
        warning_text = self._format_warning(active)
        messages.append(ModelRequest(parts=[UserPromptPart(content=warning_text)]))

        request_context.messages = messages
        return request_context


# ---------------------------------------------------------------------------
# Compaction (LLM-powered summarization)
# ---------------------------------------------------------------------------

_DEFAULT_SUMMARY_PROMPT = """\
You are a context summarization assistant.  Extract the most important \
information from the conversation below.

The conversation history will be replaced with your summary, so include all \
facts, decisions, and outcomes that are necessary for continuing the task.  \
Do NOT repeat completed actions — focus on results and open questions.

Respond ONLY with the summary.  No preamble, no markdown fences.

<messages>
{messages}
</messages>\
"""

_SUMMARY_PREFIX = 'Summary of previous conversation:\n\n'


def _format_messages(messages: Sequence[ModelMessage]) -> str:
    """Render messages into a human-readable string for summarization."""
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    lines.append(f'User: {_user_prompt_text(part)}')
                elif isinstance(part, SystemPromptPart):
                    lines.append(f'System: {part.content}')
                elif isinstance(part, ToolReturnPart):
                    content_str = str(part.content)[:500]
                    if len(str(part.content)) > 500:
                        content_str += '...'
                    lines.append(f'Tool [{part.tool_name}]: {content_str}')
        else:
            for part in msg.parts:
                if isinstance(part, TextPart):
                    lines.append(f'Assistant: {part.content}')
                elif isinstance(part, ToolCallPart):
                    lines.append(f'Tool Call [{part.tool_name}]: {part.args}')
    return '\n'.join(lines)


def _user_prompt_text(part: UserPromptPart) -> str:
    """Extract text content from a user prompt part."""
    if isinstance(part.content, str):
        return part.content
    texts: list[str] = []
    for item in part.content:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, TextContent):
            texts.append(item.content)
    return ' '.join(texts) if texts else ''


def _extract_system_prompts(messages: list[ModelMessage]) -> list[SystemPromptPart]:
    """Extract leading system-prompt parts from the conversation."""
    parts: list[SystemPromptPart] = []
    for msg in messages:
        if not isinstance(msg, ModelRequest):
            break
        for part in msg.parts:
            if isinstance(part, SystemPromptPart):
                parts.append(part)
            else:
                return parts
    return parts


def _extract_previous_summary(messages: list[ModelMessage]) -> str | None:
    """Extract the most recent compaction summary from the message history.

    Looks for a ``SystemPromptPart`` whose content starts with the summary prefix,
    which indicates it was produced by a prior compaction pass.
    """
    for msg in messages:
        if not isinstance(msg, ModelRequest):
            continue
        for part in msg.parts:
            if isinstance(part, SystemPromptPart) and part.content.startswith(_SUMMARY_PREFIX):
                return part.content[len(_SUMMARY_PREFIX) :]
    return None


@dataclass
class Compaction(AbstractCapability[AgentDepsT]):
    """LLM-powered conversation compaction.

    When the conversation exceeds a configurable threshold, older messages are
    summarized using a dedicated model call and replaced with a compact summary
    message, preserving recent context and tool-call integrity.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_harness.compaction import Compaction

        agent = Agent(
            'openai:gpt-4o',
            capabilities=[Compaction(
                model='openai:gpt-4o-mini',
                max_messages=60,
                keep_messages=20,
            )],
        )
        ```
    """

    model: str
    """Model to use for generating summaries (e.g. ``'openai:gpt-4o-mini'``)."""

    max_messages: int | None = None
    """Trigger compaction when message count exceeds this value."""

    max_tokens: int | None = None
    """Trigger compaction when estimated token count exceeds this value."""

    keep_messages: int = 20
    """Number of tail messages to preserve after compaction (message-count trigger)."""

    keep_tokens: int | None = None
    """Target token budget to preserve after compaction (token-count trigger).

    When ``None``, falls back to ``keep_messages``.
    """

    summary_prompt: str = _DEFAULT_SUMMARY_PROMPT
    """Prompt template for generating summaries.

    Must contain a ``{messages}`` placeholder.
    """

    tokenizer: Callable[[str], int] | None = None
    """Optional tokenizer for accurate token counting.

    A callable that returns the token count for a given string.
    When ``None``, uses a ~4 characters-per-token heuristic.
    """

    preserve_first_user_message: bool = True
    """When ``True``, the first ``ModelRequest`` containing a ``UserPromptPart``
    is always kept after compaction, in addition to system prompts.
    """

    incremental: bool = True
    """When ``True``, include any existing summary from a prior compaction in the
    summarization prompt so that it is extended rather than regenerated from scratch.
    """

    def __post_init__(self) -> None:  # noqa: D105
        if self.max_messages is None and self.max_tokens is None:
            raise ValueError('At least one of max_messages or max_tokens must be set.')
        if self.max_messages is not None and self.max_messages < 1:
            raise ValueError('max_messages must be positive.')
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError('max_tokens must be positive.')
        if self.keep_messages < 0:
            raise ValueError('keep_messages must be non-negative.')
        if self.keep_tokens is not None and self.keep_tokens < 0:
            raise ValueError('keep_tokens must be non-negative.')

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Summarize older messages when the threshold is exceeded."""
        messages: list[ModelMessage] = list(request_context.messages)
        triggered = False

        if self.max_messages is not None and len(messages) > self.max_messages:
            triggered = True
        if not triggered and self.max_tokens is not None:
            if estimate_token_count(messages, self.tokenizer) > self.max_tokens:
                triggered = True

        if not triggered:
            return request_context

        if self.keep_tokens is not None:
            cutoff = _find_token_cutoff(messages, self.keep_tokens, self.tokenizer)
        else:
            cutoff = _find_safe_cutoff(messages, self.keep_messages)

        if cutoff <= 0:
            return request_context

        system_parts = _extract_system_prompts(messages)
        to_summarize = messages[:cutoff]
        preserved = messages[cutoff:]

        previous_summary = _extract_previous_summary(messages) if self.incremental else None
        summary = await self._summarize(to_summarize, previous_summary=previous_summary)

        summary_part = SystemPromptPart(content=f'{_SUMMARY_PREFIX}{summary}')
        summary_message = ModelRequest(parts=[*system_parts, summary_part])

        first_user: list[ModelMessage] = []
        if self.preserve_first_user_message:
            first_user_msg = _find_first_user_message(messages)
            if first_user_msg is not None:
                idx = messages.index(first_user_msg)
                if idx < cutoff and first_user_msg not in preserved:
                    first_user = [first_user_msg]

        request_context.messages = [summary_message, *first_user, *preserved]
        return request_context

    async def _summarize(
        self,
        messages: list[ModelMessage],
        *,
        previous_summary: str | None = None,
    ) -> str:
        """Generate a summary for the given messages using the configured model."""
        from pydantic_ai import Agent

        formatted = _format_messages(messages)
        prompt = self.summary_prompt.format(messages=formatted)

        if previous_summary is not None:
            prompt = f'{prompt}\n\n<previous_summary>\n{previous_summary}\n</previous_summary>'

        agent: Agent[None, str] = Agent(
            self.model,
            instructions='You are a context summarization assistant. Extract the most important information from conversations.',
        )
        result = await agent.run(prompt)
        return result.output.strip()


__all__ = [
    'Compaction',
    'LimitWarner',
    'SlidingWindow',
    'WarningKind',
    'estimate_token_count',
]
