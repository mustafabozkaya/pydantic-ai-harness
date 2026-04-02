"""Stuck loop detection capability for PydanticAI agents.

Detects when an agent is stuck repeating the same actions and either warns the
model via a retry prompt or raises an error to abort the run.

Detection scenarios:
    1. **Repeated calls**: The same tool is called with the same arguments
       `max_repeated_calls` times consecutively.
    2. **Alternating calls**: Two distinct tool calls alternate back and forth
       for `max_repeated_calls` full cycles (i.e. `max_repeated_calls * 2`
       consecutive tool calls forming an A-B-A-B pattern).
    3. **No-op calls**: The same tool returns the same result
       `max_repeated_calls` times consecutively, regardless of whether the
       arguments differ.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelResponse, ToolCallPart

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.tools import RunContext


class StuckLoopError(Exception):
    """Raised when the agent is detected to be stuck in a loop.

    Attributes:
        reason: A human-readable description of why the loop was detected.
    """

    reason: str

    def __init__(self, reason: str) -> None:
        """Initialize with a human-readable description of the detected loop."""
        self.reason = reason
        super().__init__(reason)


def _normalize_args(args: str | dict[str, Any] | None) -> str:
    """Produce a stable string representation of tool call arguments for comparison."""
    if args is None:
        return ''
    if isinstance(args, str):
        # Try to parse and re-serialize for canonical ordering.
        try:
            return json.dumps(json.loads(args), sort_keys=True)
        except (json.JSONDecodeError, ValueError):
            return args
    return json.dumps(args, sort_keys=True)


def _tool_call_key(part: ToolCallPart) -> str:
    """Return a hashable key representing the tool name + normalized arguments."""
    return f'{part.tool_name}::{_normalize_args(part.args)}'


def _detect_repeated(history: list[str], threshold: int) -> str | None:
    """Detect if the last *threshold* entries are all identical."""
    if len(history) < threshold:
        return None
    tail = history[-threshold:]
    if len(set(tail)) == 1:
        return tail[0]
    return None


def _detect_alternating(history: list[str], threshold: int) -> tuple[str, str] | None:
    """Detect an A-B-A-B pattern in the tail of *history*.

    Returns the two alternating keys if found, otherwise ``None``.
    A full "cycle" is A-B, so we need ``threshold * 2`` entries.
    """
    needed = threshold * 2
    if len(history) < needed:
        return None
    tail = history[-needed:]
    a, b = tail[0], tail[1]
    if a == b:
        return None
    for i, key in enumerate(tail):
        expected = a if i % 2 == 0 else b
        if key != expected:
            return None
    return (a, b)


DEFAULT_WARNING_MESSAGE = 'You appear to be stuck in a loop, repeating the same action(s). Try a different approach.'


@dataclass
class StuckLoopDetection(AbstractCapability[Any]):
    """Detects when an agent is stuck repeating the same tool calls.

    Monitors model responses for repetitive tool-call patterns and either
    sends a retry prompt asking the model to change strategy (``action='warn'``)
    or raises :class:`StuckLoopError` to abort the run (``action='error'``).

    Example::

        from pydantic_ai import Agent
        from pydantic_harness.stuck_loop_detection import StuckLoopDetection

        agent = Agent(
            'openai:gpt-4o',
            capabilities=[StuckLoopDetection(max_repeated_calls=3)],
        )
    """

    max_repeated_calls: int = 3
    """Number of consecutive repetitions before detection triggers."""

    action: Literal['warn', 'error'] = 'warn'
    """What to do when a loop is detected.

    - ``'warn'``: Raise :class:`~pydantic_ai.exceptions.ModelRetry` so the model
      receives a retry prompt asking it to try a different approach.
    - ``'error'``: Raise :class:`StuckLoopError` to abort the run.
    """

    warning_message: str = DEFAULT_WARNING_MESSAGE
    """The message sent to the model (or included in the error) when a loop is detected."""

    # --- Per-run state (populated by ``for_run``) ---

    _call_history: list[str] = field(default_factory=lambda: list[str](), repr=False)
    """Keys of recent tool calls (tool_name::normalized_args)."""

    _result_history: list[tuple[str, str]] = field(default_factory=lambda: list[tuple[str, str]](), repr=False)
    """Pairs of (tool_name, repr(result)) for no-op detection."""

    async def for_run(self, ctx: RunContext[Any]) -> StuckLoopDetection:
        """Return a fresh instance with empty history for each agent run."""
        return StuckLoopDetection(
            max_repeated_calls=self.max_repeated_calls,
            action=self.action,
            warning_message=self.warning_message,
        )

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Track tool calls from the model response and check for loops."""
        tool_calls = [p for p in response.parts if isinstance(p, ToolCallPart)]
        if not tool_calls:
            return response

        for tc in tool_calls:
            self._call_history.append(_tool_call_key(tc))

        # --- Check for repeated identical calls ---
        reason = self._check_repeated()
        if reason is None:
            reason = self._check_alternating()

        if reason is not None:
            self._trigger(reason)

        return response

    async def after_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: Any,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        """Track tool results for no-op detection."""
        result_repr = repr(result)
        self._result_history.append((call.tool_name, result_repr))

        reason = self._check_noop()
        if reason is not None:
            self._trigger(reason)

        return result

    # --- Detection helpers ---

    def _check_repeated(self) -> str | None:
        match = _detect_repeated(self._call_history, self.max_repeated_calls)
        if match is not None:
            name = match.split('::')[0]
            return f'Tool `{name}` called {self.max_repeated_calls} times with identical arguments.'
        return None

    def _check_alternating(self) -> str | None:
        match = _detect_alternating(self._call_history, self.max_repeated_calls)
        if match is not None:
            a_name = match[0].split('::')[0]
            b_name = match[1].split('::')[0]
            return f'Alternating between `{a_name}` and `{b_name}` for {self.max_repeated_calls} cycles.'
        return None

    def _check_noop(self) -> str | None:
        if len(self._result_history) < self.max_repeated_calls:
            return None
        tail = self._result_history[-self.max_repeated_calls :]
        names = {t[0] for t in tail}
        results = {t[1] for t in tail}
        if len(names) == 1 and len(results) == 1:
            return f'Tool `{next(iter(names))}` returned the same result {self.max_repeated_calls} times.'
        return None

    def _trigger(self, reason: str) -> None:
        """Trigger the configured action."""
        message = f'{self.warning_message}\n\nDetected: {reason}'
        if self.action == 'error':
            raise StuckLoopError(message)
        raise ModelRetry(message)
