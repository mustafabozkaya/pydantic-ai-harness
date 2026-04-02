"""Adaptive reasoning effort capability.

Dynamically adjusts the model's thinking effort level per step based on
task complexity signals, reducing token usage on simple steps while
preserving deep reasoning for complex decisions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from pydantic_ai._run_context import RunContext
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, RetryPromptPart
from pydantic_ai.settings import ModelSettings, ThinkingEffort

EffortLevel: TypeAlias = Literal['low', 'medium', 'high']
"""The coarse effort levels used by adaptive reasoning.

Mapped to the full ``ThinkingEffort`` scale when applied to model settings.
"""

_EFFORT_TO_THINKING: dict[str, ThinkingEffort] = {
    'low': 'low',
    'medium': 'medium',
    'high': 'high',
}


def _has_tool_errors(messages: Sequence[ModelMessage]) -> bool:
    """Check whether the most recent request message contains retry prompts (tool errors)."""
    for msg in reversed(messages):
        if isinstance(msg, ModelRequest):
            return any(isinstance(part, RetryPromptPart) for part in msg.parts)
    return False


def default_effort_fn(ctx: RunContext[Any]) -> Literal['low', 'medium', 'high']:
    """Built-in heuristic effort selector.

    Rules (evaluated in order):
    1. First step (``run_step == 1``): ``'high'`` -- understand the task.
    2. After tool errors (retry prompts in the latest request): ``'high'`` -- reason about failures.
    3. Steps 2+ with no errors: ``'low'`` -- simple follow-ups incorporating tool results.
    4. Default: ``'medium'``.
    """
    if ctx.run_step <= 1:
        return 'high'

    if _has_tool_errors(ctx.messages):
        return 'high'

    # Later steps without errors are typically straightforward follow-ups.
    if ctx.run_step >= 2:
        return 'low'

    return 'medium'  # pragma: no cover


@dataclass
class AdaptiveReasoning(AbstractCapability[Any]):
    """Dynamically adjusts model thinking effort per step.

    By default a built-in heuristic is used:

    * **First step** -> ``'high'`` (understand the task)
    * **After tool errors** -> ``'high'`` (reason about what went wrong)
    * **Simple follow-ups** -> ``'low'`` (just incorporating tool results)

    Supply a custom ``effort_fn`` to override these rules::

        def my_effort(ctx: RunContext[MyDeps]) -> Literal['low', 'medium', 'high']:
            if ctx.run_step > 5:
                return 'high'  # wrap-up needs careful thought
            return 'medium'

        agent = Agent(..., capabilities=[AdaptiveReasoning(effort_fn=my_effort)])
    """

    effort_fn: Callable[[RunContext[Any]], Literal['low', 'medium', 'high']] = field(default=default_effort_fn)
    """Callable that receives the current ``RunContext`` and returns an effort level."""

    def get_model_settings(self) -> Callable[[RunContext[Any]], ModelSettings]:
        """Return a dynamic model-settings callable that sets ``thinking`` per step."""

        def _resolve(ctx: RunContext[Any]) -> ModelSettings:
            effort = self.effort_fn(ctx)
            return ModelSettings(thinking=_EFFORT_TO_THINKING[effort])

        return _resolve
