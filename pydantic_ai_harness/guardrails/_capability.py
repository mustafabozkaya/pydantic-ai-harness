"""Input and output guardrail capabilities.

`InputGuard` intercepts the first model request and lets a user-supplied
callable decide whether the user prompt is safe to send to the model. A guard
that reports the input unsafe triggers a graceful refusal: the LLM call is
skipped via [`SkipModelRequest`][pydantic_ai.exceptions.SkipModelRequest] and
a refusal message becomes the model response for that step. A guard that
raises propagates the exception so the caller can observe a hard failure.

`OutputGuard` runs once the run completes and validates the final output.
A guard that reports the output unsafe raises
[`OutputBlocked`][pydantic_ai_harness.guardrails.OutputBlocked].

A guard returns either a bare `bool` (`True` = safe) or a
[`GuardResult`][pydantic_ai_harness.guardrails.GuardResult] carrying a refusal
`message`. Guards may be sync or async, and may optionally take a
[`RunContext`][pydantic_ai.tools.RunContext] as their first argument —
detected from the signature, the way pydantic-ai treats output validators.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability, WrapModelRequestHandler
from pydantic_ai.exceptions import SkipModelRequest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.tools import AgentDepsT, RunContext

from pydantic_ai_harness.guardrails._exceptions import OutputBlocked

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.run import AgentRunResult


_DEFAULT_INPUT_BLOCK_MESSAGE = 'Request blocked by input guardrail.'
_DEFAULT_OUTPUT_BLOCK_MESSAGE = 'Output blocked by output guardrail.'


@dataclass
class GuardResult:
    """Verdict returned by a guard.

    A guard may return a bare `bool` (`True` = safe) for the common case, or a
    `GuardResult` when it wants to attach a refusal `message` describing what
    tripped it — produced at the moment the guard decides, so it can reflect
    the guard's own reasoning. The `message` is used only when `safe` is
    `False`; when it is `None` a capability-specific default is used.
    """

    safe: bool
    """`True` when the inspected value is safe to proceed with."""

    message: str | None = None
    """Refusal text surfaced when `safe` is `False`. `None` falls back to a default."""


GuardOutcome = bool | GuardResult
"""What a guard callable returns: a bare `bool` (`True` = safe), or a `GuardResult`."""


InputGuardFunc = (
    Callable[[str], GuardOutcome | Awaitable[GuardOutcome]]
    | Callable[[RunContext[AgentDepsT], str], GuardOutcome | Awaitable[GuardOutcome]]
)
"""Signature of the callable passed to `InputGuard`.

The callable receives the user prompt and returns `True` / `GuardResult` when
the input is safe to send. It may optionally take a
[`RunContext`][pydantic_ai.tools.RunContext] as a first argument — for `deps`,
message history, or other run state — and may be sync or async. Raising an
exception is treated as a hard failure and propagates up to the caller.
"""

OutputGuardFunc = (
    Callable[[object], GuardOutcome | Awaitable[GuardOutcome]]
    | Callable[[RunContext[AgentDepsT], object], GuardOutcome | Awaitable[GuardOutcome]]
)
"""Signature of the callable passed to `OutputGuard`.

The callable receives `result.output` unchanged — for typed outputs this is
the Pydantic model (not a stringified form), so the guard can read fields
directly or serialize with `model_dump_json()`. It may optionally take a
[`RunContext`][pydantic_ai.tools.RunContext] first, and may be sync or async.
"""


def _takes_ctx(func: Callable[..., object]) -> bool:
    """Return `True` when `func` declares a leading `RunContext` parameter.

    Detected by parameter count: a guard always takes the guarded value, so a
    second parameter means it also wants the run context. This matches
    pydantic-ai's own optional-`ctx` convention for output validators.
    """
    return len(inspect.signature(func).parameters) > 1


async def _evaluate(
    guard: Callable[..., GuardOutcome | Awaitable[GuardOutcome]],
    ctx: RunContext[AgentDepsT],
    value: object,
) -> GuardResult:
    """Call `guard` (passing `ctx` when declared), await it, and normalize to `GuardResult`."""
    outcome = guard(ctx, value) if _takes_ctx(guard) else guard(value)
    if inspect.isawaitable(outcome):
        outcome = await outcome
    if isinstance(outcome, GuardResult):
        return outcome
    return GuardResult(safe=outcome)


def _extract_prompt(ctx: RunContext[AgentDepsT], messages: Sequence[ModelMessage]) -> str | None:
    """Return the text of the most recent user prompt, or `None` if absent.

    Prefers `ctx.prompt` (set at run start) and falls back to scanning the
    message history for the last [`UserPromptPart`][pydantic_ai.messages.UserPromptPart]
    so that sub-agent calls or resumed runs without a fresh prompt still work.
    """
    if ctx.prompt is not None:
        return ctx.prompt if isinstance(ctx.prompt, str) else str(ctx.prompt)
    for message in reversed(messages):
        for part in reversed(message.parts):
            if isinstance(part, UserPromptPart):
                return part.content if isinstance(part.content, str) else str(part.content)
    return None


@dataclass
class InputGuard(AbstractCapability[AgentDepsT]):
    """Validate the user prompt before it reaches the model.

    The `guard` callable receives the prompt text and reports whether the
    input is safe. Reporting it unsafe triggers a graceful refusal: the
    current model request is short-circuited via
    [`SkipModelRequest`][pydantic_ai.exceptions.SkipModelRequest], and the
    refusal message becomes the model response, so the agent returns cleanly
    to the caller. Raising an exception from the guard propagates it as-is.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import InputGuard


    def no_secrets(prompt: str) -> bool:
        return 'api_key' not in prompt.lower()


    agent = Agent('openai:gpt-5.4', capabilities=[InputGuard(guard=no_secrets)])
    ```

    Return a [`GuardResult`][pydantic_ai_harness.guardrails.GuardResult]
    instead of a bare `bool` to attach a refusal message describing what
    tripped the guard:

    ```python
    from pydantic_ai_harness import GuardResult


    def no_secrets(prompt: str) -> GuardResult:
        if 'api_key' in prompt.lower():
            return GuardResult(safe=False, message='Your message looks like it contains an API key.')
        return GuardResult(safe=True)
    ```

    The guard may take a [`RunContext`][pydantic_ai.tools.RunContext] as a
    first parameter when it needs run state — `deps` for tenant/role-aware
    policy, `messages` for conversation-aware checks. The parameter is
    detected from the signature, so prompt-only guards stay as-is.

    Set `parallel=True` to start the guard alongside the model call. The
    handler is cancelled as soon as the guard reports a violation, which saves
    tokens when the guard is slower than the provider round-trip.

    Scope: the guard runs exactly once per run — on the first model request —
    and evaluates the original user prompt. Subsequent model requests in the
    same run (e.g. after tool calls) are not re-checked, since the user input
    has not changed. Validation of tool results or other mid-run content
    belongs in a separate capability hooking `after_model_request`.
    """

    guard: InputGuardFunc[AgentDepsT]
    """Callable that reports whether the prompt is safe to send to the model."""

    parallel: bool = False
    """Run the guard concurrently with the model request and cancel the model call on failure."""

    async def _run_guard(self, ctx: RunContext[AgentDepsT], prompt: str) -> None:
        """Evaluate the guard and raise `SkipModelRequest` when it blocks the prompt."""
        verdict = await _evaluate(self.guard, ctx, prompt)
        if not verdict.safe:
            message = verdict.message or _DEFAULT_INPUT_BLOCK_MESSAGE
            raise SkipModelRequest(ModelResponse(parts=[TextPart(content=message)]))

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        """Check the prompt before the first model call.

        Sequential mode runs the guard then the model. `parallel=True` races
        the guard against the model call and cancels it on a violation.
        """
        if ctx.run_step > 1:
            return await handler(request_context)
        prompt = _extract_prompt(ctx, request_context.messages)
        if prompt is None:
            return await handler(request_context)
        if not self.parallel:
            await self._run_guard(ctx, prompt)
            return await handler(request_context)

        async def run_handler() -> ModelResponse:
            return await handler(request_context)

        guard_task: asyncio.Task[None] = asyncio.create_task(self._run_guard(ctx, prompt))
        handler_task: asyncio.Task[ModelResponse] = asyncio.create_task(run_handler())
        try:
            done, _ = await asyncio.wait(
                {guard_task, handler_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if guard_task in done:
                await guard_task
                return await handler_task

            response = await handler_task
            await guard_task
            return response
        finally:
            for task in (guard_task, handler_task):
                if not task.done():
                    task.cancel()

            await asyncio.gather(guard_task, handler_task, return_exceptions=True)


@dataclass
class OutputGuard(AbstractCapability[AgentDepsT]):
    """Validate the final agent output.

    The `guard` callable receives `result.output` unchanged — no automatic
    stringification — and reports whether the output is safe to expose.
    Reporting it unsafe raises
    [`OutputBlocked`][pydantic_ai_harness.guardrails.OutputBlocked]. Raising an
    exception from the guard propagates it.

    For string outputs the guard works directly on the text. For typed
    (Pydantic model) outputs the guard receives the model instance, so
    choose the serialization that fits your check: read a field directly,
    or call `model_dump_json()` to match against JSON text. Defaulting to
    `str(model)` would produce a `MyModel(field=...)` repr rather than JSON
    and easily hide fields from regex-based checks.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import OutputGuard


    def no_pii(output: object) -> bool:
        return 'SSN' not in str(output)


    agent = Agent('openai:gpt-5.4', capabilities=[OutputGuard(guard=no_pii)])
    ```

    Return a [`GuardResult`][pydantic_ai_harness.guardrails.GuardResult] to
    attach a refusal message. Like `InputGuard`, the guard may take a
    [`RunContext`][pydantic_ai.tools.RunContext] as a first parameter; it is
    detected from the signature.
    """

    guard: OutputGuardFunc[AgentDepsT]
    """Callable that reports whether the output is safe."""

    async def after_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        """Validate `result.output` and raise `OutputBlocked` when the guard blocks it."""
        verdict = await _evaluate(self.guard, ctx, result.output)
        if not verdict.safe:
            raise OutputBlocked(verdict.message or _DEFAULT_OUTPUT_BLOCK_MESSAGE)
        return result
