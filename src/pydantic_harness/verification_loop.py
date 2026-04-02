"""Verification loop capability for PydanticAI agents.

Runs configurable verification checks after the agent completes and retries
with failure feedback if any check fails, up to a configurable maximum number
of retries.

Example::

    from pydantic_ai import Agent
    from pydantic_harness import VerificationLoop, Verifier, VerificationResult

    async def check_lint() -> VerificationResult:
        # Run linting, return pass/fail
        return VerificationResult(passed=True, message='No lint errors.')

    agent = Agent(
        'openai:gpt-4o',
        capabilities=[
            VerificationLoop(
                verifiers=[Verifier(name='lint', check_fn=check_lint)],
                max_retries=3,
            ),
        ],
    )
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.capabilities.abstract import AbstractCapability, WrapRunHandler
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import RunContext

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """The outcome of a single verification check.

    Attributes:
        passed: Whether the check passed.
        message: A human-readable description of the outcome.
    """

    passed: bool
    message: str


@dataclass
class Verifier:
    """A named verification check to run after agent completion.

    Attributes:
        name: A short identifier for this verifier (e.g. ``'lint'``, ``'test'``).
        check_fn: An async callable that returns a :class:`VerificationResult`.
    """

    name: str
    check_fn: Callable[[], Awaitable[VerificationResult]]


@dataclass
class VerificationLoop(AbstractCapability[Any]):
    """Runs verification checks after agent completion and retries on failure.

    After the agent produces a result, each :class:`Verifier` is run in order.
    If any verifier fails, the agent is re-run with the failure messages
    appended as context so the model can fix the issues.  This repeats up to
    ``max_retries`` times.  If all retries are exhausted the last result is
    returned and a warning is emitted.

    Example::

        from pydantic_ai import Agent
        from pydantic_harness import VerificationLoop, Verifier, VerificationResult

        async def check_tests() -> VerificationResult:
            ...

        agent = Agent(
            'openai:gpt-4o',
            capabilities=[
                VerificationLoop(
                    verifiers=[Verifier(name='tests', check_fn=check_tests)],
                ),
            ],
        )
    """

    verifiers: list[Verifier] = field(default_factory=lambda: list[Verifier]())
    """Verifiers to run after each agent completion."""

    max_retries: int = 3
    """Maximum number of retry attempts when verification fails."""

    # --- Per-run state ---

    _in_retry: bool = field(default=False, repr=False)
    """When ``True``, :meth:`wrap_run` skips verification (retry pass-through)."""

    async def wrap_run(
        self,
        ctx: RunContext[Any],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        """Run the agent, then verify.  Retry with feedback on failure.

        When the agent is re-run for a retry, this hook fires again on
        the new run.  The ``_in_retry`` flag prevents recursive verification:
        retry runs pass straight through to the handler.
        """
        result = await handler()

        # Retry runs skip verification to avoid infinite recursion.
        if self._in_retry:
            return result

        agent = ctx.agent

        for attempt in range(1, self.max_retries + 1):
            failures = await self._run_verifiers()
            if not failures:
                return result

            failure_summary = '; '.join(f'{name}: {msg}' for name, msg in failures)
            feedback = self._build_feedback(failures, attempt)
            logger.info(
                'Verification failed (attempt %d/%d): %s',
                attempt,
                self.max_retries,
                failure_summary,
            )

            if agent is None:  # pragma: no cover — defensive; agent is always set in practice
                warnings.warn(
                    'Verification failed but agent is not available on RunContext for retry. Returning last result.',
                    stacklevel=2,
                )
                return result

            # Mark that the next run is a retry so wrap_run passes through.
            self._in_retry = True
            try:
                result = await agent.run(
                    feedback,
                    message_history=result.all_messages(),
                )
            finally:
                self._in_retry = False

        # Final verification after last retry.
        failures = await self._run_verifiers()
        if not failures:
            return result

        warnings.warn(
            f'Verification still failing after {self.max_retries} retries: '
            + '; '.join(f'{name}: {msg}' for name, msg in failures),
            stacklevel=2,
        )
        return result

    async def _run_verifiers(self) -> list[tuple[str, str]]:
        """Run all verifiers and return a list of ``(name, message)`` for failures."""
        failures: list[tuple[str, str]] = []
        for verifier in self.verifiers:
            vr = await verifier.check_fn()
            if not vr.passed:
                failures.append((verifier.name, vr.message))
        return failures

    @staticmethod
    def _build_feedback(failures: list[tuple[str, str]], attempt: int) -> str:
        """Build a feedback prompt from verification failures."""
        parts = [f'Verification failed (attempt {attempt}). Please fix the issues:']
        for name, message in failures:
            parts.append(f'- {name}: {message}')
        return '\n'.join(parts)
