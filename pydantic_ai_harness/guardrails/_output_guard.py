"""OutputGuard — capability that validates model outputs."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from inspect import signature
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.guardrails._guard_result import GuardResult

GuardCallable = Callable[..., bool | GuardResult | Awaitable[bool | GuardResult]]


def _is_async_guard(fn: Callable[..., Any]) -> bool:
    from inspect import iscoroutinefunction

    return iscoroutinefunction(fn)


def _takes_run_context(fn: Callable[..., Any]) -> bool:
    params = list(signature(fn).parameters.keys())
    return params and params[0] == 'ctx'


@dataclass
class OutputGuard(AbstractCapability[AgentDepsT]):
    """Capability that validates model outputs after processing.

    The ``guard`` callable receives the output value (or ``RunContext`` + output)
    and returns:
    - ``True`` or ``GuardResult.allow()`` — return the output
    - ``False`` or ``GuardResult.block(message)`` — raise OutputBlocked
    - ``GuardResult.replace(value)`` — return the replacement value
    - ``GuardResult.retry(message)`` — send back to model for retry

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.guardrails import OutputGuard

    def no_pii(output: str) -> bool:
        return '@' not in output  # simple PII check

    agent = Agent('openai:gpt-5', capabilities=[OutputGuard(guard=no_pii)])
    ```
    """

    guard: GuardCallable

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='innermost')

    async def wrap_output_process(self, ctx: RunContext[AgentDepsT], output: Any, handler: Any, **kwargs: Any) -> Any:
        """Run the guard on the output after processing."""
        result = await self._run_guard(ctx, output)

        if result is None or result.is_allow:
            return await handler(output)

        if result.is_block:
            message = result._message or 'Output blocked by guardrail.'
            raise OutputBlocked(message)

        if result.is_replace:
            return await handler(result._value)

        if result.is_retry:
            message = result._message or 'Output rejected, retrying.'
            raise ModelRetry(message)

        return await handler(output)

    async def _run_guard(self, ctx: RunContext[AgentDepsT], output: Any) -> GuardResult | None:
        try:
            if _is_async_guard(self.guard):
                if _takes_run_context(self.guard):
                    result = await self.guard(ctx, output)
                else:
                    result = await self.guard(output)
            else:
                if _takes_run_context(self.guard):
                    result = self.guard(ctx, output)
                else:
                    result = self.guard(output)

            if result is True:
                return GuardResult.allow()
            elif result is False:
                return GuardResult.block()
            elif isinstance(result, GuardResult):
                return result
            else:
                return GuardResult.allow()
        except Exception:
            raise


class OutputBlocked(Exception):
    """Raised when an OutputGuard blocks the output."""
