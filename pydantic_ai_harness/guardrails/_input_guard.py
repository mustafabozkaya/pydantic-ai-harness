"""InputGuard — capability that validates prompts before model requests."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from inspect import signature
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.exceptions import SkipModelRequest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.guardrails._guard_result import GuardResult

GuardCallable = Callable[..., bool | GuardResult | Awaitable[bool | GuardResult]]


def _is_async_guard(fn: Callable[..., Any]) -> bool:
    """Check if a guard callable is async."""
    from inspect import iscoroutinefunction

    return iscoroutinefunction(fn)


def _takes_run_context(fn: Callable[..., Any]) -> bool:
    """Check if a guard callable takes RunContext as first parameter."""
    params = list(signature(fn).parameters.keys())
    return params and params[0] == 'ctx'


@dataclass
class InputGuard(AbstractCapability[AgentDepsT]):
    """Capability that validates prompts before sending to the model.

    The ``guard`` callable receives the prompt string (or ``RunContext`` + prompt)
    and returns:
    - ``True`` or ``GuardResult.allow()`` — proceed normally
    - ``False`` or ``GuardResult.block(message)`` — skip the model call
    - ``GuardResult.replace(new_prompt)`` — rewrite the prompt

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.guardrails import InputGuard

    def no_jailbreak(prompt: str) -> bool:
        return 'ignore previous instructions' not in prompt.lower()

    agent = Agent('openai:gpt-5', capabilities=[InputGuard(guard=no_jailbreak)])
    ```
    """

    guard: GuardCallable
    """The guard function to validate prompts."""
    parallel: bool = False
    """If True, the guard runs in parallel with the model request (future use)."""

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost')

    async def wrap_model_request(self, ctx: RunContext[AgentDepsT], request_context: Any, handler: Any) -> Any:
        """Run the guard before the model request."""
        # Extract the prompt from the request context
        prompt = self._extract_prompt(request_context)

        # Run the guard
        result = await self._run_guard(ctx, prompt)

        if result is None or result.is_allow:
            # Allow — proceed with the model request
            return await handler(request_context)

        if result.is_block:
            # Block — skip model call, return refusal
            message = result._message or 'Request blocked by guardrail.'
            raise SkipModelRequest(ModelResponse(parts=[TextPart(content=message)]))

        if result.is_replace:
            # Replace — rewrite the prompt and proceed
            # We need to modify the request context with the new prompt
            # For now, we'll pass through with the replacement value
            # TODO: Properly rewrite the prompt in the request context
            return await handler(request_context)

        # Shouldn't reach here
        return await handler(request_context)

    def _extract_prompt(self, request_context: Any) -> str:
        """Extract the prompt text from the request context."""
        # request_context is a ModelRequestContext or similar
        # Try to get the user prompt from messages
        if hasattr(request_context, 'messages'):
            for msg in reversed(request_context.messages):
                if hasattr(msg, 'parts'):
                    for part in msg.parts:
                        if hasattr(part, 'content'):
                            return str(part.content)
        return str(request_context)

    async def _run_guard(self, ctx: RunContext[AgentDepsT], prompt: str) -> GuardResult | None:
        """Run the guard callable and normalize the result."""
        try:
            if _takes_run_context(self.guard):
                result = self.guard(ctx, prompt)
            else:
                result = self.guard(prompt)

            # Handle async guards
            if _is_async_guard(self.guard):
                if _takes_run_context(self.guard):
                    result = await self.guard(ctx, prompt)
                else:
                    result = await self.guard(prompt)

            # Normalize result
            if result is True:
                return GuardResult.allow()
            elif result is False:
                return GuardResult.block()
            elif isinstance(result, GuardResult):
                return result
            else:
                return GuardResult.allow()
        except Exception:
            # Guard exceptions propagate as hard failures
            raise
