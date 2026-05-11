"""Guardrail capabilities for Pydantic AI agents.

Reusable capabilities for input/output validation, cost/token budget enforcement,
per-tool permission control, and concurrent model-request guardrails.
Built on Pydantic AI's native capabilities API.

Example:
    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import InputGuardrail, OutputGuardrail, CostGuard, ToolGuard

    agent = Agent(
        'openai:gpt-4.1',
        capabilities=[
            InputGuardrail(guard=lambda text: 'DROP TABLE' not in text),
            OutputGuardrail(guard=lambda text: 'password' not in text.lower()),
            CostGuard(max_total_tokens=100_000),
            ToolGuard(blocked=['execute_sql'], require_approval=['delete_file']),
        ],
    )
    ```
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias, TypeGuard

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import RunContext, ToolDefinition

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GuardrailError(Exception):
    """Base exception for guardrail violations."""


class InputBlocked(GuardrailError):
    """Raised when user input fails a guardrail check."""


class OutputBlocked(GuardrailError):
    """Raised when model output fails a guardrail check."""


class BudgetExceededError(GuardrailError):
    """Raised when token or cost budget is exceeded.

    Attributes:
        detail: A human-readable description of which limit was breached.
    """

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class ToolBlocked(GuardrailError):
    """Raised when a tool call is denied by a guardrail.

    Attributes:
        tool_name: The name of the blocked tool.
        reason: Why the tool was blocked.
    """

    def __init__(self, tool_name: str, *, reason: str = '') -> None:
        self.tool_name = tool_name
        self.reason = reason
        msg = f"Tool '{tool_name}' blocked"
        if reason:
            msg += f': {reason}'
        super().__init__(msg)


class GuardrailFailed(GuardrailError):
    """Raised when an async guardrail check fails.

    Attributes:
        result: The :class:`GuardrailResult` that triggered the failure.
    """

    def __init__(self, result: GuardrailResult) -> None:
        self.result = result
        super().__init__(f'Guardrail failed: {result.reason}')


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuardrailResult:
    """Result of a guardrail check.

    Attributes:
        passed: `True` if the check passed.
        reason: Human-readable explanation (used in error messages and logs).
    """

    passed: bool
    reason: str = ''


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

GuardFunc: TypeAlias = Callable[[str], bool] | Callable[[str], Awaitable[bool]]
"""A sync or async function that receives a text string and returns `True` if safe."""

ContextGuardFunc: TypeAlias = Callable[[RunContext[Any], str], bool] | Callable[[RunContext[Any], str], Awaitable[bool]]
"""A sync or async guard that receives `RunContext` and the text string, returns `True` if safe."""

ApprovalFunc: TypeAlias = Callable[[str, dict[str, Any]], bool] | Callable[[str, dict[str, Any]], Awaitable[bool]]
"""A sync or async function `(tool_name, args) -> bool` that grants or denies tool execution."""

AsyncGuardFunc: TypeAlias = (
    Callable[[list[ModelMessage]], Awaitable[GuardrailResult]]
    | Callable[[RunContext[Any], list[ModelMessage]], Awaitable[GuardrailResult]]
)
"""An async guard for :class:`AsyncGuardrail`.

Accepts either `(messages) -> GuardrailResult` or `(ctx, messages) -> GuardrailResult`.
"""

GuardrailMode: TypeAlias = Literal['concurrent', 'blocking', 'monitoring']
"""Execution mode for :class:`AsyncGuardrail`.

- `concurrent`: run guard and model call in parallel; cancel model if guard fails.
- `blocking`: run guard before the model call; raise on failure.
- `monitoring`: run guard after the model call; log failures without raising.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_awaitable_bool(val: bool | Awaitable[bool]) -> TypeGuard[Awaitable[bool]]:
    """Narrow a sync-or-async guard result to the awaitable case."""
    return inspect.isawaitable(val)


def _is_context_async_guard(
    func: AsyncGuardFunc,
) -> TypeGuard[Callable[[RunContext[Any], list[ModelMessage]], Awaitable[GuardrailResult]]]:
    """Narrow an async guard function to the 2-arg (context-aware) variant."""
    sig = inspect.signature(func)
    params = [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return len(params) >= 2


async def _call_guard(func: GuardFunc, text: str) -> bool:
    """Call a sync or async guard function and return its bool result."""
    result = func(text)
    if _is_awaitable_bool(result):
        return await result
    return result  # type: ignore[return-value]


async def _call_context_guard(func: ContextGuardFunc, ctx: RunContext[Any], text: str) -> bool:
    """Call a sync or async context-aware guard function and return its bool result."""
    result = func(ctx, text)
    if _is_awaitable_bool(result):
        return await result
    return result  # type: ignore[return-value]


async def _call_approval(func: ApprovalFunc, tool_name: str, args: dict[str, Any]) -> bool:
    """Call a sync or async approval function and return its bool result."""
    result = func(tool_name, args)
    if _is_awaitable_bool(result):
        return await result
    return result  # type: ignore[return-value]


async def _call_async_guard(
    func: AsyncGuardFunc, ctx: RunContext[Any], messages: list[ModelMessage]
) -> GuardrailResult:
    """Call an async guard function, auto-detecting the 1-arg or 2-arg signature."""
    if _is_context_async_guard(func):
        return await func(ctx, messages)
    return await func(messages)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# InputGuardrail
# ---------------------------------------------------------------------------


@dataclass
class InputGuardrail(AbstractCapability[Any]):
    """Validate user input before the agent run starts.

    The guard function receives the user prompt as a string (or `RunContext`
    and the string when `context_guard` is used) and returns `True`
    if the input is acceptable.  When it returns `False`, an
    `InputBlocked` exception is raised and the run never starts,
    unless `on_fail='warn'` in which case a warning is logged instead.

    Both sync and async guard functions are accepted.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_harness import InputGuardrail

        async def check_toxicity(text: str) -> bool:
            # Call a moderation API ...
            return True

        agent = Agent('openai:gpt-4.1', capabilities=[InputGuardrail(guard=check_toxicity)])
        ```
    """

    guard: GuardFunc | None = None
    """Function `(text) -> bool` that checks input safety.  Returns `True` if safe."""

    context_guard: ContextGuardFunc | None = None
    """Function `(ctx, text) -> bool` that checks input safety with access to `RunContext`."""

    on_fail: Literal['raise', 'warn'] = 'raise'
    """Action when the guard fails: `'raise'` (default) raises `InputBlocked`; `'warn'` logs a warning."""

    def __post_init__(self) -> None:
        """Validate that exactly one guard is provided."""
        if self.guard is None and self.context_guard is None:
            raise ValueError('Either guard or context_guard must be provided')
        if self.guard is not None and self.context_guard is not None:
            raise ValueError('Only one of guard or context_guard may be provided')

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not spec-serializable (takes a callable)."""
        return None

    async def before_run(self, ctx: RunContext[Any]) -> None:
        """Check user input before the run starts."""
        prompt = ctx.prompt
        if prompt is None:
            return

        prompt_str = str(prompt) if not isinstance(prompt, str) else prompt
        if self.context_guard is not None:
            passed = await _call_context_guard(self.context_guard, ctx, prompt_str)
        else:
            assert self.guard is not None
            passed = await _call_guard(self.guard, prompt_str)

        if not passed:
            msg = f'Input blocked by guardrail: {prompt_str[:100]}'
            if self.on_fail == 'warn':
                logger.warning(msg)
            else:
                raise InputBlocked(msg)


# ---------------------------------------------------------------------------
# OutputGuardrail
# ---------------------------------------------------------------------------


@dataclass
class OutputGuardrail(AbstractCapability[Any]):
    """Validate model output after the agent run completes.

    The guard function receives the stringified output (or `RunContext`
    and the string when `context_guard` is used) and returns `True`
    if the output is acceptable.  When it returns `False`, an
    `OutputBlocked` exception is raised, unless `on_fail='warn'`
    in which case a warning is logged instead and the result passes through.

    Both sync and async guard functions are accepted.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_harness import OutputGuardrail

        def no_secrets(text: str) -> bool:
            return 'sk-' not in text

        agent = Agent('openai:gpt-4.1', capabilities=[OutputGuardrail(guard=no_secrets)])
        ```
    """

    guard: GuardFunc | None = None
    """Function `(text) -> bool` that checks output safety.  Returns `True` if safe."""

    context_guard: ContextGuardFunc | None = None
    """Function `(ctx, text) -> bool` that checks output safety with access to `RunContext`."""

    on_fail: Literal['raise', 'warn'] = 'raise'
    """Action when the guard fails: `'raise'` (default) raises `OutputBlocked`; `'warn'` logs a warning."""

    def __post_init__(self) -> None:
        """Validate that exactly one guard is provided."""
        if self.guard is None and self.context_guard is None:
            raise ValueError('Either guard or context_guard must be provided')
        if self.guard is not None and self.context_guard is not None:
            raise ValueError('Only one of guard or context_guard may be provided')

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not spec-serializable (takes a callable)."""
        return None

    async def after_run(self, ctx: RunContext[Any], *, result: Any) -> Any:
        """Check model output after the run completes."""
        output_str = str(result.output)
        if self.context_guard is not None:
            passed = await _call_context_guard(self.context_guard, ctx, output_str)
        else:
            assert self.guard is not None
            passed = await _call_guard(self.guard, output_str)

        if not passed:
            msg = f'Output blocked by guardrail: {output_str[:100]}'
            if self.on_fail == 'warn':
                logger.warning(msg)
            else:
                raise OutputBlocked(msg)
        return result


# ---------------------------------------------------------------------------
# CostGuard
# ---------------------------------------------------------------------------


@dataclass
class CostGuard(AbstractCapability[Any]):
    """Enforce token budget limits during an agent run.

    Checks cumulative token usage via `ctx.usage` before each model request
    and raises `BudgetExceededError` when a configured threshold is
    exceeded.

    At least one of `max_input_tokens`, `max_output_tokens`, or
    `max_total_tokens` must be set.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_harness import CostGuard

        agent = Agent(
            'openai:gpt-4.1',
            capabilities=[CostGuard(max_total_tokens=50_000)],
        )
        ```
    """

    max_input_tokens: int | None = None
    """Maximum cumulative input tokens allowed.  `None` means unlimited."""

    max_output_tokens: int | None = None
    """Maximum cumulative output tokens allowed.  `None` means unlimited."""

    max_total_tokens: int | None = None
    """Maximum cumulative total tokens (input + output) allowed.  `None` means unlimited."""

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Return serialization name for spec construction."""
        return 'CostGuard'

    async def before_model_request(self, ctx: RunContext[Any], request_context: Any) -> Any:
        """Check token budget before each model request."""
        usage = ctx.usage

        if self.max_input_tokens is not None and usage.input_tokens > self.max_input_tokens:
            raise BudgetExceededError(f'Input token budget exceeded: {usage.input_tokens}/{self.max_input_tokens}')

        if self.max_output_tokens is not None and usage.output_tokens > self.max_output_tokens:
            raise BudgetExceededError(f'Output token budget exceeded: {usage.output_tokens}/{self.max_output_tokens}')

        if self.max_total_tokens is not None:
            total = usage.input_tokens + usage.output_tokens
            if total > self.max_total_tokens:
                raise BudgetExceededError(f'Total token budget exceeded: {total}/{self.max_total_tokens}')

        return request_context


# ---------------------------------------------------------------------------
# ToolGuard
# ---------------------------------------------------------------------------


@dataclass
class ToolGuard(AbstractCapability[Any]):
    """Control per-tool access: block tools or require approval before execution.

    Blocked tools are hidden from the model entirely via `prepare_tools`.
    Tools requiring approval trigger the `approval_callback` before execution;
    if no callback is configured or the callback returns `False`, a
    `ToolBlocked` exception is raised.

    Both sync and async approval callbacks are accepted.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_harness import ToolGuard

        async def ask_user(tool_name: str, args: dict) -> bool:
            return input(f'Allow {tool_name}? (y/n) ').lower() == 'y'

        agent = Agent(
            'openai:gpt-4.1',
            capabilities=[ToolGuard(
                blocked=['execute_sql'],
                require_approval=['delete_file', 'send_email'],
                approval_callback=ask_user,
            )],
        )
        ```
    """

    blocked: list[str] = field(default_factory=list[str])
    """Tool names to hide from the model entirely."""

    require_approval: list[str] = field(default_factory=list[str])
    """Tool names that require approval before execution."""

    approval_callback: ApprovalFunc | None = None
    """Callback `(tool_name, args) -> bool`.  Required when `require_approval` is non-empty."""

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not spec-serializable (takes a callable)."""
        return None

    async def prepare_tools(
        self,
        ctx: RunContext[Any],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Hide blocked tools from the model."""
        if not self.blocked:
            return tool_defs
        blocked_set = frozenset(self.blocked)
        return [td for td in tool_defs if td.name not in blocked_set]

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Enforce approval for tools in `require_approval`."""
        if call.tool_name not in self.require_approval:
            return args

        if self.approval_callback is None:
            raise ToolBlocked(call.tool_name, reason='approval required but no callback configured')

        if not await _call_approval(self.approval_callback, call.tool_name, args):
            raise ToolBlocked(call.tool_name, reason='approval denied')

        return args


# ---------------------------------------------------------------------------
# AsyncGuardrail
# ---------------------------------------------------------------------------


@dataclass
class AsyncGuardrail(AbstractCapability[Any]):
    """Run a guard function alongside model requests.

    Uses `AbstractCapability.wrap_model_request`
    to intercept each model call and execute the guard concurrently, before, or
    after the model request depending on `mode`.

    Modes:
        - `concurrent` (default): guard and model run in parallel via
          `asyncio.create_task`. If the guard fails first, the model task
          is cancelled and `GuardrailFailed` is raised.
        - `blocking`: guard runs *before* the model call. If the guard
          fails, the model is never called.
        - `monitoring`: model runs first, then the guard runs. Guard
          failures are logged but do not raise.

    The guard function may accept one or two positional arguments:

    - `async (messages: list[ModelMessage]) -> GuardrailResult`
    - `async (ctx: RunContext, messages: list[ModelMessage]) -> GuardrailResult`

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.messages import ModelMessage
        from pydantic_ai_harness import AsyncGuardrail, GuardrailResult

        async def prompt_injection_check(messages: list[ModelMessage]) -> GuardrailResult:
            # Run a classifier ...
            return GuardrailResult(passed=True)

        agent = Agent(
            'openai:gpt-4.1',
            capabilities=[AsyncGuardrail(guard=prompt_injection_check, mode='concurrent')],
        )
        ```
    """

    guard: AsyncGuardFunc
    """Async guard function to run on each model request."""

    mode: GuardrailMode = 'concurrent'
    """Execution mode: `'concurrent'`, `'blocking'`, or `'monitoring'`."""

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not spec-serializable (takes a callable)."""
        return None

    async def wrap_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        handler: Callable[[ModelRequestContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Intercept model requests to run the guard according to `mode`."""
        messages = request_context.messages
        if self.mode == 'blocking':
            return await self._run_blocking(ctx, messages, request_context, handler)
        elif self.mode == 'monitoring':
            return await self._run_monitoring(ctx, messages, request_context, handler)
        else:
            return await self._run_concurrent(ctx, messages, request_context, handler)

    async def _run_blocking(
        self,
        ctx: RunContext[Any],
        messages: list[ModelMessage],
        request_context: ModelRequestContext,
        handler: Callable[[ModelRequestContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Run guard before the model call; raise on failure."""
        result = await _call_async_guard(self.guard, ctx, messages)
        if not result.passed:
            raise GuardrailFailed(result)
        return await handler(request_context)

    async def _run_monitoring(
        self,
        ctx: RunContext[Any],
        messages: list[ModelMessage],
        request_context: ModelRequestContext,
        handler: Callable[[ModelRequestContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Run model first, then guard; log failures without raising."""
        response = await handler(request_context)
        result = await _call_async_guard(self.guard, ctx, messages)
        if not result.passed:
            logger.warning('AsyncGuardrail (monitoring): %s', result.reason)
        return response

    async def _run_concurrent(
        self,
        ctx: RunContext[Any],
        messages: list[ModelMessage],
        request_context: ModelRequestContext,
        handler: Callable[[ModelRequestContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Run guard and model in parallel; cancel model if guard fails first."""

        async def _call_model() -> ModelResponse:
            return await handler(request_context)

        guard_task: asyncio.Task[GuardrailResult] = asyncio.create_task(_call_async_guard(self.guard, ctx, messages))
        model_task: asyncio.Task[ModelResponse] = asyncio.create_task(_call_model())

        done: set[asyncio.Task[Any]] = set()
        done, _ = await asyncio.wait(
            {guard_task, model_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if guard_task in done:
            guard_result = guard_task.result()
            if not guard_result.passed:
                model_task.cancel()
                try:
                    await model_task
                except asyncio.CancelledError:
                    pass
                raise GuardrailFailed(guard_result)
            # Guard passed; wait for model to finish
            return await model_task

        # Model finished first; still check the guard result
        model_response: ModelResponse = model_task.result()
        guard_result = await guard_task
        if not guard_result.passed:
            raise GuardrailFailed(guard_result)
        return model_response


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Capabilities
    'InputGuardrail',
    'OutputGuardrail',
    'CostGuard',
    'ToolGuard',
    'AsyncGuardrail',
    # Data
    'GuardrailResult',
    # Exceptions
    'GuardrailError',
    'InputBlocked',
    'OutputBlocked',
    'BudgetExceededError',
    'ToolBlocked',
    'GuardrailFailed',
    # Type aliases
    'GuardrailMode',
    'AsyncGuardFunc',
    'ContextGuardFunc',
]
