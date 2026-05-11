"""The batteries for your Pydantic AI agent -- the official capability library."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .code_mode import CodeMode
    from .guardrails import (
        AsyncGuardFunc,
        AsyncGuardrail,
        BudgetExceededError,
        ContextGuardFunc,
        CostGuard,
        GuardrailError,
        GuardrailFailed,
        GuardrailMode,
        GuardrailResult,
        InputBlocked,
        InputGuardrail,
        OutputBlocked,
        OutputGuardrail,
        ToolBlocked,
        ToolGuard,
    )

__all__ = [
    'AsyncGuardFunc',
    'AsyncGuardrail',
    'BudgetExceededError',
    'CodeMode',
    'ContextGuardFunc',
    'CostGuard',
    'GuardrailError',
    'GuardrailFailed',
    'GuardrailMode',
    'GuardrailResult',
    'InputBlocked',
    'InputGuardrail',
    'OutputBlocked',
    'OutputGuardrail',
    'ToolBlocked',
    'ToolGuard',
]

_GUARDRAIL_NAMES = frozenset(
    {
        'AsyncGuardFunc',
        'AsyncGuardrail',
        'BudgetExceededError',
        'ContextGuardFunc',
        'CostGuard',
        'GuardrailError',
        'GuardrailFailed',
        'GuardrailMode',
        'GuardrailResult',
        'InputBlocked',
        'InputGuardrail',
        'OutputBlocked',
        'OutputGuardrail',
        'ToolBlocked',
        'ToolGuard',
    }
)


def __getattr__(name: str) -> object:
    if name == 'CodeMode':
        from .code_mode import CodeMode

        return CodeMode
    if name in _GUARDRAIL_NAMES:
        from . import guardrails

        return getattr(guardrails, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
