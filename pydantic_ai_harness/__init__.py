"""The batteries for your Pydantic AI agent -- the official capability library."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .code_mode import CodeMode
    from .guardrails import (
        GuardrailError,
        GuardResult,
        InputBlocked,
        InputGuard,
        InputGuardFunc,
        OutputBlocked,
        OutputGuard,
        OutputGuardFunc,
    )

__all__ = [
    'CodeMode',
    'GuardResult',
    'GuardrailError',
    'InputBlocked',
    'InputGuard',
    'InputGuardFunc',
    'OutputBlocked',
    'OutputGuard',
    'OutputGuardFunc',
]

_GUARDRAIL_EXPORTS = {
    'GuardResult',
    'GuardrailError',
    'InputBlocked',
    'InputGuard',
    'InputGuardFunc',
    'OutputBlocked',
    'OutputGuard',
    'OutputGuardFunc',
}


def __getattr__(name: str) -> object:
    if name == 'CodeMode':
        from .code_mode import CodeMode

        return CodeMode
    if name in _GUARDRAIL_EXPORTS:
        from . import guardrails

        return getattr(guardrails, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
