"""The batteries for your Pydantic AI agent -- the official capability library."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .code_mode import CodeMode
    from .guardrails import GuardResult, InputGuard, OutputBlocked, OutputGuard, llm_input_guard, llm_output_guard
    from .memory import MemoryCapability

__all__ = ['CodeMode', 'GuardResult', 'InputGuard', 'MemoryCapability', 'OutputBlocked', 'OutputGuard',
           'llm_input_guard', 'llm_output_guard']


def __getattr__(name: str) -> object:
    if name == 'CodeMode':
        from .code_mode import CodeMode

        return CodeMode
    if name == 'MemoryCapability':
        from .memory import MemoryCapability

        return MemoryCapability
    if name == 'GuardResult':
        from .guardrails import GuardResult

        return GuardResult
    if name == 'InputGuard':
        from .guardrails import InputGuard

        return InputGuard
    if name == 'OutputGuard':
        from .guardrails import OutputGuard

        return OutputGuard
    if name == 'OutputBlocked':
        from .guardrails import OutputBlocked

        return OutputBlocked
    if name == 'llm_input_guard':
        from .guardrails import llm_input_guard

        return llm_input_guard
    if name == 'llm_output_guard':
        from .guardrails import llm_output_guard

        return llm_output_guard
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
