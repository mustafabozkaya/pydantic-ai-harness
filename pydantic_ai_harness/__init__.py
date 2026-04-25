"""The batteries for your Pydantic AI agent -- the official capability library."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .background_tools import BackgroundTools
    from .code_mode import CodeMode

__all__ = ['BackgroundTools', 'CodeMode']


def __getattr__(name: str) -> object:
    if name == 'BackgroundTools':
        from .background_tools import BackgroundTools

        return BackgroundTools
    if name == 'CodeMode':
        from .code_mode import CodeMode

        return CodeMode
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
