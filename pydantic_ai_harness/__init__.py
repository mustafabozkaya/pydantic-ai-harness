"""The batteries for your Pydantic AI agent -- the official capability library."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .code_mode import CodeMode
    from .memory import (
        DictMemoryStore,
        FileMemoryStore,
        Memory,
        MemoryEntry,
        MemoryEntryDict,
        MemoryStore,
        RecencyScorer,
        exponential_decay,
    )

__all__ = [
    'CodeMode',
    'DictMemoryStore',
    'FileMemoryStore',
    'Memory',
    'MemoryEntry',
    'MemoryEntryDict',
    'MemoryStore',
    'RecencyScorer',
    'exponential_decay',
]


_MEMORY_NAMES = {
    'DictMemoryStore',
    'FileMemoryStore',
    'Memory',
    'MemoryEntry',
    'MemoryEntryDict',
    'MemoryStore',
    'RecencyScorer',
    'exponential_decay',
}


def __getattr__(name: str) -> object:
    if name == 'CodeMode':
        from .code_mode import CodeMode

        return CodeMode
    if name in _MEMORY_NAMES:
        from . import memory

        return getattr(memory, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
