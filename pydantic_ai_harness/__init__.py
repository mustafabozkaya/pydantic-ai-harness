"""The batteries for your Pydantic AI agent -- the official capability library."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .code_mode import CodeMode
    from .step_persistence import (
        ContinuableSnapshot,
        EventKind,
        FileStepStore,
        InMemoryStepStore,
        RunRecord,
        StepEvent,
        StepPersistence,
        StepStore,
        ToolEffectRecord,
        ToolEffectStatus,
        annotate_tool_effect,
        continue_run,
        fork_run,
        is_provider_valid,
    )

__all__ = [
    'CodeMode',
    'ContinuableSnapshot',
    'EventKind',
    'FileStepStore',
    'InMemoryStepStore',
    'RunRecord',
    'StepEvent',
    'StepPersistence',
    'StepStore',
    'ToolEffectRecord',
    'ToolEffectStatus',
    'annotate_tool_effect',
    'continue_run',
    'fork_run',
    'is_provider_valid',
]

_STEP_PERSISTENCE_NAMES = {
    'ContinuableSnapshot',
    'EventKind',
    'FileStepStore',
    'InMemoryStepStore',
    'RunRecord',
    'StepEvent',
    'StepPersistence',
    'StepStore',
    'ToolEffectRecord',
    'ToolEffectStatus',
    'annotate_tool_effect',
    'continue_run',
    'fork_run',
    'is_provider_valid',
}


def __getattr__(name: str) -> object:
    if name == 'CodeMode':
        from .code_mode import CodeMode

        return CodeMode
    if name in _STEP_PERSISTENCE_NAMES:
        from . import step_persistence

        return getattr(step_persistence, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
