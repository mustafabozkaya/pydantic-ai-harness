"""Shared async-context state for `StepPersistence` cross-capability coordination."""

from __future__ import annotations

from contextvars import ContextVar

current_run_id: ContextVar[str | None] = ContextVar(
    'pydantic_ai_harness.step_persistence.current_run_id',
    default=None,
)
"""Async-context-local pointer to the active `StepPersistence` `run_id`.

Set by `StepPersistence.wrap_run` for the duration of a run; read by a
nested capability's `for_run` to auto-fill `parent_run_id`, and by
`annotate_tool_effect` to find the in-flight tool's run scope.

Module-level rather than a class attribute so the helpers in `_helpers.py`
and the capability in `_capability.py` can share it without a circular
import.
"""
