# Planning Capability

## Summary

A `Planning` capability (`AbstractCapability` subclass) that gives agents structured task planning and tracking tools, with dynamic system prompt injection of current plan state.

Addresses #39 (Planning capability) and #65 (Task Tracking capability).

## Design

### Tools provided

- **`create_plan(steps: list[str])`** -- Create or replace the plan with a list of step descriptions. All steps start as `pending`.
- **`update_task(index: int, status: TaskStatus)`** -- Update a task's status by zero-based index. Validates bounds and returns an error message for invalid indices.
- **`get_plan()`** -- Return the current plan with status indicators and progress summary.

### Task statuses

`pending`, `in_progress`, `completed`, `skipped` -- modeled as a `str` enum (`TaskStatus`).

### Per-run isolation

`for_run()` returns a fresh `Planning` instance each run so plan state never leaks between runs.

### Dynamic instructions

`get_instructions()` returns a callable that reads the live task list and formats it into the system prompt on every model request, so the model always sees the current plan state.

### Architecture

Tool logic is extracted into module-level `*_impl` functions for testability:
- `create_plan_impl(tasks, steps)`
- `update_task_impl(tasks, index, status)`
- `get_plan_impl(tasks)`

The `get_toolset()` method registers thin closures that capture the shared task list and delegate to these functions.

## Files

- `src/pydantic_harness/planning.py` -- Capability implementation
- `src/pydantic_harness/__init__.py` -- Public exports
- `tests/test_planning.py` -- 32 tests, 100% coverage

## Quality

- ruff lint: clean
- ruff format: clean
- pyright strict: clean
- pytest: 32 passed, 100% branch coverage
