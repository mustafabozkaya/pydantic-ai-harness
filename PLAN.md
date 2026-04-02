# StuckLoopDetection Capability

Closes #71.

## Summary

A `StuckLoopDetection` capability that monitors agent tool-call patterns via
capability hooks and detects when the agent is stuck in a repetitive loop.

## Detection scenarios

1. **Repeated calls** -- the same tool is called with the same arguments N times
   consecutively (tracked in `after_model_request`).
2. **Alternating calls** -- two distinct tool+args pairs alternate A-B-A-B for N
   full cycles (tracked in `after_model_request`).
3. **No-op calls** -- the same tool returns the same result N times consecutively,
   even if the arguments differ (tracked in `after_tool_execute`).

N is configurable via `max_repeated_calls` (default 3).

## Recovery actions

| `action` | Behavior |
|----------|----------|
| `'warn'` (default) | Raises `ModelRetry` with a descriptive message so the model receives a retry prompt asking it to change approach. |
| `'error'` | Raises `StuckLoopError` to abort the run. |

## Per-run state

Uses `for_run()` to return a fresh instance with empty history lists, ensuring
concurrent runs don't interfere.

## API

```python
from pydantic_ai import Agent
from pydantic_harness.stuck_loop_detection import StuckLoopDetection

agent = Agent(
    'openai:gpt-4o',
    capabilities=[
        StuckLoopDetection(
            max_repeated_calls=3,
            action='warn',
            warning_message='You appear to be stuck. Try something else.',
        ),
    ],
)
```

## Files

- `src/pydantic_harness/stuck_loop_detection.py` -- capability implementation
- `src/pydantic_harness/__init__.py` -- re-exports `StuckLoopDetection` and `StuckLoopError`
- `tests/test_stuck_loop_detection.py` -- 32 tests, 100% coverage
