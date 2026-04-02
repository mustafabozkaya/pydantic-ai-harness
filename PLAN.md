# SystemReminders Capability

Closes #83

## Problem

Long-running agents suffer from *instruction fade-out* -- the phenomenon where agents progressively ignore system prompt guidelines after many turns of tool use. A single system prompt at the start of a session is insufficient for maintaining behavioral consistency across extended interactions.

## Solution

A `SystemReminders` capability that injects periodic `SystemPromptPart` entries into model conversations via the `before_model_request` hook. This is a focused first implementation that provides the core mechanism for periodic reminders, which more advanced features (trigger-based reminders, cooldowns, priorities) can be layered on top of.

## Design

### Two kinds of reminders

- **Static** (`Reminder`): a fixed message string injected every N model requests (configurable `interval`).
- **Dynamic** (callable): a sync or async function receiving `RunContext` and returning `str | None`. Called on every model request; returns `None` to skip injection.

### Injection mechanism

Reminder parts are appended as `SystemPromptPart` entries to the last `ModelRequest` in the message history. This places them close to the model's attention window without creating separate messages.

### Per-run isolation

`for_run()` returns a fresh instance with a reset request counter, ensuring concurrent runs on the same agent don't interfere with each other.

### Not spec-serializable

`get_serialization_name()` returns `None` because dynamic reminders take callables which cannot be serialized.

## Files

- `src/pydantic_harness/system_reminders.py` -- `Reminder`, `DynamicReminder`, `AsyncDynamicReminder`, `SystemReminders`
- `src/pydantic_harness/__init__.py` -- public exports
- `tests/test_system_reminders.py` -- 27 tests covering all code paths

## Future work

The issue (#83) describes a richer system with trigger-based reminders (loop detection, token budget warnings, post-compaction re-injection), cooldowns, fire limits, priority ordering, and template substitution. This implementation provides the foundational interval-based and dynamic-callable mechanisms that those features can build on.
