# StepPersistence

`StepPersistence` records what an agent did at each boundary, separate from
whether the run can be safely resumed. It is the persistence substrate for
orchestrators that delegate to sub-agents (e.g. an AICA orchestrator spawning
a `code_librarian` to investigate one symbol, then continuing that delegate's
investigation with a follow-up question).

It is not a full graph-state checkpoint. Capability-state restore, workspace
snapshots, and graph-node resume are out of scope and tracked separately
(see `pydantic-ai-harness` issues #149 and #196).

## What it gives you

1. **Append-only step events** — every interesting boundary (run start/end,
   model request, tool call, failure) appends a `StepEvent`. A run killed
   mid-tool-call still leaves a usable event trail.
2. **Continuable snapshots** — a `ContinuableSnapshot` is saved only at
   boundaries where the message history is **provider-valid**: every
   `ToolCallPart` has a matching `ToolReturnPart` or `RetryPromptPart`. Pass
   one back to `Agent.run(message_history=...)` to continue or fork.
3. **Tool-effect ledger** — every tool call's lifecycle (`started`,
   `completed`, `failed`) is recorded against its `tool_call_id`. After a
   crash, a tool with a `started` record and no terminal update should be
   treated as `unknown_after_crash`: the side effect may or may not have
   happened.
4. **Lineage metadata** — `parent_run_id` ties delegate runs back to the
   orchestrator; `agent_name` distinguishes multiple runs of the same logical
   delegate type.

## Quick start

```python
from pydantic_ai import Agent
from pydantic_ai_harness import StepPersistence, InMemoryStepStore

store = InMemoryStepStore()
orchestrator = Agent(
    'openai:gpt-5',
    capabilities=[
        StepPersistence(store=store, agent_name='orchestrator', run_id='orch-1'),
    ],
)

# Delegate, tied to the orchestrator via parent_run_id
librarian = Agent(
    'openai:gpt-5',
    capabilities=[
        StepPersistence(
            store=store,
            agent_name='code_librarian',
            run_id='libr-1',
            parent_run_id='orch-1',
        ),
    ],
)
```

## Continuing a delegate's investigation

```python
from pydantic_ai_harness import continue_run

# First delegate run records investigation
await librarian.run('Find ThinkingPartDelta and confirm the callable allowance')

# Follow-up uses the prior snapshot instead of restarting
history = await continue_run(store, run_id='libr-1')
await librarian.run(
    'Read _apply_provider_details_delta and check the path',
    message_history=history,
)
```

`fork_run(store, run_id=...)` returns the same shape but is intended when the
caller wants a branched, new logical run rather than a continuation.

## Backends

- `InMemoryStepStore` — process-local; great for tests.
- `FileStepStore(directory)` — JSON/JSONL layout under
  `<directory>/<run_id>/`:
    - `run.json` — `RunRecord` (lineage)
    - `events.jsonl` — append-only `StepEvent`s
    - `tool_effects.jsonl` — append-only `ToolEffectRecord`s
    - `snapshots/<step_index>.json` — `ContinuableSnapshot`s
- Both implement the same async `StepStore` protocol, so capability hooks
  never block the event loop on the file backend (I/O is dispatched via
  `anyio.to_thread`).

`FileStepStore` validates `run_id` against `[A-Za-z0-9_.-]{1,200}` and rejects
`..` to prevent path traversal — callers passing user-controlled IDs should
still sanitise first.

## Inspecting a crashed run

```python
events = await store.list_events(run_id='libr-1')
unresolved = await store.list_unresolved_tool_effects(run_id='libr-1')
latest = await store.latest_snapshot(run_id='libr-1')
```

A run that died after `tool_call_started` but before `tool_call_completed`
fired will show:

- a `tool_call_started` event with no matching `tool_call_completed` /
  `tool_call_failed`,
- a `ToolEffectRecord` whose `status == 'started'`,
- a `latest_snapshot()` that is **older** than the failed tool call (the
  history at that point is not provider-valid).

That is the contract: visible event trail, no false-continuable snapshot.

## What this capability does **not** do

- It does not restore capability per-run state, graph-node state, retry
  counters, or in-flight streaming responses.
- It does not deduplicate replayed side effects automatically. Tools that
  write artifacts, labels, PRs, or external state should populate
  `idempotency_key` and `effect_summary` on their own `ToolEffectRecord`
  entries (or wrap the tool to do so) so the orchestrator can decide whether
  replay is safe.
- It does not clean up old snapshots/events. Retention is the caller's
  responsibility.
