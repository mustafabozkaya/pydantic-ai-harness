"""Helpers for continuation, forking, and provider-validity checks."""

from __future__ import annotations

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)

from pydantic_ai_harness.step_persistence._store import StepStore


def is_provider_valid(messages: list[ModelMessage]) -> bool:
    """Return True when `messages` can be safely passed to `Agent.run(message_history=...)`.

    A history is provider-valid when every `ToolCallPart` has a matching
    `ToolReturnPart` or `RetryPromptPart` later in the conversation.
    Unmatched calls are how providers reject "I asked for a tool result and
    you skipped it" requests — they must be resolved before resume.
    """
    open_calls: set[str] = set()
    for msg in messages:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    open_calls.add(part.tool_call_id)
        else:
            for part in msg.parts:
                if isinstance(part, (ToolReturnPart, RetryPromptPart)):
                    open_calls.discard(part.tool_call_id)
    return not open_calls


async def continue_run(store: StepStore, *, run_id: str) -> list[ModelMessage]:
    """Load the latest continuable snapshot for `run_id` as a message history.

    Pass the return value to `Agent.run(message_history=...)` to continue
    a delegate's prior investigation instead of starting fresh.

    Raises `LookupError` if no continuable snapshot exists for `run_id` — the
    run may have crashed mid-tool-call, in which case there is event-log data
    but no safe resume point.
    """
    snapshot = await store.latest_snapshot(run_id=run_id)
    if snapshot is None:
        raise LookupError(f'no continuable snapshot for run_id {run_id!r}')
    return list(snapshot.messages)


async def fork_run(store: StepStore, *, run_id: str) -> list[ModelMessage]:
    """Return a copy of the latest snapshot's messages, intended for a new logical run.

    Semantically identical to `continue_run` at the data layer; the
    distinction is in how the caller treats the returned history (new
    `run_id`, new lineage entry, branching off prior context).
    """
    return await continue_run(store, run_id=run_id)
