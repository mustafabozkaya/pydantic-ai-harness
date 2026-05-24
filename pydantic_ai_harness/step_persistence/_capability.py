"""StepPersistence capability: append-only event log + continuable snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic_ai import CallToolsNode
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities.abstract import AgentNode, NodeResult
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition

from pydantic_ai_harness.step_persistence._helpers import is_provider_valid
from pydantic_ai_harness.step_persistence._store import InMemoryStepStore, StepStore
from pydantic_ai_harness.step_persistence._types import (
    ContinuableSnapshot,
    EventKind,
    RunRecord,
    StepEvent,
    ToolEffectRecord,
)


def _empty_metadata() -> dict[str, str]:
    return {}


@dataclass
class StepPersistence(AbstractCapability[AgentDepsT]):
    """Append-only step log + continuable snapshots + tool-effect ledger.

    The capability emits a `StepEvent` at every interesting boundary
    (run/model-request/tool-call start, completion, failure), records a
    `ToolEffectRecord` for every tool call so the orchestrator can decide
    whether replay is safe, and saves a `ContinuableSnapshot` only at
    boundaries where the message history is provider-valid.

    A run that crashes between `before_tool_execute` and `after_tool_execute`
    leaves a visible event trail and a `started` tool-effect record, but no
    new continuable snapshot — the latest snapshot reflects the last
    provider-valid state.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import StepPersistence, InMemoryStepStore

    store = InMemoryStepStore()
    agent = Agent(
        'openai:gpt-5',
        capabilities=[StepPersistence(store=store, agent_name='code-librarian')],
    )
    ```

    To continue a prior run's history, pass `continue_from=<prior_run_id>` or
    use the `continue_run` / `fork_run` helpers and pass the result to
    `Agent.run(message_history=...)`.
    """

    store: StepStore = field(default_factory=InMemoryStepStore)
    """Backend that records events, snapshots, and tool effects."""

    run_id: str | None = None
    """Identifier for this run.

    Set explicitly to give the orchestrator a deterministic name (e.g.
    `'code-librarian-001'`), then later resume with
    `continue_run(store, run_id='code-librarian-001')`. Leave as `None` and
    `for_run` resolves it from `ctx.run_id` (or a fresh UUID4) once per
    `Agent.run`, so reusing the same capability instance across runs does
    not silently merge them.
    """

    parent_run_id: str | None = None
    """Run that spawned this one (e.g. orchestrator → delegate)."""

    agent_name: str | None = None
    """Logical agent name (e.g. `code_librarian`, `reproducer`)."""

    metadata: dict[str, str] = field(default_factory=_empty_metadata)
    """Free-form metadata recorded with the `RunRecord` and on each event."""

    continue_from: str | None = None
    """Run ID whose latest continuable snapshot should preload `ctx.messages`.

    When set, `before_run` looks up the snapshot and prepends its messages
    so the delegate sees its prior investigation. Skipped silently if no
    snapshot exists yet.
    """

    @classmethod
    def from_spec(cls, *args: Any, **kwargs: Any) -> StepPersistence[Any]:
        """Construct from a serialised spec.

        Supports `backend='memory'` (default) or `backend='file'` with `directory`.
        """
        backend = kwargs.pop('backend', 'memory')
        if backend == 'file':
            from pydantic_ai_harness.step_persistence._store import FileStepStore

            directory = kwargs.pop('directory', '.step-persistence')
            return cls(store=FileStepStore(directory), **kwargs)
        return cls(store=InMemoryStepStore(), **kwargs)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        """Materialise `run_id` per run when not supplied explicitly.

        Sharing one capability instance across multiple `Agent.run` calls
        without setting `run_id` would otherwise merge their events under a
        single auto-generated ID — almost never what the caller wants.
        """
        if self.run_id is not None:
            return self
        resolved = ctx.run_id or str(uuid4())
        return replace(self, run_id=resolved)

    def _effective_run_id(self, ctx: RunContext[AgentDepsT]) -> str:
        if self.run_id is not None:
            return self.run_id
        return ctx.run_id or str(uuid4())

    def _make_event(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        kind: EventKind,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        error: str | None = None,
    ) -> StepEvent:
        return StepEvent(
            run_id=self._effective_run_id(ctx),
            kind=kind,
            step_index=ctx.run_step,
            parent_run_id=self.parent_run_id,
            agent_name=self.agent_name,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            error=error,
            metadata=dict(self.metadata),
        )

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Register run lineage, load prior snapshot, and emit `run_started`."""
        await self.store.register_run(
            RunRecord(
                run_id=self._effective_run_id(ctx),
                parent_run_id=self.parent_run_id,
                agent_name=self.agent_name,
                metadata=dict(self.metadata),
            )
        )
        if self.continue_from is not None:
            snapshot = await self.store.latest_snapshot(run_id=self.continue_from)
            if snapshot is not None:
                ctx.messages[:0] = list(snapshot.messages)
        await self.store.append_event(self._make_event(ctx, kind='run_started'))

    async def after_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        """Save a final continuable snapshot and emit `run_completed`."""
        messages = result.all_messages()
        if is_provider_valid(messages):
            await self.store.save_snapshot(
                ContinuableSnapshot(
                    run_id=self._effective_run_id(ctx),
                    step_index=ctx.run_step,
                    messages=list(messages),
                    conversation_id=ctx.conversation_id,
                    parent_run_id=self.parent_run_id,
                    agent_name=self.agent_name,
                )
            )
        await self.store.append_event(self._make_event(ctx, kind='run_completed'))
        return result

    async def on_run_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        error: BaseException,
    ) -> AgentRunResult[Any]:
        """Emit `run_failed` so a killed run leaves a visible event trail."""
        await self.store.append_event(self._make_event(ctx, kind='run_failed', error=repr(error)))
        raise error

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        await self.store.append_event(self._make_event(ctx, kind='model_request_started'))
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        await self.store.append_event(self._make_event(ctx, kind='model_request_completed'))
        return response

    async def on_model_request_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        error: Exception,
    ) -> ModelResponse:
        await self.store.append_event(self._make_event(ctx, kind='model_request_failed', error=repr(error)))
        raise error

    async def before_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        run_id = self._effective_run_id(ctx)
        await self.store.record_tool_effect(
            ToolEffectRecord(
                tool_call_id=call.tool_call_id,
                tool_name=tool_def.name,
                run_id=run_id,
                status='started',
            )
        )
        await self.store.append_event(
            self._make_event(
                ctx,
                kind='tool_call_started',
                tool_call_id=call.tool_call_id,
                tool_name=tool_def.name,
            )
        )
        return args

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        run_id = self._effective_run_id(ctx)
        prior = await self.store.get_tool_effect(tool_call_id=call.tool_call_id)
        started_at = prior.started_at if prior is not None else None
        await self.store.record_tool_effect(
            ToolEffectRecord(
                tool_call_id=call.tool_call_id,
                tool_name=tool_def.name,
                run_id=run_id,
                status='completed',
                started_at=started_at or datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
            )
        )
        await self.store.append_event(
            self._make_event(
                ctx,
                kind='tool_call_completed',
                tool_call_id=call.tool_call_id,
                tool_name=tool_def.name,
            )
        )
        return result

    async def on_tool_execute_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        error: Exception,
    ) -> Any:
        run_id = self._effective_run_id(ctx)
        prior = await self.store.get_tool_effect(tool_call_id=call.tool_call_id)
        started_at = prior.started_at if prior is not None else datetime.now(timezone.utc)
        await self.store.record_tool_effect(
            ToolEffectRecord(
                tool_call_id=call.tool_call_id,
                tool_name=tool_def.name,
                run_id=run_id,
                status='failed',
                started_at=started_at,
                ended_at=datetime.now(timezone.utc),
                effect_summary=repr(error),
            )
        )
        await self.store.append_event(
            self._make_event(
                ctx,
                kind='tool_call_failed',
                tool_call_id=call.tool_call_id,
                tool_name=tool_def.name,
                error=repr(error),
            )
        )
        raise error

    async def after_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
        result: NodeResult[AgentDepsT],
    ) -> NodeResult[AgentDepsT]:
        """Save a mid-run continuable snapshot after `CallToolsNode` succeeds.

        At that boundary every tool call from the preceding `ModelRequestNode`
        has a matching tool return, so the history is provider-valid.
        Snapshots are filtered through `is_provider_valid` defensively in case
        a custom node reshapes history.
        """
        if isinstance(node, CallToolsNode):
            messages = list(ctx.messages)
            if is_provider_valid(messages):
                await self.store.save_snapshot(
                    ContinuableSnapshot(
                        run_id=self._effective_run_id(ctx),
                        step_index=ctx.run_step,
                        messages=messages,
                        conversation_id=ctx.conversation_id,
                        parent_run_id=self.parent_run_id,
                        agent_name=self.agent_name,
                    )
                )
        return result
