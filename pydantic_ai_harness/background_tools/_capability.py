"""Background tools capability that spawns selected tools as fire-and-forget tasks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, PendingMessageDrainCapability
from pydantic_ai.messages import SystemPromptPart, ToolCallPart
from pydantic_ai.tools import (
    AgentDepsT,
    RunContext,
    ToolDefinition,
    ToolSelector,
    matches_tool_selector,
)

if TYPE_CHECKING:
    from pydantic_ai import _agent_graph
    from pydantic_ai.capabilities.abstract import WrapToolExecuteHandler
    from pydantic_ai.result import FinalResult
    from pydantic_graph import End


_DEFAULT_SELECTOR: dict[str, Any] = {'background': True}

_INSTRUCTIONS = """\
Some tools run in the background: when you call them you'll get an immediate \
acknowledgment, and the real result will be delivered automatically as a follow-up \
message when the task completes. Continue working on other things in the meantime; \
do not block waiting for the result.\
"""


@dataclass
class BackgroundTools(AbstractCapability[AgentDepsT]):
    """Run selected tools as fire-and-forget asyncio tasks.

    When the model calls a tool that matches the selector, the capability spawns the
    tool's handler in an `asyncio.Task` and immediately returns an acknowledgment
    string to the agent. When the task completes, its result (or error) is enqueued
    via [`RunContext.enqueue`][pydantic_ai.tools.RunContext.enqueue] as a `'follow_up'`
    message — Pydantic AI's pending message queue redirects the agent to a fresh
    `ModelRequest` instead of ending, so the model receives the result and can act on it.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import BackgroundTools

    # Default: any tool with `metadata={'background': True}` runs in the background.
    agent = Agent('openai:gpt-5', capabilities=[BackgroundTools()])

    @agent.tool_plain(metadata={'background': True})
    async def slow_research(query: str) -> str:
        return await do_expensive_research(query)
    ```

    Combine with [`SetToolMetadata`][pydantic_ai.capabilities.SetToolMetadata] to mark
    every tool from a specific MCP server, or with `FunctionToolset.with_metadata(...)`
    to mark a whole toolset. Or pass a name list / predicate via `tools=...` to ignore
    metadata entirely.
    """

    tools: ToolSelector[AgentDepsT] = field(default_factory=lambda: dict(_DEFAULT_SELECTOR))
    """Which tools should run in the background.

    - `dict[str, Any]` (default `{'background': True}`): tools whose metadata deeply
      includes the given key-value pairs.
    - `'all'`: every tool in the agent's toolset (rarely what you want).
    - `Sequence[str]`: tools with matching names.
    - Callable `(ctx, tool_def) -> bool | Awaitable[bool]`: custom predicate.
    """

    _tasks: dict[str, asyncio.Task[None]] = field(
        default_factory=dict[str, 'asyncio.Task[None]'], init=False, repr=False
    )
    _completion_event: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)

    def get_ordering(self) -> CapabilityOrdering:
        # `after_node_run` runs in reverse order (outermost runs last). We need to
        # wait for at least one background task BEFORE the core
        # `PendingMessageDrainCapability` checks the queue for follow-ups, so
        # drain must be outermost relative to us.
        return CapabilityOrdering(wrapped_by=[PendingMessageDrainCapability])

    def get_instructions(self) -> str:
        return _INSTRUCTIONS

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> BackgroundTools[AgentDepsT]:
        # Fresh per-run state so concurrent runs don't share tasks.
        return BackgroundTools(tools=self.tools)

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        if not await matches_tool_selector(self.tools, ctx, tool_def):
            return await handler(args)

        task_id = call.tool_call_id
        tool_name = call.tool_name

        async def _run() -> None:
            try:
                result = await handler(args)
                ctx.enqueue(
                    SystemPromptPart(f"Background tool '{tool_name}' (task {task_id}) completed.\nResult: {result}"),
                    priority='follow_up',
                )
            except asyncio.CancelledError:
                # Run cleanup cancelled us; don't enqueue a spurious failure follow-up.
                raise
            except Exception as e:
                ctx.enqueue(
                    SystemPromptPart(f"Background tool '{tool_name}' (task {task_id}) failed: {e}"),
                    priority='follow_up',
                )
            finally:
                self._tasks.pop(task_id, None)
                self._completion_event.set()

        self._tasks[task_id] = asyncio.create_task(_run())
        return (
            f"Tool '{tool_name}' is running in background (task {task_id}). "
            f'You will receive the result automatically when it completes. '
            f'Continue with other work in the meantime.'
        )

    async def after_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: _agent_graph.AgentNode[AgentDepsT, Any],
        result: _agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]],
    ) -> _agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]:
        from pydantic_graph import End

        if not isinstance(result, End) or not self._tasks:
            return result

        # Hold End until at least one task completes so the drain capability
        # (which runs after us in reverse order) has a follow-up to deliver.
        self._completion_event.clear()
        await self._completion_event.wait()
        return result

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: Any,
    ) -> Any:
        try:
            return await handler()
        finally:
            for task in self._tasks.values():
                task.cancel()
            if self._tasks:
                await asyncio.gather(*self._tasks.values(), return_exceptions=True)
