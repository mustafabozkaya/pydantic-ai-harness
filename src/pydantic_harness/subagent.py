"""SubAgent capability: delegate tasks from a parent agent to specialized sub-agents."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.tools import AgentDepsT, RunContext, Tool
from pydantic_ai.toolsets import AgentToolset
from pydantic_ai.toolsets.function import FunctionToolset

__all__ = ('SubAgent',)


def _resolve_description(name: str, agent: Agent[Any, Any]) -> str:
    """Derive a description for a sub-agent from its metadata."""
    if agent.description:
        return agent.description
    if agent.name:
        return agent.name
    return f'Sub-agent: {name}'


def _shareable_history(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Return a copy of the message history safe to pass to a sub-agent.

    The parent's ``ctx.messages`` may end with a ``ModelResponse`` containing
    the ``ToolCallPart`` currently being executed, which the sub-agent cannot
    process (it would conflict with its own user prompt).  This helper strips
    such trailing responses to yield a clean conversation history.
    """
    history = list(messages)
    while history and isinstance(history[-1], ModelResponse):
        if any(isinstance(p, ToolCallPart) for p in history[-1].parts):
            history.pop()
        else:
            break
    return history


def _format_output(output: Any) -> str:
    """Format a sub-agent's output as a string for the parent agent.

    Preserves structured data by JSON-serializing Pydantic models, dicts, and
    lists, and using `repr()` for other non-string types.
    """
    if isinstance(output, str):
        return output
    if isinstance(output, BaseModel):
        return output.model_dump_json()
    if isinstance(output, (dict, list)):
        return json.dumps(output)
    return repr(output)


@dataclass
class SubAgent(AbstractCapability[AgentDepsT]):
    """Capability that lets a parent agent delegate tasks to named sub-agents.

    Each sub-agent is an independent `Agent` instance. The parent agent receives
    a `delegate_task` tool that runs a named sub-agent with a given prompt and
    returns its text output as the tool result.

    Example:
    ```python
    from pydantic_ai import Agent
    from pydantic_harness.subagent import SubAgent

    researcher = Agent('openai:gpt-4o', description='Researches topics thoroughly.')
    coder = Agent('openai:gpt-4o', description='Writes and reviews code.')

    orchestrator = Agent(
        'openai:gpt-4o',
        capabilities=[
            SubAgent(agents={'researcher': researcher, 'coder': coder}),
        ],
    )
    ```
    """

    agents: dict[str, Agent[Any, Any]]
    """Mapping of agent name to `Agent` instance.

    Sub-agents may have any output type; structured outputs are automatically
    serialized to strings for the parent agent.

    Names are used by the parent agent in the `delegate_task` tool to select
    which sub-agent to run.
    """

    descriptions: dict[str, str] = field(default_factory=dict[str, str])
    """Optional explicit descriptions for each sub-agent.

    These are included in the system prompt and in the `delegate_task` tool
    description so the parent agent knows what each sub-agent does.

    When a name is not present in this dict, the description is derived from
    `agent.description`, `agent.name`, or a default.
    """

    pass_deps: bool = True
    """Whether to forward the parent agent's `deps` to sub-agents.

    When True (the default), sub-agents receive the same dependency object
    as the parent. Set to False if sub-agents use incompatible dependency types.
    """

    share_history: bool = False
    """Whether to pass the parent agent's message history to sub-agents.

    When True, the parent's conversation history is forwarded as
    ``message_history`` to each sub-agent run, giving it access to the
    full conversation context. When False (the default), sub-agents start
    with a fresh conversation.
    """

    _resolved_descriptions: dict[str, str] = field(default_factory=dict[str, str], init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve descriptions for all registered sub-agents."""
        for name, agent in self.agents.items():
            if name in self.descriptions:
                self._resolved_descriptions[name] = self.descriptions[name]
            else:
                self._resolved_descriptions[name] = _resolve_description(name, agent)

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not spec-serializable (takes Agent instances)."""
        return None

    def get_instructions(self) -> str | None:
        """Inject descriptions of available sub-agents into the system prompt."""
        if not self.agents:
            return None

        lines = [
            'You can delegate tasks to the following sub-agents using the '
            '`delegate_task` tool (one at a time) or the `delegate_tasks` tool (multiple in parallel):'
        ]
        for name in self.agents:
            desc = self._resolved_descriptions[name]
            lines.append(f'- **{name}**: {desc}')
        return '\n'.join(lines)

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Provide the `delegate_task` and `delegate_tasks` tools."""
        if not self.agents:
            return None

        agents = self.agents
        pass_deps = self.pass_deps
        share_history = self.share_history

        async def _run_sub_agent(ctx: RunContext[AgentDepsT], agent_name: str, task: str) -> str:
            """Run a single sub-agent, returning its formatted output."""
            agent = agents.get(agent_name)
            if agent is None:
                available = ', '.join(sorted(agents))
                raise ModelRetry(f'Unknown agent {agent_name!r}. Available agents: {available}')

            deps = ctx.deps if pass_deps else None
            message_history = _shareable_history(ctx.messages) if share_history else None
            result = await agent.run(task, deps=deps, message_history=message_history)
            return _format_output(result.output)

        async def delegate_task(ctx: RunContext[AgentDepsT], agent_name: str, task: str) -> str:
            """Delegate a task to a named sub-agent and return its output.

            Args:
                ctx: The run context from the parent agent.
                agent_name: The name of the sub-agent to run. Must be one of the registered agent names.
                task: The prompt describing the task to delegate.
            """
            return await _run_sub_agent(ctx, agent_name, task)

        async def delegate_tasks(
            ctx: RunContext[AgentDepsT],
            tasks: list[dict[str, str]],
        ) -> list[str]:
            """Delegate multiple tasks to sub-agents in parallel and return their outputs.

            Args:
                ctx: The run context from the parent agent.
                tasks: A list of task objects, each with ``agent`` (sub-agent name) and ``task`` (prompt).
            """
            coros = [_run_sub_agent(ctx, t['agent'], t['task']) for t in tasks]
            return list(await asyncio.gather(*coros))

        agent_desc = self._delegate_task_description()
        tools: list[Tool[AgentDepsT]] = [
            Tool[AgentDepsT](
                delegate_task,
                name='delegate_task',
                description=agent_desc,
            ),
            Tool[AgentDepsT](
                delegate_tasks,
                name='delegate_tasks',
                description=f'Delegate multiple tasks in parallel. Each item needs "agent" and "task" keys. {agent_desc}',
            ),
        ]
        return FunctionToolset[AgentDepsT](tools)

    def _delegate_task_description(self) -> str:
        """Build a description for the delegate_task tool including available agent names."""
        parts: list[str] = []
        for name in self.agents:
            desc = self._resolved_descriptions[name]
            parts.append(f'{name} ({desc})')
        agent_list = ', '.join(parts)
        return f'Delegate a task to a sub-agent. Available agents: {agent_list}'
