"""SubAgent capability: delegate tasks from a parent agent to specialized sub-agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT, RunContext, Tool
from pydantic_ai.toolsets import AgentToolset
from pydantic_ai.toolsets.function import FunctionToolset

__all__ = ('SubAgent',)


def _resolve_description(name: str, agent: Agent[Any]) -> str:
    """Derive a description for a sub-agent from its metadata."""
    if agent.description:
        return agent.description
    if agent.name:
        return agent.name
    return f'Sub-agent: {name}'


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

    agents: dict[str, Agent[Any]]
    """Mapping of agent name to `Agent` instance.

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

        lines = ['You can delegate tasks to the following sub-agents using the `delegate_task` tool:']
        for name in self.agents:
            desc = self._resolved_descriptions[name]
            lines.append(f'- **{name}**: {desc}')
        return '\n'.join(lines)

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Provide the `delegate_task` tool."""
        if not self.agents:
            return None

        agents = self.agents
        pass_deps = self.pass_deps

        async def delegate_task(ctx: RunContext[AgentDepsT], agent_name: str, task: str) -> str:
            """Delegate a task to a named sub-agent and return its text output.

            Args:
                ctx: The run context from the parent agent.
                agent_name: The name of the sub-agent to run. Must be one of the registered agent names.
                task: The prompt describing the task to delegate.
            """
            agent = agents.get(agent_name)
            if agent is None:
                available = ', '.join(sorted(agents))
                raise ModelRetry(f'Unknown agent {agent_name!r}. Available agents: {available}')

            deps = ctx.deps if pass_deps else None
            result = await agent.run(task, deps=deps)
            return str(result.output)

        tool = Tool[AgentDepsT](
            delegate_task,
            name='delegate_task',
            description=self._delegate_task_description(),
        )
        return FunctionToolset[AgentDepsT]([tool])

    def _delegate_task_description(self) -> str:
        """Build a description for the delegate_task tool including available agent names."""
        parts: list[str] = []
        for name in self.agents:
            desc = self._resolved_descriptions[name]
            parts.append(f'{name} ({desc})')
        agent_list = ', '.join(parts)
        return f'Delegate a task to a sub-agent. Available agents: {agent_list}'
