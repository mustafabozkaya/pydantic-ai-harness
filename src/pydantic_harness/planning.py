"""Planning capability for structured task planning and tracking.

Provides tools for creating and managing a step-by-step plan during agent runs,
with dynamic system prompt injection of current plan state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pydantic_ai._instructions import AgentInstructions
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AgentToolset
from pydantic_ai.toolsets.function import FunctionToolset


class TaskStatus(str, Enum):
    """Status of a task in the plan."""

    pending = 'pending'
    in_progress = 'in_progress'
    completed = 'completed'
    skipped = 'skipped'


@dataclass
class Task:
    """A single task in the plan."""

    description: str
    status: TaskStatus = TaskStatus.pending


def format_plan(tasks: list[Task]) -> str:
    """Format the current plan as a readable string.

    Args:
        tasks: The list of tasks to format.

    Returns:
        A human-readable string representation of the plan.
    """
    if not tasks:
        return 'No plan created yet.'

    status_icons = {
        TaskStatus.pending: '[ ]',
        TaskStatus.in_progress: '[~]',
        TaskStatus.completed: '[x]',
        TaskStatus.skipped: '[-]',
    }

    lines: list[str] = []
    for i, task in enumerate(tasks):
        icon = status_icons[task.status]
        lines.append(f'{i}. {icon} {task.description}')

    total = len(tasks)
    completed = sum(1 for t in tasks if t.status == TaskStatus.completed)
    skipped = sum(1 for t in tasks if t.status == TaskStatus.skipped)
    in_progress = sum(1 for t in tasks if t.status == TaskStatus.in_progress)
    pending = sum(1 for t in tasks if t.status == TaskStatus.pending)

    lines.append('')
    lines.append(
        f'Progress: {completed}/{total} completed, {in_progress} in progress, {pending} pending, {skipped} skipped'
    )

    return '\n'.join(lines)


def create_plan_impl(tasks: list[Task], steps: list[str]) -> str:
    """Create a new plan, replacing any existing one.

    Args:
        tasks: The shared task list to modify.
        steps: Step descriptions for the new plan.

    Returns:
        A confirmation message with the formatted plan.
    """
    tasks.clear()
    tasks.extend(Task(description=step) for step in steps)
    return f'Plan created with {len(tasks)} steps.\n\n{format_plan(tasks)}'


def update_task_impl(tasks: list[Task], index: int, status: TaskStatus) -> str:
    """Update the status of a task.

    Args:
        tasks: The shared task list to modify.
        index: Zero-based index of the task to update.
        status: The new status.

    Returns:
        A confirmation message or an error description.
    """
    if not tasks:
        return 'No plan exists. Use create_plan first.'
    if index < 0 or index >= len(tasks):
        return f'Invalid task index {index}. Valid range: 0-{len(tasks) - 1}.'
    tasks[index].status = status
    return f'Task {index} updated to {status.value}.\n\n{format_plan(tasks)}'


def get_plan_impl(tasks: list[Task]) -> str:
    """Get the current plan.

    Args:
        tasks: The task list to format.

    Returns:
        The formatted plan.
    """
    return format_plan(tasks)


@dataclass
class Planning(AbstractCapability[AgentDepsT]):
    """Structured task planning and tracking capability.

    Provides tools for the agent to create a step-by-step plan, update task
    statuses as work progresses, and review the current plan. The current plan
    state is dynamically injected into the system prompt so the model always
    has context on what has been done and what remains.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_harness.planning import Planning

        agent = Agent('openai:gpt-4o', capabilities=[Planning()])
        ```
    """

    plan_tasks: list[Task] = field(default_factory=lambda: list[Task]())
    """Per-run task list. Populated via ``for_run()``."""

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> Planning[AgentDepsT]:
        """Return a fresh instance with isolated per-run state."""
        return Planning[AgentDepsT]()

    def get_instructions(self) -> AgentInstructions[AgentDepsT]:
        """Return a dynamic instruction that injects the current plan state."""
        tasks = self.plan_tasks

        def _instructions(ctx: RunContext[AgentDepsT]) -> str:
            plan_text = format_plan(tasks)
            return (
                'You have a planning capability. Use the planning tools to break complex tasks '
                'into steps and track your progress. Before starting work, create a plan. '
                'Update task statuses as you make progress.\n\n'
                f'Current plan:\n{plan_text}'
            )

        return _instructions

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Return a toolset with plan management tools."""
        tasks = self.plan_tasks
        toolset: FunctionToolset[AgentDepsT] = FunctionToolset()

        @toolset.tool
        def create_plan(ctx: RunContext[AgentDepsT], steps: list[str]) -> str:  # pyright: ignore[reportUnusedFunction]
            """Create a new plan with the given steps. Replaces any existing plan.

            Args:
                ctx: The run context.
                steps: A list of step descriptions for the plan.
            """
            return create_plan_impl(tasks, steps)

        @toolset.tool
        def update_task(ctx: RunContext[AgentDepsT], index: int, status: TaskStatus) -> str:  # pyright: ignore[reportUnusedFunction]
            """Update the status of a task in the plan.

            Args:
                ctx: The run context.
                index: The zero-based index of the task to update.
                status: The new status for the task.
            """
            return update_task_impl(tasks, index, status)

        @toolset.tool
        def get_plan(ctx: RunContext[AgentDepsT]) -> str:  # pyright: ignore[reportUnusedFunction]
            """Get the current plan with all task statuses.

            Args:
                ctx: The run context.
            """
            return get_plan_impl(tasks)

        return toolset

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Return the serialization name for spec support."""
        return 'Planning'

    @property
    def tasks(self) -> list[Task]:
        """Read-only access to the current task list."""
        return list(self.plan_tasks)
