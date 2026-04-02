"""ToolBudget capability — limits how many times tools can be called per run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic_ai.capabilities.abstract import (
    AbstractCapability,
    ValidatedToolArgs,
    WrapToolExecuteHandler,
)
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition


class ToolBudgetExceeded(Exception):
    """Raised when a tool call exceeds the configured budget."""

    def __init__(self, tool_name: str, message: str) -> None:
        """Initialize with the tool name and a descriptive message."""
        self.tool_name = tool_name
        super().__init__(message)


@dataclass
class ToolBudget(AbstractCapability[AgentDepsT]):
    """Capability that limits how many times tools can be called per run.

    Enforces both a total call budget across all tools and optional per-tool
    limits. When a tool's budget is exhausted, it is removed from the model's
    view via `prepare_tools` so the model stops attempting to call it. If the
    model somehow still calls an exhausted tool (e.g. via a cached plan),
    the call is intercepted in `wrap_tool_execute`.

    The `action` parameter controls what happens on budget violation:

    - `'inform'` (default): returns a synthetic message to the model so it
      can adapt its plan.
    - `'error'`: raises `ToolBudgetExceeded`.

    ```python
    from pydantic_ai import Agent
    from pydantic_harness import ToolBudget

    agent = Agent(
        'openai:gpt-4o',
        capabilities=[
            ToolBudget(max_total_calls=10, max_per_tool={'web_search': 3}),
        ],
    )
    ```
    """

    max_total_calls: int | None = None
    """Maximum total tool calls across all tools in a single run. `None` means unlimited."""

    max_per_tool: dict[str, int] = field(default_factory=dict[str, int])
    """Per-tool call limits, keyed by tool name. Tools not listed are unlimited
    (unless constrained by `max_total_calls`)."""

    action: Literal['inform', 'error'] = 'inform'
    """What to do when a budget is exceeded.

    - `'inform'`: return a message to the model describing the exceeded budget.
    - `'error'`: raise `ToolBudgetExceeded`.
    """

    # --- Per-run state ---

    _total_calls: int = field(default=0, init=False, repr=False)
    _calls_by_tool: dict[str, int] = field(default_factory=dict[str, int], init=False, repr=False)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> ToolBudget[AgentDepsT]:
        """Return a fresh instance for per-run state isolation."""
        return ToolBudget(
            max_total_calls=self.max_total_calls,
            max_per_tool=self.max_per_tool,
            action=self.action,
        )

    def _is_tool_exhausted(self, tool_name: str) -> bool:
        """Check whether a specific tool has exhausted its budget."""
        per_tool_limit = self.max_per_tool.get(tool_name)
        if per_tool_limit is not None and self._calls_by_tool.get(tool_name, 0) >= per_tool_limit:
            return True
        return False

    def _is_total_exhausted(self) -> bool:
        """Check whether the total call budget is exhausted."""
        return self.max_total_calls is not None and self._total_calls >= self.max_total_calls

    async def prepare_tools(
        self,
        ctx: RunContext[AgentDepsT],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Remove tools whose budget is exhausted so the model stops trying to call them."""
        if self._is_total_exhausted():
            return [td for td in tool_defs if td.kind != 'function']
        return [td for td in tool_defs if not self._is_tool_exhausted(td.name)]

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        """Count tool calls and enforce budgets before execution."""
        tool_name = tool_def.name

        # Check per-tool limit
        if self._is_tool_exhausted(tool_name):
            return self._handle_exceeded(
                tool_name,
                f"Tool '{tool_name}' has reached its budget of {self.max_per_tool[tool_name]} calls.",
            )

        # Check total limit
        if self._is_total_exhausted():
            return self._handle_exceeded(
                tool_name,
                f'Total tool call budget of {self.max_total_calls} has been reached.',
            )

        # Execute the tool and record the call
        result = await handler(args)
        self._total_calls += 1
        self._calls_by_tool[tool_name] = self._calls_by_tool.get(tool_name, 0) + 1
        return result

    def _handle_exceeded(self, tool_name: str, message: str) -> str:
        """Handle a budget violation according to the configured action."""
        if self.action == 'error':
            raise ToolBudgetExceeded(tool_name, message)
        return f'Tool call budget exceeded: {message}'
