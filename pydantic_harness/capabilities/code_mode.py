"""Code mode capability that routes selected tools through a Monty sandbox."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from pydantic_ai import AbstractToolset, CombinedToolset, FilteredToolset, RunContext, ToolDefinition
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

from pydantic_harness.toolsets import CodeExecutionToolset


@dataclass
class CodeMode(AbstractCapability[AgentDepsT]):
    """Capability that exposes selected tools as callables inside a `run_code` sandbox.

    By default (`tools='all'`) every tool the agent has is wrapped behind a single
    `run_code` tool — the model writes Python that calls them as functions instead
    of issuing tool calls directly.

    Pass a callable to `tools` to split the toolset: tools the predicate accepts
    become callables inside the sandbox, and the rest stay visible to the model
    as normal tool calls. The callable shape matches
    [`FilteredToolset.filter_func`][pydantic_ai.toolsets.FilteredToolset], so the
    same predicate can be reused with either.
    """

    # Inline `Callable[[RunContext[AgentDepsT], ToolDefinition], bool]` to match the
    # spelling pydantic-ai uses on `FilteredToolset.filter_func` and friends — there
    # is no exported `ToolFilter` alias upstream, so we don't introduce one here.
    tools: Literal['all'] | Callable[[RunContext[AgentDepsT], ToolDefinition], bool] = field(default='all')
    """Which wrapped tools should be sandboxed inside `run_code`.

    - `'all'` (default): every tool the agent has is sandboxed.
    - Callable `(ctx, tool_def) -> bool`: tools where the callable returns `True`
      are sandboxed; tools where it returns `False` stay visible to the model
      as native tool calls.
    """

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        """Wrap the agent's assembled toolset, splitting it into native + sandboxed subsets if needed."""
        if self.tools == 'all':
            return CodeExecutionToolset(wrapped=toolset)

        tool_filter = self.tools
        sandboxed = FilteredToolset(wrapped=toolset, filter_func=tool_filter)
        native = FilteredToolset(
            wrapped=toolset,
            filter_func=lambda ctx, td: not tool_filter(ctx, td),
        )
        return CombinedToolset([native, CodeExecutionToolset(wrapped=sandboxed)])
