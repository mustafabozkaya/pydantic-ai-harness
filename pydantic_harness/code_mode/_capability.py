"""Code mode capability that routes selected tools through a Monty sandbox."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from pydantic_ai import AbstractToolset, DeferredToolRequests, RunContext
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.capabilities._tool_search import ToolSearch as _ToolSearch
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, ToolSelector

from pydantic_harness.code_mode._toolset import CodeModeToolset

from ._toolset import _RUN_CODE_TOOL_NAME


@dataclass
class CodeMode(AbstractCapability[AgentDepsT]):
    """Capability that exposes selected tools as callables inside a `run_code` sandbox.

    By default (`tools='all'`) every tool the agent has is wrapped behind a single
    `run_code` tool -- the model writes Python that calls them as functions instead
    of issuing tool calls directly.

    Pass a list of tool names or a callable predicate to `tools` to split the
    toolset: matching tools become callables inside the sandbox, and the rest
    stay visible to the model as normal tool calls.

    ```python
    from pydantic_ai import Agent
    from pydantic_harness import CodeMode

    # Sandbox all tools
    agent = Agent('openai:gpt-5', capabilities=[CodeMode()])

    # Sandbox only specific tools
    agent = Agent('openai:gpt-5', capabilities=[CodeMode(tools=['search', 'fetch'])])
    ```
    """

    tools: ToolSelector[AgentDepsT] = field(default='all')
    """Which wrapped tools should be sandboxed inside `run_code`.

    - `'all'` (default): every tool the agent has is sandboxed.
    - `Sequence[str]`: only tools whose names are listed are sandboxed.
    - Callable `(ctx, tool_def) -> bool | Awaitable[bool]`: tools where the
      callable returns `True` are sandboxed; the rest stay as native tool calls.
    """

    max_retries: int = 3
    """Maximum number of retries for the `run_code` tool (syntax errors count as retries)."""

    def get_ordering(self) -> CapabilityOrdering:
        """CodeMode wraps around ToolSearch so that search_tools stays native."""
        return CapabilityOrdering(position='outermost', wraps=[_ToolSearch])

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        """Wrap the agent's assembled toolset, splitting it into native + sandboxed subsets if needed."""
        return CodeModeToolset(wrapped=toolset, tool_selector=self.tools, max_retries=self.max_retries)

    async def after_run(self, ctx: RunContext[AgentDepsT], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
        output = result.output
        if not isinstance(output, DeferredToolRequests):
            return result

        for i, part in enumerate(output.approvals):
            if part.tool_name != _RUN_CODE_TOOL_NAME:
                continue
            metadata = result.output.metadata.get(part.tool_call_id, {})
            tool_name = metadata.get('tool_name')
            kwargs = metadata.get('kwargs')
            if isinstance(tool_name, str) and isinstance(kwargs, dict):
                output.approvals[i] = replace(part, tool_name=tool_name, args=kwargs)

        return result
