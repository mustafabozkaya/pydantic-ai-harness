"""Code mode capability that routes selected tools through a Monty sandbox."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai import AbstractToolset
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.capabilities._tool_search import ToolSearch as _ToolSearch
from pydantic_ai.tools import AgentDepsT, ToolSelector

from pydantic_ai_harness.code_mode._toolset import CodeModeToolset, MontyMount, MontyOS


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
    from pydantic_ai_harness import CodeMode

    # Sandbox all tools
    agent = Agent('openai:gpt-5', capabilities=[CodeMode()])

    # Sandbox only specific tools
    agent = Agent('openai:gpt-5', capabilities=[CodeMode(tools=['search', 'fetch'])])
    ```

    Pass `mount` for host filesystem access and/or `os` for environment/clock
    (plus filesystem) access -- without them, `pathlib`/`os` I/O and
    `datetime.now()` are unavailable inside `run_code`:

    ```python
    from pydantic_monty import MountDir

    agent = Agent('openai:gpt-5', capabilities=[CodeMode(mount=MountDir('/work', '/tmp/agent-work'))])
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

    os: MontyOS | None = None
    """Host-backed OS access for sandboxed code.

    Pass a `pydantic_monty.AbstractOS` instance or a raw Monty OS callback
    `(function_name, args, kwargs) -> result`. When set, `pathlib.Path`, `os`,
    `datetime.datetime.now()`, and `datetime.date.today()` calls inside `run_code`
    are routed to it instead of being unavailable. Fixed at construction, so build
    `CodeMode` per request to scope access per request.
    """

    mount: MontyMount | None = None
    """Host directory mount(s) exposed inside the sandbox as `pydantic_monty.MountDir`."""

    def get_ordering(self) -> CapabilityOrdering:
        """CodeMode wraps around ToolSearch so that search_tools stays native."""
        return CapabilityOrdering(position='outermost', wraps=[_ToolSearch])

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        """Wrap the agent's assembled toolset, splitting it into native + sandboxed subsets if needed."""
        return CodeModeToolset(
            wrapped=toolset,
            tool_selector=self.tools,
            max_retries=self.max_retries,
            os=self.os,
            mount=self.mount,
        )
