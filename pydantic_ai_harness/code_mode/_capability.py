"""Code mode capability that routes selected tools through a Monty sandbox."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

from pydantic_ai import AbstractToolset
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.capabilities._tool_search import ToolSearch as _ToolSearch
from pydantic_ai.messages import ModelRequest, ModelResponse, NativeToolSearchReturnPart, SystemPromptPart
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition, ToolSelector

from pydantic_ai_harness.code_mode._toolset import CodeModeToolset

if TYPE_CHECKING:
    from pydantic_ai.capabilities.abstract import ValidatedToolArgs
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.models import ModelRequestContext


_DISCOVERY_ANNOUNCEMENT_PREFIX = (
    'New functions are now available inside `run_code`. Their signatures have been '
    'added to the available-functions catalog in the system prompt'
)


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

    dynamic_catalog: bool = False
    """Keep the `run_code` tool definition cache-stable as the sandboxed toolset grows.

    By default the signatures of all sandboxed tools are rendered into `run_code`'s
    description, which lives in the prompt-cache-keyed tool-definitions block. When the
    toolset changes mid-run -- e.g. [`ToolSearch`][pydantic_ai.capabilities.ToolSearch]
    reveals a new tool that then gets folded into `run_code` -- the description changes and
    busts the prefix cache from that point on.

    Set `dynamic_catalog=True` to instead:

    - keep only the static base prose (sandbox restrictions, return-value contract) in
      `run_code.description`, so the tool-definitions block stays byte-stable across
      discoveries;
    - move the "available functions" catalog (TypedDict definitions + signatures) into
      agent instructions as a dynamic
      [`InstructionPart`][pydantic_ai.messages.InstructionPart], which providers with
      static/dynamic instruction splitting (Anthropic, Bedrock) place after the cache
      breakpoint;
    - announce newly-discovered tools via a short
      [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart] enqueued through
      [`RunContext.enqueue`][pydantic_ai.tools.RunContext.enqueue], so the model knows the
      new functions are callable without rewriting the cached description.

    This pays off when paired with [`ToolSearch`][pydantic_ai.capabilities.ToolSearch]: the
    tool-definitions cache survives discoveries at the cost of a larger (but
    cache-friendly) system prompt. With a fixed toolset and no `ToolSearch`, the default
    keeps the system prompt shorter and is the better choice.
    """

    _announced_tools: set[str] = field(default_factory=set[str], init=False, repr=False)

    def get_ordering(self) -> CapabilityOrdering:
        """CodeMode wraps around ToolSearch so that search_tools stays native."""
        return CapabilityOrdering(position='outermost', wraps=[_ToolSearch])

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> CodeMode[AgentDepsT]:
        """Return a fresh instance so concurrent runs don't share `_announced_tools`."""
        if not self.dynamic_catalog:
            return self
        return replace(self)

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        """Wrap the agent's assembled toolset, splitting it into native + sandboxed subsets if needed."""
        return CodeModeToolset(
            wrapped=toolset,
            tool_selector=self.tools,
            max_retries=self.max_retries,
            dynamic_catalog=self.dynamic_catalog,
        )

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        result: Any,
    ) -> Any:
        """Announce newly-discovered tools from a local `search_tools` return.

        Only active with `dynamic_catalog=True`. The native-search path is handled by
        [`after_model_request`][pydantic_ai_harness.CodeMode.after_model_request] instead
        (server-side search emits a `NativeToolSearchReturnPart` rather than a regular tool
        execute result).
        """
        if self.dynamic_catalog and tool_def.tool_kind == 'tool-search':
            self._announce_newly_discovered(ctx, _extract_discovered_names(result))
        return result

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Announce newly-discovered tools from a native (server-side) tool-search return.

        Only active with `dynamic_catalog=True`.
        """
        if self.dynamic_catalog:
            for part in response.parts:
                if isinstance(part, NativeToolSearchReturnPart):
                    self._announce_newly_discovered(ctx, _extract_discovered_names(part.content))
        return response

    def _announce_newly_discovered(self, ctx: RunContext[AgentDepsT], names: Sequence[str]) -> None:
        """Enqueue a system-prompt announcement for any names we haven't already announced."""
        fresh = [n for n in names if n not in self._announced_tools]
        if not fresh:
            return
        self._announced_tools.update(fresh)
        listing = ', '.join(f'`{name}`' for name in fresh)
        # Enqueue as a `ModelRequest(SystemPromptPart)` so it's framed as system-level
        # context. `RunContext.enqueue` doesn't accept a bare `SystemPromptPart` (provider
        # mappings for mid-conversation system content vary -- see pydantic/pydantic-ai#5437),
        # but a `ModelRequest` passthrough is allowed and rendered inline. Once #5437 lands,
        # providers that currently hoist mid-conversation system content will instead inline
        # it as an XML-wrapped user prompt, making this cache-safe across providers.
        ctx.enqueue(ModelRequest(parts=[SystemPromptPart(content=f'{_DISCOVERY_ANNOUNCEMENT_PREFIX}: {listing}.')]))


def _extract_discovered_names(content: Any) -> list[str]:
    """Read newly-discovered tool names from a tool-search return content.

    Accepts both the local `ToolSearchReturnContent` (TypedDict shape) and the same shape
    on a `NativeToolSearchReturnPart`. Returns `[]` for any malformed/unexpected input --
    the announcement is a courtesy nudge, not load-bearing logic.
    """
    if not isinstance(content, dict):
        return []
    typed = cast(dict[str, Any], content)
    raw = typed.get('discovered_tools')
    if not isinstance(raw, list):
        return []
    raw_list = cast(list[Any], raw)
    names: list[str] = []
    for match in raw_list:
        if isinstance(match, dict):
            name = cast(dict[str, Any], match).get('name')
            if isinstance(name, str):
                names.append(name)
    return names
