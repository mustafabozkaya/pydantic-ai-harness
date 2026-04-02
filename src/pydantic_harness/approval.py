"""Approval capability for human-in-the-loop tool approval workflows.

Intercepts tool execution to require approval before tools run. Supports
configurable approval modes (always ask, ask once then remember, or
auto-approve) and glob-based tool name matching.

Example:
    ```python
    from pydantic_ai import Agent
    from pydantic_harness import Approval

    async def ask_user(tool_name: str, args: dict) -> bool:
        return input(f'Allow {tool_name}({args})? (y/n) ').lower() == 'y'

    agent = Agent(
        'openai:gpt-4.1',
        capabilities=[Approval(
            tool_patterns=['delete_*', 'send_email'],
            callback=ask_user,
            mode='once',
        )],
    )
    ```
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from fnmatch import fnmatch
from typing import Any, Literal, TypeAlias

from pydantic_ai.capabilities.abstract import AbstractCapability, ValidatedToolArgs, WrapToolExecuteHandler
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition

ApprovalMode: TypeAlias = Literal['always', 'once', 'never']
"""How often to request approval for a matched tool.

- ``'always'``: Ask every time the tool is called.
- ``'once'``: Ask the first time, then auto-approve for the rest of the run.
- ``'never'``: Auto-approve all calls (useful for testing or trusted contexts).
"""

ApprovalCallback: TypeAlias = 'Callable[[str, dict[str, Any]], bool | Awaitable[bool]]'
"""Sync or async function ``(tool_name, args) -> bool`` that decides whether a tool call is approved."""

DENIED_MESSAGE = 'Tool execution was denied by user.'
"""Synthetic result returned to the model when approval is denied."""


async def _call_callback(func: ApprovalCallback, tool_name: str, args: dict[str, Any]) -> bool:
    """Call a sync or async approval callback and return its bool result."""
    result = func(tool_name, args)
    if inspect.isawaitable(result):
        return await result
    return result  # type: ignore[return-value]


def _matches_any_pattern(tool_name: str, patterns: tuple[str, ...]) -> bool:
    """Return True if ``tool_name`` matches any of the glob patterns."""
    return any(fnmatch(tool_name, pattern) for pattern in patterns)


@dataclass
class Approval(AbstractCapability[AgentDepsT]):
    """Require human approval before executing matched tools.

    Uses :meth:`~pydantic_ai.capabilities.AbstractCapability.wrap_tool_execute`
    to intercept tool execution. When a tool matches one of the configured
    ``tool_patterns``, the ``callback`` is invoked to request approval.

    If the callback returns ``False`` (denied), the tool is not executed and a
    synthetic denial message is returned to the model instead.

    Per-run state isolation is handled via
    :meth:`~pydantic_ai.capabilities.AbstractCapability.for_run`, which resets
    the set of already-approved tools for ``mode='once'``.
    """

    tool_patterns: list[str] = field(default_factory=list[str])
    """Glob patterns for tool names that require approval.

    Supports ``fnmatch``-style wildcards: ``*`` matches everything,
    ``delete_*`` matches any tool starting with ``delete_``, etc.
    An empty list means no tools require approval.
    """

    callback: ApprovalCallback | None = None
    """The approval callback.  Required when ``mode`` is not ``'never'``."""

    mode: ApprovalMode = 'always'
    """When to ask for approval.  See :data:`ApprovalMode`."""

    # --- Internal per-run state ---

    _patterns: tuple[str, ...] = field(default=(), init=False, repr=False)
    """Frozen copy of ``tool_patterns`` for efficient matching."""

    _approved_tools: set[str] = field(default_factory=set[str], init=False, repr=False)
    """Tool names already approved this run (used by ``mode='once'``)."""

    def __post_init__(self) -> None:  # noqa: D105
        self._patterns = tuple(self.tool_patterns)

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not spec-serializable (takes a callable)."""
        return None

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> Approval[AgentDepsT]:
        """Return a fresh instance with empty approved-tools state."""
        new = replace(self)
        new._patterns = self._patterns
        new._approved_tools = set()
        return new

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        """Intercept tool execution to check approval."""
        if not self._requires_approval(call.tool_name):
            return await handler(args)

        if self.mode == 'never':
            return await handler(args)

        if self.mode == 'once' and call.tool_name in self._approved_tools:
            return await handler(args)

        # Ask for approval
        if self.callback is None:
            return DENIED_MESSAGE

        approved = await _call_callback(self.callback, call.tool_name, args)

        if not approved:
            return DENIED_MESSAGE

        # Approved
        if self.mode == 'once':
            self._approved_tools.add(call.tool_name)

        return await handler(args)

    def _requires_approval(self, tool_name: str) -> bool:
        """Check if the tool matches any configured pattern."""
        if not self._patterns:
            return False
        return _matches_any_pattern(tool_name, self._patterns)


__all__ = ['Approval', 'ApprovalMode', 'ApprovalCallback', 'DENIED_MESSAGE']
