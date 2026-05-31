"""Memory capability that provides persistent storage for Pydantic AI agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic_ai import AbstractToolset
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.memory._toolset import MemoryToolset

if TYPE_CHECKING:
    from pydantic_ai_harness.memory._abstract import AbstractMemoryBackend


@dataclass
class MemoryCapability(AbstractCapability[AgentDepsT]):
    """Capability that provides persistent memory for agents.

    Adds five tools to the agent:

    - ``memory_store``: Save information with a key, value, optional tags and metadata.
    - ``memory_retrieve``: Full-text search across stored memories.
    - ``memory_list``: List entries with optional glob pattern and tag filtering.
    - ``memory_delete``: Remove an entry by key.
    - ``memory_compact``: Clean up old, rarely-accessed entries.

    By default, data is persisted in ``~/.pydantic-ai/memory.db`` (SQLite).
    Pass a custom :class:`AbstractMemoryBackend` for other storage engines.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import MemoryCapability

    agent = Agent('openai:gpt-5', capabilities=[MemoryCapability()])

    # The agent can now remember things across conversations
    result = agent.run_sync('Remember that my favorite color is blue')
    result = agent.run_sync('What is my favorite color?')  # -> 'blue'
    ```

    Custom backend:

    ```python
    from pydantic_ai_harness.memory import MemoryCapability, SQLiteMemoryBackend

    backend = SQLiteMemoryBackend('/path/to/custom.db')
    agent = Agent('openai:gpt-5', capabilities=[MemoryCapability(backend=backend)])
    ```
    """

    backend: AbstractMemoryBackend | None = field(default=None)
    """The memory backend to use. Defaults to SQLite at ``~/.pydantic-ai/memory.db``."""

    def __post_init__(self) -> None:
        if self.backend is None:
            from pydantic_ai_harness.memory._sqlite import SQLiteMemoryBackend

            self.backend = SQLiteMemoryBackend()

    def get_ordering(self) -> CapabilityOrdering:
        """Memory sits innermost — it adds tools but doesn't wrap others."""
        return CapabilityOrdering(position='innermost')

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        """Wrap the agent's toolset with memory tools."""
        assert self.backend is not None
        return MemoryToolset(wrapped=toolset, backend=self.backend)

    def get_instructions(self) -> str | None:
        """Provide instructions about memory usage to the agent."""
        return (
            'You have access to persistent memory via memory tools. '
            'Use them to remember important information between conversations.\n'
            '- Use memory_store to save facts, preferences, and context\n'
            '- Use memory_retrieve to recall previously stored information\n'
            '- Use memory_list to browse stored memories\n'
            '- Use memory_delete to remove outdated information\n'
            '- Use memory_compact periodically to clean up old entries'
        )
