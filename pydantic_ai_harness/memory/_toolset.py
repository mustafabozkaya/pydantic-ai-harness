"""Memory toolset that provides persistent storage tools to the agent."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

from pydantic_ai import AbstractToolset, RunContext, ToolDefinition, WrapperToolset
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets.abstract import ToolsetTool

from pydantic_ai_harness.memory._abstract import AbstractMemoryBackend
from pydantic_ai_harness.memory._models import (
    MemoryDeleteInput,
    MemoryEntry,
    MemoryListInput,
    MemoryRetrieveInput,
    MemoryStoreInput,
)

# Tool name constants
_TOOL_STORE = 'memory_store'
_TOOL_RETRIEVE = 'memory_retrieve'
_TOOL_LIST = 'memory_list'
_TOOL_DELETE = 'memory_delete'
_TOOL_COMPACT = 'memory_compact'

# Type adapters for argument validation
_store_adapter = MemoryStoreInput
_retrieve_adapter = MemoryRetrieveInput
_list_adapter = MemoryListInput
_delete_adapter = MemoryDeleteInput

# JSON schemas for tool definitions
_STORE_JSON_SCHEMA = MemoryStoreInput.model_json_schema()
_RETRIEVE_JSON_SCHEMA = MemoryRetrieveInput.model_json_schema()
_LIST_JSON_SCHEMA = MemoryListInput.model_json_schema()
_DELETE_JSON_SCHEMA = MemoryDeleteInput.model_json_schema()
_COMPACT_JSON_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'max_age_days': {
            'type': 'integer',
            'description': 'Remove entries older than this many days',
            'default': 90,
            'minimum': 1,
        },
        'min_access': {
            'type': 'integer',
            'description': 'Keep entries accessed at least this many times',
            'default': 2,
            'minimum': 0,
        },
    },
}

_DESCRIPTION_STORE = """\
Store information in persistent memory for future recall.

Use this to remember important facts, user preferences, conversation context,
or any information that should persist across conversations.

Examples:
- User preferences: name, location, interests
- Important decisions or agreements
- Key facts from research or analysis
- Project context and progress updates
"""

_DESCRIPTION_RETRIEVE = """\
Search and retrieve information from persistent memory.

Uses full-text search to find relevant memories. Returns the most
relevant entries ranked by similarity to the query.

Examples:
- "What is the user's name?"
- "Find notes about the project deadline"
- "Search for information about Python best practices"
"""

_DESCRIPTION_LIST = """\
List stored memory entries with optional filtering.

Use pattern matching (glob syntax) and/or tag filtering to find entries.
Examples:
- List all user entries: pattern="user:*"
- List all project notes: tags=["project"]
- List recent entries: (no filter, sorted by most recent)
"""

_DESCRIPTION_DELETE = """\
Delete a memory entry by its key.

Use this to remove outdated or incorrect information from memory.
"""

_DESCRIPTION_COMPACT = """\
Clean up old, rarely-accessed memory entries to save space.

Entries that are older than max_age_days AND have been accessed
fewer than min_access times will be removed. This helps keep
memory lean while preserving important information.
"""


def _utcnow() -> datetime:
    """Current UTC time."""
    return datetime.now(timezone.utc)


@dataclass(kw_only=True)
class MemoryToolset(WrapperToolset[AgentDepsT]):
    """Toolset that provides persistent memory tools.

    Wraps the agent's existing toolset and adds memory management tools:
    store, retrieve, list, delete, and compact.
    """

    backend: AbstractMemoryBackend
    """The memory backend to use for storage."""

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Return a fresh toolset for this agent run."""
        wrapped = await self.wrapped.for_run(ctx)
        return replace(self, wrapped=wrapped)

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        """Return the wrapped tools plus memory tools."""
        wrapped_tools = await self.wrapped.get_tools(ctx)

        # Add memory tools
        wrapped_tools[_TOOL_STORE] = ToolsetTool(
            toolset=self,
            tool_def=ToolDefinition(
                name=_TOOL_STORE,
                description=_DESCRIPTION_STORE,
                parameters_json_schema=_STORE_JSON_SCHEMA,
            ),
            max_retries=2,
        )
        wrapped_tools[_TOOL_RETRIEVE] = ToolsetTool(
            toolset=self,
            tool_def=ToolDefinition(
                name=_TOOL_RETRIEVE,
                description=_DESCRIPTION_RETRIEVE,
                parameters_json_schema=_RETRIEVE_JSON_SCHEMA,
            ),
            max_retries=2,
        )
        wrapped_tools[_TOOL_LIST] = ToolsetTool(
            toolset=self,
            tool_def=ToolDefinition(
                name=_TOOL_LIST,
                description=_DESCRIPTION_LIST,
                parameters_json_schema=_LIST_JSON_SCHEMA,
            ),
            max_retries=1,
        )
        wrapped_tools[_TOOL_DELETE] = ToolsetTool(
            toolset=self,
            tool_def=ToolDefinition(
                name=_TOOL_DELETE,
                description=_DESCRIPTION_DELETE,
                parameters_json_schema=_DELETE_JSON_SCHEMA,
            ),
            max_retries=1,
        )
        wrapped_tools[_TOOL_COMPACT] = ToolsetTool(
            toolset=self,
            tool_def=ToolDefinition(
                name=_TOOL_COMPACT,
                description=_DESCRIPTION_COMPACT,
                parameters_json_schema=_COMPACT_JSON_SCHEMA,
            ),
            max_retries=1,
        )

        return wrapped_tools

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        """Dispatch memory tool calls or pass through to the wrapped toolset."""
        if name == _TOOL_STORE:
            return await self._store(tool_args)
        if name == _TOOL_RETRIEVE:
            return await self._retrieve(tool_args)
        if name == _TOOL_LIST:
            return await self._list(tool_args)
        if name == _TOOL_DELETE:
            return await self._delete(tool_args)
        if name == _TOOL_COMPACT:
            return await self._compact(tool_args)
        return await self.wrapped.call_tool(name, tool_args, ctx, tool)

    async def _store(self, args: dict[str, Any]) -> dict[str, Any]:
        """Store a memory entry."""
        parsed = MemoryStoreInput(**args)
        now = _utcnow()

        # Check if entry already exists to preserve access_count
        existing = await self.backend.retrieve(parsed.key)
        access_count = (existing.access_count if existing else 0) + 1

        entry = MemoryEntry(
            key=parsed.key,
            value=parsed.value,
            metadata=parsed.metadata,
            tags=parsed.tags,
            created_at=existing.created_at if existing else now,
            updated_at=now,
            access_count=access_count,
            last_accessed=now,
        )
        await self.backend.store(entry)

        return {
            'status': 'stored',
            'key': entry.key,
            'created': existing is None,
        }

    async def _retrieve(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        """Search and retrieve memory entries."""
        parsed = MemoryRetrieveInput(**args)
        entries = await self.backend.search(
            query=parsed.query,
            top_k=parsed.top_k,
            tags=parsed.tags,
        )
        return [
            {
                'key': e.key,
                'value': e.value,
                'tags': e.tags,
                'metadata': e.metadata,
                'created_at': e.created_at.isoformat(),
                'score': i + 1,  # Rank position
            }
            for i, e in enumerate(entries)
        ]

    async def _list(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        """List memory entries."""
        parsed = MemoryListInput(**args)
        entries = await self.backend.list_all(
            pattern=parsed.pattern,
            tags=parsed.tags,
            limit=parsed.limit,
        )
        return [
            {
                'key': e.key,
                'value': e.value[:100] + ('...' if len(e.value) > 100 else ''),
                'tags': e.tags,
                'access_count': e.access_count,
                'updated_at': e.updated_at.isoformat(),
            }
            for e in entries
        ]

    async def _delete(self, args: dict[str, Any]) -> dict[str, Any]:
        """Delete a memory entry."""
        parsed = MemoryDeleteInput(**args)
        deleted = await self.backend.delete(parsed.key)
        return {
            'status': 'deleted' if deleted else 'not_found',
            'key': parsed.key,
        }

    async def _compact(self, args: dict[str, Any]) -> dict[str, Any]:
        """Compact old memory entries."""
        max_age_days = args.get('max_age_days', 90)
        min_access = args.get('min_access', 2)
        removed = await self.backend.compact(
            max_age_days=max_age_days,
            min_access=min_access,
        )
        remaining = await self.backend.count()
        return {
            'removed': removed,
            'remaining': remaining,
        }
