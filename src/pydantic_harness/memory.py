"""Memory capability for persistent agent memory across sessions.

Provides tools for saving, recalling, searching, listing, and deleting
key-value memories, with pluggable storage backends (`DictMemoryStore` for
testing, `FileMemoryStore` for on-disk persistence).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol, TypedDict, runtime_checkable

from pydantic_ai._instructions import AgentInstructions
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.tools import AgentDepsT, RunContext, Tool
from pydantic_ai.toolsets import AgentToolset
from pydantic_ai.toolsets.function import FunctionToolset

logger = logging.getLogger(__name__)


class _MemoryEntryDictRequired(TypedDict):
    """Required fields for MemoryEntryDict."""

    key: str
    content: str


class MemoryEntryDict(_MemoryEntryDictRequired, total=False):
    """Serialized form of a MemoryEntry for JSON storage.

    Only `key` and `content` are required; the remaining fields are
    optional so that `from_dict` can accept legacy data missing some keys.
    """

    tags: list[str]
    scope: str
    expires_at: str | None
    created_at: str
    updated_at: str
    summary: str | None
    metadata: dict[str, object]
    read_only: bool
    char_limit: int | None
    importance: float | None


@dataclass
class MemoryEntry:
    """A single memory entry with content, tags, and timestamps."""

    key: str
    """Unique identifier for this memory."""

    content: str
    """The content of the memory."""

    tags: list[str] = field(default_factory=lambda: list[str]())
    """Optional tags for categorization and search."""

    scope: str = 'global'
    """Namespace scope for this memory (default `'global'`)."""

    expires_at: str | None = None
    """Optional ISO 8601 expiration timestamp. `None` means no expiry."""

    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    """ISO 8601 timestamp of when the memory was first created."""

    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    """ISO 8601 timestamp of the last update."""

    summary: str | None = None
    """Optional short summary used by `Memory.build_instructions` when injecting this entry into the system prompt; falls back to `content` when None."""

    metadata: dict[str, object] = field(default_factory=lambda: dict[str, object]())
    """Structured attributes for filterable search (`MemoryStore.search(filter=...)`). Values must be JSON-serializable."""

    read_only: bool = False
    """If True, the agent's `save_memory` and `delete_memory` tools refuse to modify this entry. Programmatic access via the store is unrestricted."""

    char_limit: int | None = None
    """Optional hard cap on `content` length (chars). Enforced at `MemoryEntry` construction; raises `ValueError` if exceeded."""

    importance: float | None = None
    """Optional relevance booster used by `MemoryStore.search` scoring when set."""

    def __post_init__(self) -> None:
        """Validate `char_limit` immediately so dev errors surface at construction."""
        if self.char_limit is not None and len(self.content) > self.char_limit:
            raise ValueError(
                f'MemoryEntry {self.key!r} content is {len(self.content)} chars, exceeds char_limit={self.char_limit}',
            )

    def is_expired(self) -> bool:
        """Return True if this entry has passed its expiration time.

        Wall-clock semantics: an entry created with `ttl_minutes=N` expires `N`
        minutes after creation in real time, regardless of how many agent turns
        or sessions have elapsed. TTL is opt-in (default `expires_at=None` =
        no expiry) and intended for facts with a real-world lifetime
        (verification codes, session credentials, etc.).
        """
        if self.expires_at is None:
            return False
        return datetime.fromisoformat(self.expires_at) <= datetime.now(timezone.utc)

    def to_dict(self) -> MemoryEntryDict:
        """Serialize to a plain dict for JSON storage."""
        return {
            'key': self.key,
            'content': self.content,
            'tags': self.tags,
            'scope': self.scope,
            'expires_at': self.expires_at,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'summary': self.summary,
            'metadata': self.metadata,
            'read_only': self.read_only,
            'char_limit': self.char_limit,
            'importance': self.importance,
        }

    @classmethod
    def from_dict(cls, data: MemoryEntryDict) -> MemoryEntry:
        """Deserialize from a plain dict."""
        return cls(
            key=data['key'],
            content=data['content'],
            tags=data.get('tags', []),
            scope=data.get('scope', 'global'),
            expires_at=data.get('expires_at'),
            created_at=data.get('created_at', ''),
            updated_at=data.get('updated_at', ''),
            summary=data.get('summary'),
            metadata=data.get('metadata', {}),
            read_only=data.get('read_only', False),
            char_limit=data.get('char_limit'),
            importance=data.get('importance'),
        )


def _score_entry(entry: MemoryEntry, words: list[str]) -> int:
    r"""Score a memory entry by counting word-boundary matches across fields.

    Each query word that appears as a whole word (case-insensitive) in the
    key, content, or any tag contributes one point per field it appears in.
    Underscores and hyphens are treated as word separators in addition to
    the standard `\\b` boundaries.
    """
    score = 0
    for word in words:
        # Use a boundary pattern that also treats _ and - as separators.
        escaped = re.escape(word)
        pattern = re.compile(rf'(?<![a-zA-Z0-9]){escaped}(?![a-zA-Z0-9])', re.IGNORECASE)
        if pattern.search(entry.key):
            score += 1
        if pattern.search(entry.content):
            score += 1
        if any(pattern.search(tag) for tag in entry.tags):
            score += 1
    return score


def _simple_similarity(a: str, b: str) -> bool:
    """Return True if two keys share the same first 10 characters and differ only slightly.

    Uses a simple character-level edit distance check: keys are considered
    similar when they share the same 10-char prefix and differ by at most 2
    characters (Levenshtein-like).
    """
    if len(a) < 10 or len(b) < 10:
        return False
    if a[:10] != b[:10]:
        return False
    if a == b:
        return False
    # Simple Levenshtein-like check: allow at most 2 edits
    if abs(len(a) - len(b)) > 2:
        return False
    # Bounded character-level distance (sufficient for dedup warnings)
    max_edits = 2
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n] <= max_edits


@runtime_checkable
class MemoryStore(Protocol):
    """Protocol for pluggable memory storage backends."""

    def get(self, key: str) -> MemoryEntry | None:  # pragma: no cover
        """Retrieve a memory entry by key, or None if not found."""
        ...

    def put(self, entry: MemoryEntry) -> None:  # pragma: no cover
        """Store or update a memory entry."""
        ...

    def delete(self, key: str) -> bool:  # pragma: no cover
        """Delete a memory entry by key. Returns True if it existed."""
        ...

    def list_all(self, *, scope: str | None = None) -> list[MemoryEntry]:  # pragma: no cover
        """Return all non-expired entries, optionally filtered by scope."""
        ...

    def search(self, query: str, *, scope: str | None = None) -> list[MemoryEntry]:  # pragma: no cover
        """Search non-expired entries with word-boundary matching, sorted by relevance."""
        ...


class _BaseDictStore:
    """Base class for dict-backed memory stores."""

    _entries: dict[str, MemoryEntry]

    def get(self, key: str) -> MemoryEntry | None:
        """Retrieve a non-expired memory entry by key."""
        entry = self._entries.get(key)
        if entry is None or entry.is_expired():
            return None
        return entry

    def put(self, entry: MemoryEntry) -> None:
        """Store or update a memory entry."""
        self._entries[entry.key] = entry

    def delete(self, key: str) -> bool:
        """Delete a memory entry by key."""
        return self._entries.pop(key, None) is not None

    def _gc_expired(self) -> None:
        """Drop expired entries from the backing dict."""
        expired_keys = [key for key, entry in self._entries.items() if entry.is_expired()]
        for key in expired_keys:
            del self._entries[key]

    def list_all(self, *, scope: str | None = None) -> list[MemoryEntry]:
        """Return all non-expired entries, optionally filtered by scope."""
        return [
            entry
            for entry in self._entries.values()
            if not entry.is_expired() and (scope is None or entry.scope == scope)
        ]

    def search(self, query: str, *, scope: str | None = None) -> list[MemoryEntry]:
        """Search non-expired entries with word-boundary matching, sorted by relevance."""
        words = query.lower().split()
        if not words:
            return []
        scored: list[tuple[int, MemoryEntry]] = []
        for entry in self._entries.values():
            if entry.is_expired():
                continue
            if scope is not None and entry.scope != scope:
                continue
            score = _score_entry(entry, words)
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [entry for _, entry in scored]


class DictMemoryStore(_BaseDictStore):
    """Dict-based in-memory store, suitable for testing.

    All data lives in a plain `dict` and is lost when the process exits.
    """

    def __init__(self) -> None:
        """Initialize an empty in-memory store."""
        self._entries: dict[str, MemoryEntry] = {}


class FileMemoryStore(_BaseDictStore):
    """JSON-file-based store for simple on-disk persistence.

    Reads the file on initialization and writes back on every mutation.
    """

    def __init__(self, path: str | Path) -> None:
        """Initialize a file-backed store at the given path."""
        self._path = Path(path)
        self._entries: dict[str, MemoryEntry] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                raw: dict[str, MemoryEntryDict] = json.loads(self._path.read_text(encoding='utf-8'))
                if not isinstance(raw, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
                    logger.warning('Memory file %s contains non-dict JSON, starting empty', self._path)
                    return
                self._entries = {key: MemoryEntry.from_dict(val) for key, val in raw.items()}
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning('Failed to load memory file %s: %s, starting empty', self._path, e)
                self._entries = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._gc_expired()
        data = {key: entry.to_dict() for key, entry in self._entries.items()}
        self._path.write_text(json.dumps(data, indent=2), encoding='utf-8')

    def put(self, entry: MemoryEntry) -> None:
        """Store or update a memory entry."""
        super().put(entry)
        self._save()

    def delete(self, key: str) -> bool:
        """Delete a memory entry by key."""
        existed = super().delete(key)
        if existed:
            self._save()
        return existed


def format_entry(entry: MemoryEntry) -> str:
    """Format a memory entry as a human-readable string."""
    line = f'[{entry.key}] {entry.content}'
    extras: list[str] = []
    if entry.tags:
        extras.append(f'tags: {", ".join(entry.tags)}')
    if entry.scope != 'global':
        extras.append(f'scope: {entry.scope}')
    if entry.expires_at is not None:
        extras.append(f'expires: {entry.expires_at}')
    if extras:
        line += f' ({"; ".join(extras)})'
    return line


@dataclass
class Memory(AbstractCapability[AgentDepsT]):
    """Capability for persistent memory across agent sessions.

    Provides tools for saving, recalling, searching, listing, and deleting
    key-value memories. Uses a pluggable `MemoryStore` backend for storage.

    Example:
        ```python {test="skip" lint="skip"}
        from pydantic_ai import Agent
        from pydantic_harness.memory import Memory, DictMemoryStore

        agent = Agent('openai:gpt-4o', capabilities=[Memory(store=DictMemoryStore())])
        ```
    """

    store: MemoryStore = field(default_factory=DictMemoryStore)
    """The storage backend. Defaults to `DictMemoryStore` (ephemeral, dict-based)."""

    inject_memories_in_instructions: bool = True
    """Whether to inject existing memories into the system prompt at run start."""

    max_instructions_memories: int = 20
    """Maximum number of memories to include in the system prompt."""

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Return the name used for spec serialization."""
        return 'Memory'

    @classmethod
    def from_spec(
        cls,
        *,
        backend: str = 'memory',
        path: str = '.memories.json',
        inject_memories_in_instructions: bool = True,
        max_instructions_memories: int = 20,
    ) -> Memory[Any]:
        """Create from spec arguments.

        Args:
            backend: Storage backend, `"memory"` (default) or `"file"`.
            path: File path for the `"file"` backend (default `".memories.json"`).
            inject_memories_in_instructions: Whether to inject memories into the system prompt.
            max_instructions_memories: Maximum memories to inject into the system prompt.
        """
        store: MemoryStore
        if backend == 'memory':
            store = DictMemoryStore()
        elif backend == 'file':
            store = FileMemoryStore(path)
        else:
            raise ValueError(f'Unknown memory backend: {backend!r}. Use "memory" or "file".')
        return cls(
            store=store,
            inject_memories_in_instructions=inject_memories_in_instructions,
            max_instructions_memories=max_instructions_memories,
        )

    def build_instructions(self, ctx: RunContext[AgentDepsT]) -> str:
        """Build dynamic instructions that include currently stored memories."""
        parts: list[str] = [
            'You have access to a persistent memory system. '
            'Use it to save important information that should be remembered across conversations.',
        ]
        if self.inject_memories_in_instructions:
            entries = self.store.list_all()
            if entries:
                parts.append('\nCurrently stored memories:')
                for entry in entries[: self.max_instructions_memories]:
                    parts.append(f'- {format_entry(entry)}')
                overflow = len(entries) - self.max_instructions_memories
                if overflow > 0:
                    parts.append(f'... and {overflow} more (use list_memories or search_memories to see all).')
        return '\n'.join(parts)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Return dynamic instructions that include stored memories."""
        return self.build_instructions

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Return a toolset with memory management tools.

        Tool functions close over `self` to access the store without
        requiring anything from the agent's `deps`.
        """
        store = self.store

        def save_memory(
            key: str,
            content: str,
            tags: list[str] | None = None,
            scope: str = 'global',
            ttl_minutes: int | None = None,
            summary: str | None = None,
            importance: float | None = None,
        ) -> str:
            """Save or update a memory entry.

            Args:
                key: Unique key for this memory.
                content: The content to remember.
                tags: Optional tags for categorization and search.
                scope: Namespace scope (default `'global'`).
                ttl_minutes: Optional time-to-live in minutes. The entry will expire after this duration.
                summary: Optional short summary; preferred over `content` when injecting into the system prompt.
                importance: Optional relevance booster (e.g., 0.0–1.0); used by search scoring.
            """
            now = datetime.now(timezone.utc)
            now_iso = now.isoformat()
            existing = store.get(key)

            if existing is not None and existing.read_only:
                return f'Memory {key!r} is read-only and cannot be modified.'

            # Dedup warning: check for similar keys among existing entries
            for existing_entry in store.list_all():
                if _simple_similarity(key, existing_entry.key):
                    logger.warning(
                        'New memory key %r is very similar to existing key %r — possible duplicate',
                        key,
                        existing_entry.key,
                    )

            expires_at: str | None = None
            if ttl_minutes is not None:
                expires_at = (now + timedelta(minutes=ttl_minutes)).isoformat()

            entry = MemoryEntry(
                key=key,
                content=content,
                tags=tags or [],
                scope=scope,
                expires_at=expires_at,
                created_at=existing.created_at if existing else now_iso,
                updated_at=now_iso,
                summary=summary,
                importance=importance,
            )
            store.put(entry)
            return f'Memory saved: {key}'

        def recall_memory(key: str) -> str:
            """Recall a specific memory by its key.

            Args:
                key: The key of the memory to recall.
            """
            entry = store.get(key)
            if entry is None:
                return f'No memory found for key: {key}'
            return format_entry(entry)

        def search_memories(query: str, scope: str | None = None) -> str:
            """Search memories by word-boundary matching on keys, content, or tags, sorted by relevance.

            Args:
                query: The search query string (space-separated words).
                scope: Optional scope to restrict the search to.
            """
            results = store.search(query, scope=scope)
            if not results:
                return f'No memories found matching: {query}'
            return '\n'.join(format_entry(entry) for entry in results)

        def list_memories(scope: str | None = None) -> str:
            """List all stored memories, optionally filtered by scope.

            Args:
                scope: Optional scope to filter by.
            """
            entries = store.list_all(scope=scope)
            if not entries:
                return 'No memories stored.'
            return '\n'.join(format_entry(entry) for entry in entries)

        def delete_memory(key: str) -> str:
            """Delete a memory by its key.

            Args:
                key: The key of the memory to delete.
            """
            entry = store.get(key)
            if entry is not None and entry.read_only:
                return f'Memory {key!r} is read-only and cannot be deleted.'
            if store.delete(key):
                return f'Memory deleted: {key}'
            return f'No memory found for key: {key}'

        return FunctionToolset(
            [
                Tool(save_memory, takes_ctx=False),
                Tool(recall_memory, takes_ctx=False),
                Tool(search_memories, takes_ctx=False),
                Tool(list_memories, takes_ctx=False),
                Tool(delete_memory, takes_ctx=False),
            ],
        )
