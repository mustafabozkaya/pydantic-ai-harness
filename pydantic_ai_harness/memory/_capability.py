"""Memory capability for persistent agent memory across sessions.

Provides tools for saving, recalling, searching, listing, and deleting
key-value memories, with pluggable storage backends (`DictMemoryStore` for
testing, `FileMemoryStore` for on-disk persistence).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol, TypeAlias, TypedDict, runtime_checkable

from pydantic_ai._instructions import AgentInstructions
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
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
    namespace: list[str]
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

    namespace: tuple[str, ...] = ('global',)
    """Hierarchical namespace for this memory.

    A tuple of strings forming a path-like namespace (e.g., `('users', 'alice')`,
    `('agents', 'planner', 'facts')`). Filters in `list_all`/`search` use prefix
    matching: `namespace=('users',)` matches `('users', 'alice')` and
    `('users', 'bob')`. Default `('global',)` is a single-segment namespace.
    """

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
            'namespace': list(self.namespace),
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
            namespace=tuple(data.get('namespace', ('global',))),
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


RecencyScorer: TypeAlias = Callable[['MemoryEntry'], float]
"""Callable that maps a `MemoryEntry` to a recency score (typically in `[0, 1]`).

Added to the keyword-match score in `MemoryStore.search` to bias results toward fresher entries.
Use the built-in `exponential_decay` factory or supply any callable.
"""


def exponential_decay(*, half_life_days: float = 30.0, weight: float = 1.0) -> RecencyScorer:
    """Build a recency scorer with exponential decay over `entry.updated_at`.

    Args:
        half_life_days: Age (in days) at which the decay value is halved. Default `30.0`.
        weight: Multiplier applied to the decay value. Default `1.0`.

    Returns:
        A `RecencyScorer` callable. Entries with an unparsable `updated_at`
        return `0.0`; future-dated entries return the full `weight`.
    """

    def scorer(entry: MemoryEntry) -> float:
        try:
            updated = datetime.fromisoformat(entry.updated_at)
        except ValueError:
            return 0.0
        age_seconds = (datetime.now(timezone.utc) - updated).total_seconds()
        if age_seconds < 0:
            return weight
        age_days = age_seconds / 86400
        return weight * (2 ** (-age_days / half_life_days))

    return scorer


def _saves_in_history(messages: list[ModelMessage]) -> dict[str, str]:
    """Scan tool history for `save_memory` calls; return `{key: last saved content}`.

    Used by `Memory.build_instructions` to suppress re-injecting entries the LLM
    just saved — when the saved content still matches the current store entry,
    the LLM has already seen the value via the tool call/result in history,
    so re-injecting it wastes tokens. If a key was saved multiple times, the
    most recent save wins.
    """
    last: dict[str, str] = {}
    for msg in messages:
        if not isinstance(msg, ModelResponse):
            continue
        for part in msg.parts:
            if not isinstance(part, ToolCallPart):
                continue
            if part.tool_name != 'save_memory':
                continue
            args = part.args_as_dict()
            key = args.get('key')
            content = args.get('content')
            if isinstance(key, str) and isinstance(content, str):
                last[key] = content
    return last


def _matches_filter(entry: MemoryEntry, filter_: dict[str, object]) -> bool:
    """Return True if all filter keys match `entry.metadata` values exactly."""
    for key, value in filter_.items():
        if entry.metadata.get(key) != value:
            return False
    return True


def _namespace_matches(entry_ns: tuple[str, ...], filter_prefix: tuple[str, ...]) -> bool:
    """Return True if `entry_ns` starts with `filter_prefix`.

    Empty `filter_prefix` matches any namespace (use `None` in callers to mean
    "no filter" — this helper assumes the caller has decided to filter).
    """
    if len(entry_ns) < len(filter_prefix):
        return False
    return entry_ns[: len(filter_prefix)] == filter_prefix


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

    def list_all(  # pragma: no cover
        self,
        *,
        namespace: tuple[str, ...] | None = None,
        filter: dict[str, object] | None = None,
    ) -> list[MemoryEntry]:
        """Return all non-expired entries, optionally filtered by namespace prefix and metadata equality."""
        ...

    def search(  # pragma: no cover
        self,
        query: str,
        *,
        namespace: tuple[str, ...] | None = None,
        filter: dict[str, object] | None = None,
        recency_scorer: RecencyScorer | None = None,
    ) -> list[MemoryEntry]:
        """Search non-expired entries, sorted by relevance.

        Score = keyword-match count + entry.importance (if set) + recency_scorer(entry) (if provided).
        Entries with zero keyword match are excluded regardless of recency or importance.
        """
        ...

    def list_namespaces(  # pragma: no cover
        self,
        *,
        prefix: tuple[str, ...] | None = None,
        suffix: tuple[str, ...] | None = None,
        max_depth: int | None = None,
    ) -> list[tuple[str, ...]]:
        """List unique namespaces among non-expired entries, optionally filtered.

        Args:
            prefix: Only include namespaces starting with this prefix.
            suffix: Only include namespaces ending with this suffix.
            max_depth: Truncate each namespace to at most this many segments before deduplication.
        """
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

    def list_all(
        self,
        *,
        namespace: tuple[str, ...] | None = None,
        filter: dict[str, object] | None = None,
    ) -> list[MemoryEntry]:
        """Return all non-expired entries, optionally filtered by namespace prefix and metadata equality."""
        return [
            entry
            for entry in self._entries.values()
            if not entry.is_expired()
            and (namespace is None or _namespace_matches(entry.namespace, namespace))
            and (filter is None or _matches_filter(entry, filter))
        ]

    def search(
        self,
        query: str,
        *,
        namespace: tuple[str, ...] | None = None,
        filter: dict[str, object] | None = None,
        recency_scorer: RecencyScorer | None = None,
    ) -> list[MemoryEntry]:
        """Search non-expired entries with word-boundary matching, sorted by relevance.

        Score = keyword-match count + `entry.importance` (if set) + `recency_scorer(entry)` (if provided).
        """
        words = query.lower().split()
        if not words:
            return []
        scored: list[tuple[float, MemoryEntry]] = []
        for entry in self._entries.values():
            if entry.is_expired():
                continue
            if namespace is not None and not _namespace_matches(entry.namespace, namespace):
                continue
            if filter is not None and not _matches_filter(entry, filter):
                continue
            base = _score_entry(entry, words)
            if base == 0:
                continue
            score: float = float(base)
            if entry.importance is not None:
                score += entry.importance
            if recency_scorer is not None:
                score += recency_scorer(entry)
            scored.append((score, entry))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [entry for _, entry in scored]

    def list_namespaces(
        self,
        *,
        prefix: tuple[str, ...] | None = None,
        suffix: tuple[str, ...] | None = None,
        max_depth: int | None = None,
    ) -> list[tuple[str, ...]]:
        """List unique namespaces among non-expired entries."""
        seen: set[tuple[str, ...]] = set()
        for entry in self._entries.values():
            if entry.is_expired():
                continue
            ns = entry.namespace
            if max_depth is not None:
                ns = ns[:max_depth]
            if prefix is not None and not _namespace_matches(ns, prefix):
                continue
            if suffix is not None and (len(ns) < len(suffix) or ns[-len(suffix) :] != suffix):
                continue
            seen.add(ns)
        return sorted(seen)


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


def format_entry(entry: MemoryEntry, *, prefer_summary: bool = False) -> str:
    """Format a memory entry as a human-readable string.

    Args:
        entry: The entry to format.
        prefer_summary: If True and `entry.summary` is set, render the summary
            in place of the full content. Used by `Memory.build_instructions`
            to keep system-prompt injection short. Defaults to False (full content).
    """
    body = entry.summary if (prefer_summary and entry.summary is not None) else entry.content
    line = f'[{entry.key}] {body}'
    extras: list[str] = []
    if entry.tags:
        extras.append(f'tags: {", ".join(entry.tags)}')
    if entry.namespace != ('global',):
        extras.append(f'namespace: {"/".join(entry.namespace)}')
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
        from pydantic_ai_harness.memory import Memory, DictMemoryStore

        agent = Agent('openai:gpt-4o', capabilities=[Memory(store=DictMemoryStore())])
        ```

    Multi-agent shared store:
        ```python {test="skip" lint="skip"}
        from pydantic_ai import Agent
        from pydantic_ai_harness.memory import FileMemoryStore, Memory

        shared = FileMemoryStore('/var/lib/myapp/memory.json')
        planner = Agent('openai:gpt-4o', capabilities=[Memory(store=shared, byte_budget=2000)])
        worker = Agent('openai:gpt-4o-mini', capabilities=[Memory(store=shared, byte_budget=500)])
        ```
        Both agents see the same entries; use distinct `namespace` tuples on
        saves to keep their workspaces separate (e.g., `('agents', 'planner')`
        vs `('agents', 'worker')`).
    """

    store: MemoryStore = field(default_factory=DictMemoryStore)
    """The storage backend. Defaults to `DictMemoryStore` (ephemeral, dict-based)."""

    inject_memories_in_instructions: bool = True
    """Whether to inject existing memories into the system prompt at run start."""

    max_instructions_memories: int = 20
    """Maximum number of non-pinned memories to include in the system prompt.

    `read_only=True` entries always inject regardless of this cap.
    """

    byte_budget: int | None = None
    """Optional UTF-8 byte cap on the injected memories block.

    When set, non-pinned entries are skipped once adding the next would exceed
    the budget. `read_only=True` entries always inject regardless of this cap.
    Default `None` = no byte cap (only the count cap applies).
    """

    recency_scorer: RecencyScorer | None = field(
        default_factory=lambda: exponential_decay(half_life_days=30.0, weight=0.5),
    )
    """Recency scorer threaded into `search_memories` to bias results toward fresher entries.

    Defaults to `exponential_decay(half_life_days=30, weight=0.5)`. Set to `None` to disable.
    Pass any `Callable[[MemoryEntry], float]` for custom decay shapes.
    """

    tool_descriptions: dict[str, str] = field(default_factory=lambda: dict[str, str]())
    """Per-tool description overrides. Keys are tool names (`save_memory`, `recall_memory`,
    `search_memories`, `list_memories`, `delete_memory`); values replace the docstring used
    by the LLM. Useful for nudging the agent (e.g., "Save aggressively, even small facts")."""

    dedup_recent_saves: bool = True
    """When True, suppress injection of entries that match a `save_memory` call
    in the current run's tool history (the LLM has already seen the value).

    The check is content-aware: if the store entry's `content` differs from the
    most recent saved content (e.g., another process updated the entry), the
    entry is injected so the LLM sees the current value. `read_only=True`
    entries are never suppressed.
    """

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
        """Build dynamic instructions that include currently stored memories.

        Selection rules:
        - `read_only=True` entries always inject (bypass count cap, byte budget, and dedup).
        - Non-pinned entries respect `max_instructions_memories` and `byte_budget`.
        - When `entry.summary` is set, it's preferred over `entry.content` to save tokens.
        - When `dedup_recent_saves` is True, entries whose current content matches
          the most recent `save_memory` call in this run's tool history are suppressed
          (the LLM has already seen the value via the tool call).
        - Pinned entries are listed first.
        """
        parts: list[str] = [
            'You have access to a persistent memory system. '
            'Use it to save important information that should be remembered across conversations.',
        ]
        if not self.inject_memories_in_instructions:
            return '\n'.join(parts)

        entries = self.store.list_all()
        if not entries:
            return '\n'.join(parts)

        parts.append('\nCurrently stored memories:')

        recent_saves: dict[str, str] = _saves_in_history(ctx.messages) if self.dedup_recent_saves else {}

        # Pinned first, then the rest in store order
        ordered = sorted(entries, key=lambda e: not e.read_only)

        formatted: list[str] = []
        used_bytes = 0
        consumed_non_pinned = 0
        for entry in ordered:
            # read_only entries bypass dedup, count cap, and byte budget
            if not entry.read_only:
                saved_content = recent_saves.get(entry.key)
                if saved_content is not None and saved_content == entry.content:
                    continue
            line = f'- {format_entry(entry, prefer_summary=True)}'
            line_bytes = len(line.encode('utf-8'))
            if entry.read_only:
                formatted.append(line)
                used_bytes += line_bytes
                continue
            if consumed_non_pinned >= self.max_instructions_memories:
                break
            if self.byte_budget is not None and used_bytes + line_bytes > self.byte_budget:
                break
            formatted.append(line)
            used_bytes += line_bytes
            consumed_non_pinned += 1

        parts.extend(formatted)

        overflow = len(entries) - len(formatted)
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
        recency_scorer = self.recency_scorer

        def save_memory(
            key: str,
            content: str,
            tags: list[str] | None = None,
            namespace: list[str] | None = None,
            ttl_minutes: int | None = None,
            summary: str | None = None,
            importance: float | None = None,
        ) -> str:
            """Save or update a memory entry.

            Args:
                key: Unique key for this memory.
                content: The content to remember.
                tags: Optional tags for categorization and search.
                namespace: Optional hierarchical namespace as a list of segments
                    (e.g., `['users', 'alice']`). Defaults to `['global']`.
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

            ns: tuple[str, ...] = tuple(namespace) if namespace else ('global',)
            entry = MemoryEntry(
                key=key,
                content=content,
                tags=tags or [],
                namespace=ns,
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

        def search_memories(query: str, namespace: list[str] | None = None) -> str:
            """Search memories by word-boundary matching on keys, content, or tags, sorted by relevance.

            Args:
                query: The search query string (space-separated words).
                namespace: Optional namespace prefix to restrict the search to (e.g., `['users']`).
            """
            ns: tuple[str, ...] | None = tuple(namespace) if namespace else None
            results = store.search(query, namespace=ns, recency_scorer=recency_scorer)
            if not results:
                return f'No memories found matching: {query}'
            return '\n'.join(format_entry(entry) for entry in results)

        def list_memories(namespace: list[str] | None = None) -> str:
            """List all stored memories, optionally filtered by namespace prefix.

            Args:
                namespace: Optional namespace prefix to filter by (e.g., `['users']`).
            """
            ns: tuple[str, ...] | None = tuple(namespace) if namespace else None
            entries = store.list_all(namespace=ns)
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

        descs = self.tool_descriptions
        return FunctionToolset(
            [
                Tool(save_memory, takes_ctx=False, description=descs.get('save_memory')),
                Tool(recall_memory, takes_ctx=False, description=descs.get('recall_memory')),
                Tool(search_memories, takes_ctx=False, description=descs.get('search_memories')),
                Tool(list_memories, takes_ctx=False, description=descs.get('list_memories')),
                Tool(delete_memory, takes_ctx=False, description=descs.get('delete_memory')),
            ],
        )
