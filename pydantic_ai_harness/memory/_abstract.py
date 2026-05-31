"""Abstract base class for memory backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic_ai_harness.memory._models import MemoryEntry


class AbstractMemoryBackend(ABC):
    """Abstract base class for memory storage backends.

    Implement this interface to create a custom memory backend (e.g. Redis, PostgreSQL).
    The default backend is :class:`SQLiteMemoryBackend`.
    """

    @abstractmethod
    async def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry, overwriting if the key already exists."""
        ...

    @abstractmethod
    async def retrieve(self, key: str) -> MemoryEntry | None:
        """Retrieve a memory entry by key. Returns ``None`` if not found."""
        ...

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 5,
        tags: list[str] | None = None,
    ) -> list[MemoryEntry]:
        """Search memory entries by keyword matching.

        For simple backends, this can be a case-insensitive substring match.
        More advanced backends may implement semantic search.
        """
        ...

    @abstractmethod
    async def list_all(
        self,
        pattern: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> list[MemoryEntry]:
        """List all entries, optionally filtered by glob pattern and/or tags."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a memory entry by key. Returns ``True`` if the entry existed."""
        ...

    @abstractmethod
    async def compact(self, max_age_days: int = 90, min_access: int = 2) -> int:
        """Remove old, rarely-accessed entries.

        Returns the number of entries removed.
        """
        ...

    @abstractmethod
    async def count(self) -> int:
        """Return the total number of stored entries."""
        ...
