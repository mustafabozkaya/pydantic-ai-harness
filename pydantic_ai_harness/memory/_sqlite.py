"""SQLite-based memory backend."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic_ai_harness.memory._abstract import AbstractMemoryBackend
from pydantic_ai_harness.memory._models import MemoryEntry


def _utcnow() -> datetime:
    """Current UTC time."""
    return datetime.now(timezone.utc)


class SQLiteMemoryBackend(AbstractMemoryBackend):
    """SQLite-backed persistent memory storage.

    Data is stored in a single SQLite database file. The schema is designed
    for efficient keyword search and tag-based filtering.

    Args:
        db_path: Path to the SQLite database file. Defaults to
            ``~/.pydantic-ai/memory.db``. Use ``':memory:'`` for in-memory
            testing.
    """

    def __init__(self, db_path: str | Path = '~/.pydantic-ai/memory.db') -> None:
        if db_path == ':memory:':
            self.db_path = ':memory:'
        else:
            self.db_path = str(Path(db_path).expanduser().resolve())
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db: Any | None = None

    async def _get_db(self) -> Any:
        """Lazily initialize the SQLite connection and schema."""
        if self._db is None:
            import aiosqlite

            self._db = await aiosqlite.connect(self.db_path)
            await self._init_tables()
        return self._db

    async def _init_tables(self) -> None:
        """Create the memory table and indexes if they don't exist."""
        db = await self._get_db()
        await db.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                tags TEXT DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT
            )
        """)
        # FTS5 virtual table for full-text search
        await db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                key, value, tags, content=memory, content_rowid=rowid
            )
        """)
        # Trigger to keep FTS in sync on INSERT
        await db.execute("""
            CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memory BEGIN
                INSERT INTO memory_fts(rowid, key, value, tags)
                VALUES (new.rowid, new.key, new.value, new.tags);
            END
        """)
        # Trigger to keep FTS in sync on DELETE
        await db.execute("""
            CREATE TRIGGER IF NOT EXISTS memory_ad AFTER DELETE ON memory BEGIN
                INSERT INTO memory_fts(memory_fts, rowid, key, value, tags)
                VALUES ('delete', old.rowid, old.key, old.value, old.tags);
            END
        """)
        # Trigger to keep FTS in sync on UPDATE
        await db.execute("""
            CREATE TRIGGER IF NOT EXISTS memory_au AFTER UPDATE ON memory BEGIN
                INSERT INTO memory_fts(memory_fts, rowid, key, value, tags)
                VALUES ('delete', old.rowid, old.key, old.value, old.tags);
                INSERT INTO memory_fts(rowid, key, value, tags)
                VALUES (new.rowid, new.key, new.value, new.tags);
            END
        """)
        await db.commit()

    def _row_to_entry(self, row: Any) -> MemoryEntry:
        """Convert a database row to a MemoryEntry."""
        return MemoryEntry(
            key=row[0],
            value=row[1],
            metadata=json.loads(row[2]) if row[2] else {},
            tags=json.loads(row[3]) if row[3] else [],
            created_at=datetime.fromisoformat(row[4]),
            updated_at=datetime.fromisoformat(row[5]),
            access_count=row[6] or 0,
            last_accessed=datetime.fromisoformat(row[7]) if row[7] else None,
        )

    async def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry, overwriting if the key already exists."""
        db = await self._get_db()
        await db.execute(
            """INSERT OR REPLACE INTO memory
               (key, value, metadata, tags, created_at, updated_at, access_count, last_accessed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.key,
                entry.value,
                json.dumps(entry.metadata),
                json.dumps(entry.tags),
                entry.created_at.isoformat(),
                entry.updated_at.isoformat(),
                entry.access_count,
                entry.last_accessed.isoformat() if entry.last_accessed else None,
            ),
        )
        await db.commit()

    async def retrieve(self, key: str) -> MemoryEntry | None:
        """Retrieve a memory entry by key, updating access metadata."""
        db = await self._get_db()
        cursor = await db.execute('SELECT * FROM memory WHERE key = ?', (key,))
        row = await cursor.fetchone()
        if row is None:
            return None

        # Update access count and last accessed time
        now = _utcnow()
        await db.execute(
            'UPDATE memory SET access_count = access_count + 1, last_accessed = ? WHERE key = ?',
            (now.isoformat(), key),
        )
        await db.commit()

        entry = self._row_to_entry(row)
        entry.access_count += 1
        entry.last_accessed = now
        return entry

    async def search(
        self,
        query: str,
        top_k: int = 5,
        tags: list[str] | None = None,
    ) -> list[MemoryEntry]:
        """Search memory entries using FTS5 full-text search."""
        db = await self._get_db()

        # Build FTS5 query
        fts_query = self._build_fts_query(query)

        try:
            cursor = await db.execute(
                """SELECT m.* FROM memory m
                   INNER JOIN memory_fts fts ON m.rowid = fts.rowid
                   WHERE memory_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (fts_query, top_k * 3),  # Fetch extra for tag filtering
            )
            rows = await cursor.fetchall()
        except Exception:
            # Fallback to LIKE search if FTS fails
            like_pattern = f'%{query}%'
            cursor = await db.execute(
                """SELECT * FROM memory
                   WHERE value LIKE ? OR key LIKE ?
                   LIMIT ?""",
                (like_pattern, like_pattern, top_k * 3),
            )
            rows = await cursor.fetchall()

        entries = [self._row_to_entry(row) for row in rows]

        # Filter by tags if specified
        if tags:
            entries = [e for e in entries if any(t in e.tags for t in tags)]

        return entries[:top_k]

    def _build_fts_query(self, query: str) -> str:
        """Build an FTS5 query from a search string.

        Converts simple queries to FTS5 syntax:
        - Multi-word queries use AND by default
        - Exact phrases can be quoted
        """
        terms = query.split()
        if len(terms) == 1:
            return f'"{terms[0]}"'
        # Join with implicit AND (FTS5 default)
        return ' '.join(f'"{term}"' for term in terms)

    async def list_all(
        self,
        pattern: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> list[MemoryEntry]:
        """List all entries with optional pattern and tag filtering."""
        db = await self._get_db()

        if pattern:
            # Convert glob pattern to SQL LIKE
            like_pattern = pattern.replace('*', '%').replace('?', '_')
            cursor = await db.execute(
                'SELECT * FROM memory WHERE key LIKE ? ORDER BY updated_at DESC LIMIT ?',
                (like_pattern, limit * 3),
            )
        else:
            cursor = await db.execute(
                'SELECT * FROM memory ORDER BY updated_at DESC LIMIT ?',
                (limit * 3,),
            )

        rows = await cursor.fetchall()
        entries = [self._row_to_entry(row) for row in rows]

        # Filter by tags
        if tags:
            entries = [e for e in entries if any(t in e.tags for t in tags)]

        return entries[:limit]

    async def delete(self, key: str) -> bool:
        """Delete a memory entry by key."""
        db = await self._get_db()
        cursor = await db.execute('DELETE FROM memory WHERE key = ?', (key,))
        await db.commit()
        return cursor.rowcount > 0

    async def compact(self, max_age_days: int = 90, min_access: int = 2) -> int:
        """Remove old, rarely-accessed entries."""
        db = await self._get_db()
        cutoff = (_utcnow() - timedelta(days=max_age_days)).isoformat()

        # Delete entries that are old AND rarely accessed
        # Keep entries that have been accessed at least min_access times
        cursor = await db.execute(
            """DELETE FROM memory
               WHERE updated_at < ? AND access_count < ?""",
            (cutoff, min_access),
        )
        await db.commit()
        return cursor.rowcount

    async def count(self) -> int:
        """Return the total number of stored entries."""
        db = await self._get_db()
        cursor = await db.execute('SELECT COUNT(*) FROM memory')
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None
