"""Postgres backend for the Memory capability — reference implementation.

Shows how to implement `MemoryStore` against Postgres using `psycopg`. This is
a starting point, not a production-ready backend: adapt to your deployment
(connection pooling, schema migrations, async, full-text search via tsvector).

For semantic retrieval, swap the `search` implementation for one that runs
`SELECT ... ORDER BY embedding <=> %s` against a pgvector column.

Schema:
    CREATE TABLE memories (
        key       TEXT PRIMARY KEY,
        namespace TEXT[] NOT NULL DEFAULT ARRAY['global'],
        data      JSONB NOT NULL
    );

Usage:
    pip install 'psycopg[binary]'

    import psycopg
    from pydantic_ai import Agent
    from pydantic_ai_harness.memory import Memory
    from examples.memory.postgres_store import PostgresMemoryStore

    conn = psycopg.connect('postgresql://localhost/myapp')
    agent = Agent('openai:gpt-4o', capabilities=[Memory(store=PostgresMemoryStore(conn))])
"""

from __future__ import annotations

import json
import re
from typing import Any

import psycopg

from pydantic_ai_harness.memory import MemoryEntry, RecencyScorer

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    key       TEXT PRIMARY KEY,
    namespace TEXT[] NOT NULL DEFAULT ARRAY['global'],
    data      JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS memories_namespace_idx ON memories USING GIN (namespace);
"""


class PostgresMemoryStore:
    """`MemoryStore` backed by Postgres via psycopg.

    Implements the full `MemoryStore` Protocol: `get`, `put`, `delete`,
    `list_all`, `search`, `list_namespaces`. Filtering happens in SQL
    (namespace prefix via array slicing, metadata equality via JSONB ops);
    keyword scoring runs in Python after the DB pre-filter.
    """

    def __init__(self, conn: psycopg.Connection[Any]) -> None:
        self._conn = conn
        with self._conn.cursor() as cur:
            cur.execute(SCHEMA)
        self._conn.commit()

    def get(self, key: str) -> MemoryEntry | None:
        with self._conn.cursor() as cur:
            cur.execute('SELECT data FROM memories WHERE key = %s', (key,))
            row = cur.fetchone()
        if row is None:
            return None
        entry = MemoryEntry.from_dict(row[0])
        return None if entry.is_expired() else entry

    def put(self, entry: MemoryEntry) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                'INSERT INTO memories (key, namespace, data) VALUES (%s, %s, %s) '
                'ON CONFLICT (key) DO UPDATE SET namespace = EXCLUDED.namespace, data = EXCLUDED.data',
                (entry.key, list(entry.namespace), json.dumps(entry.to_dict())),
            )
        self._conn.commit()

    def delete(self, key: str) -> bool:
        with self._conn.cursor() as cur:
            cur.execute('DELETE FROM memories WHERE key = %s', (key,))
            deleted = cur.rowcount > 0
        self._conn.commit()
        return deleted

    def list_all(
        self,
        *,
        namespace: tuple[str, ...] | None = None,
        filter: dict[str, object] | None = None,
    ) -> list[MemoryEntry]:
        sql = 'SELECT data FROM memories WHERE TRUE'
        params: list[Any] = []
        if namespace is not None:
            # Prefix match: entry.namespace[1:N] = supplied namespace tuple
            sql += ' AND namespace[1:%s] = %s'
            params.extend([len(namespace), list(namespace)])
        if filter is not None:
            for k, v in filter.items():
                sql += " AND (data -> 'metadata' ->> %s) = %s"
                params.extend([k, str(v)])
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        entries = [MemoryEntry.from_dict(r[0]) for r in rows]
        return [e for e in entries if not e.is_expired()]

    def search(
        self,
        query: str,
        *,
        namespace: tuple[str, ...] | None = None,
        filter: dict[str, object] | None = None,
        recency_scorer: RecencyScorer | None = None,
    ) -> list[MemoryEntry]:
        # SQL-side filter, Python-side keyword scoring matching DictMemoryStore semantics.
        # Production: replace with full-text or pgvector ranking.
        words = query.lower().split()
        if not words:
            return []
        candidates = self.list_all(namespace=namespace, filter=filter)
        scored: list[tuple[float, MemoryEntry]] = []
        for entry in candidates:
            base = 0
            for word in words:
                pattern = re.compile(rf'(?<![a-zA-Z0-9]){re.escape(word)}(?![a-zA-Z0-9])', re.IGNORECASE)
                if pattern.search(entry.key):
                    base += 1
                if pattern.search(entry.content):
                    base += 1
                if any(pattern.search(t) for t in entry.tags):
                    base += 1
            if base == 0:
                continue
            score: float = float(base)
            if entry.importance is not None:
                score += entry.importance
            if recency_scorer is not None:
                score += recency_scorer(entry)
            scored.append((score, entry))
        scored.sort(key=lambda p: p[0], reverse=True)
        return [e for _, e in scored]

    def list_namespaces(
        self,
        *,
        prefix: tuple[str, ...] | None = None,
        suffix: tuple[str, ...] | None = None,
        max_depth: int | None = None,
    ) -> list[tuple[str, ...]]:
        with self._conn.cursor() as cur:
            cur.execute('SELECT DISTINCT namespace FROM memories')
            rows = cur.fetchall()
        seen: set[tuple[str, ...]] = set()
        for row in rows:
            ns: tuple[str, ...] = tuple(row[0])
            if max_depth is not None:
                ns = ns[:max_depth]
            if prefix is not None and (len(ns) < len(prefix) or ns[: len(prefix)] != prefix):
                continue
            if suffix is not None and (len(ns) < len(suffix) or ns[-len(suffix) :] != suffix):
                continue
            seen.add(ns)
        return sorted(seen)
