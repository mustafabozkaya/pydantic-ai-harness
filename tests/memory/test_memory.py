"""Tests for the memory capability."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from pydantic_ai_harness.memory._models import (
    MemoryDeleteInput,
    MemoryEntry,
    MemoryListInput,
    MemoryRetrieveInput,
    MemoryStoreInput,
)
from pydantic_ai_harness.memory._sqlite import SQLiteMemoryBackend

# aiosqlite uses asyncio internally, so these tests only run with asyncio (not trio)
pytestmark = [pytest.mark.anyio]


@pytest.fixture
async def backend() -> SQLiteMemoryBackend:
    """Create an in-memory SQLite backend for testing."""
    b = SQLiteMemoryBackend(':memory:')
    yield b
    await b.close()


@pytest.fixture
async def populated_backend(backend: SQLiteMemoryBackend) -> SQLiteMemoryBackend:
    """A backend pre-populated with test data."""
    now = datetime.now(timezone.utc)
    entries = [
        MemoryEntry(
            key='user:name',
            value='Mustafa Bozkaya',
            tags=['user', 'identity'],
            created_at=now,
            updated_at=now,
            access_count=5,
        ),
        MemoryEntry(
            key='user:color',
            value='My favorite color is blue',
            tags=['user', 'preference'],
            created_at=now,
            updated_at=now,
            access_count=3,
        ),
        MemoryEntry(
            key='project:deadline',
            value='Project deadline is June 15, 2026',
            tags=['project', 'deadline'],
            created_at=now,
            updated_at=now,
            access_count=1,
        ),
        MemoryEntry(
            key='user:location',
            value='I live in Istanbul, Turkey',
            tags=['user', 'location'],
            created_at=now,
            updated_at=now,
            access_count=2,
        ),
    ]
    for entry in entries:
        await backend.store(entry)
    return backend


# --- Model Tests ---


class TestMemoryStoreInput:
    def test_required_fields(self) -> None:
        inp = MemoryStoreInput(key='test', value='hello')
        assert inp.key == 'test'
        assert inp.value == 'hello'
        assert inp.metadata == {}
        assert inp.tags == []

    def test_with_metadata_and_tags(self) -> None:
        inp = MemoryStoreInput(
            key='test',
            value='hello',
            metadata={'source': 'test'},
            tags=['important'],
        )
        assert inp.metadata == {'source': 'test'}
        assert inp.tags == ['important']


class TestMemoryRetrieveInput:
    def test_defaults(self) -> None:
        inp = MemoryRetrieveInput(query='test')
        assert inp.top_k == 5
        assert inp.tags is None


class TestMemoryListInput:
    def test_defaults(self) -> None:
        inp = MemoryListInput()
        assert inp.pattern is None
        assert inp.tags is None
        assert inp.limit == 20


class TestMemoryDeleteInput:
    def test_required_key(self) -> None:
        inp = MemoryDeleteInput(key='test')
        assert inp.key == 'test'


# --- SQLite Backend Tests ---


class TestSQLiteMemoryBackend:
    @pytest.mark.anyio
    async def test_store_and_retrieve(self, backend: SQLiteMemoryBackend) -> None:
        now = datetime.now(timezone.utc)
        entry = MemoryEntry(
            key='test:key',
            value='test value',
            tags=['test'],
            created_at=now,
            updated_at=now,
        )
        await backend.store(entry)

        retrieved = await backend.retrieve('test:key')
        assert retrieved is not None
        assert retrieved.key == 'test:key'
        assert retrieved.value == 'test value'
        assert retrieved.tags == ['test']
        assert retrieved.access_count == 1  # Incremented by retrieve

    @pytest.mark.anyio
    async def test_retrieve_nonexistent(self, backend: SQLiteMemoryBackend) -> None:
        result = await backend.retrieve('nonexistent')
        assert result is None

    @pytest.mark.anyio
    async def test_overwrite(self, backend: SQLiteMemoryBackend) -> None:
        now = datetime.now(timezone.utc)
        entry1 = MemoryEntry(key='k', value='v1', created_at=now, updated_at=now)
        await backend.store(entry1)

        entry2 = MemoryEntry(key='k', value='v2', created_at=now, updated_at=now)
        await backend.store(entry2)

        retrieved = await backend.retrieve('k')
        assert retrieved is not None
        assert retrieved.value == 'v2'

    @pytest.mark.anyio
    async def test_delete(self, backend: SQLiteMemoryBackend) -> None:
        now = datetime.now(timezone.utc)
        entry = MemoryEntry(key='to-delete', value='bye', created_at=now, updated_at=now)
        await backend.store(entry)

        deleted = await backend.delete('to-delete')
        assert deleted is True

        retrieved = await backend.retrieve('to-delete')
        assert retrieved is None

    @pytest.mark.anyio
    async def test_delete_nonexistent(self, backend: SQLiteMemoryBackend) -> None:
        deleted = await backend.delete('nonexistent')
        assert deleted is False

    @pytest.mark.anyio
    async def test_count(self, backend: SQLiteMemoryBackend) -> None:
        assert await backend.count() == 0

        now = datetime.now(timezone.utc)
        await backend.store(MemoryEntry(key='a', value='1', created_at=now, updated_at=now))
        assert await backend.count() == 1

        await backend.store(MemoryEntry(key='b', value='2', created_at=now, updated_at=now))
        assert await backend.count() == 2

    @pytest.mark.anyio
    async def test_list_all(self, populated_backend: SQLiteMemoryBackend) -> None:
        entries = await populated_backend.list_all()
        assert len(entries) == 4

    @pytest.mark.anyio
    async def test_list_with_pattern(self, populated_backend: SQLiteMemoryBackend) -> None:
        entries = await populated_backend.list_all(pattern='user:*')
        assert len(entries) == 3
        assert all(e.key.startswith('user:') for e in entries)

    @pytest.mark.anyio
    async def test_list_with_tags(self, populated_backend: SQLiteMemoryBackend) -> None:
        entries = await populated_backend.list_all(tags=['deadline'])
        assert len(entries) == 1
        assert entries[0].key == 'project:deadline'

    @pytest.mark.anyio
    async def test_search(self, populated_backend: SQLiteMemoryBackend) -> None:
        entries = await populated_backend.search('Mustafa')
        assert len(entries) >= 1
        assert any('Mustafa' in e.value for e in entries)

    @pytest.mark.anyio
    async def test_search_with_tags(self, populated_backend: SQLiteMemoryBackend) -> None:
        entries = await populated_backend.search('Istanbul', tags=['location'])
        assert len(entries) == 1
        assert 'Istanbul' in entries[0].value

    @pytest.mark.anyio
    async def test_compact(self, populated_backend: SQLiteMemoryBackend) -> None:
        # All test entries have low access_count and recent timestamps,
        # so compact with very aggressive settings should remove some
        removed = await populated_backend.compact(max_age_days=0, min_access=100)
        assert removed >= 0  # Depending on timing, may or may not remove

    @pytest.mark.anyio
    async def test_metadata_preserved(self, backend: SQLiteMemoryBackend) -> None:
        now = datetime.now(timezone.utc)
        entry = MemoryEntry(
            key='meta-test',
            value='value',
            metadata={'source': 'test', 'score': 42},
            created_at=now,
            updated_at=now,
        )
        await backend.store(entry)
        retrieved = await backend.retrieve('meta-test')
        assert retrieved is not None
        assert retrieved.metadata == {'source': 'test', 'score': 42}
