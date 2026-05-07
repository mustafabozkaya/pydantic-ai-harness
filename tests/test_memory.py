"""Tests for the Memory capability."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic_ai._run_context import RunContext
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.usage import RunUsage

from pydantic_harness.memory import (
    DictMemoryStore,
    FileMemoryStore,
    Memory,
    MemoryEntry,
    MemoryStore,
    _score_entry,
    _simple_similarity,
    exponential_decay,
    format_entry,
)

# --- MemoryEntry ---


class TestMemoryEntry:
    def test_round_trip(self) -> None:
        entry = MemoryEntry(
            key='k',
            content='v',
            tags=['a', 'b'],
            namespace=('project',),
            expires_at='2099-01-01T00:00:00+00:00',
            created_at='t1',
            updated_at='t2',
        )
        assert MemoryEntry.from_dict(entry.to_dict()) == entry

    def test_from_dict_defaults(self) -> None:
        entry = MemoryEntry.from_dict({'key': 'k', 'content': 'v'})
        assert entry.tags == []
        assert entry.namespace == ('global',)
        assert entry.expires_at is None
        assert entry.created_at == ''
        assert entry.updated_at == ''

    def test_default_timestamps(self) -> None:
        entry = MemoryEntry(key='k', content='v')
        assert entry.created_at  # non-empty ISO string
        assert entry.updated_at

    def test_default_namespace(self) -> None:
        entry = MemoryEntry(key='k', content='v')
        assert entry.namespace == ('global',)

    def test_is_expired_no_expiry(self) -> None:
        entry = MemoryEntry(key='k', content='v')
        assert not entry.is_expired()

    def test_is_expired_future(self) -> None:
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        entry = MemoryEntry(key='k', content='v', expires_at=future)
        assert not entry.is_expired()

    def test_is_expired_past(self) -> None:
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        entry = MemoryEntry(key='k', content='v', expires_at=past)
        assert entry.is_expired()

    def test_default_new_fields(self) -> None:
        entry = MemoryEntry(key='k', content='v')
        assert entry.summary is None
        assert entry.metadata == {}
        assert entry.read_only is False
        assert entry.char_limit is None
        assert entry.importance is None

    def test_round_trip_with_new_fields(self) -> None:
        entry = MemoryEntry(
            key='k',
            content='v',
            summary='short',
            metadata={'priority': 1, 'source': 'manual'},
            read_only=True,
            char_limit=100,
            importance=0.8,
        )
        assert MemoryEntry.from_dict(entry.to_dict()) == entry

    def test_char_limit_enforced(self) -> None:
        import pytest

        with pytest.raises(ValueError, match='exceeds char_limit'):
            MemoryEntry(key='k', content='hello world', char_limit=5)

    def test_char_limit_allows_exact(self) -> None:
        # Exactly at the limit is allowed
        MemoryEntry(key='k', content='hello', char_limit=5)


# --- _score_entry ---


class TestScoreEntry:
    def test_no_match(self) -> None:
        entry = MemoryEntry(key='greeting', content='hello world')
        assert _score_entry(entry, ['zzz']) == 0

    def test_key_match(self) -> None:
        entry = MemoryEntry(key='greeting', content='some text')
        assert _score_entry(entry, ['greeting']) == 1

    def test_content_match(self) -> None:
        entry = MemoryEntry(key='k', content='hello world')
        assert _score_entry(entry, ['hello']) == 1

    def test_tag_match(self) -> None:
        entry = MemoryEntry(key='k', content='text', tags=['important'])
        assert _score_entry(entry, ['important']) == 1

    def test_multiple_field_match(self) -> None:
        entry = MemoryEntry(key='hello', content='hello world', tags=['hello'])
        # 'hello' appears in key (1) + content (1) + tags (1) = 3
        assert _score_entry(entry, ['hello']) == 3

    def test_multiple_words(self) -> None:
        entry = MemoryEntry(key='user', content='Alice likes blue')
        # 'alice' in content (1), 'blue' in content (1) = 2
        assert _score_entry(entry, ['alice', 'blue']) == 2

    def test_word_boundary_no_partial(self) -> None:
        # 'fox' should NOT match 'foxes' with word-boundary matching
        entry = MemoryEntry(key='k', content='foxes jump')
        assert _score_entry(entry, ['fox']) == 0

    def test_regex_metacharacters_in_query(self) -> None:
        entry = MemoryEntry(key='lang', content='I use c++ daily')
        assert _score_entry(entry, ['c++']) == 1

    def test_empty_words_list(self) -> None:
        entry = MemoryEntry(key='k', content='hello')
        assert _score_entry(entry, []) == 0

    def test_underscore_word_boundary(self) -> None:
        entry = MemoryEntry(key='user_name', content='text')
        assert _score_entry(entry, ['name']) == 1

    def test_hyphen_word_boundary(self) -> None:
        entry = MemoryEntry(key='my-project', content='text')
        assert _score_entry(entry, ['project']) == 1

    def test_partial_word_match(self) -> None:
        entry = MemoryEntry(key='k', content='alice likes blue')
        # 'alice' matches (1), 'zzz' does not (0) = score 1
        assert _score_entry(entry, ['alice', 'zzz']) == 1


# --- _simple_similarity ---


class TestSimpleSimilarity:
    def test_identical_keys_not_similar(self) -> None:
        assert not _simple_similarity('abcdefghij', 'abcdefghij')

    def test_short_keys_not_similar(self) -> None:
        assert not _simple_similarity('abc', 'abd')

    def test_similar_long_keys(self) -> None:
        # Differ by 2 characters ('fo' vs 'ba') — within the edit-distance threshold
        assert _simple_similarity('abcdefghij_fo', 'abcdefghij_ba')

    def test_different_prefix(self) -> None:
        assert not _simple_similarity('xxxxxxxxxxfoo', 'yyyyyyyyyyfoo')

    def test_same_prefix_large_edit(self) -> None:
        assert not _simple_similarity('abcdefghijklmnop', 'abcdefghijzzzzzz')

    def test_length_diff_too_large(self) -> None:
        # Same 10-char prefix but length differs by more than 2
        assert not _simple_similarity('abcdefghij_x', 'abcdefghij_xyzw')

    def test_one_char_diff(self) -> None:
        assert _simple_similarity('abcdefghij_x', 'abcdefghij_y')

    def test_edit_distance_exactly_three(self) -> None:
        # Just over the threshold -- should NOT be similar
        assert not _simple_similarity('abcdefghij_abc', 'abcdefghij_xyz')

    def test_nine_char_keys(self) -> None:
        # Just below the 10-char minimum
        assert not _simple_similarity('abcdefghi', 'abcdefghj')

    def test_exactly_ten_char_keys_not_similar(self) -> None:
        # 10-char keys differing at position 10 do NOT share a 10-char prefix
        assert not _simple_similarity('abcdefghij', 'abcdefghik')


# --- DictMemoryStore ---


class TestDictMemoryStore:
    def test_put_and_get(self) -> None:
        store = DictMemoryStore()
        entry = MemoryEntry(key='greeting', content='hello')
        store.put(entry)
        assert store.get('greeting') is entry

    def test_get_missing(self) -> None:
        store = DictMemoryStore()
        assert store.get('nope') is None

    def test_put_overwrites(self) -> None:
        store = DictMemoryStore()
        store.put(MemoryEntry(key='k', content='v1'))
        store.put(MemoryEntry(key='k', content='v2'))
        result = store.get('k')
        assert result is not None
        assert result.content == 'v2'

    def test_delete_existing(self) -> None:
        store = DictMemoryStore()
        store.put(MemoryEntry(key='k', content='v'))
        assert store.delete('k') is True
        assert store.get('k') is None

    def test_delete_missing(self) -> None:
        store = DictMemoryStore()
        assert store.delete('nope') is False

    def test_list_all_empty(self) -> None:
        store = DictMemoryStore()
        assert store.list_all() == []

    def test_list_all(self) -> None:
        store = DictMemoryStore()
        store.put(MemoryEntry(key='a', content='alpha'))
        store.put(MemoryEntry(key='b', content='beta'))
        entries = store.list_all()
        assert len(entries) == 2
        assert {e.key for e in entries} == {'a', 'b'}

    def test_list_all_filters_expired(self) -> None:
        store = DictMemoryStore()
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        store.put(MemoryEntry(key='alive', content='fresh'))
        store.put(MemoryEntry(key='dead', content='stale', expires_at=past))
        entries = store.list_all()
        assert len(entries) == 1
        assert entries[0].key == 'alive'

    def test_get_filters_expired(self) -> None:
        store = DictMemoryStore()
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        store.put(MemoryEntry(key='dead', content='stale', expires_at=past))
        assert store.get('dead') is None

    def test_list_all_scope_filter(self) -> None:
        store = DictMemoryStore()
        store.put(MemoryEntry(key='a', content='x', namespace=('project',)))
        store.put(MemoryEntry(key='b', content='y', namespace=('global',)))
        entries = store.list_all(namespace=('project',))
        assert len(entries) == 1
        assert entries[0].key == 'a'

    def test_list_all_scope_none_returns_all(self) -> None:
        store = DictMemoryStore()
        store.put(MemoryEntry(key='a', content='x', namespace=('project',)))
        store.put(MemoryEntry(key='b', content='y', namespace=('global',)))
        assert len(store.list_all(namespace=None)) == 2

    def test_search_by_key(self) -> None:
        store = DictMemoryStore()
        store.put(MemoryEntry(key='user_name', content='Alice'))
        store.put(MemoryEntry(key='color', content='blue'))
        results = store.search('user')
        assert len(results) == 1
        assert results[0].key == 'user_name'

    def test_search_by_content(self) -> None:
        store = DictMemoryStore()
        store.put(MemoryEntry(key='k1', content='the quick brown fox'))
        store.put(MemoryEntry(key='k2', content='lazy dog'))
        results = store.search('fox')
        assert len(results) == 1
        assert results[0].key == 'k1'

    def test_search_by_tag(self) -> None:
        store = DictMemoryStore()
        store.put(MemoryEntry(key='k1', content='x', tags=['important']))
        store.put(MemoryEntry(key='k2', content='y', tags=['trivial']))
        results = store.search('important')
        assert len(results) == 1
        assert results[0].key == 'k1'

    def test_search_case_insensitive(self) -> None:
        store = DictMemoryStore()
        store.put(MemoryEntry(key='K1', content='Hello World'))
        results = store.search('hello')
        assert len(results) == 1

    def test_search_no_results(self) -> None:
        store = DictMemoryStore()
        store.put(MemoryEntry(key='k', content='v'))
        assert store.search('zzz') == []

    def test_search_filters_expired(self) -> None:
        store = DictMemoryStore()
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        store.put(MemoryEntry(key='alive', content='hello world'))
        store.put(MemoryEntry(key='dead', content='hello world', expires_at=past))
        results = store.search('hello')
        assert len(results) == 1
        assert results[0].key == 'alive'

    def test_search_scope_filter(self) -> None:
        store = DictMemoryStore()
        store.put(MemoryEntry(key='a', content='hello world', namespace=('project',)))
        store.put(MemoryEntry(key='b', content='hello world', namespace=('global',)))
        results = store.search('hello', namespace=('project',))
        assert len(results) == 1
        assert results[0].key == 'a'

    def test_search_relevance_ordering(self) -> None:
        store = DictMemoryStore()
        # 'hello' appears in key + content = score 2
        store.put(MemoryEntry(key='hello', content='hello there'))
        # 'hello' appears only in content = score 1
        store.put(MemoryEntry(key='other', content='hello world'))
        results = store.search('hello')
        assert len(results) == 2
        assert results[0].key == 'hello'  # higher score first
        assert results[1].key == 'other'

    def test_search_empty_query(self) -> None:
        store = DictMemoryStore()
        store.put(MemoryEntry(key='k', content='v'))
        assert store.search('') == []

    def test_list_namespaces(self) -> None:
        store = DictMemoryStore()
        store.put(MemoryEntry(key='a', content='x', namespace=('users', 'alice')))
        store.put(MemoryEntry(key='b', content='y', namespace=('users', 'bob')))
        store.put(MemoryEntry(key='c', content='z', namespace=('agents', 'planner')))
        assert store.list_namespaces() == [('agents', 'planner'), ('users', 'alice'), ('users', 'bob')]

    def test_list_namespaces_prefix(self) -> None:
        store = DictMemoryStore()
        store.put(MemoryEntry(key='a', content='x', namespace=('users', 'alice')))
        store.put(MemoryEntry(key='b', content='y', namespace=('agents', 'planner')))
        assert store.list_namespaces(prefix=('users',)) == [('users', 'alice')]

    def test_list_namespaces_max_depth(self) -> None:
        store = DictMemoryStore()
        store.put(MemoryEntry(key='a', content='x', namespace=('users', 'alice', 'prefs')))
        store.put(MemoryEntry(key='b', content='y', namespace=('users', 'bob', 'prefs')))
        # Truncate to depth 1 → both collapse to ('users',)
        assert store.list_namespaces(max_depth=1) == [('users',)]

    def test_list_namespaces_suffix(self) -> None:
        store = DictMemoryStore()
        store.put(MemoryEntry(key='a', content='x', namespace=('users', 'alice', 'prefs')))
        store.put(MemoryEntry(key='b', content='y', namespace=('agents', 'planner', 'prefs')))
        assert store.list_namespaces(suffix=('prefs',)) == [
            ('agents', 'planner', 'prefs'),
            ('users', 'alice', 'prefs'),
        ]

    def test_list_all_namespace_prefix_match(self) -> None:
        store = DictMemoryStore()
        store.put(MemoryEntry(key='a', content='x', namespace=('users', 'alice')))
        store.put(MemoryEntry(key='b', content='y', namespace=('users', 'bob')))
        store.put(MemoryEntry(key='c', content='z', namespace=('agents',)))
        # Prefix ('users',) matches both alice and bob
        results = store.list_all(namespace=('users',))
        assert {e.key for e in results} == {'a', 'b'}

    def test_list_all_with_filter(self) -> None:
        store = DictMemoryStore()
        store.put(MemoryEntry(key='a', content='x', metadata={'priority': 1}))
        store.put(MemoryEntry(key='b', content='y', metadata={'priority': 2}))
        results = store.list_all(filter={'priority': 1})
        assert len(results) == 1
        assert results[0].key == 'a'

    def test_search_with_filter(self) -> None:
        store = DictMemoryStore()
        store.put(MemoryEntry(key='a', content='hello world', metadata={'source': 'manual'}))
        store.put(MemoryEntry(key='b', content='hello world', metadata={'source': 'auto'}))
        results = store.search('hello', filter={'source': 'manual'})
        assert len(results) == 1
        assert results[0].key == 'a'

    def test_search_filter_no_match(self) -> None:
        store = DictMemoryStore()
        store.put(MemoryEntry(key='a', content='hello world', metadata={'source': 'manual'}))
        assert store.search('hello', filter={'source': 'nonexistent'}) == []

    def test_search_importance_boosts(self) -> None:
        store = DictMemoryStore()
        # Both match 'hello' once in content; importance differentiates them
        store.put(MemoryEntry(key='boring', content='hello there'))
        store.put(MemoryEntry(key='vip', content='hello there', importance=2.0))
        results = store.search('hello')
        assert results[0].key == 'vip'
        assert results[1].key == 'boring'

    def test_search_with_recency_scorer(self) -> None:
        store = DictMemoryStore()
        # Identical keyword scores; recent entry should rank first via recency_scorer.
        old = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
        new = datetime.now(timezone.utc).isoformat()
        store.put(MemoryEntry(key='ancient', content='hello there', updated_at=old))
        store.put(MemoryEntry(key='fresh', content='hello there', updated_at=new))
        results = store.search('hello', recency_scorer=exponential_decay(half_life_days=30.0))
        assert results[0].key == 'fresh'
        assert results[1].key == 'ancient'


class TestExponentialDecay:
    def test_fresh_entry_full_weight(self) -> None:
        scorer = exponential_decay(half_life_days=30.0, weight=1.0)
        entry = MemoryEntry(key='k', content='v')  # updated_at = now
        # Should be very close to 1.0 (essentially zero seconds elapsed)
        assert 0.99 < scorer(entry) <= 1.0

    def test_one_half_life_old(self) -> None:
        scorer = exponential_decay(half_life_days=30.0, weight=1.0)
        thirty_days_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        entry = MemoryEntry(key='k', content='v', updated_at=thirty_days_ago)
        # ~0.5 (within float-precision tolerance)
        assert 0.49 < scorer(entry) < 0.51

    def test_invalid_updated_at_returns_zero(self) -> None:
        scorer = exponential_decay()
        entry = MemoryEntry(key='k', content='v', updated_at='')
        assert scorer(entry) == 0.0

    def test_weight_multiplier(self) -> None:
        scorer = exponential_decay(half_life_days=30.0, weight=0.5)
        entry = MemoryEntry(key='k', content='v')  # fresh
        assert 0.49 < scorer(entry) <= 0.5


# --- FileMemoryStore ---


class TestFileMemoryStore:
    def test_put_and_get(self, tmp_path: Path) -> None:
        path = tmp_path / 'mem.json'
        store = FileMemoryStore(path)
        store.put(MemoryEntry(key='k', content='v'))
        assert store.get('k') is not None
        assert store.get('k').content == 'v'  # type: ignore[union-attr]

    def test_persistence(self, tmp_path: Path) -> None:
        path = tmp_path / 'mem.json'
        store1 = FileMemoryStore(path)
        store1.put(MemoryEntry(key='k', content='persisted'))

        # New store instance should load from disk
        store2 = FileMemoryStore(path)
        result = store2.get('k')
        assert result is not None
        assert result.content == 'persisted'

    def test_delete_saves(self, tmp_path: Path) -> None:
        path = tmp_path / 'mem.json'
        store = FileMemoryStore(path)
        store.put(MemoryEntry(key='k', content='v'))
        store.delete('k')

        # Reload and verify deletion persisted
        store2 = FileMemoryStore(path)
        assert store2.get('k') is None

    def test_delete_missing(self, tmp_path: Path) -> None:
        path = tmp_path / 'mem.json'
        store = FileMemoryStore(path)
        assert store.delete('nope') is False

    def test_list_all(self, tmp_path: Path) -> None:
        path = tmp_path / 'mem.json'
        store = FileMemoryStore(path)
        store.put(MemoryEntry(key='a', content='alpha'))
        store.put(MemoryEntry(key='b', content='beta'))
        assert len(store.list_all()) == 2

    def test_search(self, tmp_path: Path) -> None:
        path = tmp_path / 'mem.json'
        store = FileMemoryStore(path)
        store.put(MemoryEntry(key='k1', content='hello', tags=['greeting']))
        store.put(MemoryEntry(key='k2', content='world'))
        assert len(store.search('greeting')) == 1
        assert len(store.search('hello')) == 1

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / 'mem.json'
        # File does not exist yet
        store = FileMemoryStore(path)
        assert store.list_all() == []

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / 'sub' / 'dir' / 'mem.json'
        store = FileMemoryStore(path)
        store.put(MemoryEntry(key='k', content='v'))
        assert path.exists()

    def test_file_format(self, tmp_path: Path) -> None:
        path = tmp_path / 'mem.json'
        store = FileMemoryStore(path)
        store.put(MemoryEntry(key='k', content='v', tags=['t'], created_at='c', updated_at='u'))
        raw = json.loads(path.read_text())
        assert raw == {
            'k': {
                'key': 'k',
                'content': 'v',
                'tags': ['t'],
                'namespace': ['global'],
                'expires_at': None,
                'created_at': 'c',
                'updated_at': 'u',
                'summary': None,
                'metadata': {},
                'read_only': False,
                'char_limit': None,
                'importance': None,
            }
        }

    def test_list_all_filters_expired(self, tmp_path: Path) -> None:
        path = tmp_path / 'mem.json'
        store = FileMemoryStore(path)
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        store.put(MemoryEntry(key='alive', content='x'))
        store.put(MemoryEntry(key='dead', content='y', expires_at=past))
        assert len(store.list_all()) == 1

    def test_search_filters_expired(self, tmp_path: Path) -> None:
        path = tmp_path / 'mem.json'
        store = FileMemoryStore(path)
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        store.put(MemoryEntry(key='alive', content='hello world'))
        store.put(MemoryEntry(key='dead', content='hello world', expires_at=past))
        assert len(store.search('hello')) == 1

    def test_list_all_scope(self, tmp_path: Path) -> None:
        path = tmp_path / 'mem.json'
        store = FileMemoryStore(path)
        store.put(MemoryEntry(key='a', content='x', namespace=('project',)))
        store.put(MemoryEntry(key='b', content='y', namespace=('global',)))
        assert len(store.list_all(namespace=('project',))) == 1

    def test_search_scope(self, tmp_path: Path) -> None:
        path = tmp_path / 'mem.json'
        store = FileMemoryStore(path)
        store.put(MemoryEntry(key='a', content='hello world', namespace=('project',)))
        store.put(MemoryEntry(key='b', content='hello world', namespace=('global',)))
        assert len(store.search('hello', namespace=('project',))) == 1

    def test_search_empty_query(self, tmp_path: Path) -> None:
        path = tmp_path / 'mem.json'
        store = FileMemoryStore(path)
        store.put(MemoryEntry(key='k', content='v'))
        assert store.search('') == []

    def test_load_malformed_json(self, tmp_path: Path) -> None:
        path = tmp_path / 'mem.json'
        path.write_text('not json at all', encoding='utf-8')
        store = FileMemoryStore(path)
        assert store.list_all() == []

    def test_load_wrong_structure(self, tmp_path: Path) -> None:
        path = tmp_path / 'mem.json'
        path.write_text('["a", "b"]', encoding='utf-8')
        store = FileMemoryStore(path)
        assert store.list_all() == []

    def test_load_missing_entry_fields(self, tmp_path: Path) -> None:
        path = tmp_path / 'mem.json'
        path.write_text('{"k": {"not_a_key": "oops"}}', encoding='utf-8')
        store = FileMemoryStore(path)
        assert store.list_all() == []

    def test_namespace_persists(self, tmp_path: Path) -> None:
        path = tmp_path / 'mem.json'
        store1 = FileMemoryStore(path)
        store1.put(MemoryEntry(key='k', content='v', namespace=('session',)))
        store2 = FileMemoryStore(path)
        entry = store2.get('k')
        assert entry is not None
        assert entry.namespace == ('session',)

    def test_nested_namespace_persists(self, tmp_path: Path) -> None:
        path = tmp_path / 'mem.json'
        store1 = FileMemoryStore(path)
        store1.put(MemoryEntry(key='k', content='v', namespace=('users', 'alice', 'prefs')))
        store2 = FileMemoryStore(path)
        entry = store2.get('k')
        assert entry is not None
        assert entry.namespace == ('users', 'alice', 'prefs')

    def test_expires_at_persists(self, tmp_path: Path) -> None:
        path = tmp_path / 'mem.json'
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        store1 = FileMemoryStore(path)
        store1.put(MemoryEntry(key='k', content='v', expires_at=future))
        store2 = FileMemoryStore(path)
        entry = store2.get('k')
        assert entry is not None
        assert entry.expires_at == future

    def test_save_drops_expired(self, tmp_path: Path) -> None:
        path = tmp_path / 'mem.json'
        store = FileMemoryStore(path)
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        store.put(MemoryEntry(key='dead', content='stale', expires_at=past))
        store.put(MemoryEntry(key='alive', content='fresh'))

        raw = json.loads(path.read_text(encoding='utf-8'))
        assert 'dead' not in raw
        assert 'alive' in raw

    def test_get_filters_expired(self, tmp_path: Path) -> None:
        path = tmp_path / 'mem.json'
        store = FileMemoryStore(path)
        future = (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat()
        store.put(MemoryEntry(key='soon', content='v', expires_at=future))
        # Manually backdate by mutating the in-memory entry's expires_at
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        store._entries['soon'].expires_at = past
        assert store.get('soon') is None


# --- format_entry ---


class TestFormatEntry:
    def test_no_tags(self) -> None:
        entry = MemoryEntry(key='k', content='hello')
        assert format_entry(entry) == '[k] hello'

    def test_with_tags(self) -> None:
        entry = MemoryEntry(key='k', content='hello', tags=['a', 'b'])
        assert format_entry(entry) == '[k] hello (tags: a, b)'

    def test_with_namespace(self) -> None:
        entry = MemoryEntry(key='k', content='hello', namespace=('project',))
        assert format_entry(entry) == '[k] hello (namespace: project)'

    def test_with_nested_namespace(self) -> None:
        entry = MemoryEntry(key='k', content='hello', namespace=('users', 'alice'))
        assert format_entry(entry) == '[k] hello (namespace: users/alice)'

    def test_global_namespace_omitted(self) -> None:
        entry = MemoryEntry(key='k', content='hello', namespace=('global',))
        assert format_entry(entry) == '[k] hello'

    def test_with_expires_at(self) -> None:
        entry = MemoryEntry(key='k', content='hello', expires_at='2099-01-01T00:00:00+00:00')
        assert format_entry(entry) == '[k] hello (expires: 2099-01-01T00:00:00+00:00)'

    def test_all_extras(self) -> None:
        entry = MemoryEntry(
            key='k',
            content='hello',
            tags=['t'],
            namespace=('project',),
            expires_at='2099-01-01T00:00:00+00:00',
        )
        assert format_entry(entry) == '[k] hello (tags: t; namespace: project; expires: 2099-01-01T00:00:00+00:00)'

    def test_empty_content(self) -> None:
        entry = MemoryEntry(key='k', content='')
        assert format_entry(entry) == '[k] '

    def test_empty_key(self) -> None:
        entry = MemoryEntry(key='', content='hello')
        assert format_entry(entry) == '[] hello'


# --- Memory capability ---


class TestMemoryCapability:
    def test_serialization_name(self) -> None:
        assert Memory.get_serialization_name() == 'Memory'

    def test_from_spec_default(self) -> None:
        cap = Memory.from_spec()
        assert isinstance(cap.store, DictMemoryStore)

    def test_from_spec_file(self, tmp_path: Path) -> None:
        path = tmp_path / 'mem.json'
        cap = Memory.from_spec(backend='file', path=str(path))
        assert isinstance(cap.store, FileMemoryStore)

    def test_from_spec_unknown_backend(self) -> None:
        import pytest

        with pytest.raises(ValueError, match='Unknown memory backend'):
            Memory.from_spec(backend='redis')

    def test_from_spec_explicit_memory_backend(self) -> None:
        cap = Memory.from_spec(backend='memory')
        assert isinstance(cap.store, DictMemoryStore)

    def test_from_spec_with_options(self, tmp_path: Path) -> None:
        cap = Memory.from_spec(
            backend='file',
            path=str(tmp_path / 'mem.json'),
            inject_memories_in_instructions=False,
            max_instructions_memories=10,
        )
        assert isinstance(cap.store, FileMemoryStore)
        assert cap.inject_memories_in_instructions is False
        assert cap.max_instructions_memories == 10

    def test_default_store(self) -> None:
        cap: Memory[None] = Memory()
        assert isinstance(cap.store, DictMemoryStore)

    def test_get_toolset_returns_function_toolset(self) -> None:
        cap: Memory[None] = Memory()
        toolset = cap.get_toolset()
        assert isinstance(toolset, FunctionToolset)

    def test_toolset_has_expected_tools(self) -> None:
        cap: Memory[None] = Memory()
        toolset = cap.get_toolset()
        assert isinstance(toolset, FunctionToolset)
        tool_names = set(toolset.tools.keys())
        assert tool_names == {'save_memory', 'recall_memory', 'search_memories', 'list_memories', 'delete_memory'}


# --- Tool functions (via closure) ---


class TestMemoryTools:
    """Test the tool functions exposed by the Memory capability."""

    @staticmethod
    def _get_tools(store: DictMemoryStore | None = None) -> dict[str, Any]:
        cap: Memory[None] = Memory(store=store or DictMemoryStore())
        toolset = cap.get_toolset()
        assert isinstance(toolset, FunctionToolset)
        return {name: tool.function for name, tool in toolset.tools.items()}

    def test_save_and_recall(self) -> None:
        store = DictMemoryStore()
        tools = self._get_tools(store)
        result = tools['save_memory']('greeting', 'hello world')
        assert result == 'Memory saved: greeting'

        recalled = tools['recall_memory']('greeting')
        assert '[greeting] hello world' in recalled

    def test_recall_missing(self) -> None:
        tools = self._get_tools()
        assert 'No memory found' in tools['recall_memory']('nope')

    def test_recall_expired(self) -> None:
        store = DictMemoryStore()
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        store.put(MemoryEntry(key='old', content='stale', expires_at=past))
        tools = self._get_tools(store)
        assert 'No memory found' in tools['recall_memory']('old')

    def test_save_updates_existing(self) -> None:
        store = DictMemoryStore()
        tools = self._get_tools(store)
        tools['save_memory']('k', 'v1')
        original = store.get('k')
        assert original is not None
        original_created = original.created_at

        tools['save_memory']('k', 'v2')
        updated = store.get('k')
        assert updated is not None
        assert updated.content == 'v2'
        # created_at should be preserved
        assert updated.created_at == original_created

    def test_save_with_tags(self) -> None:
        store = DictMemoryStore()
        tools = self._get_tools(store)
        tools['save_memory']('k', 'v', ['tag1', 'tag2'])
        entry = store.get('k')
        assert entry is not None
        assert entry.tags == ['tag1', 'tag2']

    def test_save_with_namespace(self) -> None:
        store = DictMemoryStore()
        tools = self._get_tools(store)
        tools['save_memory']('k', 'v', None, ['project'])
        entry = store.get('k')
        assert entry is not None
        assert entry.namespace == ('project',)

    def test_save_with_nested_namespace(self) -> None:
        store = DictMemoryStore()
        tools = self._get_tools(store)
        tools['save_memory']('k', 'v', None, ['users', 'alice'])
        entry = store.get('k')
        assert entry is not None
        assert entry.namespace == ('users', 'alice')

    def test_save_with_ttl(self) -> None:
        store = DictMemoryStore()
        tools = self._get_tools(store)
        tools['save_memory']('k', 'v', None, ['global'], 60)
        entry = store.get('k')
        assert entry is not None
        assert entry.expires_at is not None
        expires = datetime.fromisoformat(entry.expires_at)
        # Should expire roughly 60 minutes from now
        assert expires > datetime.now(timezone.utc) + timedelta(minutes=59)
        assert expires < datetime.now(timezone.utc) + timedelta(minutes=61)

    def test_search(self) -> None:
        store = DictMemoryStore()
        tools = self._get_tools(store)
        tools['save_memory']('user_name', 'Alice')
        tools['save_memory']('color', 'blue')

        result = tools['search_memories']('Alice')
        assert 'Alice' in result
        assert 'blue' not in result

    def test_search_no_results(self) -> None:
        tools = self._get_tools()
        assert 'No memories found' in tools['search_memories']('zzz')

    def test_search_with_scope(self) -> None:
        store = DictMemoryStore()
        tools = self._get_tools(store)
        tools['save_memory']('a', 'hello world', None, ['project'])
        tools['save_memory']('b', 'hello world', None, ['global'])
        result = tools['search_memories']('hello', ['project'])
        assert '[a]' in result
        assert '[b]' not in result

    def test_list_empty(self) -> None:
        tools = self._get_tools()
        assert tools['list_memories']() == 'No memories stored.'

    def test_list_with_entries(self) -> None:
        store = DictMemoryStore()
        tools = self._get_tools(store)
        tools['save_memory']('a', 'alpha')
        tools['save_memory']('b', 'beta')
        result = tools['list_memories']()
        assert '[a] alpha' in result
        assert '[b] beta' in result

    def test_list_with_scope(self) -> None:
        store = DictMemoryStore()
        tools = self._get_tools(store)
        tools['save_memory']('a', 'alpha', None, ['project'])
        tools['save_memory']('b', 'beta', None, ['global'])
        result = tools['list_memories'](['project'])
        assert '[a] alpha' in result
        assert '[b]' not in result

    def test_delete_existing(self) -> None:
        store = DictMemoryStore()
        tools = self._get_tools(store)
        tools['save_memory']('k', 'v')
        assert tools['delete_memory']('k') == 'Memory deleted: k'
        assert store.get('k') is None

    def test_delete_missing(self) -> None:
        tools = self._get_tools()
        assert 'No memory found' in tools['delete_memory']('nope')

    def test_save_with_ttl_zero(self) -> None:
        store = DictMemoryStore()
        tools = self._get_tools(store)
        tools['save_memory']('k', 'v', None, ['global'], 0)
        # TTL=0 expires immediately; get() filters it out
        assert store.get('k') is None
        # recall_memory should likewise report no memory
        assert 'No memory found' in tools['recall_memory']('k')

    def test_save_with_summary_and_importance(self) -> None:
        store = DictMemoryStore()
        tools = self._get_tools(store)
        tools['save_memory']('k', 'long content here', None, ['global'], None, 'short', 0.9)
        entry = store.get('k')
        assert entry is not None
        assert entry.summary == 'short'
        assert entry.importance == 0.9

    def test_save_refuses_read_only(self) -> None:
        store = DictMemoryStore()
        store.put(MemoryEntry(key='persona', content='locked', read_only=True))
        tools = self._get_tools(store)
        result = tools['save_memory']('persona', 'overwrite attempt')
        assert 'read-only' in result.lower()
        # Original content preserved
        entry = store.get('persona')
        assert entry is not None
        assert entry.content == 'locked'

    def test_delete_refuses_read_only(self) -> None:
        store = DictMemoryStore()
        store.put(MemoryEntry(key='persona', content='locked', read_only=True))
        tools = self._get_tools(store)
        result = tools['delete_memory']('persona')
        assert 'read-only' in result.lower()
        assert store.get('persona') is not None


# --- Dedup warning ---


class TestDedupWarning:
    def test_similar_key_logs_warning(self, caplog: Any) -> None:
        store = DictMemoryStore()
        tools = TestMemoryTools._get_tools(store)
        tools['save_memory']('abcdefghij_x', 'first value')
        with caplog.at_level(logging.WARNING, logger='pydantic_harness.memory'):
            tools['save_memory']('abcdefghij_y', 'second value')
        assert any('possible duplicate' in record.message.lower() for record in caplog.records)

    def test_different_keys_no_warning(self, caplog: Any) -> None:
        store = DictMemoryStore()
        tools = TestMemoryTools._get_tools(store)
        tools['save_memory']('first_key_long', 'first value')
        with caplog.at_level(logging.WARNING, logger='pydantic_harness.memory'):
            tools['save_memory']('other_key_long', 'second value')
        assert not any('possible duplicate' in record.message.lower() for record in caplog.records)

    def test_short_keys_no_warning(self, caplog: Any) -> None:
        store = DictMemoryStore()
        tools = TestMemoryTools._get_tools(store)
        tools['save_memory']('abc', 'first value')
        with caplog.at_level(logging.WARNING, logger='pydantic_harness.memory'):
            tools['save_memory']('abd', 'second value')
        assert not any('possible duplicate' in record.message.lower() for record in caplog.records)


# --- Instructions ---


class TestMemoryInstructions:
    @staticmethod
    def _make_ctx() -> RunContext[None]:
        from unittest.mock import MagicMock

        return RunContext(
            deps=None,
            model=MagicMock(),
            usage=RunUsage(),
        )

    def test_get_instructions_is_callable(self) -> None:
        cap: Memory[None] = Memory()
        assert callable(cap.get_instructions())

    def test_instructions_with_no_memories(self) -> None:
        cap: Memory[None] = Memory()
        text = cap.build_instructions(self._make_ctx())
        assert 'persistent memory system' in text
        assert 'Currently stored memories' not in text

    def test_instructions_with_memories(self) -> None:
        store = DictMemoryStore()
        store.put(MemoryEntry(key='user', content='Alice'))
        cap: Memory[None] = Memory(store=store)
        text = cap.build_instructions(self._make_ctx())
        assert 'Currently stored memories' in text
        assert '[user] Alice' in text

    def test_instructions_respects_max(self) -> None:
        store = DictMemoryStore()
        for i in range(25):
            store.put(MemoryEntry(key=f'k{i}', content=f'v{i}'))
        cap: Memory[None] = Memory(store=store, max_instructions_memories=5)
        text = cap.build_instructions(self._make_ctx())
        assert '... and 20 more' in text

    def test_instructions_disabled(self) -> None:
        store = DictMemoryStore()
        store.put(MemoryEntry(key='k', content='v'))
        cap: Memory[None] = Memory(store=store, inject_memories_in_instructions=False)
        text = cap.build_instructions(self._make_ctx())
        assert 'Currently stored memories' not in text

    def test_instructions_exact_max_no_overflow(self) -> None:
        store = DictMemoryStore()
        for i in range(5):
            store.put(MemoryEntry(key=f'k{i}', content=f'v{i}'))
        cap: Memory[None] = Memory(store=store, max_instructions_memories=5)
        text = cap.build_instructions(self._make_ctx())
        assert '... and' not in text
        assert '[k0]' in text
        assert '[k4]' in text


# --- MemoryStore protocol ---


class TestMemoryStoreProtocol:
    def test_in_memory_store_satisfies_protocol(self) -> None:
        assert isinstance(DictMemoryStore(), MemoryStore)

    def test_file_store_satisfies_protocol(self, tmp_path: Path) -> None:
        assert isinstance(FileMemoryStore(tmp_path / 'mem.json'), MemoryStore)


# --- AbstractCapability conformance ---


class TestAbstractCapabilityConformance:
    def test_is_abstract_capability_subclass(self) -> None:
        from pydantic_ai.capabilities.abstract import AbstractCapability

        assert issubclass(Memory, AbstractCapability)

    def test_instance_is_abstract_capability(self) -> None:
        from pydantic_ai.capabilities.abstract import AbstractCapability

        assert isinstance(Memory(), AbstractCapability)
