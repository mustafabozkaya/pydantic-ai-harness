# Memory Capability

## Summary

Implements a `Memory` capability (`AbstractCapability` subclass) that provides persistent key-value memory across agent sessions, referencing issues #30 and #31.

## Design

### Architecture

- **`Memory`** dataclass extends `AbstractCapability[AgentDepsT]`
  - `get_instructions()` returns a dynamic callable that injects stored memories into the system prompt at run start
  - `get_toolset()` returns a `FunctionToolset` with five tools: `save_memory`, `recall_memory`, `search_memories`, `list_memories`, `delete_memory`
  - Tool functions use closures over `self.store` (no dependency on agent `deps`)

### Storage

- **`MemoryStore`** protocol: pluggable backend with `get`, `put`, `delete`, `list_all`, `search`
- **`DictMemoryStore`**: dict-based, ephemeral, for testing (default)
- **`FileMemoryStore`**: JSON file on disk, reads on init, writes on every mutation

### Memory Model

- **`MemoryEntry`** dataclass: `key`, `content`, `tags` (list[str]), `scope`, `expires_at`, `created_at`, `updated_at`
- **`MemoryEntryDict`** TypedDict for serialization
- Word-boundary search with relevance scoring (case-insensitive) across key, content, and tags
- Scoping/namespaces via `scope` field with filtering on search/list
- TTL/expiration via `expires_at` with `is_expired()` auto-filtering
- Dedup warning on save when keys are similar (Levenshtein distance <= 2)

### Spec Serialization

- `Memory.get_serialization_name()` returns `"Memory"`
- `Memory.from_spec(backend="file", path="...")` creates a `FileMemoryStore`-backed instance

## Configuration

| Field | Default | Description |
|-------|---------|-------------|
| `store` | `DictMemoryStore()` | Storage backend |
| `inject_memories_in_instructions` | `True` | Include memories in system prompt |
| `max_instructions_memories` | `20` | Cap on memories injected into prompt |

## Files

- `src/pydantic_harness/memory.py` - Capability, stores, entry model
- `src/pydantic_harness/__init__.py` - Re-exports
- `tests/test_memory.py` - 113 tests covering all code paths

## Future Work

- Semantic/vector search backend (e.g. embedding-based `MemoryStore`)
- Session-scoped memory isolation via `for_run()`
- SQLite / Redis backends for production persistence
