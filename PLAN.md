# Memory Capability

## Summary

Implements a `Memory` capability (`AbstractCapability` subclass) providing
persistent key-value memory across agent sessions. References issues #30 and #31.

User-facing docs: [`docs/capabilities/memory.md`](docs/capabilities/memory.md).

## Design

### Architecture

- **`Memory`** dataclass extends `AbstractCapability[AgentDepsT]`
  - `get_instructions()` returns a dynamic callable injecting stored memories
    into the system prompt at run start
  - `get_toolset()` returns a `FunctionToolset` with five tools: `save_memory`,
    `recall_memory`, `search_memories`, `list_memories`, `delete_memory`
  - Per-tool description overrides via `tool_descriptions: dict[str, str]`
  - Tool functions use closures over `self.store` and `self.recency_scorer`
    (no dependency on agent `deps`)

### Storage

- **`MemoryStore`** Protocol: pluggable backend with six methods — `get`, `put`,
  `delete`, `list_all`, `search`, `list_namespaces`
- **`DictMemoryStore`**: dict-based, ephemeral, for tests/scratch (default)
- **`FileMemoryStore(path)`**: JSON file on disk, reads on init, writes on every
  mutation; drops expired entries on save
- Both extend `_BaseDictStore` for shared logic
- Custom backends: implement the Protocol. See
  [`examples/memory/postgres_store.py`](examples/memory/postgres_store.py) for
  a Postgres reference.

### `MemoryEntry`

Required: `key`, `content`. Optional fields:

- `tags: list[str]` — LLM-set categorisation
- `namespace: tuple[str, ...]` — hierarchical namespace, prefix-matched
- `expires_at: str | None` — ISO 8601 wall-clock expiry (opt-in TTL)
- `created_at`, `updated_at: str` — ISO 8601 timestamps
- `summary: str | None` — preferred over `content` for prompt injection
- `metadata: dict[str, object]` — structured attributes; filterable via search
- `read_only: bool` — pin against agent modification (always injected)
- `char_limit: int | None` — hard cap on `content` length, enforced at construction
- `importance: float | None` — search-score booster

`MemoryEntryDict` TypedDict covers the JSON serialisation form.

### Search & retrieval

- `search` score = keyword-match count + `entry.importance` (if set) +
  `recency_scorer(entry)` (if provided)
- Word-boundary regex matching across `key`, `content`, `tags` (case-insensitive)
- `_` and `-` count as word separators
- Default recency scorer: `exponential_decay(half_life_days=30, weight=0.5)`
- `search`/`list_all` accept `namespace` (prefix match) and `filter` (metadata
  equality) kwargs; `search` additionally accepts `recency_scorer`

### Instructions injection

- `read_only=True` entries always inject (bypass count cap, byte budget, and dedup)
- Non-pinned entries respect `max_instructions_memories` (default 20) and
  `byte_budget: int | None` (UTF-8 byte cap)
- `entry.summary` is preferred over `entry.content` to save tokens
- `dedup_recent_saves: bool = True` suppresses entries the LLM just saved in
  this run's tool history, when the saved content still matches the store
  entry (content-aware: if external state diverged, inject the current value)
- Pinned entries are listed first
- Disabled entirely via `inject_memories_in_instructions=False` (prompt-cache
  mitigation for write-heavy workloads)

### Spec serialisation

- `Memory.get_serialization_name()` → `"Memory"`
- `Memory.from_spec(backend='memory'|'file', path=..., ...)` for declarative config

## Configuration

| Field | Default | Description |
|---|---|---|
| `store` | `DictMemoryStore()` | Storage backend |
| `inject_memories_in_instructions` | `True` | Include memories in system prompt |
| `max_instructions_memories` | `20` | Cap on non-pinned memories injected |
| `byte_budget` | `None` | Optional UTF-8 byte cap on injection block |
| `recency_scorer` | `exponential_decay(half_life_days=30, weight=0.5)` | Or `None` to disable |
| `tool_descriptions` | `{}` | Per-tool description overrides |
| `dedup_recent_saves` | `True` | Suppress injection of entries just saved in this run |

## Files

- `src/pydantic_harness/memory.py` — capability, stores, entry model, recency helpers
- `src/pydantic_harness/__init__.py` — re-exports
- `tests/test_memory.py` — 150 tests covering all code paths
- `examples/memory/*.py` — three runnable examples plus the Postgres reference
- `docs/capabilities/memory.md` — user-facing docs

## Future Work

- **Semantic retrieval** — `SemanticMemoryStore` Protocol extension and an
  `EmbeddingStore` reference (numpy/cosine, or pgvector). Deferred until a
  concrete backend drives the API design — premature design tends to lock in
  the wrong shape.
- **Deferred capability loading** (PR #5230 in pydantic-ai) — once that lands,
  declare `id`/`description` on `Memory` to opt into deferred loading.
