# Memory

Persistent key-value memory across agent sessions. Provides five tools the LLM
can call (`save_memory`, `recall_memory`, `search_memories`, `list_memories`,
`delete_memory`) and injects currently-stored memories into the system prompt
each run.

## Quick start

```python
from pydantic_ai import Agent
from pydantic_harness import Memory

agent = Agent('openai:gpt-4o', capabilities=[Memory()])
```

By default `Memory()` uses an in-process `DictMemoryStore` — entries live only
for the lifetime of the Python process. Use `FileMemoryStore` for single-user
single-process persistence, or implement `MemoryStore` for anything else.

## Built-in backends

| Backend | Persistence | Concurrency | Use case |
|---|---|---|---|
| `DictMemoryStore` | None (in-process) | Single-thread | Tests, scratch agents |
| `FileMemoryStore(path)` | JSON file on disk | Single-process | Single-user CLI agents |

For Postgres, Redis, vector DBs, etc. — implement the `MemoryStore` Protocol.
See [`examples/memory/postgres_store.py`](https://github.com/pydantic/pydantic-harness/blob/main/examples/memory/postgres_store.py)
for a reference implementation.

## `MemoryEntry` fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `key` | `str` | required | Unique identifier |
| `content` | `str` | required | The fact itself |
| `tags` | `list[str]` | `[]` | LLM-set categorisation |
| `namespace` | `tuple[str, ...]` | `('global',)` | Hierarchical namespace; prefix-matched in queries |
| `expires_at` | `str \| None` | `None` | ISO 8601 wall-clock expiry; opt-in TTL |
| `created_at`, `updated_at` | `str` | now | ISO 8601 timestamps |
| `summary` | `str \| None` | `None` | Short version preferred over `content` for prompt injection |
| `metadata` | `dict[str, object]` | `{}` | Structured attributes; filterable via `search(filter=...)` |
| `read_only` | `bool` | `False` | If True, agent's tools refuse to modify |
| `char_limit` | `int \| None` | `None` | Optional hard cap on `content` length (raises at construction) |
| `importance` | `float \| None` | `None` | Search-score booster |

## Namespaces

Hierarchical, tuple-based. Filters in `list_all`/`search` use prefix matching:

```python
store.put(MemoryEntry(key='a', content='...', namespace=('users', 'alice')))
store.put(MemoryEntry(key='b', content='...', namespace=('users', 'bob')))
store.put(MemoryEntry(key='c', content='...', namespace=('agents', 'planner')))

store.list_all(namespace=('users',))          # → [a, b]
store.list_namespaces(prefix=('users',))      # → [('users', 'alice'), ('users', 'bob')]
```

## Multi-agent shared memory

One store, two agents, separate namespaces:

```python
from pydantic_ai import Agent
from pydantic_harness import FileMemoryStore, Memory

shared = FileMemoryStore('/var/lib/myapp/memory.json')

planner = Agent('openai:gpt-4o', capabilities=[
    Memory(store=shared, byte_budget=2000),
])
worker = Agent('openai:gpt-4o-mini', capabilities=[
    Memory(store=shared, byte_budget=500),
])
```

Entries written by either agent are visible to both. Use `namespace=('agents',
'planner')` etc. on saves to keep their workspaces separate while still sharing
common facts in `('global',)`.

## Search

Word-boundary regex on key, content, and tags. Final score:

```
keyword_match_count + (entry.importance or 0) + (recency_scorer(entry) or 0)
```

Recency boost is enabled by default — `Memory` ships with
`exponential_decay(half_life_days=30, weight=0.5)`. Override with any callable:

```python
from pydantic_harness import Memory, exponential_decay

# Tighter half-life for fast-moving information
Memory(recency_scorer=exponential_decay(half_life_days=7))

# Custom: boost only entries with the 'pinned' tag
Memory(recency_scorer=lambda e: 1.0 if 'pinned' in e.tags else 0.0)

# Disable recency entirely
Memory(recency_scorer=None)
```

## Prompt-cache trade-off

Every save/delete changes the injected memories block in the system prompt,
invalidating the prompt-cache prefix. Read-heavy workloads keep the cache;
write-heavy workloads thrash.

Mitigation: `Memory(inject_memories_in_instructions=False)` skips the
injection. The LLM reads memories only via explicit `list_memories` /
`search_memories` / `recall_memory` calls — system prompt prefix stays stable
across writes.

For partial mitigation, set `byte_budget` to cap the injected block size.

## Tool description overrides

```python
Memory(tool_descriptions={
    'save_memory': 'Save anything the user mentions about themselves, '
                   'even tiny details. Tag with "user_pref".',
})
```

## Custom backends

Implement the `MemoryStore` Protocol — six methods, all positional/kwarg-only:

```python
from pydantic_harness import MemoryEntry, MemoryStore, RecencyScorer

class MyStore:
    def get(self, key: str) -> MemoryEntry | None: ...
    def put(self, entry: MemoryEntry) -> None: ...
    def delete(self, key: str) -> bool: ...
    def list_all(self, *, namespace=None, filter=None): ...
    def search(self, query, *, namespace=None, filter=None, recency_scorer=None): ...
    def list_namespaces(self, *, prefix=None, suffix=None, max_depth=None): ...
```

Drop into any `Memory(store=MyStore())`. See the Postgres example for a
working reference.

## Known followups

- **Semantic retrieval**: `SemanticMemoryStore` Protocol extension and an
  `EmbeddingStore` reference impl. Deferred until a concrete backend (Qdrant /
  pgvector / LanceDB) drives the API design.
- **Tool-history dedup**: suppress next-turn injection of memories the LLM
  just saved (already in tool history). Deferred — has subtle semantics around
  updates that need real-world telemetry to design correctly.
