## Summary

Add persistent memory capability for Pydantic AI agents. This is the first implementation of the [Persistent key-value memory](https://github.com/pydantic/pydantic-ai-harness/issues/179) feature request.

## What This Adds

### New Capability: MemoryCapability

Provides five tools for persistent memory management:

- `memory_store` — Save information with key, value, tags, and metadata
- `memory_retrieve` — Full-text search across stored memories (FTS5)
- `memory_list` — List entries with glob pattern and tag filtering
- `memory_delete` — Remove an entry by key
- `memory_compact` — Clean up old, rarely-accessed entries

### Architecture

The implementation follows the established CodeMode capability pattern:

- `_abstract.py` — AbstractMemoryBackend interface for custom backends
- `_capability.py` — MemoryCapability (AbstractCapability subclass)
- `_models.py` — Pydantic models (MemoryEntry, input schemas)
- `_sqlite.py` — SQLiteMemoryBackend (default backend with FTS5)
- `_toolset.py` — MemoryToolset (tool implementations)

### Key Features

- SQLite backend with FTS5 full-text search
- AbstractMemoryBackend interface for custom storage engines
- Tag-based filtering and glob pattern matching
- Access tracking with automatic compaction
- Lazy initialization of database connection
- 18 tests covering models, backend, and edge cases

### Usage

```python
from pydantic_ai import Agent
from pydantic_ai_harness import MemoryCapability

agent = Agent('openai:gpt-5', capabilities=[MemoryCapability()])
result = agent.run_sync('Remember that my favorite color is blue')
result = agent.run_sync('What is my favorite color?')  # blue
```

## Checklist

- Follows CodeMode capability pattern
- Ruff lint clean
- 18 tests passing
- README.md with examples
- Abstract interface for extensibility

Closes #179
