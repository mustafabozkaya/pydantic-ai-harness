# Memory Capability

Persistent memory for Pydantic AI agents — remember facts, preferences, and context across conversations.

## Overview

`MemoryCapability` adds persistent storage tools to your agent, enabling it to:

- **Store** information with keys, values, tags, and metadata
- **Retrieve** memories via full-text search
- **List** entries with glob patterns and tag filters
- **Delete** outdated information
- **Compact** old entries to save space

Data is persisted in SQLite by default (`~/.pydantic-ai/memory.db`), with optional FTS5 full-text search support.

## Quick Start

```python
from pydantic_ai import Agent
from pydantic_ai_harness import MemoryCapability

agent = Agent('openai:gpt-5', capabilities=[MemoryCapability()])

# Agent can now remember things
result = agent.run_sync('Remember that my favorite color is blue')
# -> "I've stored that your favorite color is blue."

result = agent.run_sync('What is my favorite color?')
# -> "Your favorite color is blue!"
```

## Tools

| Tool | Description |
|------|-------------|
| `memory_store` | Save information with a key, value, optional tags and metadata |
| `memory_retrieve` | Full-text search across stored memories |
| `memory_list` | List entries with optional glob pattern and tag filtering |
| `memory_delete` | Remove an entry by key |
| `memory_compact` | Clean up old, rarely-accessed entries |

## Custom Backend

Implement `AbstractMemoryBackend` for custom storage:

```python
from pydantic_ai_harness.memory import AbstractMemoryBackend, MemoryCapability, MemoryEntry

class RedisMemoryBackend(AbstractMemoryBackend):
    """Custom Redis-backed memory storage."""
    
    async def store(self, entry: MemoryEntry) -> None:
        # Implement Redis storage
        ...
    
    async def retrieve(self, key: str) -> MemoryEntry | None:
        # Implement Redis retrieval
        ...
    
    # ... implement other methods

agent = Agent(
    'openai:gpt-5',
    capabilities=[MemoryCapability(backend=RedisMemoryBackend())],
)
```

## Configuration

```python
from pydantic_ai_harness import MemoryCapability
from pydantic_ai_harness.memory import SQLiteMemoryBackend

# Custom database path
backend = SQLiteMemoryBackend('/path/to/custom.db')
agent = Agent('openai:gpt-5', capabilities=[MemoryCapability(backend=backend)])

# In-memory database (for testing)
backend = SQLiteMemoryBackend(':memory:')
agent = Agent('openai:gpt-5', capabilities=[MemoryCapability(backend=backend)])
```

## How It Works

1. `MemoryCapability` wraps the agent's toolset with memory tools
2. When the agent calls a memory tool, it goes through `MemoryToolset`
3. `MemoryToolset` delegates to the configured `AbstractMemoryBackend`
4. The default `SQLiteMemoryBackend` uses FTS5 for full-text search

## Testing

```python
import pytest
from pydantic_ai import Agent
from pydantic_ai.models import TestModel
from pydantic_ai_harness import MemoryCapability
from pydantic_ai_harness.memory import SQLiteMemoryBackend

@pytest.mark.anyio
async def test_memory_store():
    backend = SQLiteMemoryBackend(':memory:')
    agent = Agent(TestModel(), capabilities=[MemoryCapability(backend=backend)])
    
    result = await agent.run('Store the fact: Python was created by Guido')
    assert result.output is not None
```
