"""Pydantic models for the memory capability."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    """A single memory entry stored in the backend."""

    key: str = Field(description='Unique identifier for this memory entry')
    value: str = Field(description='The content stored in memory')
    metadata: dict[str, Any] = Field(default_factory=dict, description='Arbitrary metadata')
    tags: list[str] = Field(default_factory=list, description='Tags for categorization and filtering')
    created_at: datetime = Field(description='When this entry was created')
    updated_at: datetime = Field(description='When this entry was last updated')
    access_count: int = Field(default=0, description='Number of times this entry has been accessed')
    last_accessed: datetime | None = Field(default=None, description='When this entry was last accessed')


class MemoryStoreInput(BaseModel):
    """Input schema for the memory_store tool."""

    key: str = Field(description='Unique key to identify this memory')
    value: str = Field(description='Content to store')
    metadata: dict[str, Any] = Field(default_factory=dict, description='Optional metadata')
    tags: list[str] = Field(default_factory=list, description='Optional tags for categorization')


class MemoryRetrieveInput(BaseModel):
    """Input schema for the memory_retrieve tool."""

    query: str = Field(description='Search query to find relevant memories')
    top_k: int = Field(default=5, ge=1, le=50, description='Maximum number of results to return')
    tags: list[str] | None = Field(default=None, description='Filter by tags')


class MemoryListInput(BaseModel):
    """Input schema for the memory_list tool."""

    pattern: str | None = Field(default=None, description='Glob pattern to filter keys (e.g. "user:*")')
    tags: list[str] | None = Field(default=None, description='Filter by tags')
    limit: int = Field(default=20, ge=1, le=100, description='Maximum number of entries to return')


class MemoryDeleteInput(BaseModel):
    """Input schema for the memory_delete tool."""

    key: str = Field(description='Key of the memory entry to delete')


class MemoryCompactInput(BaseModel):
    """Input schema for the memory_compact tool."""

    max_age_days: int = Field(default=90, ge=1, description='Remove entries older than this many days')
    min_access: int = Field(default=2, ge=0, description='Keep entries accessed at least this many times')
