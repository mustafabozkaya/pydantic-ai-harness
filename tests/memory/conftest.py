"""Conftest for memory tests — restricts to asyncio only (aiosqlite requirement)."""

import pytest


@pytest.fixture(params=['asyncio'])
def anyio_backend(request: pytest.FixtureRequest) -> str:
    """Override anyio_backend to only use asyncio (aiosqlite doesn't support trio)."""
    return request.param
