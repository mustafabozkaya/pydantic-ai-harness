"""Conftest for guardrails tests."""

from __future__ import annotations

import pytest


@pytest.fixture(params=['asyncio'])
def anyio_backend(request: pytest.FixtureRequest) -> str:
    """Restrict to asyncio only for consistency."""
    return request.param
