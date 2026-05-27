"""Shared fixtures for execution_env capability tests."""

import pytest


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'
