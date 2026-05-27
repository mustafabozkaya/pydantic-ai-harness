"""Shared fixtures for environment backend tests.

The `environment` fixture is parametrized over backends. Today that is just
`LocalEnvironment`; when `DockerEnvironment` lands it joins the `params` list and the
entire `test_conformance.py` suite runs against it for free (see task: Slice 5).

Conformance tests seed files on the host `tmp_path` and exercise them through
`environment`. `local` roots the environment at `tmp_path` directly; a future `docker`
backend bind-mounts `tmp_path` into the container, so host-side setup keeps working
unchanged across backends.
"""

from pathlib import Path

import pytest

from pydantic_ai_harness.environments.abstract import AbstractEnvironment
from pydantic_ai_harness.environments.local import LocalEnvironment


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture(params=['local'])
def environment(request: pytest.FixtureRequest, tmp_path: Path) -> AbstractEnvironment:
    """A backend rooted at `tmp_path`, parametrized over every environment implementation."""
    if request.param == 'local':
        return LocalEnvironment(root=str(tmp_path))
    raise AssertionError(f'unknown environment backend {request.param!r}')  # pragma: no cover
