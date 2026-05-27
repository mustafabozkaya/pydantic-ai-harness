"""LocalEnvironment-specific behavior that does not generalize across backends.

The backend-agnostic contract lives in `test_conformance.py`. Here we test things tied to
LocalEnvironment's implementation: how it resolves symlinked roots, how it maps POSIX
permission errors, and the catch-all I/O error path. A Docker backend would exercise these
concerns differently, so they are deliberately not part of the shared conformance suite.
"""

import os
from pathlib import Path

import pytest

from pydantic_ai_harness.environments.exceptions import (
    EnvFilePermissionError,
    EnvFileReadError,
    PathEscapeError,
)
from pydantic_ai_harness.environments.local import LocalEnvironment

# The permission tests below assert that the OS denies us a read/write. But root ignores
# POSIX permission bits entirely -- `chmod 000` then reading still succeeds as root -- so
# those tests would fail spuriously when run as uid 0 (common in CI containers). Skip them
# there. `hasattr` guards Windows, which has no `geteuid` and no POSIX bits to begin with.
skip_if_root = pytest.mark.skipif(
    hasattr(os, 'geteuid') and os.geteuid() == 0, reason='root bypasses POSIX permission bits'
)


async def test_read_through_symlinked_root(tmp_path: Path) -> None:
    # Root the environment at a SYMLINK to the real dir (like /tmp/... on macOS). A read of a
    # file genuinely inside must SUCCEED: the jail must resolve the root before comparing.
    real_box = tmp_path / 'realbox'
    real_box.mkdir()
    (real_box / 'ok.txt').write_bytes(b'inside the box')
    link_box = tmp_path / 'linkbox'
    link_box.symlink_to(real_box)

    env = LocalEnvironment(root=str(link_box))
    assert await env.read_file('ok.txt') == b'inside the box'


async def test_write_through_symlinked_root(tmp_path: Path) -> None:
    real_box = tmp_path / 'realbox'
    real_box.mkdir()
    link_box = tmp_path / 'linkbox'
    link_box.symlink_to(real_box)

    env = LocalEnvironment(root=str(link_box))
    await env.write_file('test.txt', b'Hello, world!')
    assert (real_box / 'test.txt').read_bytes() == b'Hello, world!'


async def test_read_outside_root_absolute_raises(tmp_path: Path) -> None:
    env = LocalEnvironment(root=str(tmp_path))
    secret = tmp_path.parent / 'secret.txt'
    secret.write_bytes(b'top secret')
    with pytest.raises(PathEscapeError):
        await env.read_file(str(secret))


async def test_write_outside_root_absolute_raises(tmp_path: Path) -> None:
    env = LocalEnvironment(root=str(tmp_path))
    secret = tmp_path.parent / 'secret.txt'
    with pytest.raises(PathEscapeError):
        await env.write_file(str(secret), b'top secret')


@skip_if_root
async def test_read_unreadable_file_raises_permission(tmp_path: Path) -> None:
    target = tmp_path / 'locked.txt'
    target.write_bytes(b'secret')
    target.chmod(0o000)
    try:
        env = LocalEnvironment(root=str(tmp_path))
        with pytest.raises(EnvFilePermissionError):
            await env.read_file('locked.txt')
    finally:
        target.chmod(0o644)  # let tmp_path cleanup remove it


@skip_if_root
async def test_write_into_readonly_dir_raises_permission(tmp_path: Path) -> None:
    box = tmp_path / 'readonly'
    box.mkdir()
    box.chmod(0o555)
    try:
        env = LocalEnvironment(root=str(tmp_path))
        with pytest.raises(EnvFilePermissionError):
            await env.write_file('readonly/new.txt', b'nope')
    finally:
        box.chmod(0o755)


async def test_read_symlink_loop_raises_read_error(tmp_path: Path) -> None:
    # A symlink cycle resolves without error but fails at read time with a generic OSError
    # (ELOOP) -- not one of the mapped subclasses -- exercising the catch-all read path.
    a = tmp_path / 'a'
    b = tmp_path / 'b'
    a.symlink_to(b)
    b.symlink_to(a)

    env = LocalEnvironment(root=str(tmp_path))
    with pytest.raises(EnvFileReadError):
        await env.read_file('a')
