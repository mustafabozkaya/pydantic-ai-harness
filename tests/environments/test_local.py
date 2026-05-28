"""LocalEnvironment-specific behavior that does not generalize across backends.

The backend-agnostic contract lives in `test_conformance.py`. Here we test things tied to
LocalEnvironment's implementation: how it resolves symlinked roots, how it maps POSIX
permission errors, and the catch-all I/O error path. A Docker backend would exercise these
concerns differently, so they are deliberately not part of the shared conformance suite.
"""

import os
from pathlib import Path

import pytest
from inline_snapshot import snapshot

from pydantic_ai_harness.environments.abstract import AbstractMatch
from pydantic_ai_harness.environments.exceptions import (
    EnvPermissionError,
    EnvReadError,
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
        with pytest.raises(EnvPermissionError):
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
        with pytest.raises(EnvPermissionError):
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
    with pytest.raises(EnvReadError):
        await env.read_file('a')


@skip_if_root
async def test_ls_unlistable_directory_raises_permission(tmp_path: Path) -> None:
    box = tmp_path / 'locked'
    box.mkdir()
    box.chmod(0o000)
    try:
        env = LocalEnvironment(root=str(tmp_path))
        with pytest.raises(EnvPermissionError):
            await env.ls('locked')
    finally:
        box.chmod(0o755)  # let tmp_path cleanup remove it


async def test_ls_symlink_loop_raises_read_error(tmp_path: Path) -> None:
    # Listing through a symlink cycle fails with a generic OSError (ELOOP) -- not one of the
    # mapped subclasses -- exercising the catch-all path in `ls`.
    a = tmp_path / 'a'
    b = tmp_path / 'b'
    a.symlink_to(b)
    b.symlink_to(a)

    env = LocalEnvironment(root=str(tmp_path))
    with pytest.raises(EnvReadError):
        await env.ls('a')


@skip_if_root
async def test_grep_unreadable_file_in_tree_is_skipped(tmp_path: Path) -> None:
    # A per-file permission failure mid-walk is swallowed (same skip path as a binary file),
    # so the search continues and still returns matches from the readable files. POSIX/uid
    # specific (root bypasses the bits), hence local-only + skip_if_root.
    box = tmp_path / 'dir'
    box.mkdir()
    (box / 'unreadable.txt').write_text('NEEDLE')
    (box / 'readable.txt').write_text('NEEDLE')
    (box / 'no-match.txt').write_text('nothing')
    (box / 'unreadable.txt').chmod(0o000)
    try:
        env = LocalEnvironment(root=str(tmp_path))
        assert await env.grep('dir', 'NEEDLE') == snapshot(
            [AbstractMatch(path='dir/readable.txt', line='NEEDLE', lineno=1)]
        )
    finally:
        (box / 'unreadable.txt').chmod(0o644)  # let tmp_path cleanup remove it


@skip_if_root
async def test_grep_unreadable_top_path_raises_permission(tmp_path: Path) -> None:
    # When the TOP path (the argument) is unreadable, grep raises -- it is not a per-file
    # skip. This exercises the `os.access` guard clause that os.walk could not surface.
    box = tmp_path / 'locked'
    box.mkdir()
    box.chmod(0o000)
    try:
        env = LocalEnvironment(root=str(tmp_path))
        with pytest.raises(EnvPermissionError):
            await env.grep('locked', 'NEEDLE')
    finally:
        box.chmod(0o755)  # let tmp_path cleanup remove it
