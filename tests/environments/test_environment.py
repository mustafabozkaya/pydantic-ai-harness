from pathlib import Path

import pytest

from pydantic_ai_harness.environments.exceptions import PathEscapeError
from pydantic_ai_harness.environments.local import LocalEnvironment


async def test_read_file(tmp_path: Path) -> None:
    local_env = LocalEnvironment(root=str(tmp_path))
    (tmp_path / 'test.txt').write_text('Hello, world!')

    assert await local_env.read_file('test.txt') == b'Hello, world!'


async def test_read_file_does_not_read_outside_root_absolute(tmp_path: Path) -> None:
    local_env = LocalEnvironment(root=str(tmp_path))

    # absolute path
    secret = tmp_path.parent / 'secret.txt'
    secret.write_bytes(b'top secret')

    # try to read the files
    with pytest.raises(PathEscapeError):
        await local_env.read_file(str(secret))


async def test_read_file_does_not_read_outside_root_relative(tmp_path: Path) -> None:
    local_env = LocalEnvironment(root=str(tmp_path))

    # relative path
    secret = '../secret.txt'
    (tmp_path / secret).write_bytes(b'top secret')

    # try to read the files
    with pytest.raises(PathEscapeError):
        await local_env.read_file(str(secret))


async def test_read_file_through_symlinked_root(tmp_path: Path) -> None:
    # A real directory that actually holds the file.
    real_box = tmp_path / 'realbox'
    real_box.mkdir()
    (real_box / 'ok.txt').write_bytes(b'inside the box')

    # A symlink that points at real_box. `link_box.symlink_to(real_box)` creates
    # `link_box` as a pointer to `real_box` (like `ln -s real_box link_box`).
    link_box = tmp_path / 'linkbox'
    link_box.symlink_to(real_box)

    # Root the environment at the SYMLINK -- a perfectly valid but non-canonical
    # root, exactly like rooting at /tmp/... on macOS.
    local_env = LocalEnvironment(root=str(link_box))

    # 'ok.txt' is genuinely inside the box. Reading it must SUCCEED. If the jail
    # compares a resolved candidate against the unresolved root, it wrongly
    # rejects this legit read with PathEscapeError.
    assert await local_env.read_file('ok.txt') == b'inside the box'
