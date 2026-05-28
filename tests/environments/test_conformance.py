"""Backend conformance suite: the contract every `AbstractEnvironment` must satisfy.

Each test takes the parametrized `environment` fixture and runs against every backend.
Files are seeded on the host `tmp_path` and exercised through `environment`; the two
point at the same directory (see `conftest.py`). Backend-specific behavior (symlink
resolution, POSIX permissions) lives in the per-backend test modules, not here.
"""

from pathlib import Path

import pytest

from pydantic_ai_harness.environments.abstract import AbstractEnvironment
from pydantic_ai_harness.environments.exceptions import (
    EnvIsADirectoryError,
    EnvNotADirectoryError,
    EnvNotFoundError,
    EnvWriteError,
    PathEscapeError,
)


async def test_write_then_read_round_trips(environment: AbstractEnvironment) -> None:
    await environment.write_file('note.txt', b'hello')
    assert await environment.read_file('note.txt') == b'hello'


async def test_read_returns_raw_bytes(environment: AbstractEnvironment, tmp_path: Path) -> None:
    (tmp_path / 'data.bin').write_bytes(b'\x00\xff\xfe')
    assert await environment.read_file('data.bin') == b'\x00\xff\xfe'


async def test_write_creates_missing_file(environment: AbstractEnvironment, tmp_path: Path) -> None:
    await environment.write_file('fresh.txt', b'new')
    assert (tmp_path / 'fresh.txt').read_bytes() == b'new'


async def test_read_missing_file_raises_not_found(environment: AbstractEnvironment) -> None:
    with pytest.raises(EnvNotFoundError):
        await environment.read_file('does-not-exist.txt')


async def test_read_directory_raises_is_a_directory(environment: AbstractEnvironment, tmp_path: Path) -> None:
    (tmp_path / 'subdir').mkdir()
    with pytest.raises(EnvIsADirectoryError):
        await environment.read_file('subdir')


async def test_read_through_file_component_raises_not_a_directory(
    environment: AbstractEnvironment, tmp_path: Path
) -> None:
    # A path that treats a regular file as if it were a directory.
    (tmp_path / 'file.txt').write_bytes(b'x')
    with pytest.raises(EnvNotADirectoryError):
        await environment.read_file('file.txt/inner')


async def test_write_onto_directory_raises_write_error(environment: AbstractEnvironment, tmp_path: Path) -> None:
    # Writing bytes where a directory already exists is an I/O failure, not a model-fixable
    # path problem -> the generic write error, which the capability layer propagates.
    (tmp_path / 'adir').mkdir()
    with pytest.raises(EnvWriteError):
        await environment.write_file('adir', b'nope')


async def test_relative_escape_read_raises(environment: AbstractEnvironment) -> None:
    with pytest.raises(PathEscapeError):
        await environment.read_file('../escape.txt')


async def test_relative_escape_write_raises(environment: AbstractEnvironment) -> None:
    with pytest.raises(PathEscapeError):
        await environment.write_file('../escape.txt', b'nope')


async def test_ls_lists_entries_with_types(environment: AbstractEnvironment, tmp_path: Path) -> None:
    (tmp_path / 'a.txt').write_bytes(b'x')
    (tmp_path / 'subdir').mkdir()
    listing = await environment.ls('.')
    assert {(f.name, f.is_directory) for f in listing} == {('a.txt', False), ('subdir', True)}


async def test_ls_missing_directory_raises_not_found(environment: AbstractEnvironment) -> None:
    with pytest.raises(EnvNotFoundError):
        await environment.ls('does-not-exist')


async def test_ls_on_a_file_raises_not_a_directory(environment: AbstractEnvironment, tmp_path: Path) -> None:
    (tmp_path / 'file.txt').write_bytes(b'x')
    with pytest.raises(EnvNotADirectoryError):
        await environment.ls('file.txt')


async def test_ls_relative_escape_raises(environment: AbstractEnvironment) -> None:
    with pytest.raises(PathEscapeError):
        await environment.ls('..')
