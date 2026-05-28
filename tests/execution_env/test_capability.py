"""Capability-layer tests for ExecutionEnv: error routing, truncation, and edit logic.

These test the *capability*, not any backend, so they drive in-memory fakes:

- `_RaisingEnvironment` makes the env raise a preset exception (tests error routing:
  model-fixable errors -> `ModelRetry`, infrastructure errors -> propagate).
- `_StoreEnvironment` holds bytes in memory (tests read/edit/write behavior without a
  real filesystem).

Backend correctness (that `LocalEnvironment` actually produces these errors) is verified
once, in `tests/environments/`. The single end-to-end test below wires a real
`LocalEnvironment` through an agent to prove the capability composes.
"""

from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic_ai import Agent, ModelResponse, ModelRetry, RunContext, TextPart
from pydantic_ai.messages import ModelMessage, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.environments.abstract import AbstractEnvironment, AbstractFile
from pydantic_ai_harness.environments.exceptions import (
    EnvIsADirectoryError,
    EnvNotADirectoryError,
    EnvNotFoundError,
    EnvPermissionError,
    EnvReadError,
    EnvWriteError,
    ExecutionEnvironmentError,
    PathEscapeError,
)
from pydantic_ai_harness.environments.local import LocalEnvironment
from pydantic_ai_harness.execution_env import ExecutionEnv


@dataclass(kw_only=True)
class _RaisingEnvironment(AbstractEnvironment):
    """Environment whose `read_file`/`write_file` always raise a preset exception."""

    error: Exception

    async def read_file(self, path: str) -> bytes:
        raise self.error

    async def write_file(self, path: str, data: bytes) -> None:
        raise self.error

    async def ls(self, path: str) -> list[AbstractFile]:
        raise self.error


@dataclass(kw_only=True)
class _StoreEnvironment(AbstractEnvironment):
    """In-memory environment: read_file returns stored bytes, write_file overwrites them."""

    data: bytes

    async def read_file(self, path: str) -> bytes:
        return self.data

    async def write_file(self, path: str, data: bytes) -> None:
        self.data = data

    async def ls(self, path: str) -> list[AbstractFile]:
        return [AbstractFile(name='sub', is_directory=True), AbstractFile(name='file.txt', is_directory=False)]


def _ctx() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


async def _call_read_file(environment: AbstractEnvironment, path: str = 'f.txt') -> object:
    """Invoke the read_file tool through its toolset."""
    toolset = ExecutionEnv(environment=environment).get_toolset()
    ctx = _ctx()
    tools = await toolset.get_tools(ctx)
    return await toolset.call_tool('read_file', {'path': path}, ctx, tools['read_file'])


async def _read(data: bytes, *, offset: int | None = None, limit: int | None = None) -> str:
    """Invoke read_file with preset bytes and optional offset/limit, return the text."""
    toolset = ExecutionEnv(environment=_StoreEnvironment(root='/x', data=data)).get_toolset()
    ctx = _ctx()
    tools = await toolset.get_tools(ctx)
    args: dict[str, object] = {'path': 'f.txt'}
    if offset is not None:
        args['offset'] = offset
    if limit is not None:
        args['limit'] = limit
    return await toolset.call_tool('read_file', args, ctx, tools['read_file'])


async def _write(environment: AbstractEnvironment, *, data: str = 'hi', path: str = 'f.txt') -> object:
    """Invoke the write_file tool through its toolset."""
    toolset = ExecutionEnv(environment=environment).get_toolset()
    ctx = _ctx()
    tools = await toolset.get_tools(ctx)
    return await toolset.call_tool('write_file', {'path': path, 'data': data}, ctx, tools['write_file'])


async def _edit(
    environment: AbstractEnvironment,
    *,
    old_string: str,
    new_string: str,
    replace_all: bool | None = None,
    path: str = 'f.txt',
) -> object:
    """Invoke the edit_file tool through its toolset."""
    toolset = ExecutionEnv(environment=environment).get_toolset()
    ctx = _ctx()
    tools = await toolset.get_tools(ctx)
    args: dict[str, object] = {'path': path, 'old_string': old_string, 'new_string': new_string}
    if replace_all is not None:
        args['replace_all'] = replace_all
    return await toolset.call_tool('edit_file', args, ctx, tools['edit_file'])


async def _ls(environment: AbstractEnvironment, *, path: str = 'd', limit: int | None = None) -> object:
    """Invoke the ls tool through its toolset."""
    toolset = ExecutionEnv(environment=environment).get_toolset()
    ctx = _ctx()
    tools = await toolset.get_tools(ctx)
    args: dict[str, object] = {'path': path}
    if limit is not None:
        args['limit'] = limit
    return await toolset.call_tool('ls', args, ctx, tools['ls'])


# --- read_file: error routing -------------------------------------------------


@pytest.mark.parametrize(
    'error',
    [
        EnvNotFoundError('not found'),
        EnvPermissionError('not readable'),
        EnvIsADirectoryError('is a directory'),
        EnvNotADirectoryError('not a directory'),
        PathEscapeError('outside root'),
    ],
)
async def test_recoverable_errors_become_model_retry(error: ExecutionEnvironmentError) -> None:
    with pytest.raises(ModelRetry):
        await _call_read_file(_RaisingEnvironment(root='/x', error=error))


async def test_infra_error_propagates() -> None:
    with pytest.raises(EnvReadError):
        await _call_read_file(_RaisingEnvironment(root='/x', error=EnvReadError('disk on fire')))


async def test_non_utf8_file_becomes_model_retry() -> None:
    with pytest.raises(ModelRetry):
        await _call_read_file(_StoreEnvironment(root='/x', data=b'\xff\xfe\x00'))


async def test_toolset_decodes_and_returns_text() -> None:
    assert await _call_read_file(_StoreEnvironment(root='/x', data=b'hello')) == 'hello'


# --- read_file: offset/limit + truncation formatting --------------------------


async def test_read_no_offset_returns_full_text() -> None:
    assert await _read(b'a\nb\nc') == 'a\nb\nc'


async def test_read_offset_and_limit_window() -> None:
    out = await _read(b'a\nb\nc\nd\ne', offset=2, limit=2)
    assert out == 'b\nc\n\n[2 more lines in file. Use offset=4 to continue.]'


async def test_read_limit_stops_early_adds_more_note() -> None:
    out = await _read(b'a\nb\nc\nd\ne', limit=2)
    assert out == 'a\nb\n\n[3 more lines in file. Use offset=3 to continue.]'


@pytest.mark.parametrize('offset,limit', [(0, None), (-1, None), (None, 0)])
async def test_read_invalid_offset_or_limit_is_model_retry(offset: int | None, limit: int | None) -> None:
    with pytest.raises(ModelRetry):
        await _read(b'a\nb', offset=offset, limit=limit)


async def test_read_offset_beyond_eof_is_model_retry() -> None:
    with pytest.raises(ModelRetry):
        await _read(b'a\nb', offset=5)


async def test_read_truncates_by_line_cap_with_note() -> None:
    data = '\n'.join(str(i) for i in range(2001)).encode('utf-8')
    out = await _read(data)
    assert out.endswith('[Showing lines 1-2000 of 2001. Use offset=2001 to continue.]')


async def test_read_truncates_by_byte_cap_with_note() -> None:
    data = '\n'.join('x' * 1024 for _ in range(100)).encode('utf-8')
    out = await _read(data)
    assert '(50.0KB limit). Use offset=' in out


async def test_read_first_line_too_big_is_omitted() -> None:
    data = ('x' * (60 * 1024) + '\nrest').encode('utf-8')
    out = await _read(data)
    assert out == '[Line 1 is 60.0KB, exceeds the 50.0KB limit and was omitted.]'


# --- write_file: error routing ------------------------------------------------


async def test_write_permission_error_is_model_retry() -> None:
    with pytest.raises(ModelRetry):
        await _write(_RaisingEnvironment(root='/x', error=EnvPermissionError('read only')))


async def test_write_infra_error_propagates() -> None:
    with pytest.raises(EnvWriteError):
        await _write(_RaisingEnvironment(root='/x', error=EnvWriteError('disk on fire')))


# --- edit_file ----------------------------------------------------------------


async def test_edit_replaces_unique_occurrence() -> None:
    env = _StoreEnvironment(root='/x', data=b'hello world')
    result = await _edit(env, old_string='world', new_string='there')
    assert env.data == b'hello there'
    assert result == "Replaced 1 occurrence in 'f.txt'."


async def test_edit_replace_all_changes_every_occurrence() -> None:
    env = _StoreEnvironment(root='/x', data=b'a a a')
    result = await _edit(env, old_string='a', new_string='b', replace_all=True)
    assert env.data == b'b b b'
    assert result == "Replaced 3 occurrences in 'f.txt'."


async def test_edit_multiple_matches_without_replace_all_is_model_retry() -> None:
    env = _StoreEnvironment(root='/x', data=b'a a a')
    with pytest.raises(ModelRetry):
        await _edit(env, old_string='a', new_string='b')
    assert env.data == b'a a a'  # untouched when rejected


async def test_edit_zero_matches_is_model_retry() -> None:
    env = _StoreEnvironment(root='/x', data=b'hello')
    with pytest.raises(ModelRetry):
        await _edit(env, old_string='goodbye', new_string='hi')
    assert env.data == b'hello'


async def test_edit_noop_is_model_retry() -> None:
    with pytest.raises(ModelRetry):
        await _edit(_StoreEnvironment(root='/x', data=b'hello'), old_string='hello', new_string='hello')


async def test_edit_missing_file_is_model_retry() -> None:
    with pytest.raises(ModelRetry):
        await _edit(_RaisingEnvironment(root='/x', error=EnvNotFoundError('nope')), old_string='a', new_string='b')


async def test_edit_path_escape_is_model_retry() -> None:
    with pytest.raises(ModelRetry):
        await _edit(_RaisingEnvironment(root='/x', error=PathEscapeError('outside')), old_string='a', new_string='b')


async def test_edit_non_utf8_file_is_model_retry() -> None:
    with pytest.raises(ModelRetry):
        await _edit(_StoreEnvironment(root='/x', data=b'\xff\xfe'), old_string='a', new_string='b')


async def test_edit_read_infra_error_propagates() -> None:
    with pytest.raises(EnvReadError):
        await _edit(_RaisingEnvironment(root='/x', error=EnvReadError('disk on fire')), old_string='a', new_string='b')


async def test_edit_write_permission_error_is_model_retry() -> None:
    @dataclass(kw_only=True)
    class _ReadOkWriteDenied(_StoreEnvironment):
        async def write_file(self, path: str, data: bytes) -> None:
            raise EnvPermissionError('read only')

    with pytest.raises(ModelRetry):
        await _edit(_ReadOkWriteDenied(root='/x', data=b'hello world'), old_string='world', new_string='there')


async def test_edit_write_infra_error_propagates() -> None:
    @dataclass(kw_only=True)
    class _ReadOkWriteFails(_StoreEnvironment):
        async def write_file(self, path: str, data: bytes) -> None:
            raise EnvWriteError('disk on fire')

    with pytest.raises(EnvWriteError):
        await _edit(_ReadOkWriteFails(root='/x', data=b'hello world'), old_string='world', new_string='there')


# --- ls -----------------------------------------------------------------------


async def test_ls_formats_directory_suffix() -> None:
    # `/` appended to directories, plain name otherwise -- exercises both branches.
    assert await _ls(_StoreEnvironment(root='/x', data=b'')) == ['sub/', 'file.txt']


async def test_ls_limit_truncates_listing() -> None:
    # `limit` caps the entries at the presentation layer; the tail is dropped.
    assert await _ls(_StoreEnvironment(root='/x', data=b''), limit=1) == ['sub/']


@pytest.mark.parametrize(
    'error',
    [
        EnvNotFoundError('not found'),
        EnvPermissionError('not listable'),
        EnvIsADirectoryError('is a directory'),
        EnvNotADirectoryError('not a directory'),
        PathEscapeError('outside root'),
    ],
)
async def test_ls_recoverable_errors_become_model_retry(error: ExecutionEnvironmentError) -> None:
    with pytest.raises(ModelRetry):
        await _ls(_RaisingEnvironment(root='/x', error=error))


async def test_ls_infra_error_propagates() -> None:
    with pytest.raises(EnvReadError):
        await _ls(_RaisingEnvironment(root='/x', error=EnvReadError('disk on fire')))


# --- end-to-end: capability through a real backend + agent --------------------


async def test_execution_env_capability_read_file(tmp_path: Path) -> None:
    file_name = 'test.txt'
    (tmp_path / file_name).write_text('Hello, world!')

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        already_read = any(
            isinstance(part, ToolReturnPart) and part.tool_name == 'read_file' for msg in messages for part in msg.parts
        )
        if already_read:
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='read_file', args={'path': file_name})])

    agent = Agent(
        FunctionModel(model_fn), capabilities=[ExecutionEnv(environment=LocalEnvironment(root=str(tmp_path)))]
    )
    result = await agent.run(f'Read the file {file_name} and return the contents.')

    returns = [
        part.content
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'read_file'
    ]
    assert returns == ['Hello, world!']
