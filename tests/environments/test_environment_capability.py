from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic_ai import Agent, ModelResponse, ModelRetry, RunContext, TextPart
from pydantic_ai.messages import ModelMessage, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.environments.abstract import AbstractEnvironment
from pydantic_ai_harness.environments.exceptions import (
    EnvFileIsADirectoryError,
    EnvFileNotADirectoryError,
    EnvFileNotFoundError,
    EnvFilePermissionError,
    EnvFileReadError,
    EnvFileWriteError,
    ExecutionEnvironmentError,
    PathEscapeError,
)
from pydantic_ai_harness.environments.local import LocalEnvironment
from pydantic_ai_harness.execution_env import ExecutionEnv


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


async def test_execution_env_capability_read_file(tmp_path: Path) -> None:
    # Let us write a file into the path first
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

    result = await agent.run(
        f'Read the file {file_name} and return the contents.',
    )

    returns = [
        part.content
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'read_file'
    ]

    assert returns == ['Hello, world!']


# --- error-routing tests (toolset level) -------------------------------------
#
# These prove the capability layer's contract: errors the model can fix by
# changing its argument become `ModelRetry`; infrastructure failures propagate.
# We drive a fake environment so we test the *mapping* in isolation, independent
# of how `LocalEnvironment` happens to produce each error (tested separately).


@dataclass(kw_only=True)
class _RaisingEnvironment(AbstractEnvironment):
    """Environment whose `read_file` always raises a preset exception."""

    error: Exception

    async def read_file(self, path: str) -> bytes:
        raise self.error

    async def write_file(self, path: str, data: bytes) -> None:
        raise self.error


@dataclass(kw_only=True)
class _BytesEnvironment(AbstractEnvironment):
    """Environment whose `read_file` always returns preset bytes."""

    data: bytes

    async def read_file(self, path: str) -> bytes:
        return self.data

    async def write_file(self, path: str, data: bytes) -> None:
        return None


async def _call_read_file(environment: AbstractEnvironment, path: str = 'f.txt') -> object:
    """Invoke the capability's `read_file` tool directly through its toolset."""
    toolset = ExecutionEnv(environment=environment).get_toolset()
    ctx: RunContext[None] = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    tools = await toolset.get_tools(ctx)
    return await toolset.call_tool('read_file', {'path': path}, ctx, tools['read_file'])


@pytest.mark.parametrize(
    'error',
    [
        EnvFileNotFoundError('not found'),
        EnvFilePermissionError('not readable'),
        EnvFileIsADirectoryError('is a directory'),
        EnvFileNotADirectoryError('not a directory'),
        PathEscapeError('outside root'),
    ],
)
async def test_recoverable_errors_become_model_retry(error: ExecutionEnvironmentError) -> None:
    # The model can fix these by choosing a different path -> ModelRetry.
    with pytest.raises(ModelRetry):
        await _call_read_file(_RaisingEnvironment(root='/x', error=error))


async def test_infra_error_propagates() -> None:
    # Not the model's fault and not fixable by retrying -> propagate (no ModelRetry).
    with pytest.raises(EnvFileReadError):
        await _call_read_file(_RaisingEnvironment(root='/x', error=EnvFileReadError('disk on fire')))


async def test_non_utf8_file_becomes_model_retry() -> None:
    # The model pointed at a binary file; it can pick another -> ModelRetry.
    with pytest.raises(ModelRetry):
        await _call_read_file(_BytesEnvironment(root='/x', data=b'\xff\xfe\x00'))


async def test_toolset_decodes_and_returns_text() -> None:
    result = await _call_read_file(_BytesEnvironment(root='/x', data=b'hello'))
    assert result == 'hello'


# --- offset/limit + truncation formatting -----------------------------------
#
# These drive read_file directly with preset bytes so we test the windowing and
# the four continuation-note shapes, independent of any real filesystem.


async def _read(data: bytes, *, offset: int | None = None, limit: int | None = None) -> str:
    """Invoke read_file with preset bytes and optional offset/limit, return the text."""
    toolset = ExecutionEnv(environment=_BytesEnvironment(root='/x', data=data)).get_toolset()
    ctx: RunContext[None] = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    tools = await toolset.get_tools(ctx)
    args: dict[str, object] = {'path': 'f.txt'}
    if offset is not None:
        args['offset'] = offset
    if limit is not None:
        args['limit'] = limit
    result = await toolset.call_tool('read_file', args, ctx, tools['read_file'])
    return result


async def test_read_no_offset_returns_full_text() -> None:
    assert await _read(b'a\nb\nc') == 'a\nb\nc'


async def test_read_offset_and_limit_window() -> None:
    # 1-indexed lines: 1=a 2=b 3=c 4=d 5=e; offset=2, limit=2 -> b, c. Lines 4-5 remain,
    # so the user-limit-stopped-early note fires pointing at the next line.
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
    # 2001 short lines: over the line cap, well under the byte cap.
    data = '\n'.join(str(i) for i in range(2001)).encode('utf-8')
    out = await _read(data)
    assert out.endswith('[Showing lines 1-2000 of 2001. Use offset=2001 to continue.]')


async def test_read_truncates_by_byte_cap_with_note() -> None:
    # 100 lines of 1KB each: ~100KB but only 100 lines, so the byte cap wins.
    data = '\n'.join('x' * 1024 for _ in range(100)).encode('utf-8')
    out = await _read(data)
    assert '(50.0KB limit). Use offset=' in out


async def test_read_first_line_too_big_is_omitted() -> None:
    data = ('x' * (60 * 1024) + '\nrest').encode('utf-8')
    out = await _read(data)
    assert out == '[Line 1 is 60.0KB, exceeds the 50.0KB limit and was omitted.]'


# --- edit_file -----------------------------------------------------------------
#
# A string-replace edit is capability-layer composition over the env's read_file +
# write_file. We use a stateful fake that actually stores bytes so we can assert the
# new file contents, not just that the tool returned without error.


@dataclass(kw_only=True)
class _StoreEnvironment(AbstractEnvironment):
    """In-memory environment: read_file returns stored bytes, write_file overwrites them."""

    data: bytes

    async def read_file(self, path: str) -> bytes:
        return self.data

    async def write_file(self, path: str, data: bytes) -> None:
        self.data = data


async def _edit(
    environment: AbstractEnvironment,
    *,
    old_string: str,
    new_string: str,
    replace_all: bool | None = None,
    path: str = 'f.txt',
) -> object:
    """Invoke the capability's edit_file tool through its toolset."""
    toolset = ExecutionEnv(environment=environment).get_toolset()
    ctx: RunContext[None] = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    tools = await toolset.get_tools(ctx)
    args: dict[str, object] = {'path': path, 'old_string': old_string, 'new_string': new_string}
    if replace_all is not None:
        args['replace_all'] = replace_all
    return await toolset.call_tool('edit_file', args, ctx, tools['edit_file'])


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
    # The file is left untouched when the edit is rejected.
    assert env.data == b'a a a'


async def test_edit_zero_matches_is_model_retry() -> None:
    env = _StoreEnvironment(root='/x', data=b'hello')
    with pytest.raises(ModelRetry):
        await _edit(env, old_string='goodbye', new_string='hi')
    assert env.data == b'hello'


async def test_edit_noop_is_model_retry() -> None:
    env = _StoreEnvironment(root='/x', data=b'hello')
    with pytest.raises(ModelRetry):
        await _edit(env, old_string='hello', new_string='hello')


async def test_edit_missing_file_is_model_retry() -> None:
    with pytest.raises(ModelRetry):
        await _edit(_RaisingEnvironment(root='/x', error=EnvFileNotFoundError('nope')), old_string='a', new_string='b')


async def test_edit_path_escape_is_model_retry() -> None:
    with pytest.raises(ModelRetry):
        await _edit(_RaisingEnvironment(root='/x', error=PathEscapeError('outside')), old_string='a', new_string='b')


async def test_edit_non_utf8_file_is_model_retry() -> None:
    with pytest.raises(ModelRetry):
        await _edit(_StoreEnvironment(root='/x', data=b'\xff\xfe'), old_string='a', new_string='b')


async def test_edit_write_permission_error_is_model_retry() -> None:
    # read succeeds, the replacement is valid, but the backend refuses the write.
    @dataclass(kw_only=True)
    class _ReadOkWriteDenied(AbstractEnvironment):
        async def read_file(self, path: str) -> bytes:
            return b'hello world'

        async def write_file(self, path: str, data: bytes) -> None:
            raise EnvFilePermissionError('read only')

    with pytest.raises(ModelRetry):
        await _edit(_ReadOkWriteDenied(root='/x'), old_string='world', new_string='there')


async def test_edit_read_infra_error_propagates() -> None:
    # An infrastructure read failure is not the model's fault -> propagate, no ModelRetry.
    with pytest.raises(EnvFileReadError):
        await _edit(
            _RaisingEnvironment(root='/x', error=EnvFileReadError('disk on fire')), old_string='a', new_string='b'
        )


# --- write_file tool error routing --------------------------------------------


async def _write(environment: AbstractEnvironment, *, data: str = 'hi', path: str = 'f.txt') -> object:
    """Invoke the capability's write_file tool through its toolset."""
    toolset = ExecutionEnv(environment=environment).get_toolset()
    ctx: RunContext[None] = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    tools = await toolset.get_tools(ctx)
    return await toolset.call_tool('write_file', {'path': path, 'data': data}, ctx, tools['write_file'])


async def test_write_permission_error_is_model_retry() -> None:
    with pytest.raises(ModelRetry):
        await _write(_RaisingEnvironment(root='/x', error=EnvFilePermissionError('read only')))


async def test_write_infra_error_propagates() -> None:
    with pytest.raises(EnvFileWriteError):
        await _write(_RaisingEnvironment(root='/x', error=EnvFileWriteError('disk on fire')))


async def test_edit_write_infra_error_propagates() -> None:
    # An infrastructure write failure is not the model's fault -> propagate, no ModelRetry.
    @dataclass(kw_only=True)
    class _ReadOkWriteFails(AbstractEnvironment):
        async def read_file(self, path: str) -> bytes:
            return b'hello world'

        async def write_file(self, path: str, data: bytes) -> None:
            raise EnvFileWriteError('disk on fire')

    with pytest.raises(EnvFileWriteError):
        await _edit(_ReadOkWriteFails(root='/x'), old_string='world', new_string='there')
