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


@dataclass(kw_only=True)
class _BytesEnvironment(AbstractEnvironment):
    """Environment whose `read_file` always returns preset bytes."""

    data: bytes

    async def read_file(self, path: str) -> bytes:
        return self.data


async def _call_read_file(environment: AbstractEnvironment, path: str = 'f.txt') -> object:
    """Invoke the capability's `read_file` tool directly through its toolset."""
    toolset = ExecutionEnv(environment=environment).get_toolset()
    assert toolset is not None
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
