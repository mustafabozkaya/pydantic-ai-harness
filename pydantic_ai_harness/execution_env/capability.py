"""Capability that exposes the execution environment to the agent."""

from dataclasses import dataclass
from typing import Annotated

from pydantic import Field
from pydantic_ai import FunctionToolset, ModelRetry
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

from ..environments.abstract import AbstractEnvironment
from ..environments.exceptions import (
    EnvFileIsADirectoryError,
    EnvFileNotADirectoryError,
    EnvFileNotFoundError,
    EnvFilePermissionError,
    EnvFileReadError,
    EnvFileWriteError,
    PathEscapeError,
)


@dataclass
class ExecutionEnv(AbstractCapability[AgentDepsT]):
    """Capability that exposes the execution environment to the agent."""

    environment: AbstractEnvironment

    def get_toolset(self) -> FunctionToolset[AgentDepsT]:
        """Get the toolset for the execution environment."""
        toolset = FunctionToolset[AgentDepsT]()

        async def read_file(
            path: Annotated[str, Field(description='Path to the file to read, relative to the workspace root.')],
            offset: Annotated[int | None, Field(description='Line number to start reading from (1-indexed)')] = None,
            limit: Annotated[int | None, Field(description='Maximum number of lines to read')] = None,
        ) -> str:
            """Read a file from the execution environment."""
            # offset/limit are 1-indexed line counts at the boundary (to agree with
            # grep -n, editors, stack traces). pi leaned on JS treating 0 as falsy;
            # Python 0 is a real value, so we validate explicitly and bounce mistakes
            # back to the model instead of silently clamping.

            if offset is not None and offset < 1:
                raise ModelRetry(f'offset must be >= 1 (lines are 1-indexed), got {offset}')

            if limit is not None and limit < 1:
                raise ModelRetry(f'limit must be >= 1, got {limit}')

            try:
                data = await self.environment.read_file(path)
                text = data.decode('utf-8')

            except (
                EnvFileNotFoundError,
                EnvFilePermissionError,
                EnvFileIsADirectoryError,
                EnvFileNotADirectoryError,
                PathEscapeError,
            ) as e:
                # TODO(observability): PathEscapeError is the one security-relevant
                # case here (a boundary-crossing attempt). When we design the
                # observability story, consider emitting a Logfire/OTel span or event
                # for it before retrying -- NOT stdlib logging or warnings.warn
                # (pydantic-ai uses neither for runtime events).
                raise ModelRetry(str(e)) from e
            except (EnvFileReadError,):
                # TODO: This should be a ToolFailed error when I merge that in
                # catching and re raising here to show the boundary where we change it
                raise
            except UnicodeDecodeError as e:
                raise ModelRetry(str(e)) from e

            # Split on '\n' only, NOT str.splitlines(): splitlines() also breaks on
            # '\r', '\v', '\f', and Unicode line/paragraph separators, and collapses a
            # trailing newline. That would make our line numbers disagree with what
            # editors, grep -n, and the model expect. Plain '\n' keeps numbering honest
            # (cost: a trailing '\n' yields a final '' element, so total_lines counts it).
            lines = text.split('\n')
            total_lines = len(lines)

            start = offset - 1 if offset is not None else 0

            if start >= total_lines:
                raise ModelRetry(f'offset {offset} is beyond end of file ({total_lines} lines total)')

            end = min(start + limit, total_lines) if limit is not None else total_lines
            window = lines[start:end]

            return '\n'.join(window)

        async def write_file(
            path: Annotated[str, Field(description='Path to the file, relative to the workspace root.')],
            data: Annotated[str, Field(description='Data to write to the file.')],
        ) -> None:
            """Write a file to the execution environment."""
            try:
                await self.environment.write_file(path, data.encode('utf-8'))
            except EnvFilePermissionError as e:
                raise ModelRetry(str(e)) from e
            except (EnvFileWriteError,):
                # TODO: This should be a ToolFailed error when I merge that in
                # catching and re raising here to show the boundary where we change it
                raise

        toolset.add_function(read_file, description='Read a file from the execution environment.')
        toolset.add_function(write_file, description='Write a file to the execution environment.')

        return toolset
