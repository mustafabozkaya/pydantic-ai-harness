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
            path: Annotated[str, Field(description='Path to the file, relative to the workspace root.')],
        ) -> str:
            """Read a file from the execution environment."""
            try:
                data = await self.environment.read_file(path)
                return data.decode('utf-8')

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

        toolset.add_function(read_file, description='Read a file from the execution environment.')

        return toolset
