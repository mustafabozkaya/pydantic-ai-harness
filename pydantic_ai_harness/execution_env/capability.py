"""Capability that exposes the execution environment to the agent."""

from dataclasses import dataclass
from typing import Annotated

from pydantic import Field
from pydantic_ai import FunctionToolset, ModelRetry
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset

from ..environments.abstract import AbstractEnvironment
from ..environments.exceptions import (
    EnvFileIsADirectoryError,
    EnvFileNotADirectoryError,
    EnvFileNotFoundError,
    EnvFilePermissionError,
    EnvFileReadError,
)


@dataclass
class ExecutionEnv(AbstractCapability[AgentDepsT]):
    """Capability that exposes the execution environment to the agent."""

    environment: AbstractEnvironment

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Get the toolset for the execution environment."""
        toolset = FunctionToolset[AgentDepsT]()

        async def read_file(
            path: Annotated[str, Field(description='Path to the file, relative to the workspace root.')],
        ) -> str:
            """Read a file from the execution environment."""
            try:
                bytes = await self.environment.read_file(path)

                return bytes.decode('utf-8')

            except (
                EnvFileNotFoundError,
                EnvFilePermissionError,
                EnvFileIsADirectoryError,
                EnvFileNotADirectoryError,
            ) as e:
                raise ModelRetry(f'{str(e)}')
            except (EnvFileReadError,):
                raise

        toolset.add_function(read_file, name='read_file', description='Read a file from the execution environment.')

        return toolset
