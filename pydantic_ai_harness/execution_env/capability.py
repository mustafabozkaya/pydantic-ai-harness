"""Capability that exposes the execution environment to the agent."""

from dataclasses import dataclass

from pydantic_ai import FunctionToolset
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

from ..environments.abstract import AbstractEnvironment
from ._toolset import build_toolset


@dataclass
class ExecutionEnv(AbstractCapability[AgentDepsT]):
    """Capability that exposes the execution environment to the agent.

    Bounds are applied at this presentation layer, not in the backend: `read_file`
    fetches the whole file then windows/truncates it, and `ls` fetches the whole
    listing then caps it. On a remote backend this ships bytes/entries over the wire
    only to discard the tail -- a real cost we accept for now rather than push limits
    into the backend contract, which would grow the surface area every backend must
    implement correctly. Keeping every tool consistent here is the deliberate trade-off;
    revisit it for all of them together if a remote backend's cost says otherwise.

    The tools themselves live in `_toolset.py`; this class owns the generic `AgentDepsT`
    and the environment, and delegates tool construction to `build_toolset`.
    """

    environment: AbstractEnvironment

    def get_toolset(self) -> FunctionToolset[AgentDepsT]:
        """Get the toolset for the execution environment."""
        return build_toolset(self.environment, FunctionToolset[AgentDepsT]())
