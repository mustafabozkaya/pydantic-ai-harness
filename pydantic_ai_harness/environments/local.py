"""Local environment using the local filesystem."""

from dataclasses import dataclass
from pathlib import Path

from .abstract import AbstractEnvironment
from .exceptions import PathEscapeError


@dataclass
class LocalEnvironment(AbstractEnvironment):
    """Local environment using the local filesystem."""

    root: str

    async def read_file(self, path: str) -> bytes:
        """Read a file from the local filesystem."""
        root = Path(self.root).resolve()
        resolved_path = Path(root, path).resolve()

        if not resolved_path.is_relative_to(root):
            raise PathEscapeError

        return resolved_path.read_bytes()
