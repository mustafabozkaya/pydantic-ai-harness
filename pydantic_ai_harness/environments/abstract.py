"""Abstract base class for all execution environments."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(kw_only=True)
class AbstractEnvironment(ABC):
    """Abstract base class for all execution environments."""

    root: str
    """The environment's root. Paths are resolved relative to it and confined to it."""

    @abstractmethod
    async def read_file(self, path: str) -> bytes:
        """Return a file's raw, undecoded bytes.

        Args:
            path: File path, resolved against and confined to `root`.

        Returns:
            The file's raw bytes. Decoding to text is the caller's concern.

        Raises:
            PathEscapeError: `path` resolves outside `root`.
            EnvFileNotFoundError: No file exists at `path`.
            EnvFileIsADirectoryError: `path` is a directory, not a file.
            EnvFileNotADirectoryError: A component of `path` is not a directory.
            EnvFilePermissionError: The backend may not read `path`.
            EnvFileReadError: Any other I/O failure (nothing builtin leaks).
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def write_file(self, path: str, data: bytes) -> None:
        """Create or overwrite a file with raw bytes.

        Args:
            path: File path, resolved against and confined to `root`.
            data: Raw bytes to write.

        Raises:
            PathEscapeError: `path` resolves outside `root`.
            EnvFilePermissionError: The backend may not write `path`.
            EnvFileWriteError: Any other I/O failure (nothing builtin leaks).
        """
        raise NotImplementedError  # pragma: no cover
