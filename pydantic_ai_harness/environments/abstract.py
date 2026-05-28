"""Abstract base class for all execution environments."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(kw_only=True, frozen=True)
class AbstractFile:
    """A file in the environment."""

    name: str
    """The file's name."""

    is_directory: bool
    """Whether the file is a directory."""


@dataclass(kw_only=True, frozen=True)
class AbstractMatch:
    """A line in a file."""

    path: str
    """The path to the file."""

    line: str
    """The line's text."""

    lineno: int
    """The line's number."""


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
            EnvNotFoundError: No file exists at `path`.
            EnvIsADirectoryError: `path` is a directory, not a file.
            EnvNotADirectoryError: A component of `path` is not a directory.
            EnvPermissionError: The backend may not read `path`.
            EnvReadError: Any other I/O failure (nothing builtin leaks).
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
            EnvPermissionError: The backend may not write `path`.
            EnvWriteError: Any other I/O failure (nothing builtin leaks).
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def ls(self, path: str) -> list[AbstractFile]:
        """List the contents of a directory.

        Args:
            path: Directory path, resolved against and confined to `root`.

        Returns:
            A list of files and directories in the directory.

        Raises:
            PathEscapeError: `path` resolves outside `root`.
            EnvNotFoundError: No directory exists at `path`.
            EnvNotADirectoryError: A component of `path` is not a directory.
            EnvPermissionError: The backend may not list `path`.
            EnvReadError: Any other I/O failure (nothing builtin leaks).
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def grep(self, path: str, pattern: str) -> list[AbstractMatch]:
        """Search a file for a pattern.

        Args:
            path: File path, resolved against and confined to `root`.
            pattern: The pattern to search for.

        Returns:
            A list of matches.

        Raises:
            PathEscapeError: `path` resolves outside `root`.
            EnvNotFoundError: No file exists at `path`.
            EnvIsADirectoryError: `path` is a directory, not a file.
            EnvNotADirectoryError: A component of `path` is not a directory.
            EnvPermissionError: The backend may not grep `path`.
            EnvReadError: Any other I/O failure (nothing builtin leaks).
        """
        raise NotImplementedError  # pragma: no cover
