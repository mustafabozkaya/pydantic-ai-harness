"""Local environment using the local filesystem."""

from dataclasses import dataclass
from pathlib import Path

from .abstract import AbstractEnvironment, AbstractFile
from .exceptions import (
    EnvFileIsADirectoryError,
    EnvFileNotADirectoryError,
    EnvFileNotFoundError,
    EnvFilePermissionError,
    EnvFileReadError,
    EnvFileWriteError,
    PathEscapeError,
)


@dataclass(kw_only=True)
class LocalEnvironment(AbstractEnvironment):
    """Filesystem/shell environment backed by the local machine's filesystem.

    Security: `LocalEnvironment` is **NOT a security boundary.** Its `root` path jail is
    *advisory only* -- it guards against accidental escapes and reduces noise, but it is
    bypassable (e.g. via the shell tool, symlinks, or a TOCTOU race) and must not be relied
    on to contain untrusted code. Do not use `LocalEnvironment` to run untrusted,
    model-generated code against a machine you care about. For real isolation, use
    `DockerEnvironment`, where the container is the actual boundary. (See
    `agent_docs/confinement-security-research.md` for the full rationale and references.)
    """

    async def read_file(self, path: str) -> bytes:
        """Read a file from the local filesystem."""
        root = Path(self.root).resolve()
        resolved_path = Path(root, path).resolve()

        # Advisory jail with a KNOWN TOCTOU race: the resolve()+containment check
        # and the read_bytes() below are two separate steps. A symlink swapped in
        # between could redirect the read outside `root`. Accepted for V1 -- a
        # race-free jail needs openat2/O_NOFOLLOW-style primitives. Revisit during
        # hardening. TODO: deepen understanding before relying on this.
        if not resolved_path.is_relative_to(root):
            raise PathEscapeError(f'{path!r} resolves outside the environment root {self.root!r}')

        try:
            return resolved_path.read_bytes()
        except FileNotFoundError as e:
            raise EnvFileNotFoundError(f'{path!r} not found in the environment root {self.root!r}: {str(e)}')
        except PermissionError as e:
            raise EnvFilePermissionError(f'{path!r} is not readable by the environment root {self.root!r}: {str(e)}')
        except IsADirectoryError as e:
            raise EnvFileIsADirectoryError(f'{path!r} is a directory in the environment root {self.root!r}: {str(e)}')
        except NotADirectoryError as e:
            raise EnvFileNotADirectoryError(
                f'{path!r} is not a directory in the environment root {self.root!r}: {str(e)}'
            )
        except OSError as e:
            raise EnvFileReadError(f'{path!r} could not be read in the environment root {self.root!r}: {str(e)}')

    async def write_file(self, path: str, data: bytes) -> None:
        """Write a file to the local filesystem."""
        root = Path(self.root).resolve()
        resolved_path = Path(root, path).resolve()

        if not resolved_path.is_relative_to(root):
            raise PathEscapeError(f'{path!r} resolves outside the environment root {self.root!r}')

        try:
            resolved_path.write_bytes(data)
        # If file isn't present we should create it so we need a test to make sure that is the case
        except PermissionError as e:
            raise EnvFilePermissionError(f'{path!r} is not writable by the environment root {self.root!r}: {str(e)}')
        except OSError as e:
            raise EnvFileWriteError(f'{path!r} could not be written to the environment root {self.root!r}: {str(e)}')

    async def ls(self, path: str) -> list[AbstractFile]:
        """List the contents of a directory."""
        root = Path(self.root).resolve()
        resolved_path = Path(root, path).resolve()

        if not resolved_path.is_relative_to(root):
            raise PathEscapeError(f'{path!r} resolves outside the environment root {self.root!r}')

        try:
            return [AbstractFile(name=file.name, is_directory=file.is_dir()) for file in resolved_path.iterdir()]
        except PermissionError as e:
            raise EnvFilePermissionError(f'{path!r} is not listable by the environment root {self.root!r}: {str(e)}')
        # I am not sure if any other errors are possible here anyway?
