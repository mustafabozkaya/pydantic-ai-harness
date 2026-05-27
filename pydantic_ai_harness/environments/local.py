"""Local environment using the local filesystem."""

from dataclasses import dataclass
from pathlib import Path

from .abstract import AbstractEnvironment
from .exceptions import (
    EnvFileIsADirectoryError,
    EnvFileNotADirectoryError,
    EnvFileNotFoundError,
    EnvFilePermissionError,
    EnvFileReadError,
    PathEscapeError,
)


@dataclass(kw_only=True)
class LocalEnvironment(AbstractEnvironment):
    """Local environment using the local filesystem."""

    async def read_file(self, path: str) -> bytes:
        """Read a file from the local filesystem."""
        root = Path(self.root).resolve()
        resolved_path = Path(root, path).resolve()

        # Advisory jail with a KNOWN TOCTOU race: the resolve()+containment check
        # and the read_bytes() below are two separate steps. A symlink swapped in
        # between could redirect the read outside `root`. Accepted for V1 -- a
        # race-free jail needs openat2/O_NOFOLLOW-style primitives. Revisit during
        # hardening (Slice 6). TODO: deepen understanding before relying on this.
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
