"""Local environment using the local filesystem."""

import os
from dataclasses import dataclass
from pathlib import Path

from .abstract import AbstractEnvironment, AbstractFile, AbstractMatch
from .exceptions import (
    EnvIsADirectoryError,
    EnvNotADirectoryError,
    EnvNotFoundError,
    EnvPermissionError,
    EnvReadError,
    EnvWriteError,
    PathEscapeError,
)


async def _grep_file(root: str, path: str, pattern: str) -> list[AbstractMatch]:
    """Search a file for a pattern."""
    results: list[AbstractMatch] = []
    try:
        with open(path) as file:
            for lineno, line in enumerate(file, start=1):
                if pattern in line:
                    results.append(AbstractMatch(path=str(Path(path).relative_to(root)), line=line, lineno=lineno))
    except (OSError, UnicodeDecodeError):
        return results

    return results


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
            raise EnvNotFoundError(f'{path!r} not found in the environment root {self.root!r}: {str(e)}')
        except PermissionError as e:
            raise EnvPermissionError(f'{path!r} is not readable by the environment root {self.root!r}: {str(e)}')
        except IsADirectoryError as e:
            raise EnvIsADirectoryError(f'{path!r} is a directory in the environment root {self.root!r}: {str(e)}')
        except NotADirectoryError as e:
            raise EnvNotADirectoryError(f'{path!r} is not a directory in the environment root {self.root!r}: {str(e)}')
        except OSError as e:
            raise EnvReadError(f'{path!r} could not be read in the environment root {self.root!r}: {str(e)}')

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
            raise EnvPermissionError(f'{path!r} is not writable by the environment root {self.root!r}: {str(e)}')
        except OSError as e:
            raise EnvWriteError(f'{path!r} could not be written to the environment root {self.root!r}: {str(e)}')

    async def ls(self, path: str) -> list[AbstractFile]:
        """List the contents of a directory."""
        root = Path(self.root).resolve()
        resolved_path = Path(root, path).resolve()

        if not resolved_path.is_relative_to(root):
            raise PathEscapeError(f'{path!r} resolves outside the environment root {self.root!r}')

        try:
            # `os.scandir` returns the entry type that the underlying directory read (`getdents`)
            # already carried, so `is_dir(follow_symlinks=False)` answers from that cached type
            # without an extra `stat` per entry -- one round-trip for the whole listing, which is
            # what keeps `ls` cheap on remote backends. `Path.iterdir()` discards that type and
            # would force a `stat` per entry. `follow_symlinks=False` classifies the entry itself,
            # so a broken symlink is simply "not a directory" rather than a `stat` that raises.
            with os.scandir(resolved_path) as entries:
                return [AbstractFile(name=e.name, is_directory=e.is_dir(follow_symlinks=False)) for e in entries]
        except FileNotFoundError as e:
            raise EnvNotFoundError(f'{path!r} not found in the environment root {self.root!r}: {str(e)}')
        except PermissionError as e:
            raise EnvPermissionError(f'{path!r} is not listable by the environment root {self.root!r}: {str(e)}')
        except NotADirectoryError as e:
            raise EnvNotADirectoryError(f'{path!r} is not a directory in the environment root {self.root!r}: {str(e)}')
        except OSError as e:
            raise EnvReadError(f'{path!r} could not be listed in the environment root {self.root!r}: {str(e)}')

    async def grep(self, path: str, pattern: str) -> list[AbstractMatch]:
        """Search a file for a pattern."""
        root = Path(self.root).resolve()
        resolved_path = Path(root, path).resolve()

        if not resolved_path.is_relative_to(root):
            raise PathEscapeError(f'{path!r} resolves outside the environment root {self.root!r}')

        results: list[AbstractMatch] = []

        # grep validates the top path with guard clauses (not the wrap-the-exception idiom that
        # read_file/ls use) because os.walk/os.access don't raise -- they return empty/False. You
        # can't translate an exception that's never thrown, so the argument must be checked up front.
        if not os.path.exists(resolved_path):
            raise EnvNotFoundError(f'{path!r} not found in the environment root {self.root!r}')

        if not os.access(resolved_path, os.R_OK):
            raise EnvPermissionError(f'{path!r} is not readable by the environment root {self.root!r}')

        if os.path.isfile(resolved_path):
            results = await _grep_file(str(root), str(resolved_path), pattern)
        else:
            for dirpath, _, filenames in os.walk(resolved_path):
                for filename in filenames:
                    results.extend(await _grep_file(str(root), os.path.join(dirpath, filename), pattern))

        # Returned in filesystem walk order (unsorted). Determinism is the capability layer's
        # job -- it sorts before presenting -- so backends stay thin and only one place has to
        # implement (and be verified for) the ordering. See ExecutionEnv.get_toolset grep tool.
        return results
