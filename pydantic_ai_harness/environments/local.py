"""Local environment using the local filesystem."""

import asyncio
import os
import shutil
import signal
from dataclasses import dataclass
from pathlib import Path

from .abstract import AbstractEnvironment, AbstractFile, AbstractMatch, ShellCommandResult
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

    async def glob(self, path: str, pattern: str) -> list[str]:
        """Glob a directory for a pattern."""
        root = Path(self.root).resolve()
        resolved_path = Path(root, path).resolve()

        if not resolved_path.is_relative_to(root):
            raise PathEscapeError(f'{path!r} resolves outside the environment root {self.root!r}')

        if not os.path.exists(resolved_path):
            raise EnvNotFoundError(f'{path!r} not found in the environment root {self.root!r}')

        if not os.access(resolved_path, os.R_OK):
            raise EnvPermissionError(f'{path!r} is not readable by the environment root {self.root!r}')

        if os.path.isfile(resolved_path):
            raise EnvNotADirectoryError(f'{path!r} is a file in the environment root {self.root!r}')

        results: list[str] = []

        # rglob, not glob: a bare `*.py` matches at any depth, so the model can't fall into the
        # silent-empty trap where `*.py` finds nothing in subdirectories (raw glob's `*` stops at
        # `/`). The model learns this recursion from the `pattern` param description on the
        # capability's glob tool -- this comment is for the maintainer, that one is for the model.
        # Files only (is_file): directories are not glob results, matching grep's file orientation.
        # Returned in filesystem walk order; the capability sorts for determinism (see grep/ls).
        for match in resolved_path.rglob(pattern):
            if match.is_file():
                results.append(str(match.relative_to(root)))

        return results

    async def shell_command(self, command: str, timeout: int | None = None) -> ShellCommandResult:
        """Execute a shell command."""
        # Resolve the shell instead of hardcoding: the ABC promises "a shell," not bash specifically
        # (Alpine/minimal Docker images ship only `sh`). Prefer bash -- the model emits bash syntax
        # (`[[ ]]`, arrays) -- and fall back to POSIX sh. We do NOT use the user's $SHELL: it may be
        # fish/zsh, where the model's `&&`/`export` would break. POSIX-only for V1; a bare Windows box
        # has neither bash nor sh, so `which` returns None and we fail fast here (use WSL/Docker).
        # See agent_docs/shell-run-prior-art.md "Shell resolution & platform scope".
        if shutil.which('bash') is not None:
            shell = 'bash'
        elif shutil.which('sh') is not None:
            shell = 'sh'
        else:
            # FIXME: OSError is the builtin alias for EnvironmentError and collides with how the
            # capability layer maps env errors -- use a dedicated env exception type instead.
            raise OSError(f'No bash or sh found in the environment root {self.root!r}')

        # create_subprocess_exec (async, argv list) -- not subprocess.run (blocks the loop) and not
        # create_subprocess_shell (hardcodes /bin/sh, defeats the resolution above). `-lc` = login
        # shell so it sources profile files and the command sees the user's PATH (nvm/pyenv); this is
        # best-effort (login *bash* reads ~/.bash_profile, not ~/.zshrc). PIPE on both streams =
        # captured bytes (no text=True) -> bytes-at-core contract. cwd = root so commands run in the
        # jail. start_new_session=True calls setsid(): the child becomes its own process-GROUP leader
        # so every process it forks shares one pgid -- the matched half of the killpg() below.
        proc = await asyncio.create_subprocess_exec(
            shell,
            '-lc',
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=Path(self.root).resolve(),
            start_new_session=True,
        )

        timed_out = False
        stdout = b''
        stderr = b''

        try:
            # wait_for(coro, None) == no limit, so timeout=None is the "no timeout" path for free.
            # communicate() drains both pipes (a child that fills the ~64KB pipe buffer can't deadlock)
            # and returns bytes.
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            # TRAP: wait_for only cancelled the communicate() coroutine -- the process tree is STILL
            # ALIVE. Abandoning the await != killing it (OpenHands probe: a detached child survived).
            # killpg + getpgid, not proc.kill(): we signal the whole GROUP (created by setsid above) so
            # forked children don't orphan to init and keep running/billing. SIGTERM first (catchable)
            # gives the tree a chance to flush/clean up.
            # ProcessLookupError = the process already exited in the window between the timeout firing
            # and this signal (a TOCTOU race, same shape as the filesystem jail). Already dead is the
            # outcome we wanted, so swallow it -- there's nothing left to kill.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            # Grace period: asyncio's child watcher updates proc.returncode in the background while the
            # loop runs, so after sleeping we can tell whether SIGTERM actually killed it.
            await asyncio.sleep(5)
            if proc.returncode is None:
                # Ignored SIGTERM (or wedged) -> SIGKILL is uncatchable, guaranteed teardown. We
                # escalate rather than leak an orphan on a remote/billing backend. Same race guard:
                # SIGTERM may have landed during the grace sleep, so the group can be gone by now.
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

            # Even though proc died there will be some partial stuff in the pipes so let us get that out
            if proc.stdout is not None:
                stdout = await proc.stdout.read()
            if proc.stderr is not None:
                stderr = await proc.stderr.read()
            # Reap the zombie: we're the parent, so the dead process lingers in the process table
            # holding its exit status + fds until we wait(). Also yields the final returncode.
            await proc.wait()
            timed_out = True

        assert proc.returncode is not None

        return ShellCommandResult(
            stdout=stdout,
            stderr=stderr,
            return_code=proc.returncode,
            timed_out=timed_out,
        )
