"""LocalEnvironment-specific behavior that does not generalize across backends.

The backend-agnostic contract lives in `test_conformance.py`. Here we test things tied to
LocalEnvironment's implementation: how it resolves symlinked roots, how it maps POSIX
permission errors, and the catch-all I/O error path. A Docker backend would exercise these
concerns differently, so they are deliberately not part of the shared conformance suite.
"""

import asyncio
import os
import shutil
import signal
from pathlib import Path

import pytest
from inline_snapshot import snapshot

from pydantic_ai_harness.environments.abstract import AbstractMatch
from pydantic_ai_harness.environments.exceptions import (
    EnvPermissionError,
    EnvReadError,
    EnvShellExecutionError,
    PathEscapeError,
)
from pydantic_ai_harness.environments.local import LocalEnvironment

# The permission tests below assert that the OS denies us a read/write. But root ignores
# POSIX permission bits entirely -- `chmod 000` then reading still succeeds as root -- so
# those tests would fail spuriously when run as uid 0 (common in CI containers). Skip them
# there. `hasattr` guards Windows, which has no `geteuid` and no POSIX bits to begin with.
skip_if_root = pytest.mark.skipif(
    hasattr(os, 'geteuid') and os.geteuid() == 0, reason='root bypasses POSIX permission bits'
)


async def test_read_through_symlinked_root(tmp_path: Path) -> None:
    # Root the environment at a SYMLINK to the real dir (like /tmp/... on macOS). A read of a
    # file genuinely inside must SUCCEED: the jail must resolve the root before comparing.
    real_box = tmp_path / 'realbox'
    real_box.mkdir()
    (real_box / 'ok.txt').write_bytes(b'inside the box')
    link_box = tmp_path / 'linkbox'
    link_box.symlink_to(real_box)

    env = LocalEnvironment(root=str(link_box))
    assert await env.read_file('ok.txt') == b'inside the box'


async def test_write_through_symlinked_root(tmp_path: Path) -> None:
    real_box = tmp_path / 'realbox'
    real_box.mkdir()
    link_box = tmp_path / 'linkbox'
    link_box.symlink_to(real_box)

    env = LocalEnvironment(root=str(link_box))
    await env.write_file('test.txt', b'Hello, world!')
    assert (real_box / 'test.txt').read_bytes() == b'Hello, world!'


async def test_read_outside_root_absolute_raises(tmp_path: Path) -> None:
    env = LocalEnvironment(root=str(tmp_path))
    secret = tmp_path.parent / 'secret.txt'
    secret.write_bytes(b'top secret')
    with pytest.raises(PathEscapeError):
        await env.read_file(str(secret))


async def test_write_outside_root_absolute_raises(tmp_path: Path) -> None:
    env = LocalEnvironment(root=str(tmp_path))
    secret = tmp_path.parent / 'secret.txt'
    with pytest.raises(PathEscapeError):
        await env.write_file(str(secret), b'top secret')


@skip_if_root
async def test_read_unreadable_file_raises_permission(tmp_path: Path) -> None:
    target = tmp_path / 'locked.txt'
    target.write_bytes(b'secret')
    target.chmod(0o000)
    try:
        env = LocalEnvironment(root=str(tmp_path))
        with pytest.raises(EnvPermissionError):
            await env.read_file('locked.txt')
    finally:
        target.chmod(0o644)  # let tmp_path cleanup remove it


@skip_if_root
async def test_write_into_readonly_dir_raises_permission(tmp_path: Path) -> None:
    box = tmp_path / 'readonly'
    box.mkdir()
    box.chmod(0o555)
    try:
        env = LocalEnvironment(root=str(tmp_path))
        with pytest.raises(EnvPermissionError):
            await env.write_file('readonly/new.txt', b'nope')
    finally:
        box.chmod(0o755)


async def test_read_symlink_loop_raises_read_error(tmp_path: Path) -> None:
    # A symlink cycle resolves without error but fails at read time with a generic OSError
    # (ELOOP) -- not one of the mapped subclasses -- exercising the catch-all read path.
    a = tmp_path / 'a'
    b = tmp_path / 'b'
    a.symlink_to(b)
    b.symlink_to(a)

    env = LocalEnvironment(root=str(tmp_path))
    with pytest.raises(EnvReadError):
        await env.read_file('a')


@skip_if_root
async def test_ls_unlistable_directory_raises_permission(tmp_path: Path) -> None:
    box = tmp_path / 'locked'
    box.mkdir()
    box.chmod(0o000)
    try:
        env = LocalEnvironment(root=str(tmp_path))
        with pytest.raises(EnvPermissionError):
            await env.ls('locked')
    finally:
        box.chmod(0o755)  # let tmp_path cleanup remove it


async def test_ls_symlink_loop_raises_read_error(tmp_path: Path) -> None:
    # Listing through a symlink cycle fails with a generic OSError (ELOOP) -- not one of the
    # mapped subclasses -- exercising the catch-all path in `ls`.
    a = tmp_path / 'a'
    b = tmp_path / 'b'
    a.symlink_to(b)
    b.symlink_to(a)

    env = LocalEnvironment(root=str(tmp_path))
    with pytest.raises(EnvReadError):
        await env.ls('a')


@skip_if_root
async def test_grep_unreadable_file_in_tree_is_skipped(tmp_path: Path) -> None:
    # A per-file permission failure mid-walk is swallowed (same skip path as a binary file),
    # so the search continues and still returns matches from the readable files. POSIX/uid
    # specific (root bypasses the bits), hence local-only + skip_if_root.
    box = tmp_path / 'dir'
    box.mkdir()
    (box / 'unreadable.txt').write_text('NEEDLE')
    (box / 'readable.txt').write_text('NEEDLE')
    (box / 'no-match.txt').write_text('nothing')
    (box / 'unreadable.txt').chmod(0o000)
    try:
        env = LocalEnvironment(root=str(tmp_path))
        assert await env.grep('dir', 'NEEDLE') == snapshot(
            [AbstractMatch(path='dir/readable.txt', line='NEEDLE', lineno=1)]
        )
    finally:
        (box / 'unreadable.txt').chmod(0o644)  # let tmp_path cleanup remove it


@skip_if_root
async def test_grep_unreadable_top_path_raises_permission(tmp_path: Path) -> None:
    # When the TOP path (the argument) is unreadable, grep raises -- it is not a per-file
    # skip. This exercises the `os.access` guard clause that os.walk could not surface.
    box = tmp_path / 'locked'
    box.mkdir()
    box.chmod(0o000)
    try:
        env = LocalEnvironment(root=str(tmp_path))
        with pytest.raises(EnvPermissionError):
            await env.grep('locked', 'NEEDLE')
    finally:
        box.chmod(0o755)  # let tmp_path cleanup remove it


@skip_if_root
async def test_glob_unreadable_directory_raises_permission(tmp_path: Path) -> None:
    # An unreadable search directory raises via the `os.access` guard clause -- the same
    # guard grep uses, since rglob (like os.walk) would silently yield nothing instead.
    box = tmp_path / 'locked'
    box.mkdir()
    box.chmod(0o000)
    try:
        env = LocalEnvironment(root=str(tmp_path))
        with pytest.raises(EnvPermissionError):
            await env.glob('locked', '*.py')
    finally:
        box.chmod(0o755)  # let tmp_path cleanup remove it


async def test_timeout_kills_the_whole_tree(tmp_path: Path) -> None:
    # Proves the timeout kills the whole process GROUP, not just the bash leader we spawned.
    #
    # The command builds a small process tree:
    #   bash (leader, its own process group via start_new_session=True)
    #   |-- ( sleep 1; touch marker )  <- a SUBSHELL, backgrounded with `&` -> a separate
    #   |                                 grandchild process that shares the leader's pgid
    #   `-- sleep 30                   <- foreground, keeps the leader alive past the timeout
    #                                     so the TIMEOUT ends the run, not the script finishing
    #
    # We can't ask the OS "is the grandchild still alive?" reliably (PID reuse, races), so we
    # use a dead-man's switch: the grandchild only writes `marker` at t=1. If our killpg reaches
    # the whole group, the grandchild dies at t=0.5 and never writes -> marker absent. If we only
    # killed the leader (e.g. proc.kill()), the grandchild is orphaned, survives to t=1, and
    # writes the file -> marker present. So "marker must not exist" == "the tree really died".
    env = LocalEnvironment(root=str(tmp_path))
    marker = tmp_path / 'marker.txt'
    result = await env.shell_command(f'( sleep 1; touch {marker} ) & sleep 30', timeout=0.5)

    assert result.timed_out is True
    # Wait LONGER than the grandchild's t=1 write attempt. If an orphan survived, it has already
    # written by now -- so checking after this sleep catches it. Checking immediately (the call
    # returns at ~t=0.5 once the leader dies) would see "no marker yet" even with a live orphan
    # and pass falsely; the sleep must outlast the orphan's delay-to-write for the assert to bite.
    await asyncio.sleep(1)
    assert not marker.exists()


async def test_sigkill_escalation_kills_stubborn_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Proves the SIGKILL escalation: when a child IGNORES SIGTERM, we still force it down.
    #
    # `trap "" TERM` installs an empty SIGTERM handler -- the grandchild catches SIGTERM and does
    # nothing, so (unlike the plain `sleep` in the test above) it survives the polite kill. That is
    # exactly the case `_terminate_and_drain`'s `except TimeoutError -> SIGKILL` branch exists for:
    # SIGTERM is ignored, the grace window elapses, and we escalate to SIGKILL (uncatchable -- you
    # cannot trap it), which kills the grandchild before its t=1 `touch` -> marker stays absent.
    #
    # We shrink the grace to 0.1s by monkeypatching the private module constant (see its docstring:
    # it's deliberately not a public knob, and a test of *this* module may patch this module's
    # internals). Without this, the one test that hits the escalation branch would pay the full 5s.
    monkeypatch.setattr('pydantic_ai_harness.environments.local._SIGTERM_GRACE_SECONDS', 0.1)

    env = LocalEnvironment(root=str(tmp_path))
    marker = tmp_path / 'marker.txt'
    result = await env.shell_command(f'( trap "" TERM; sleep 1; touch {marker} ) & sleep 30', timeout=0.5)

    assert result.timed_out is True
    await asyncio.sleep(1)
    assert not marker.exists()


async def test_sigterm_already_dead_is_swallowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Covers the ProcessLookupError swallow on the SIGTERM killpg: the group can vanish in the race
    # between the timeout firing and our signal. We fake that by raising ProcessLookupError for the
    # SIGTERM call only (delegating SIGKILL to the real killpg so the process still actually dies and
    # we don't hang on the drain). Teardown must shrug it off and still return a timed-out result.
    monkeypatch.setattr('pydantic_ai_harness.environments.local._SIGTERM_GRACE_SECONDS', 0.1)
    real_killpg = os.killpg

    def fake_killpg(pgid: int, sig: int) -> None:
        if sig == signal.SIGTERM:
            raise ProcessLookupError
        real_killpg(pgid, sig)

    monkeypatch.setattr(os, 'killpg', fake_killpg)
    env = LocalEnvironment(root=str(tmp_path))
    result = await env.shell_command('sleep 30', timeout=0.3)
    assert result.timed_out is True


async def test_falls_back_to_sh_when_bash_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Covers the `sh` branch: when bash is missing we resolve to POSIX sh. Report bash as absent but
    # defer to the real lookup for everything else, so `sh` resolves and the command genuinely runs.
    real_which = shutil.which

    def only_sh(name: str) -> str | None:
        return None if name == 'bash' else real_which(name)

    monkeypatch.setattr(shutil, 'which', only_sh)
    env = LocalEnvironment(root=str(tmp_path))
    result = await env.shell_command('echo hi')
    assert result.stdout == b'hi\n'
    assert result.return_code == 0


async def test_spawn_failure_raises_shell_execution_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Covers the OSError->EnvShellExecutionError wrap: a shell resolves but the spawn itself fails
    # (e.g. a broken root, fork failure). The infra error surfaces as EnvShellExecutionError.
    async def boom(*_args: object, **_kwargs: object) -> asyncio.subprocess.Process:
        raise OSError('spawn failed')

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', boom)
    env = LocalEnvironment(root=str(tmp_path))
    with pytest.raises(EnvShellExecutionError):
        await env.shell_command('echo hi')


async def test_no_shell_raises_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = LocalEnvironment(root=str(tmp_path))

    def _no_shell(_name: str) -> None:
        return None

    monkeypatch.setattr(shutil, 'which', _no_shell)
    with pytest.raises(EnvShellExecutionError):
        await env.shell_command('echo "hello"')


async def test_cancellation_kills_the_tree(tmp_path: Path) -> None:
    # Same orphan probe as the timeout test, but the trigger is the CALLER cancelling mid-run (not a
    # timeout) -- this exercises the `finally` + `asyncio.shield` teardown path. No `timeout=`: only
    # our `task.cancel()` may end the run, isolating the cancellation path (a timeout racing the
    # cancel would let the run RETURN a result, and then `await task` wouldn't raise).
    env = LocalEnvironment(root=str(tmp_path))
    marker = tmp_path / 'marker.txt'
    task = asyncio.create_task(env.shell_command(f'( sleep 1; touch {marker} ) & sleep 30'))
    # Let the coroutine actually spawn the subprocess before cancelling; cancelling at t=0 would kill
    # it while still parked at its first await, before `proc` exists, testing nothing.
    await asyncio.sleep(0.5)
    task.cancel()
    # A cancelled task MUST end by raising CancelledError, not by returning a value -- swallowing it
    # would break callers (wait_for / TaskGroup) that rely on the cancellation propagating.
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.sleep(1)
    assert not marker.exists()
