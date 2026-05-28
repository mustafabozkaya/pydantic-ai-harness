"""Tool implementations and toolset builder for the execution environment.

The capability (`ExecutionEnv`) stays thin: it owns the generic `AgentDepsT` and delegates to
`build_toolset` here. Each tool's real logic lives in a module-level `_*` helper so its try/except
branches are its own -- mccabe rolls *nested* function complexity into the enclosing builder, so the
nested tool wrappers must stay branchless one-liners or `build_toolset` blows the C901 limit.
"""

from typing import Annotated

from opentelemetry.trace import get_current_span
from pydantic import Field
from pydantic_ai import FunctionToolset, ModelRetry
from pydantic_ai.tools import AgentDepsT

from ..environments.abstract import AbstractEnvironment
from ..environments.exceptions import (
    EnvIsADirectoryError,
    EnvNotADirectoryError,
    EnvNotFoundError,
    EnvPermissionError,
    EnvReadError,
    EnvWriteError,
    PathEscapeError,
)
from ._truncate import DEFAULT_MAX_BYTES, format_size, truncate_head


async def _read_file(environment: AbstractEnvironment, path: str, offset: int | None, limit: int | None) -> str:
    """Read, window, and truncate a file, enriching the current span as it goes."""
    # offset/limit are 1-indexed line counts at the boundary (to agree with grep -n,
    # editors, stack traces). pi leaned on JS treating 0 as falsy; Python 0 is a real
    # value, so we validate explicitly and bounce mistakes back to the model.
    if offset is not None and offset < 1:
        raise ModelRetry(f'offset must be >= 1 (lines are 1-indexed), got {offset}')

    if limit is not None and limit < 1:
        raise ModelRetry(f'limit must be >= 1, got {limit}')

    try:
        data = await environment.read_file(path)
        text = data.decode('utf-8')

    except PathEscapeError as e:
        # A boundary-crossing attempt is the one security-relevant case here. Record it
        # as a point-in-time span event (queryable as "something happened"), not just an
        # attribute, then bounce the model off it.
        get_current_span().add_event('path_escape_attempt', {'path': path})
        raise ModelRetry(str(e)) from e
    except (
        EnvNotFoundError,
        EnvPermissionError,
        EnvIsADirectoryError,
        EnvNotADirectoryError,
    ) as e:
        raise ModelRetry(str(e)) from e
    except (EnvReadError,):
        # TODO: This should be a ToolFailed error when I merge that in
        # catching and re raising here to show the boundary where we change it
        raise
    except UnicodeDecodeError as e:
        raise ModelRetry(str(e)) from e

    # Split on '\n' only, NOT str.splitlines(): splitlines() also breaks on '\r', '\v',
    # '\f', and Unicode line/paragraph separators, and collapses a trailing newline. That
    # would make our line numbers disagree with what editors, grep -n, and the model
    # expect. Plain '\n' keeps numbering honest (cost: a trailing '\n' yields a final ''
    # element, so total_lines counts it).
    lines = text.split('\n')
    total_lines = len(lines)

    start = offset - 1 if offset is not None else 0

    if start >= total_lines:
        raise ModelRetry(f'offset {offset} is beyond end of file ({total_lines} lines total)')

    end = min(start + limit, total_lines) if limit is not None else total_lines
    window = lines[start:end]

    result = truncate_head(window)
    # 1-indexed line the window starts on, for the continuation notes.
    start_display = start + 1

    # Enrich the tool-execution span pydantic-ai already opened (no-op when nothing is
    # recording). Set once here so every return path below reports the same facts.
    current_span = get_current_span()
    current_span.set_attribute('truncated_by', result.truncated_by or 'none')
    current_span.set_attribute('total_lines', total_lines)

    if result.first_line_exceeded:
        # Can't show even one line without blowing the byte cap, and we never split a
        # line. pi points at a `bash sed` fallback; we have no shell tool yet, so we just
        # report the size and omit it.
        # TODO: add the sed/head-c hint once the shell tool lands (Slice 4).
        line_size = format_size(len(lines[start].encode('utf-8')))
        return f'[Line {start_display} is {line_size}, exceeds the {format_size(DEFAULT_MAX_BYTES)} limit and was omitted.]'

    body = '\n'.join(result.truncated_lines)

    if result.truncated:
        # The safety cap stopped us. Point the model at the exact next line.
        end_display = start_display + len(result.truncated_lines) - 1
        next_offset = end_display + 1
        if result.truncated_by == 'bytes':
            note = (
                f'[Showing lines {start_display}-{end_display} of {total_lines} '
                f'({format_size(DEFAULT_MAX_BYTES)} limit). Use offset={next_offset} to continue.]'
            )
        else:
            note = (
                f'[Showing lines {start_display}-{end_display} of {total_lines}. Use offset={next_offset} to continue.]'
            )
        return f'{body}\n\n{note}'

    if limit is not None and end < total_lines:
        # The model's own limit stopped us early (not the safety cap); tell it there's
        # more and where to resume.
        remaining = total_lines - end
        return f'{body}\n\n[{remaining} more lines in file. Use offset={end + 1} to continue.]'

    return body


async def _write_file(environment: AbstractEnvironment, path: str, data: str) -> None:
    """Encode and write a file, routing recoverable errors to the model."""
    try:
        await environment.write_file(path, data.encode('utf-8'))
    except EnvPermissionError as e:
        raise ModelRetry(str(e)) from e
    except (EnvWriteError,):
        # TODO: This should be a ToolFailed error when I merge that in
        # catching and re raising here to show the boundary where we change it
        raise


async def _edit_file(
    environment: AbstractEnvironment, path: str, old_string: str, new_string: str, replace_all: bool = False
) -> str:
    """Replace a single, unique occurrence of `old_string` with `new_string` in a file.

    Capability-layer composition over the env primitives: read bytes -> decode -> verify +
    replace (text concern) -> encode -> write bytes. No new env method. Inherits the
    read-modify-write TOCTOU the jail already documents; accepted for V1.

    See `agent_docs/pi-tool-learnings.md` (edit V1 scope) for what we deliberately defer.
    """
    try:
        data = await environment.read_file(path)
        text = data.decode('utf-8')
    except PathEscapeError as e:
        get_current_span().add_event('path_escape_attempt', {'path': path})
        raise ModelRetry(str(e)) from e
    except (
        EnvNotFoundError,
        EnvPermissionError,
        EnvIsADirectoryError,
        EnvNotADirectoryError,
    ) as e:
        raise ModelRetry(str(e)) from e
    except (EnvReadError,):
        raise  # TODO: ToolFailed when merged
    except UnicodeDecodeError as e:
        raise ModelRetry(str(e)) from e

    if old_string == new_string:
        raise ModelRetry('old_string and new_string are identical — this edit is a no-op. Provide different text.')

    count = text.count(old_string)

    if count == 0:
        raise ModelRetry(
            f'old_string was not found in {path!r}. It must match the file exactly, '
            'including whitespace and indentation.'
        )

    if count > 1 and not replace_all:
        raise ModelRetry(
            f'old_string matches {count} places in {path!r}. Add surrounding context to make it '
            'unique, or pass `replace_all=true` to change every occurrence.'
        )

    if replace_all:
        new_text = text.replace(old_string, new_string)
    else:
        new_text = text.replace(old_string, new_string, 1)

    try:
        await environment.write_file(path, new_text.encode('utf-8'))
    except EnvPermissionError as e:
        raise ModelRetry(str(e)) from e
    except (EnvWriteError,):
        # TODO: This should be a ToolFailed error when I merge that in
        # catching and re raising here to show the boundary where we change it
        raise

    return f'Replaced {count} occurrence{"s" if count != 1 else ""} in {path!r}.'


async def _ls(environment: AbstractEnvironment, path: str, limit: int | None) -> list[str]:
    """List a directory, sorting then capping at the presentation layer."""
    try:
        # Sort by name (Unicode code point, deliberately NOT locale-aware) before
        # capping, so the listing is deterministic across runs and backends and the
        # `[:limit]` slice takes a stable prefix instead of an arbitrary one. `limit`
        # caps here at the presentation layer; see the class docstring for why the bound
        # lives here and not in the backend contract. (`[:None]` is the whole list.)
        ls_result = await environment.ls(path)
        ls_result.sort(key=lambda f: f.name)
        return [file.name + ('/' if file.is_directory else '') for file in ls_result[:limit]]
    except PathEscapeError as e:
        get_current_span().add_event('path_escape_attempt', {'path': path})
        raise ModelRetry(str(e)) from e
    except (
        EnvNotFoundError,
        EnvPermissionError,
        EnvIsADirectoryError,
        EnvNotADirectoryError,
    ) as e:
        raise ModelRetry(str(e)) from e
    except (EnvReadError,):
        # TODO: This should be a ToolFailed error when I merge that in
        # catching and re raising here to show the boundary where we change it
        raise


async def _grep(environment: AbstractEnvironment, path: str, pattern: str) -> list[str]:
    """Search a file/tree, formatting hits as `path:lineno:line` and sorting for determinism."""
    try:
        # Classic `grep`/`rg` line format the model has seen everywhere: `path:lineno:line`.
        # Compact (one line per hit) and the lineno lets the model jump straight to a
        # follow-up read_file(offset=...) or edit. Per-file errors (binary, unreadable)
        # are skipped inside the backend, so only top-path errors reach here.
        matches = await environment.grep(path, pattern)
        # Sort by (path, lineno) here, not in the backend: the backend returns matches in
        # filesystem walk order, which varies across machines/filesystems/backends. The
        # model doesn't need sorted order, but SEPARATE executions of the same grep must
        # produce identical output -- otherwise evals, snapshot tests, and run-to-run trace
        # comparisons flap. (This is NOT about prompt caching: within one run the tool
        # result is stored and replayed verbatim, never re-executed.) Sorting at the
        # capability means one place implements it and every backend stays thin -- same
        # choice as `ls`.
        matches.sort(key=lambda match: (match.path, match.lineno))
        return [f'{match.path}:{match.lineno}:{match.line}' for match in matches]
    except PathEscapeError as e:
        get_current_span().add_event('path_escape_attempt', {'path': path})
        raise ModelRetry(str(e)) from e
    except (EnvNotFoundError, EnvPermissionError) as e:
        raise ModelRetry(str(e)) from e
    except (EnvReadError,):
        # TODO: This should be a ToolFailed error when I merge that in
        # catching and re raising here to show the boundary where we change it
        raise


async def _glob(environment: AbstractEnvironment, path: str, pattern: str) -> list[str]:
    """Glob the environment and sort the results."""
    try:
        paths = await environment.glob(path, pattern)
        # Sort for determinism across separate runs/backends -- same reason as grep/ls
        # (stable evals, snapshots, trace comparisons), not prompt caching.
        paths.sort()
        return paths
    except PathEscapeError as e:
        get_current_span().add_event('path_escape_attempt', {'path': path})
        raise ModelRetry(str(e)) from e
    except (EnvNotFoundError, EnvPermissionError, EnvNotADirectoryError) as e:
        raise ModelRetry(str(e)) from e
    except (EnvReadError,):
        # TODO: This should be a ToolFailed error when I merge that in
        # catching and re raising here to show the boundary where we change it
        raise


def _format_shell_result(header: str | None, stdout: str, stderr: str) -> str:
    """Assemble the model-facing text: optional status header, then any stdout/stderr.

    On success we pass `header=None` so the model gets the raw output with no boilerplate. stderr is
    kept only when non-empty (warnings/progress land there) so a successful command's stderr isn't
    silently dropped, but a clean run still reads as just its stdout.
    """
    parts: list[str] = []
    if header is not None:
        parts.append(header)
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f'[stderr]\n{stderr}')
    return '\n'.join(parts)


async def _shell(environment: AbstractEnvironment, command: str, timeout: int | None) -> str:
    """Run a shell command and present its result as text.

    No try/except for EnvShellExecutionError on purpose: that means "the environment couldn't start a
    shell at all" (no bash/sh, broken root) -- infra, not model-fixable -- so it propagates rather
    than becoming a ModelRetry, unlike the path-based errors in the other tools.

    decode(errors='replace') because this is the bytes->text boundary: a command can emit arbitrary
    non-UTF-8 bytes, so decoding must never raise.

    TODO: output is untruncated -- a chatty command (verbose tests, `cat` of a large file) can flood
    the model's context. Tail-truncate (errors live at the end) + spill to a temp file once we have a
    truncate_tail helper, mirroring pi's bash output handling.
    """
    result = await environment.shell_command(command, timeout)
    stdout = result.stdout.decode(errors='replace')
    stderr = result.stderr.decode(errors='replace')

    if result.timed_out:
        # Include partial output: we deliberately captured what printed before the kill, and a model
        # debugging a hang needs it.
        return _format_shell_result(f'Command timed out after {timeout}s.', stdout, stderr)

    if result.return_code != 0:
        return _format_shell_result(f'Command exited with code {result.return_code}.', stdout, stderr)

    # Success: no status boilerplate -- the model just wants the output.
    return _format_shell_result(None, stdout, stderr)


def build_toolset(environment: AbstractEnvironment, toolset: FunctionToolset[AgentDepsT]) -> FunctionToolset[AgentDepsT]:
    """Define the tools (closing over `environment`) and register them on `toolset`.

    Takes the toolset as an argument so `AgentDepsT` is bound by the caller (`ExecutionEnv.get_toolset`
    creates the `FunctionToolset[AgentDepsT]`), keeping the generic parameter with the capability while
    the tool definitions live here. Every tool body delegates to a module-level `_*` helper so this
    builder stays branchless (see module docstring on C901).
    """

    async def read_file(
        path: Annotated[str, Field(description='Path to the file to read, relative to the workspace root.')],
        offset: Annotated[int | None, Field(description='Line number to start reading from (1-indexed)')] = None,
        limit: Annotated[int | None, Field(description='Maximum number of lines to read')] = None,
    ) -> str:
        """Read a file from the execution environment."""
        return await _read_file(environment, path, offset, limit)

    async def write_file(
        path: Annotated[
            str,
            Field(
                description='Path to the file, relative to the workspace root. Created if missing, overwritten if it exists.'
            ),
        ],
        data: Annotated[str, Field(description='Data to write to the file. Replaces the entire file contents.')],
    ) -> None:
        """Create or overwrite a file in the execution environment."""
        return await _write_file(environment, path, data)

    async def edit_file(
        path: Annotated[str, Field(description='Path to the file to edit, relative to the workspace root.')],
        old_string: Annotated[str, Field(description='Exact text to replace. Must match once, uniquely.')],
        new_string: Annotated[str, Field(description='Text to replace it with.')],
        replace_all: Annotated[
            bool, Field(description='Whether to replace all occurrences of the string, not just the first.')
        ] = False,
    ) -> str:
        """Replace a single unique occurrence of text in an existing file."""
        return await _edit_file(environment, path, old_string, new_string, replace_all)

    async def ls(
        path: Annotated[str, Field(description='Path to the directory to list, relative to the workspace root.')],
        limit: Annotated[int | None, Field(description='Maximum number of entries to list')] = None,
    ) -> list[str]:
        """List the contents of a directory."""
        return await _ls(environment, path, limit)

    async def grep(
        path: Annotated[
            str,
            Field(
                description='File or directory to search, relative to the workspace root. Directories are searched recursively.'
            ),
        ],
        pattern: Annotated[str, Field(description='The literal text to search for.')],
    ) -> list[str]:
        """Search a file or directory tree for a literal pattern."""
        return await _grep(environment, path, pattern)

    async def glob(
        path: Annotated[str, Field(description='Directory to search in, relative to the workspace root.')],
        pattern: Annotated[
            str,
            Field(
                # The model reads this to learn our dialect. Two things it cannot infer and would
                # otherwise get wrong: (1) the pattern is matched RECURSIVELY at any depth -- a bare
                # `*.py` finds .py files in every subdirectory, unlike raw Python `glob` where `*`
                # stops at `/`. We imply recursion (rglob) so the model can't fall into the
                # silent-empty trap of typing `*.py` and wrongly concluding a subtree is empty. (2)
                # the concrete example patterns teach the syntax by demonstration, pi-style.
                description=(
                    "Glob pattern, matched recursively at any depth. e.g. '*.py' (any .py file "
                    "anywhere under the directory), '**/*.json', or 'src/**/*.py'."
                )
            ),
        ],
    ) -> list[str]:
        """Find files matching a glob pattern."""
        return await _glob(environment, path, pattern)

    async def shell(
        command: Annotated[str, Field(description='The shell command to run.')],
        timeout: Annotated[
            int | None,
            Field(
                description='Seconds before the process tree is killed and the result returned with `timed_out=True`. `None` means no timeout.'
            ),
        ] = None,
    ) -> str:
        """Run a shell command and return its captured output and exit code."""
        return await _shell(environment, command, timeout)

    toolset.add_function(read_file, description='Read a file from the execution environment.')
    toolset.add_function(write_file, description='Create or overwrite a file in the execution environment.')
    toolset.add_function(edit_file, description='Replace a unique occurrence of text in a file.')
    toolset.add_function(ls, description='List the contents of a directory.')
    toolset.add_function(grep, description='Search a file or directory tree for a literal pattern.')
    toolset.add_function(glob, description='Find files matching a glob pattern.')
    toolset.add_function(shell, description='Run a shell command and return its captured output and exit code.')

    return toolset
