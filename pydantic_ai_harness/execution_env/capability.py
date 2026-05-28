"""Capability that exposes the execution environment to the agent."""

from dataclasses import dataclass
from typing import Annotated

from opentelemetry.trace import get_current_span
from pydantic import Field
from pydantic_ai import FunctionToolset, ModelRetry
from pydantic_ai.capabilities import AbstractCapability
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
    """Read, window, and truncate a file, enriching the current span as it goes.

    Lives at module scope (not nested in `get_toolset`) so its branch count is its own
    and doesn't push the toolset builder over the complexity limit.
    """
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
    """

    environment: AbstractEnvironment

    def get_toolset(self) -> FunctionToolset[AgentDepsT]:
        """Get the toolset for the execution environment."""
        toolset = FunctionToolset[AgentDepsT]()

        async def read_file(
            path: Annotated[str, Field(description='Path to the file to read, relative to the workspace root.')],
            offset: Annotated[int | None, Field(description='Line number to start reading from (1-indexed)')] = None,
            limit: Annotated[int | None, Field(description='Maximum number of lines to read')] = None,
        ) -> str:
            """Read a file from the execution environment."""
            return await _read_file(self.environment, path, offset, limit)

        async def write_file(
            path: Annotated[str, Field(description='Path to the file, relative to the workspace root.')],
            data: Annotated[str, Field(description='Data to write to the file.')],
        ) -> None:
            """Write a file to the execution environment."""
            try:
                await self.environment.write_file(path, data.encode('utf-8'))
            except EnvPermissionError as e:
                raise ModelRetry(str(e)) from e
            except (EnvWriteError,):
                # TODO: This should be a ToolFailed error when I merge that in
                # catching and re raising here to show the boundary where we change it
                raise

        async def edit_file(
            path: Annotated[str, Field(description='Path to the file to edit, relative to the workspace root.')],
            old_string: Annotated[str, Field(description='Exact text to replace. Must match once, uniquely.')],
            new_string: Annotated[str, Field(description='Text to replace it with.')],
            replace_all: Annotated[
                bool, Field(description='Whether to replace all occurrences of the string, not just the first.')
            ] = False,
        ) -> str:
            """Replace a single unique occurrence of text in an existing file."""
            return await _edit_file(self.environment, path, old_string, new_string, replace_all)

        async def ls(
            path: Annotated[str, Field(description='Path to the directory to list, relative to the workspace root.')],
            limit: Annotated[int | None, Field(description='Maximum number of entries to list')] = None,
        ) -> list[str]:
            """List the contents of a directory."""
            try:
                # `limit` caps the listing here at the presentation layer; see the class
                # docstring for why the bound lives here and not in the backend contract.
                # (`[:None]` is the whole list, so this single expression covers the unbounded case.)
                ls_result = await self.environment.ls(path)
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

        toolset.add_function(read_file, description='Read a file from the execution environment.')
        toolset.add_function(write_file, description='Write a file to the execution environment.')
        toolset.add_function(edit_file, description='Replace a unique occurrence of text in a file.')
        toolset.add_function(ls, description='List the contents of a directory.')

        return toolset
