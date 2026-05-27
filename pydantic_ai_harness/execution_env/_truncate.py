"""Shared output-truncation helpers for the execution-environment toolset.

Truncation is presentation, not backend truth, so it lives in the capability layer
(here), never in `environments/`. Ported from pi-mono's `truncate.ts`
(see `agent_docs/pi-prompts.md` for attribution).
"""

from dataclasses import dataclass
from typing import Literal, TypeAlias

# Two independent caps; whichever is hit first wins. Mirrors pi's defaults so the tool
# descriptions (which quote these numbers to the model) stay truthful.
DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024  # 50KB

# Defined once, reused on the dataclass field and the local in truncate_head.
TruncatedBy: TypeAlias = Literal['lines', 'bytes'] | None


@dataclass(kw_only=True, frozen=True)
class HeadTruncationResult:
    """What fit under the caps, and why we stopped. The caller turns this into a note."""

    truncated_lines: list[str]
    truncated_by: TruncatedBy = None
    # A single line wider than the byte cap can't be shown partially; this lets the
    # caller emit a "line too big" message instead of a normal continuation note.
    first_line_exceeded: bool = False

    @property
    def truncated(self) -> bool:
        # Derived, never stored, so it can't disagree with truncated_by.
        return self.truncated_by is not None


def format_size(num_bytes: int) -> str:
    """Render a byte count the way the continuation notes show it to the model."""
    if num_bytes < 1024:
        return f'{num_bytes}B'
    if num_bytes < 1024 * 1024:
        return f'{num_bytes / 1024:.1f}KB'
    return f'{num_bytes / (1024 * 1024):.1f}MB'


def truncate_head(lines: list[str]) -> HeadTruncationResult:
    """Keep the first lines that fit under both caps; never emit a partial line."""
    # A line wider than the byte cap can't be shown partially (we never split a line),
    # so keep nothing and flag it -- the caller reports the line's size and omits it.
    if lines and len(lines[0].encode('utf-8')) > DEFAULT_MAX_BYTES:
        return HeadTruncationResult(truncated_lines=[], truncated_by='bytes', first_line_exceeded=True)

    kept: list[str] = []
    running_byte_size = 0
    truncated_by: TruncatedBy = None

    for line in lines:
        if len(kept) >= DEFAULT_MAX_LINES:
            truncated_by = 'lines'
            break
        # +1 for the '\n' that '\n'.join inserts before every line except the first,
        # so the budget matches the bytes actually emitted.
        cost = len(line.encode('utf-8')) + (1 if kept else 0)
        if running_byte_size + cost > DEFAULT_MAX_BYTES:
            truncated_by = 'bytes'
            break
        kept.append(line)
        running_byte_size += cost

    # Loop completed without breaking => kept everything => truncated_by stays None.
    return HeadTruncationResult(truncated_lines=kept, truncated_by=truncated_by)
