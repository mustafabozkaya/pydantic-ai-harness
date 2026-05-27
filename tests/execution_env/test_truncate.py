"""Unit tests for the head-truncation helper.

These pin the line/byte boundary behavior so the off-by-one and byte-accounting
decisions never silently regress. Pure-sync -- no agent, no event loop.
"""

from pydantic_ai_harness.execution_env._truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    format_size,
    truncate_head,
)


def test_small_input_not_truncated() -> None:
    result = truncate_head(['a', 'b', 'c'])
    assert result.truncated is False
    assert result.truncated_by is None
    assert result.truncated_lines == ['a', 'b', 'c']


def test_exactly_max_lines_not_truncated() -> None:
    result = truncate_head(['x'] * DEFAULT_MAX_LINES)
    assert result.truncated is False
    assert len(result.truncated_lines) == DEFAULT_MAX_LINES


def test_one_over_max_lines_truncates_by_lines() -> None:
    result = truncate_head(['x'] * (DEFAULT_MAX_LINES + 1))
    assert result.truncated_by == 'lines'
    assert len(result.truncated_lines) == DEFAULT_MAX_LINES


def test_byte_cap_truncates_by_bytes() -> None:
    # 100 lines of 1KB each is ~100KB but only 100 lines, so bytes must win.
    result = truncate_head(['x' * 1024 for _ in range(100)])
    assert result.truncated_by == 'bytes'
    # What we keep must actually fit under the byte cap (newlines included).
    assert len('\n'.join(result.truncated_lines).encode('utf-8')) <= DEFAULT_MAX_BYTES


def test_giant_first_line_flagged_and_omitted() -> None:
    result = truncate_head(['x' * (DEFAULT_MAX_BYTES + 1), 'rest'])
    assert result.first_line_exceeded is True
    assert result.truncated_by == 'bytes'
    assert result.truncated_lines == []


def test_truncated_property_tracks_truncated_by() -> None:
    assert truncate_head(['a']).truncated is False
    assert truncate_head(['x'] * (DEFAULT_MAX_LINES + 1)).truncated is True


def test_format_size() -> None:
    assert format_size(512) == '512B'
    assert format_size(1536) == '1.5KB'
    assert format_size(50 * 1024) == '50.0KB'
    assert format_size(2 * 1024 * 1024) == '2.0MB'
