"""Tests for the StuckLoopDetection capability."""
# pyright: reportPrivateUsage=false, reportArgumentType=false

from __future__ import annotations

import pytest
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

from pydantic_harness.stuck_loop_detection import (
    DEFAULT_WARNING_MESSAGE,
    StuckLoopDetection,
    StuckLoopError,
    _detect_alternating,
    _detect_repeated,
    _normalize_args,
    _tool_call_key,
)

# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestNormalizeArgs:
    def test_none(self):
        assert _normalize_args(None) == ''

    def test_dict(self):
        assert _normalize_args({'b': 2, 'a': 1}) == '{"a": 1, "b": 2}'

    def test_json_string(self):
        assert _normalize_args('{"b": 2, "a": 1}') == '{"a": 1, "b": 2}'

    def test_non_json_string(self):
        assert _normalize_args('not json') == 'not json'

    def test_empty_dict(self):
        assert _normalize_args({}) == '{}'


class TestToolCallKey:
    def test_basic(self):
        part = ToolCallPart(tool_name='read_file', args={'path': '/foo'})
        assert _tool_call_key(part) == 'read_file::{"path": "/foo"}'

    def test_no_args(self):
        part = ToolCallPart(tool_name='get_time', args=None)
        assert _tool_call_key(part) == 'get_time::'


class TestDetectRepeated:
    def test_below_threshold(self):
        assert _detect_repeated(['a', 'a'], 3) is None

    def test_at_threshold(self):
        assert _detect_repeated(['a', 'a', 'a'], 3) == 'a'

    def test_above_threshold(self):
        assert _detect_repeated(['b', 'a', 'a', 'a'], 3) == 'a'

    def test_no_repeat(self):
        assert _detect_repeated(['a', 'b', 'a'], 3) is None

    def test_mixed_then_repeat(self):
        assert _detect_repeated(['x', 'y', 'z', 'z', 'z'], 3) == 'z'


class TestDetectAlternating:
    def test_below_threshold(self):
        assert _detect_alternating(['a', 'b', 'a'], 2) is None

    def test_at_threshold(self):
        assert _detect_alternating(['a', 'b', 'a', 'b'], 2) == ('a', 'b')

    def test_not_alternating(self):
        assert _detect_alternating(['a', 'b', 'c', 'b'], 2) is None

    def test_same_keys(self):
        # a == b should return None (that's a repeat, not alternation)
        assert _detect_alternating(['a', 'a', 'a', 'a'], 2) is None

    def test_longer_pattern(self):
        assert _detect_alternating(['a', 'b', 'a', 'b', 'a', 'b'], 3) == ('a', 'b')


# ---------------------------------------------------------------------------
# Integration-style tests for the capability hooks
# ---------------------------------------------------------------------------


def _make_response(*tool_calls: ToolCallPart) -> ModelResponse:
    return ModelResponse(parts=list(tool_calls))


def _make_text_response(text: str = 'hello') -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def _make_tc(name: str, args: dict[str, object] | None = None) -> ToolCallPart:
    return ToolCallPart(tool_name=name, args=args)


class _FakeCtx:
    """Minimal stand-in for RunContext — the capability only receives it, never inspects it."""


class _FakeRequestContext:
    """Minimal stand-in for ModelRequestContext."""


class _FakeToolDef:
    """Minimal stand-in for ToolDefinition."""


@pytest.fixture()
def cap_warn() -> StuckLoopDetection:
    """A fresh warn-mode capability with threshold 3."""
    return StuckLoopDetection(max_repeated_calls=3, action='warn')


@pytest.fixture()
def cap_error() -> StuckLoopDetection:
    """A fresh error-mode capability with threshold 3."""
    return StuckLoopDetection(max_repeated_calls=3, action='error')


# --- for_run isolation ---


@pytest.mark.anyio()
async def test_for_run_returns_fresh_instance(cap_warn: StuckLoopDetection):
    cap_warn._call_history.append('something')
    fresh = await cap_warn.for_run(_FakeCtx())  # type: ignore[arg-type]
    assert fresh is not cap_warn
    assert fresh._call_history == []
    assert fresh.max_repeated_calls == cap_warn.max_repeated_calls
    assert fresh.action == cap_warn.action
    assert fresh.warning_message == cap_warn.warning_message


# --- Repeated call detection ---


@pytest.mark.anyio()
async def test_repeated_calls_warn(cap_warn: StuckLoopDetection):
    ctx: object = _FakeCtx()
    rctx: object = _FakeRequestContext()
    tc = _make_tc('read_file', {'path': '/a'})
    resp = _make_response(tc)

    # First two calls are fine.
    for _ in range(2):
        result = await cap_warn.after_model_request(ctx, request_context=rctx, response=resp)  # type: ignore[arg-type]
        assert result is resp

    # Third triggers ModelRetry.
    with pytest.raises(ModelRetry, match='read_file.*identical arguments'):
        await cap_warn.after_model_request(ctx, request_context=rctx, response=resp)  # type: ignore[arg-type]


@pytest.mark.anyio()
async def test_repeated_calls_error(cap_error: StuckLoopDetection):
    ctx: object = _FakeCtx()
    rctx: object = _FakeRequestContext()
    tc = _make_tc('bash', {'cmd': 'ls'})
    resp = _make_response(tc)

    for _ in range(2):
        await cap_error.after_model_request(ctx, request_context=rctx, response=resp)  # type: ignore[arg-type]

    with pytest.raises(StuckLoopError, match='bash.*identical arguments'):
        await cap_error.after_model_request(ctx, request_context=rctx, response=resp)  # type: ignore[arg-type]


@pytest.mark.anyio()
async def test_different_calls_do_not_trigger(cap_warn: StuckLoopDetection):
    ctx: object = _FakeCtx()
    rctx: object = _FakeRequestContext()

    for i in range(5):
        tc = _make_tc('read_file', {'path': f'/file_{i}'})
        resp = _make_response(tc)
        result = await cap_warn.after_model_request(ctx, request_context=rctx, response=resp)  # type: ignore[arg-type]
        assert result is resp


# --- Alternating call detection ---


@pytest.mark.anyio()
async def test_alternating_calls_warn(cap_warn: StuckLoopDetection):
    ctx: object = _FakeCtx()
    rctx: object = _FakeRequestContext()
    tc_a = _make_tc('read_file', {'path': '/a'})
    tc_b = _make_tc('write_file', {'path': '/b'})

    calls = [tc_a, tc_b, tc_a, tc_b, tc_a, tc_b]
    # First 5 are fine (need 6 = 3*2 for alternating detection).
    for tc in calls[:5]:
        resp = _make_response(tc)
        await cap_warn.after_model_request(ctx, request_context=rctx, response=resp)  # type: ignore[arg-type]

    with pytest.raises(ModelRetry, match='Alternating'):
        resp = _make_response(calls[5])
        await cap_warn.after_model_request(ctx, request_context=rctx, response=resp)  # type: ignore[arg-type]


# --- No-op call detection ---


@pytest.mark.anyio()
async def test_noop_detection_warn(cap_warn: StuckLoopDetection):
    ctx: object = _FakeCtx()
    tc = _make_tc('search', {'query': 'foo'})
    td: object = _FakeToolDef()

    for _ in range(2):
        result = await cap_warn.after_tool_execute(
            ctx, call=tc, tool_def=td, args={'query': 'foo'}, result='same result'
        )  # type: ignore[arg-type]
        assert result == 'same result'

    with pytest.raises(ModelRetry, match='same result'):
        await cap_warn.after_tool_execute(ctx, call=tc, tool_def=td, args={'query': 'foo'}, result='same result')  # type: ignore[arg-type]


@pytest.mark.anyio()
async def test_noop_different_results_do_not_trigger(cap_warn: StuckLoopDetection):
    ctx: object = _FakeCtx()
    tc = _make_tc('search', {'query': 'foo'})
    td: object = _FakeToolDef()

    for i in range(5):
        result = await cap_warn.after_tool_execute(
            ctx, call=tc, tool_def=td, args={'query': 'foo'}, result=f'result_{i}'
        )  # type: ignore[arg-type]
        assert result == f'result_{i}'


@pytest.mark.anyio()
async def test_noop_different_tools_do_not_trigger(cap_warn: StuckLoopDetection):
    """Even with the same result, different tool names should not trigger no-op detection."""
    ctx: object = _FakeCtx()
    td: object = _FakeToolDef()

    for i in range(5):
        tc = _make_tc(f'tool_{i}', {'x': 1})
        result = await cap_warn.after_tool_execute(ctx, call=tc, tool_def=td, args={'x': 1}, result='same')  # type: ignore[arg-type]
        assert result == 'same'


# --- Text-only responses are ignored ---


@pytest.mark.anyio()
async def test_text_response_ignored(cap_warn: StuckLoopDetection):
    ctx: object = _FakeCtx()
    rctx: object = _FakeRequestContext()
    resp = _make_text_response('hello')

    for _ in range(5):
        result = await cap_warn.after_model_request(ctx, request_context=rctx, response=resp)  # type: ignore[arg-type]
        assert result is resp


# --- Custom warning message ---


@pytest.mark.anyio()
async def test_custom_warning_message():
    cap = StuckLoopDetection(max_repeated_calls=2, action='warn', warning_message='Stop looping!')
    ctx: object = _FakeCtx()
    rctx: object = _FakeRequestContext()
    tc = _make_tc('x')
    resp = _make_response(tc)

    await cap.after_model_request(ctx, request_context=rctx, response=resp)  # type: ignore[arg-type]
    with pytest.raises(ModelRetry, match='Stop looping!'):
        await cap.after_model_request(ctx, request_context=rctx, response=resp)  # type: ignore[arg-type]


# --- Default values ---


def test_defaults():
    cap = StuckLoopDetection()
    assert cap.max_repeated_calls == 3
    assert cap.action == 'warn'
    assert cap.warning_message == DEFAULT_WARNING_MESSAGE
    assert cap._call_history == []
    assert cap._result_history == []


# --- StuckLoopError attributes ---


def test_stuck_loop_error():
    err = StuckLoopError('test reason')
    assert err.reason == 'test reason'
    assert str(err) == 'test reason'


# --- Multiple tool calls in a single response ---


@pytest.mark.anyio()
async def test_multiple_tool_calls_per_response(cap_warn: StuckLoopDetection):
    """When a model response contains multiple tool calls, all are tracked."""
    ctx: object = _FakeCtx()
    rctx: object = _FakeRequestContext()
    tc = _make_tc('do_thing', {'a': 1})
    # Response with 3 identical tool calls should trigger immediately.
    resp = _make_response(tc, tc, tc)

    with pytest.raises(ModelRetry, match='do_thing.*identical arguments'):
        await cap_warn.after_model_request(ctx, request_context=rctx, response=resp)  # type: ignore[arg-type]


# --- Threshold of 1 ---


@pytest.mark.anyio()
async def test_threshold_one():
    """With max_repeated_calls=1, the very first call triggers detection."""
    cap = StuckLoopDetection(max_repeated_calls=1, action='error')
    ctx: object = _FakeCtx()
    rctx: object = _FakeRequestContext()
    tc = _make_tc('any_tool')
    resp = _make_response(tc)

    with pytest.raises(StuckLoopError):
        await cap.after_model_request(ctx, request_context=rctx, response=resp)  # type: ignore[arg-type]
