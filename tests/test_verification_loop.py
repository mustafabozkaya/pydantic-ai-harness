"""Tests for the VerificationLoop capability."""
# pyright: reportPrivateUsage=false

from __future__ import annotations

import warnings

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_harness.verification_loop import (
    VerificationLoop,
    VerificationResult,
    Verifier,
)


@pytest.fixture(params=['asyncio'])
def anyio_backend(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------


class TestBuildFeedback:
    def test_single_failure(self):
        feedback = VerificationLoop._build_feedback([('lint', 'unused import on line 5')], attempt=1)
        assert 'attempt 1' in feedback
        assert '- lint: unused import on line 5' in feedback

    def test_multiple_failures(self):
        failures = [('lint', 'error A'), ('test', 'error B')]
        feedback = VerificationLoop._build_feedback(failures, attempt=2)
        assert 'attempt 2' in feedback
        assert '- lint: error A' in feedback
        assert '- test: error B' in feedback


class TestRunVerifiers:
    @pytest.mark.anyio()
    async def test_all_pass(self):
        cap = VerificationLoop(
            verifiers=[
                Verifier(name='lint', check_fn=_pass_verifier),
                Verifier(name='test', check_fn=_pass_verifier),
            ],
        )
        failures = await cap._run_verifiers()
        assert failures == []

    @pytest.mark.anyio()
    async def test_one_fails(self):
        cap = VerificationLoop(
            verifiers=[
                Verifier(name='lint', check_fn=_pass_verifier),
                Verifier(name='test', check_fn=_fail_verifier('2 tests failed')),
            ],
        )
        failures = await cap._run_verifiers()
        assert len(failures) == 1
        assert failures[0] == ('test', '2 tests failed')

    @pytest.mark.anyio()
    async def test_all_fail(self):
        cap = VerificationLoop(
            verifiers=[
                Verifier(name='lint', check_fn=_fail_verifier('lint error')),
                Verifier(name='test', check_fn=_fail_verifier('test error')),
            ],
        )
        failures = await cap._run_verifiers()
        assert len(failures) == 2

    @pytest.mark.anyio()
    async def test_empty_verifiers(self):
        cap = VerificationLoop(verifiers=[])
        failures = await cap._run_verifiers()
        assert failures == []


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_defaults():
    cap = VerificationLoop()
    assert cap.verifiers == []
    assert cap.max_retries == 3


# ---------------------------------------------------------------------------
# Integration tests with a real agent
# ---------------------------------------------------------------------------


@pytest.mark.anyio()
async def test_all_pass_no_retry():
    """When all verifiers pass on the first run, the result is returned without retries."""
    call_count = 0

    async def always_pass() -> VerificationResult:
        nonlocal call_count
        call_count += 1
        return VerificationResult(passed=True, message='OK')

    cap = VerificationLoop(
        verifiers=[Verifier(name='check', check_fn=always_pass)],
        max_retries=3,
    )
    agent = Agent(TestModel(), output_type=str, capabilities=[cap])
    result = await agent.run('Do something')
    assert isinstance(result.output, str)
    # Verifiers called once (after the initial run).
    assert call_count == 1


@pytest.mark.anyio()
async def test_retry_on_failure_then_pass():
    """When verification fails once, the agent retries and succeeds."""
    attempts = 0

    async def pass_on_second() -> VerificationResult:
        nonlocal attempts
        attempts += 1
        if attempts <= 1:
            return VerificationResult(passed=False, message='lint error on line 5')
        return VerificationResult(passed=True, message='OK')

    cap = VerificationLoop(
        verifiers=[Verifier(name='lint', check_fn=pass_on_second)],
        max_retries=3,
    )
    agent = Agent(TestModel(), output_type=str, capabilities=[cap])
    result = await agent.run('Fix the code')
    assert isinstance(result.output, str)
    # Verifier called twice: once after initial run (fail), once after retry (pass).
    assert attempts == 2


@pytest.mark.anyio()
async def test_max_retries_exceeded():
    """When verification keeps failing, a warning is emitted and last result is returned."""
    call_count = 0

    async def always_fail() -> VerificationResult:
        nonlocal call_count
        call_count += 1
        return VerificationResult(passed=False, message='still broken')

    cap = VerificationLoop(
        verifiers=[Verifier(name='test', check_fn=always_fail)],
        max_retries=2,
    )
    agent = Agent(TestModel(), output_type=str, capabilities=[cap])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        result = await agent.run('Fix the code')

    assert isinstance(result.output, str)
    # 2 (in-loop checks, one per retry attempt) + 1 (final check after loop) = 3
    assert call_count == 3
    assert len(w) == 1
    assert 'after 2 retries' in str(w[0].message)
    assert 'test: still broken' in str(w[0].message)


@pytest.mark.anyio()
async def test_multiple_verifiers_partial_failure():
    """Only failing verifiers appear in the retry feedback."""
    lint_calls = 0
    test_calls = 0

    async def lint_check() -> VerificationResult:
        nonlocal lint_calls
        lint_calls += 1
        return VerificationResult(passed=True, message='OK')

    async def test_check() -> VerificationResult:
        nonlocal test_calls
        test_calls += 1
        if test_calls <= 1:
            return VerificationResult(passed=False, message='1 test failed')
        return VerificationResult(passed=True, message='OK')

    cap = VerificationLoop(
        verifiers=[
            Verifier(name='lint', check_fn=lint_check),
            Verifier(name='test', check_fn=test_check),
        ],
        max_retries=3,
    )
    agent = Agent(TestModel(), output_type=str, capabilities=[cap])
    result = await agent.run('Fix things')
    assert isinstance(result.output, str)
    # lint called twice (initial + after retry), test called twice (initial fail + retry pass).
    assert lint_calls == 2
    assert test_calls == 2


@pytest.mark.anyio()
async def test_no_verifiers_passthrough():
    """With no verifiers configured, the run proceeds without any verification."""
    cap = VerificationLoop(verifiers=[], max_retries=3)
    agent = Agent(TestModel(), output_type=str, capabilities=[cap])
    result = await agent.run('Hello')
    assert isinstance(result.output, str)


@pytest.mark.anyio()
async def test_feedback_message_contains_verifier_info():
    """Verify that the feedback message sent on retry contains the verifier name and error."""
    check_calls = 0

    async def fail_once() -> VerificationResult:
        nonlocal check_calls
        check_calls += 1
        if check_calls <= 1:
            return VerificationResult(passed=False, message='type error on line 10')
        return VerificationResult(passed=True, message='OK')

    cap = VerificationLoop(
        verifiers=[Verifier(name='typecheck', check_fn=fail_once)],
        max_retries=3,
    )
    agent = Agent(TestModel(), output_type=str, capabilities=[cap])

    result = await agent.run('Fix the code')
    # The retry run produces a new message history that includes the feedback prompt.
    # Serialize to JSON and check the feedback string is present.
    history_json = result.all_messages_json().decode()
    assert 'typecheck' in history_json
    assert 'type error on line 10' in history_json


@pytest.mark.anyio()
async def test_passes_on_final_check_after_loop():
    """When verification fails during retries but passes on the final check, no warning is emitted."""
    check_calls = 0

    async def pass_on_third() -> VerificationResult:
        nonlocal check_calls
        check_calls += 1
        # Fail on calls 1 and 2 (in-loop), pass on call 3 (final check after loop).
        if check_calls < 3:
            return VerificationResult(passed=False, message='still failing')
        return VerificationResult(passed=True, message='OK')

    cap = VerificationLoop(
        verifiers=[Verifier(name='build', check_fn=pass_on_third)],
        max_retries=2,
    )
    agent = Agent(TestModel(), output_type=str, capabilities=[cap])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        result = await agent.run('Fix the build')

    assert isinstance(result.output, str)
    assert check_calls == 3
    # No warning should have been emitted since the final check passed.
    assert len(w) == 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _pass_verifier() -> VerificationResult:
    return VerificationResult(passed=True, message='OK')


def _fail_verifier(message: str):
    async def _check() -> VerificationResult:
        return VerificationResult(passed=False, message=message)

    return _check
