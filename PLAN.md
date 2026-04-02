# VerificationLoop Capability

## Problem

Coding agents that make changes and hope they work are unreliable. The most
successful coding agents (Aider, Spotify's agent fleet, etc.) converge on
correctness by running an automated **verify-fix-repeat** loop after changes.
Without this, verification only happens when the agent remembers to check.

## Design

A `VerificationLoop` capability that uses the `wrap_run` hook to:

1. Run the agent normally via `handler()`
2. Execute a list of `Verifier` checks (e.g. lint, test, build)
3. If any verifier fails, re-run the agent with failure feedback appended to
   the conversation, so the model can fix the issues
4. Repeat until all verifiers pass or `max_retries` is exhausted

### Key types

- **`VerificationResult(passed: bool, message: str)`** -- outcome of a single
  check
- **`Verifier(name: str, check_fn: async () -> VerificationResult)`** -- a
  named check
- **`VerificationLoop(verifiers, max_retries=3)`** -- the capability

### Retry mechanics

Retries call `ctx.agent.run()` with the previous run's `message_history` plus
a feedback prompt containing the verifier names and failure messages. An
`_in_retry` flag prevents recursive verification when the retry run triggers
`wrap_run` again on the same capability instance.

If all retries are exhausted, the last result is returned and a
`UserWarning` is emitted.

## Files

- `src/pydantic_harness/verification_loop.py` -- capability implementation
- `src/pydantic_harness/__init__.py` -- public exports
- `tests/test_verification_loop.py` -- 15 tests, 100% coverage

## References

- pydantic-harness #79
