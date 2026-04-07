# VERIFY Phase

Run tests and fix mechanical lint issues. Single validation gate between CODE and REVIEW.

## Input

- `.claude/skills/pr-loop/goals.json` -- test expectations to validate
- `.claude/skills/pr-loop/plan-output.md` -- implementation plan (context on failures)

## Process

### Step 1: Run Tests

1. Read `goals.json` and extract all `test_expectations` entries
2. Run targeted tests:
   ```bash
   uv run pytest <test_files> -x --tb=short 2>&1 | tee .claude/skills/pr-loop/test-output.txt
   ```
3. If test file list is empty, run tests for changed modules instead:
   ```bash
   CHANGED=$(git diff --name-only origin/main...HEAD | grep '\.py$' | head -20)
   # Map source files to test files, run whatever exists
   uv run pytest <mapped_test_files> -x --tb=short 2>&1 | tee .claude/skills/pr-loop/test-output.txt
   ```
4. If tests fail:
   - Read `test-report.json` for `test_loop_count` (default 0 if missing)
   - If `test_loop_count < 3`: append failures to `plan-output.md` under `## Test Failures (iteration N)`, write `test-report.json`, transition to **CODE**
   - If `test_loop_count >= 3`: enter **BLOCKED_QUESTIONS** ("Tests have failed 3 times. Fix manually, skip, or abort?")

### Step 2: Regression Test Gate

Check `git diff --name-only origin/main...HEAD` for high-risk file patterns (see glossary). If high-risk files changed and zero test files in the diff, enter **BLOCKED_QUESTIONS**:
```json
{
  "id": "test-gate",
  "prompt": "High-risk files changed without test modifications. Add tests or skip?",
  "options": [
    {"id": "test-gate.add", "label": "Loop back to CODE to add tests"},
    {"id": "test-gate.skip", "label": "Skip -- change doesn't need tests"}
  ]
}
```

### Step 3: Lint

1. Run `make format && make lint`
2. Fix any flagged mechanical/style issues
3. Re-run until clean
4. Run `make typecheck` if source files were modified

## Output

Write `.claude/skills/pr-loop/test-report.json` (see glossary for format).

## Rules

- Run tests BEFORE lint -- no point linting code that doesn't pass tests
- Only fix mechanical/style issues in the lint step -- design issues are for REVIEW
- Don't introduce new logic changes during lint fixes
- Always write `test-report.json` even if all tests pass

## Next Phase

- If tests PASSING and lint clean -> **REVIEW**
- If tests FAILING and `test_loop_count < 3` -> **CODE**
- If tests FAILING and `test_loop_count >= 3` -> **BLOCKED_QUESTIONS** (resume_phase: VERIFY)
- If regression gate triggered -> **BLOCKED_QUESTIONS** (resume_phase: VERIFY)
