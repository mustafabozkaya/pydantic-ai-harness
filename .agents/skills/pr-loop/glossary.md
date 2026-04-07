# Ralph Loop Glossary — pydantic-harness

Shared vocabulary for `/pr-loop`. Read this file at startup.

## Phases

- **TRIAGE**: Fetch PR comments + CI status, classify each thread using DDD+
- **GOALS**: Define acceptance criteria and test expectations per actionable item. Output: `goals.json`. For bugs, write MRE scripts. For high-risk file changes without tests, auto-add a test requirement goal
- **PLAN**: Read-only analysis. Design implementation to satisfy `goals.json`. Write plan, don't edit source. After the initial plan, `run-loop.sh` runs a review/research sub-loop (max 3 iterations) before transitioning to CODE
  - **PLAN_REVIEW** (sub-phase): Read-only analysis of `plan-output.md`. Identifies researchable gaps (research directives for PLAN_RESEARCH) and user questions (-> BLOCKED_QUESTIONS). Output: `plan-gaps.json`
  - **PLAN_RESEARCH** (sub-phase): Researches gaps via source reading, doc search, `uv run` test scripts. Updates `plan-output.md` with evidence-backed findings. Evidence saved to `plan-research/`
- **CODE**: Implement changes per plan, run `make format && make lint` until clean
- **VERIFY**: Run targeted tests from `goals.json` expectations. Tests first (fail -> CODE, max 3 loops). Enforces regression test gate for high-risk files. Output: `test-report.json`
- **REVIEW**: Review diff against plan and project standards. If blocking issues -> loop back to CODE. If clean -> proceed
- **PUBLISH**: Commit, push, write thread resolution + dismiss confirmations to `questions.json`, enter BLOCKED_QUESTIONS
- **BLOCKED_QUESTIONS**: Waiting for answers to `questions.json`. Interactive: `run-loop.sh` opens `$EDITOR`. Headless: `notify.sh` posts to configured channel, then polls for answers. On answers: resume to `resume_phase`
- **WAIT**: Checks for merge conflicts with main (auto-resolves or blocks). Then pause (10min). Then transition to TRIAGE with incremented iteration
- **DONE**: Terminal state. No more actions needed

## Classifications (DDD+ Protocol)

- **do**: Actionable feedback -- implement it
- **dismiss**: Incorrect, outdated, or already addressed -- draft a reply explaining why
- **discuss**: Ambiguous or needs user decision -- write to `questions.json`, enter BLOCKED_QUESTIONS
- **waiting**: A "discuss" item where we're waiting on another person's response -- no action from us
- **done**: Previously classified item now resolved -- no further action

## Reviewer Priority

1. **DouweM** (maintainer) -- highest authority, always address
2. **samuelcolvin** (creator) -- high authority
3. **dmontagu** (maintainer) -- high authority
4. **dsfaccini** (maintainer) -- address
5. **adtyavrdhn** (contributor) -- address
6. **Trusted bots** (github-actions[bot]) -- address CI-related feedback
7. **Other contributors** -- address unless contradicts maintainer guidance

## Loop Modes

- **auto**: Launched via `run-loop.sh`. Each phase is a fresh `claude -p` process. WAIT sleeps 10min then continues
- **manual**: User invokes `/pr-loop` directly. Runs one phase per invocation. User reviews output, re-invokes when ready
- **managed**: Launched via `/manage-ralph` from an interactive Claude session. Claude acts as the loop manager. `run-loop.sh` exits on BLOCKED_QUESTIONS (exit 42) and WAIT (exit 43). The managing Claude handles questions, controls iteration pacing, and relaunches.
- **headless**: Launched via `single-ralph.sh --headless`. Like auto mode but designed for non-interactive environments. Key differences:
  - Skips `read -rp` prompts -- resumes silently on active state
  - Redirects stdout/stderr to `ralph-loop.log`
  - BLOCKED_QUESTIONS notifies via `notify.sh` and polls for answers instead of opening `$EDITOR`
  - Supports escalation: answer `"<id>.escalate"` to post a question to the PR for maintainer input
  - Configurable poll timeout via `RALPH_POLL_TIMEOUT` (default: 4 hours)

## State Files

All live in `.claude/skills/pr-loop/`:

| File | Purpose |
|------|---------|
| `ralph-state.json` | Current phase, iteration count, phase history, resume_phase, PID, headless flag |
| `ralph-loop.pid` | PID lockfile -- prevents duplicate loops on the same worktree |
| `ralph-loop.log` | Combined stdout/stderr log (headless mode only) |
| `ralph-stderr.log` | Stderr from `claude -p` invocations (always written) |
| `triage-input.json` | Pre-fetched PR data (30min TTL) |
| `triage.json` | Classifications output from TRIAGE phase |
| `goals.json` | Acceptance criteria + test expectations from GOALS phase |
| `questions.json` | Questions for user when BLOCKED_QUESTIONS (see format below) |
| `plan-output.md` | Implementation plan from PLAN phase |
| `plan-stream.jsonl` | Temporary: raw stream-json from PLAN phase (deleted after extraction) |
| `plan-gaps.json` | Gaps identified by PLAN_REVIEW, findings filled by PLAN_RESEARCH (see format below) |
| `plan-research/` | Evidence directory: test scripts (`GAP{id}-test.py`), outputs (`GAP{id}-output.txt`) |
| `test-output.txt` | Raw pytest output from VERIFY phase |
| `test-report.json` | Structured test results from VERIFY phase (see format below) |
| `review-report.md` | Review findings from REVIEW phase |

## questions.json Format

```json
{
  "asked_at": "2026-04-06T10:00:00Z",
  "asked_by_phase": "TRIAGE",
  "questions": [
    {
      "id": "A",
      "prompt": "This reviewer says X but CI says Y. Which to follow?",
      "options": [
        {"id": "A.1", "label": "Follow reviewer"},
        {"id": "A.2", "label": "Follow CI"}
      ],
      "answer": null
    }
  ]
}
```

- `options` is optional -- if absent, `answer` accepts free text
- User fills `answer` with an option id (`"A.1"`) or free text
- All `answer` fields must be non-null for BLOCKED_QUESTIONS to resolve
- **Escalation** (headless only): answer `"<id>.escalate"` to have the question posted to the PR with full context for maintainer input. The answer is reset to `null` and polling continues until a real answer is provided.

## plan-gaps.json Format

```json
{
  "review_iteration": 1,
  "reviewed_at": "2026-04-06T10:00:00Z",
  "plan_is_solid": false,
  "gaps": [
    {
      "id": "GAP1",
      "description": "Plan assumes X but this is not verified",
      "category": "api_behavior",
      "research_strategy": "Read the source for X, write a test script to verify behavior",
      "finding": null,
      "evidence_file": null
    }
  ],
  "user_questions": [
    {
      "id": "UQ1",
      "description": "Should we support X or drop it?",
      "reason": "Scope decision -- not researchable"
    }
  ],
  "quality_flags": [
    {
      "id": "QF1",
      "description": "Item 3 over-specifies exact code",
      "severity": "info"
    }
  ]
}
```

- `gaps`: research directives for PLAN_RESEARCH. `finding` and `evidence_file` are filled by the researcher
- `category`: one of `api_behavior`, `compatibility`, `configuration`, `error_handling`, `ambiguous_path`
- `user_questions`: things only a human can decide -> triggers BLOCKED_QUESTIONS
- `quality_flags`: non-blocking, informational observations
- `plan_is_solid`: `true` when no unresolved gaps remain

## ralph-state.json Format

```json
{
  "version": 1,
  "pr_number": 42,
  "branch": "capability/memory",
  "iteration": 0,
  "max_iterations": 12,
  "current_phase": "TRIAGE",
  "resume_phase": null,
  "publish_step": null,
  "phase_history": [],
  "started_at": "2026-04-06T10:00:00Z",
  "last_updated": "2026-04-06T10:00:00Z",
  "loop_mode": "auto",
  "headless": false,
  "pid": null,
  "log_file": null,
  "plan_review_iteration": null,
  "last_notification": null
}
```

- `resume_phase`: set when entering BLOCKED_QUESTIONS -- the phase to return to after answers
- `publish_step`: tracks sub-step within PUBLISH (e.g. `"confirm"` after commit+push, awaiting thread resolution answers)
- `plan_review_iteration`: tracks position in the plan review/research sub-loop for BLOCKED_QUESTIONS resume. Deleted on CODE transition
- `headless`: whether the loop was started in headless mode
- `pid`: PID of the running `run-loop.sh` process (`null` when not running)
- `log_file`: path to the log file (headless mode only, `null` otherwise)
- `last_notification`: last notification sent during BLOCKED_QUESTIONS

## goals.json Format

```json
{
  "defined_at": "2026-04-06T10:00:00Z",
  "goals": [
    {
      "id": "GOAL1",
      "triage_ref": "<comment_id, thread_id, or ci_job>",
      "summary": "what needs to change",
      "acceptance_criteria": ["criterion 1", "criterion 2"],
      "test_expectations": [
        {
          "test_file": "tests/test_foo.py",
          "test_name": "test_bar_handles_edge_case",
          "description": "Verify that bar returns X when given Y"
        }
      ],
      "mre_script": "plan-research/mre-GOAL1.py"
    }
  ]
}
```

- Goal IDs use `GOAL` prefix; plan-gaps use `GAP` prefix
- `triage_ref`: links back to triage item
- `mre_script`: path to MRE script for bugs, `null` otherwise
- `test_expectations[].test_file`: prefer existing test files

## test-report.json Format

```json
{
  "tested_at": "2026-04-06T10:00:00Z",
  "status": "PASSING",
  "test_loop_count": 1,
  "goals_validated": [
    {
      "goal_id": "GOAL1",
      "tests_run": ["tests/test_foo.py::test_bar"],
      "result": "pass",
      "failure_detail": null
    }
  ],
  "regressions": [],
  "summary": "4/4 goals passing, 0 regressions"
}
```

- `status`: PASSING (all goals met), FAILING (any goal fails), PARTIAL (some tests missing)
- `test_loop_count`: incremented each CODE->VERIFY cycle

## High-Risk File Patterns

Files matching these patterns trigger the regression test gate in the VERIFY phase (require test changes in the diff):

- `src/pydantic_harness/*.py`
- `pyproject.toml`

## Project-Specific Commands

```bash
make format     # ruff format
make lint       # ruff check
make typecheck  # pyright strict
make test       # pytest
make testcov    # pytest with coverage
```

## Enforcement

- When the user uses a term not in this glossary for a defined concept, correct them
- When expanding the glossary, ensure no two concepts share the same name
- Define alternative phrasings as aliases

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RALPH_MANAGED` | `0` | Set to `1` by `--managed` flag or `/manage-ralph` skill. Exits on BLOCKED_QUESTIONS (42) and WAIT (43). |
| `RALPH_HEADLESS` | `0` | Set to `1` by `--headless` flag. Exported to child processes. |
| `RALPH_NOTIFY` | `github` | Notification backend for headless BLOCKED_QUESTIONS. |
| `RALPH_POLL_TIMEOUT` | `14400` (4h) | Max seconds to poll for answers before forcing DONE with `blocked_timeout`. |
