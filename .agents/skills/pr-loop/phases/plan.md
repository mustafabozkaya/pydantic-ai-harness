# PLAN Phase

Design the implementation for all actionable items. **Read-only** -- do NOT edit source files.

## Input

- `.claude/skills/pr-loop/goals.json` -- acceptance criteria and test expectations (primary input)
- `.claude/skills/pr-loop/triage.json` -- classified items and CI failures (reference)

## Process

You are running in `--permission-mode plan` -- you can read/search the codebase but cannot write files. Your stdout is captured by `run-loop.sh` as `plan-output.md`.

1. Read `goals.json` to get acceptance criteria and test expectations for each item
2. Read `triage.json` for the full context behind each goal (comments, CI failures)
3. For each goal, read the relevant source files and test files referenced
4. Read `CLAUDE.md` and `CLAUDE.local.md` for conventions to respect
5. Design file-specific, actionable instructions that satisfy each goal's acceptance criteria AND create the specified tests
6. Output the plan in the format below -- your stdout becomes `plan-output.md`

**Do NOT update `ralph-state.json`** -- `run-loop.sh` handles the state transition after capturing your output.

## Output format

```markdown
# Implementation Plan

## Goal G1: [triage_ref] -- <summary>
- Acceptance criteria: <from goals.json>
- Files: <paths to modify>
- Changes: <specific instructions>
- Tests: <test file + test name from goals.json, with implementation guidance>
- Rationale: <why this approach satisfies the criteria>

## Goal G2: [triage_ref] -- <summary>
...
```

## Rules

- Do NOT edit any source files in this phase
- Do NOT run make/lint/tests -- that's for CODE phase
- Focus on understanding the codebase context and designing correct fixes

## Next Phase

Always -> **PLAN_REVIEW** (managed by `run-loop.sh` sub-loop, which eventually transitions to CODE)
