# GOALS Phase

Define acceptance criteria and test expectations before planning. Transforms triage output into a contract that PLAN designs against and VERIFY validates.

## Input

Read `.claude/skills/pr-loop/triage.json` for classified items and CI failures.

## Process

1. Read `triage.json` to get all **do** items and **ci_failures**
2. For each actionable item, read the relevant source files and existing tests
3. Define what "done" looks like and what tests should verify it

### Per-Item Goal Definition

For each **do** item and CI failure:

1. **Acceptance criteria**: 1-3 concrete, verifiable statements of expected behavior after the fix/change
2. **Test expectations**: specify the test file, test name, and what it validates. Check for existing test files/patterns first -- extend them rather than creating new ones
3. **Bug items**: if the triage item describes a bug or CI failure, write a minimal reproduction script to `plan-research/mre-GOAL{id}.py`. Run it to confirm it reproduces the problem

### High-Risk File Detection

Check `git diff --name-only origin/main...HEAD` against high-risk patterns (see glossary). If any high-risk files are changed and no test file modifications exist in the diff, add an automatic goal requiring test changes.

## Output

Write `.claude/skills/pr-loop/goals.json` (see glossary for format).

## Rules

- Every **do** item and CI failure MUST have at least one goal
- Test expectations should be specific enough to write -- not vague ("test it works")
- Prefer extending existing test files over creating new ones
- Goal IDs use `GOAL` prefix; plan-gaps use `GAP` prefix
- MRE scripts go in `plan-research/` alongside plan evidence
- Do NOT edit source files or implement anything -- that's for CODE phase
- Do NOT write the actual tests -- just specify what they should verify

## Next Phase

Always -> **PLAN**
