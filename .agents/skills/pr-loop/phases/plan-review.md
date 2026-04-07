# PLAN_REVIEW Sub-Phase

Analyze the implementation plan for researchable gaps. **Read-only** -- do NOT edit files.

## Input

- `.claude/skills/pr-loop/plan-output.md` -- the current plan
- `.claude/skills/pr-loop/triage.json` -- original actionable items
- `.claude/skills/pr-loop/plan-gaps.json` -- previous findings (if this is iteration 2+)

## Process

You are running in `--permission-mode plan`. Your stdout is captured by `run-loop.sh` as `plan-gaps.json`. Output **only** valid JSON -- no preamble, no markdown fences, no commentary outside the JSON.

1. Read `plan-output.md` and `triage.json`
2. For each plan item, read the source files it references to verify claims
3. If `plan-gaps.json` exists from a previous iteration, check whether prior findings resolved each gap
4. Identify **researchable gaps** -- claims or assumptions in the plan that can be verified by Claude but weren't. Categories:
   - **api_behavior**: plan references methods, parameters, or behaviors without verifying they exist or work as assumed
   - **compatibility**: version or dependency assumptions not checked against `pyproject.toml` or changelogs
   - **configuration**: config options assumed without verifying the schema
   - **error_handling**: success path designed but failure/edge cases not considered
   - **ambiguous_path**: "investigate whether..." or "check if..." left as TODOs instead of resolved
5. Identify **user questions** -- things **only a human can decide** that Claude has no way to resolve through research. These must pass a strict test: if Claude could answer it by reading code, docs, running a test script, or reasoning about the codebase, it is NOT a user question -- it belongs in `gaps` instead
6. Flag **quality issues** (non-blocking, informational)

**Do NOT update `ralph-state.json`** -- `run-loop.sh` manages the sub-loop state.

## Output format

Output valid JSON to stdout. Your stdout becomes `plan-gaps.json` (see glossary for format).

Set `plan_is_solid: true` when there are no unresolved gaps requiring research.

## Rules

- **Gaps are research directives**, not questions. Frame each gap as: "the plan assumes X -- verify X by doing Y"
- Each gap must include a concrete `research_strategy`
- Only flag gaps that are **researchable by Claude**
- Apply the strict test for `user_questions`: if Claude could answer it through research, it belongs in `gaps`
- Do NOT edit any files -- output JSON to stdout only

## Next Sub-Phase

Determined by `run-loop.sh`:
- `plan_is_solid: true` or no gaps -> **CODE**
- `user_questions` present -> **BLOCKED_QUESTIONS** (resume to PLAN)
- Researchable gaps -> **PLAN_RESEARCH**
