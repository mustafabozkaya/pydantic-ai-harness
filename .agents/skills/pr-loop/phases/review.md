# REVIEW Phase

Review changes for design and architecture issues against the plan and project standards.

## Input

1. Read `.claude/skills/pr-loop/plan-output.md` -- the plan that was implemented
2. Read `CLAUDE.md` for project coding standards

The plan provides context for *why* changes were made. Distinguish "deviates from plan intent" (potential BLOCKING) from "I would have done it differently" (not BLOCKING if the plan was intentional).

## Process

1. Run `git diff origin/main...HEAD` to see all changes
2. For each changed file, review against:
   - **Plan alignment**: does the implementation match the plan's intent?
   - **Project standards**: pyright strict compliance, ruff rules, 100% coverage requirement, single quotes, no `Any` types
   - **Security**: no command injection, no secrets in code, no unsafe deserialization
   - **API design**: consistent with existing patterns in `src/pydantic_harness/`
3. Write findings to `.claude/skills/pr-loop/review-report.md`
4. Evaluate findings:
   - If any **BLOCKING** issues -> append them to `plan-output.md` as new items, set next phase to **CODE**
   - If only warnings/info or clean -> set next phase to **PUBLISH**

## Severity Levels

- **BLOCKING**: security issues, type safety violations, broken API contracts, missing test coverage for high-risk changes
- **WARNING**: style inconsistencies, suboptimal patterns, minor naming issues
- **INFO**: suggestions for future improvement

## Rules

- Don't edit source files -- this is evaluation only
- Only BLOCKING severity triggers a loop-back to CODE
- Warnings and info are recorded in the report but don't block progress

## Next Phase

- If BLOCKING issues -> **CODE** (loop back)
- If clean or only warnings -> **PUBLISH**
