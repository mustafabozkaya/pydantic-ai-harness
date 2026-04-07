# PLAN_RESEARCH Sub-Phase

Research the gaps identified by the plan reviewer. Produce evidence-backed findings and update the plan.

## Input

- `.claude/skills/pr-loop/plan-gaps.json` -- gaps to research
- `.claude/skills/pr-loop/plan-output.md` -- current plan to update
- `.claude/skills/pr-loop/questions.json` -- user answers (if resuming from BLOCKED_QUESTIONS)

## Process

1. Read `plan-gaps.json` and `plan-output.md`
2. If `questions.json` exists with answered questions, incorporate user answers into `plan-output.md`
3. For each gap where `finding` is null, research it using the `research_strategy`:

   **Research techniques by category:**

   - **api_behavior**: Read the actual class/function definition. Check method signatures, return types, parameter names. If behavior is unclear from source, write a `uv run` test script:
     ```bash
     uv run python -c "
     from pydantic_ai.capabilities.abstract import AbstractCapability
     # test the actual behavior
     print('Result:', ...)
     "
     ```
   - **compatibility**: Check `pyproject.toml` for dependency versions. Search the codebase for existing patterns that solve the same problem
   - **configuration**: Read config schemas, dataclass/model definitions. Search for existing usage of the config option
   - **error_handling**: Read the source for exception types raised. Check existing error handling patterns in the codebase
   - **ambiguous_path**: Read related code to determine which path is correct. Write a test script if the behavior can be verified programmatically

4. For each gap researched:
   - Save test scripts to `.claude/skills/pr-loop/plan-research/GAP{id}-test.py`
   - Save script outputs to `.claude/skills/pr-loop/plan-research/GAP{id}-output.txt`
   - Fill in the `finding` field in `plan-gaps.json` with a concrete, evidence-backed answer
   - Set `evidence_file` to the path of the evidence artifact

5. Update `plan-output.md` in place:
   - Replace speculative language with concrete instructions based on findings
   - If research reveals the plan approach is wrong, rewrite the plan item with the correct approach
   - Cite evidence: file paths, function signatures, test script results

6. Write the updated `plan-gaps.json`

## Rules

- Every finding **must cite evidence**: source file path + relevant signature, or test script + output
- Do NOT update `ralph-state.json` -- `run-loop.sh` manages the sub-loop state
- Do NOT introduce new plan items -- only refine existing ones with research findings
- Do NOT modify source files outside the plan-research directory, `plan-output.md`, and `plan-gaps.json`

## Next Sub-Phase

Determined by `run-loop.sh`: always loops back to **PLAN_REVIEW** for verification.
