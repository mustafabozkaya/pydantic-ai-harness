# CODE Phase

Implement all changes from the plan.

## Input

- `.claude/skills/pr-loop/plan-output.md` -- implementation instructions
- `.claude/skills/pr-loop/goals.json` -- acceptance criteria and test expectations (write the tests specified here)

## Process

1. Implement each item from the plan
2. After all changes, run formatting and linting:

```bash
make format && make lint
```

3. If lint fails, fix the issues and re-run until clean
4. If type checking is relevant to the changes, run:

```bash
make typecheck 2>&1 | tee /tmp/typecheck-output.txt
```

NOTE: only run it globally once, after that, `uv run pyright` only on the files you know are affected

## Rules

- Follow the plan
- Implement both source changes AND the tests specified in `goals.json` / `plan-output.md`
- If an instruction is impossible to implement (contradicts the codebase or itself), set `current_phase` to `"PLAN"` in `ralph-state.json` with a note in `phase_history` explaining why, then exit
- Don't run tests in this phase -- VERIFY phase handles that

## Next Phase

Always -> **VERIFY**
