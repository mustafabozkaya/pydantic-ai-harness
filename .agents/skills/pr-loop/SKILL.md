---
name: pr-loop
description: Execute one phase of the Ralph Loop for pydantic-harness. Reads state, runs current phase, updates state, exits. Invoke manually or via run-loop.sh.
user-invocable: true
allowed-tools: Bash(git:*), Bash(gh:*), Bash(jq:*), Bash(make:*), Bash(uv:*), Bash(source:*), Bash(ls:*), Bash(cat:*), Bash(date:*), Bash(.claude/skills/github-workflows/*), Read, Write, Edit, Glob, Grep, Skill
---

# Start Worktree Loop — Phase Dispatcher

Execute one phase of the Ralph Loop. Each invocation runs exactly one phase, updates state, and exits.

## Startup

1. Read `.claude/skills/pr-loop/glossary.md` for vocabulary and rules
2. Read `CLAUDE.md` and `CLAUDE.local.md` for project context

## State Check

Read `.claude/skills/pr-loop/ralph-state.json`:
- If missing → initialize with `current_phase: "TRIAGE"`, `iteration: 0`, `max_iterations: 12`, generate `started_at` timestamp
- If `iteration >= max_iterations` → force `current_phase: "DONE"`, print warning: "Max iterations (N) reached. Forcing DONE.", update state file, exit
- If `current_phase` is `DONE` → print "Loop complete.", exit
- If `current_phase` is `WAIT` → print "In WAIT state. run-loop.sh handles the sleep. If running manually, change phase to TRIAGE when ready.", exit
- If `current_phase` is `BLOCKED_QUESTIONS`:
  - Read `questions.json` — if all questions have non-null `answer` fields → set `current_phase` to `resume_phase`, clear `resume_phase` from state, proceed to that phase
  - If any answers are missing → print "BLOCKED_QUESTIONS: unanswered questions in questions.json. Answer them and re-run.", exit

## Phase Dispatch

Read `.claude/skills/pr-loop/phases/{current_phase}.md` for this phase's instructions (lowercase, underscores become hyphens — e.g. `PLAN_REVIEW` → `plan-review.md`). Execute the phase.

## Entering BLOCKED_QUESTIONS State

Any phase that needs user input (instead of using AskUserQuestion, which doesn't work in `-p` mode):
1. Write `questions.json` to the skill dir (see glossary for format)
2. Set `current_phase: "BLOCKED_QUESTIONS"` and `resume_phase: "<current_phase>"` in ralph-state.json
3. Exit immediately — `run-loop.sh` will open `$EDITOR` for the user

## State Update

After phase execution, update `ralph-state.json`:
- Set `current_phase` to the next phase (determined by the phase instructions)
- Append to `phase_history`: `{"phase": "<completed_phase>", "completed_at": "<ISO timestamp>", "result": "<brief summary>"}`
- Update `last_updated` timestamp
- For TRIAGE: increment `iteration`

**Exception — PLAN phase**: You run under `--permission-mode plan` (no Write/Edit/Bash). `run-loop.sh` captures your stdout as `plan-output.md` and updates `ralph-state.json` via `jq`. Just output the plan and exit. After capturing the plan, `run-loop.sh` runs a review/research sub-loop (max 3 iterations): PLAN_REVIEW identifies researchable gaps, PLAN_RESEARCH resolves them with evidence. The sub-phases dispatch via `current_phase` in `ralph-state.json` — read `phases/{current_phase}.md` as usual.

Then exit. The next invocation (manual or via run-loop.sh) will pick up the new phase.

## Quick Start: single-ralph.sh

Bootstrap and launch the loop from within a worktree without needing external orchestration:

```bash
cd /path/to/worktree
.claude/skills/pr-loop/single-ralph.sh                    # foreground (interactive)
.claude/skills/pr-loop/single-ralph.sh --tmux             # in tmux window
.claude/skills/pr-loop/single-ralph.sh --headless         # headless (no terminal needed)
.claude/skills/pr-loop/single-ralph.sh --headless --reset # headless, fresh start
```

Detects branch/PR automatically, writes `ralph-state.json`, and starts `run-loop.sh`. If state already exists with an active phase, prompts to reset or resume (interactive) or resumes silently (headless).

Flags:
- `--tmux`: launch in a tmux window
- `--headless`: non-interactive mode — skips prompts, redirects to log, uses notify+poll for questions
- `--reset`: delete existing `ralph-state.json` before initializing (replaces the interactive reset prompt)
