---
name: manage-ralph
description: Launch and manage the Ralph Loop from an interactive Claude session. Handles BLOCKED_QUESTIONS inline and controls pacing.
user-invocable: true
allowed-tools: Bash(git:*), Bash(gh:*), Bash(jq:*), Bash(.claude/skills/*), Read, Write, Edit, AskUserQuestion
---

# Manage Ralph — Interactive Loop Manager

Launch the Ralph Loop in managed mode from this interactive session. You act as the loop manager: run phases, handle questions inline, and control iteration pacing.

## Setup

1. Read `.claude/skills/pr-loop/glossary.md`
2. Verify `.claude/skills/pr-loop/ralph-state.json` exists (if not, ask user for PR number or run `detect-pr.sh`)

## Launch

```bash
RALPH_MANAGED=1 .claude/skills/pr-loop/run-loop.sh
```

## Exit Code Handling

- **Exit 0** — Loop complete (DONE). Report summary of all phases from `ralph-state.json` phase_history
- **Exit 42** — BLOCKED_QUESTIONS. Read `questions.json`, handle each question:
  - `resolve-*` questions: auto-approve (moderate autonomy)
  - `dismiss-*` questions: review the draft reply, use AskUserQuestion to confirm
  - discuss/other questions: use AskUserQuestion to get user input
  - Write answers back to `questions.json`, relaunch loop
- **Exit 43** — WAIT phase reached. Summarize the iteration, use AskUserQuestion to ask if user wants to continue or stop

## Rules

- Always show the user a summary after each BLOCKED_QUESTIONS resolution
- For moderate autonomy: auto-approve thread resolutions, ask for dismissals
- Keep the user informed of phase transitions
