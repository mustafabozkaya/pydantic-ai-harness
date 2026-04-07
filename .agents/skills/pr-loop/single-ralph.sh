#!/usr/bin/env bash
# single-ralph — Bootstrap and launch the Ralph Loop for the current worktree.
# Run from worktree root. Detects PR info, writes ralph-state.json, starts loop.
set -euo pipefail

SKILL_DIR=".claude/skills/pr-loop"
STATE="$SKILL_DIR/ralph-state.json"
TMUX_MODE=false
HEADLESS_MODE=false
RESET_MODE=false
MANAGED_MODE=false

for arg in "$@"; do
  case "$arg" in
    --tmux) TMUX_MODE=true ;;
    --headless) HEADLESS_MODE=true ;;
    --managed) MANAGED_MODE=true ;;
    --reset) RESET_MODE=true ;;
    *) echo "Unknown arg: $arg"; echo "Usage: $0 [--tmux] [--headless] [--managed] [--reset]"; exit 1 ;;
  esac
done

# Export mode flags for child processes
if $HEADLESS_MODE; then
  export RALPH_HEADLESS=1
fi
if $MANAGED_MODE; then
  export RALPH_MANAGED=1
fi

# Verify we're in a repo with the skill dir
if [[ ! -d "$SKILL_DIR" ]]; then
  echo "Error: $SKILL_DIR not found. Run this from the pydantic-harness repo root."
  exit 1
fi

# Detect branch
BRANCH=$(git branch --show-current)
if [[ -z "$BRANCH" ]]; then
  echo "Error: Could not detect current branch (detached HEAD?)."
  exit 1
fi
echo "Branch: $BRANCH"

# Detect PR number
PR_NUMBER=$("$SKILL_DIR/detect-pr.sh") || {
  echo "Error: No PR found for branch '$BRANCH'. Create a PR first."
  exit 1
}
echo "PR: #$PR_NUMBER"

# Handle --reset: delete existing state before initializing
if $RESET_MODE && [[ -f "$STATE" ]]; then
  echo "Reset flag: deleting existing ralph-state.json."
  rm -f "$STATE"
fi

# Check existing state
if [[ -f "$STATE" ]]; then
  CURRENT_PHASE=$(jq -r '.current_phase // "UNKNOWN"' "$STATE")
  if [[ "$CURRENT_PHASE" != "DONE" && "$CURRENT_PHASE" != "WAIT" ]]; then
    if $HEADLESS_MODE; then
      echo "Headless: resuming from existing state (phase: $CURRENT_PHASE)."
    else
      echo "Warning: ralph-state.json exists with active phase '$CURRENT_PHASE'."
      read -rp "Reset state and start fresh? (y/n) " ANSWER
      if [[ "$ANSWER" != "y" && "$ANSWER" != "Y" ]]; then
        echo "Resuming from existing state."
      else
        rm -f "$STATE"
      fi
    fi

    # If state file still exists, resume
    if [[ -f "$STATE" ]]; then
      if $HEADLESS_MODE; then
        LOG_FILE="$SKILL_DIR/ralph-loop.log"
        echo "Headless: redirecting output to $LOG_FILE"
        exec >>"$LOG_FILE" 2>&1
      fi
      if $TMUX_MODE; then
        exec tmux new-window -n "PR-$PR_NUMBER" "cd $(pwd) && $SKILL_DIR/run-loop.sh"
      else
        exec "$SKILL_DIR/run-loop.sh"
      fi
    fi
  fi
fi

# Write initial ralph-state.json
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
LOG_FILE=""
if $HEADLESS_MODE; then
  LOG_FILE="$SKILL_DIR/ralph-loop.log"
fi

cat > "$STATE" <<EOF
{
  "version": 1,
  "pr_number": $PR_NUMBER,
  "branch": "$BRANCH",
  "iteration": 0,
  "max_iterations": 12,
  "current_phase": "TRIAGE",
  "phase_history": [],
  "started_at": "$TIMESTAMP",
  "last_updated": "$TIMESTAMP",
  "loop_mode": "auto",
  "headless": $($HEADLESS_MODE && echo "true" || echo "false"),
  "log_file": $(if [[ -n "$LOG_FILE" ]]; then echo "\"$LOG_FILE\""; else echo "null"; fi),
  "pid": null
}
EOF
echo "Wrote $STATE (TRIAGE, iteration 0)"

# Redirect output in headless mode
if $HEADLESS_MODE; then
  echo "Headless: redirecting output to $LOG_FILE"
  exec >>"$LOG_FILE" 2>&1
fi

# Launch
if $TMUX_MODE; then
  tmux new-window -n "PR-$PR_NUMBER" "cd $(pwd) && $SKILL_DIR/run-loop.sh"
  echo "Launched in tmux window PR-$PR_NUMBER"
else
  exec "$SKILL_DIR/run-loop.sh"
fi
