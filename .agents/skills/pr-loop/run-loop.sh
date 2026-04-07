#!/usr/bin/env bash
# Ralph Loop dispatcher — runs one claude -p per phase until DONE.
# Adapted for pydantic-harness.
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "$0")" && pwd)"
STATE="$SKILL_DIR/ralph-state.json"
QUESTIONS="$SKILL_DIR/questions.json"

# Session naming: loop-{project}-{phase}-{run}
RUN_COUNT=0
PROJECT_NAME=$(basename "$(pwd)")

# PID lockfile — prevent duplicate loops on the same worktree
LOCKFILE="$SKILL_DIR/ralph-loop.pid"
if [[ -f "$LOCKFILE" ]] && kill -0 "$(cat "$LOCKFILE")" 2>/dev/null; then
  echo "ERROR: Ralph Loop already running (PID $(cat "$LOCKFILE")). Exiting."
  exit 1
fi
echo $$ > "$LOCKFILE"

# Write PID into ralph-state.json
if [[ -f "$STATE" ]]; then
  jq --argjson pid $$ '.pid = $pid' "$STATE" > "$STATE.tmp" && mv "$STATE.tmp" "$STATE"
fi

# Signal handling — track child PID for clean Ctrl+C exit
CHILD_PID=""
cleanup() {
  echo ""
  echo "Ralph Loop interrupted."
  if [[ -n "$CHILD_PID" ]]; then
    kill -- -"$CHILD_PID" 2>/dev/null || true
  fi
  rm -f "$LOCKFILE"
  # Clear PID from state
  if [[ -f "$STATE" ]]; then
    jq '.pid = null' "$STATE" > "$STATE.tmp" && mv "$STATE.tmp" "$STATE" 2>/dev/null || true
  fi
  exit 130
}
trap cleanup SIGINT SIGTERM

# Per-phase model override (bash 3 compatible)
phase_model() {
  case "$1" in
    TRIAGE) echo "sonnet" ;;
    *) echo "" ;;
  esac
}

# jq filter to render stream-json as human-readable terminal output.
JQ_RENDER='
if .type == "assistant" then
  (.message.content[]? |
    if .type == "text" then .text
    elif .type == "thinking" then "\n[thinking] " + (.thinking | split("\n")[0]) + "\n"
    elif .type == "tool_use" then "\n[tool] " + .name + "(" + (.input | tostring | .[0:80]) + ")\n"
    else empty end)
elif .type == "result" then
  "\n--- result ---\n" + .result + "\n"
else empty end
'

# Stream claude -p output with visible tool calls and thinking in the terminal.
run_claude() {
  local model_flag=""
  local model
  model=$(phase_model "$CURRENT_PHASE")
  if [[ -n "$model" ]]; then
    model_flag="--model $model"
  fi

  RUN_COUNT=$((RUN_COUNT + 1))
  local session_name="loop-${PROJECT_NAME}-$(echo "$CURRENT_PHASE" | tr '[:upper:]' '[:lower:]')-${RUN_COUNT}"

  ( claude $model_flag "$@" -p '/pr-loop' -n "$session_name" \
      --output-format stream-json --verbose 2>>"$SKILL_DIR/ralph-stderr.log" | \
    jq --unbuffered -rj "$JQ_RENDER" ) &
  CHILD_PID=$!
  wait $CHILD_PID
  CHILD_PID=""
}

# Extract just the final result text from stream-json (for plan capture).
extract_result() {
  jq -r 'select(.type == "result") | .result'
}

while true; do
  CURRENT_PHASE=$(jq -r '.current_phase // "TRIAGE"' "$STATE" 2>/dev/null || echo "TRIAGE")
  echo "=== Ralph Loop: phase $CURRENT_PHASE ==="

  case "$CURRENT_PHASE" in
    DONE)
      echo "Ralph Loop complete."
      rm -f "$LOCKFILE"
      if [[ -f "$STATE" ]]; then
        jq '.pid = null' "$STATE" > "$STATE.tmp" && mv "$STATE.tmp" "$STATE" 2>/dev/null || true
      fi
      break
      ;;
    WAIT)
      # --- Conflict check: auto-resolve or block ---
      CONFLICT_DONE=$(jq -r '.conflict_check_done // false' "$STATE" 2>/dev/null)
      if [[ "$CONFLICT_DONE" == "true" ]]; then
        jq 'del(.conflict_check_done)' "$STATE" > "$STATE.tmp" && mv "$STATE.tmp" "$STATE"
      else
        echo "Checking for merge conflicts with main..."
        git fetch origin main 2>/dev/null || true
        CONFLICT_STATUS=$(.claude/skills/github-workflows/check-branch-conflicts "." | jq -r '.status')
        if [[ "$CONFLICT_STATUS" == "CONFLICTS" ]]; then
          echo "Merge conflicts detected. Attempting auto-resolve..."
          if git merge origin/main --no-edit 2>/dev/null; then
            echo "Auto-merge succeeded. Pushing..."
            git push
          else
            git merge --abort
            CONFLICT_FILES=$(.claude/skills/github-workflows/check-branch-conflicts "." | jq -r '.conflict_files | join(", ")')
            echo "Auto-merge failed. Conflicting files: $CONFLICT_FILES"
            TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
            jq -n --arg files "$CONFLICT_FILES" --arg ts "$TIMESTAMP" '{
              asked_at: $ts,
              asked_by_phase: "WAIT",
              questions: [{
                id: "merge-conflict",
                prompt: ("Branch has merge conflicts with origin/main. Conflicting files: " + $files + "\n\nPlease resolve conflicts and push, then mark as answered."),
                answer: null
              }]
            }' > "$QUESTIONS"
            jq --arg ts "$TIMESTAMP" '
              .current_phase = "BLOCKED_QUESTIONS"
              | .resume_phase = "WAIT"
              | .conflict_check_done = true
              | .last_updated = $ts
            ' "$STATE" > "$STATE.tmp" && mv "$STATE.tmp" "$STATE"
            continue
          fi
        fi
      fi

      # --- Sleep 10 minutes ---
      echo "Sleeping 10 minutes..."
      WAIT_ELAPSED=0
      while [[ "$WAIT_ELAPSED" -lt 600 ]]; do
        sleep 60 &
        CHILD_PID=$!
        wait $CHILD_PID
        CHILD_PID=""
        WAIT_ELAPSED=$((WAIT_ELAPSED + 60))
        echo "[heartbeat] WAIT phase: ${WAIT_ELAPSED}s / 600s elapsed"
      done

      # --- Transition to next iteration ---
      jq '.current_phase = "TRIAGE" | .iteration += 1 | .last_updated = (now | strftime("%Y-%m-%dT%H:%M:%SZ"))' "$STATE" > "$STATE.tmp" && mv "$STATE.tmp" "$STATE"
      if [[ "${RALPH_MANAGED:-0}" == "1" ]]; then
        rm -f "$LOCKFILE"
        jq '.pid = null' "$STATE" > "$STATE.tmp" && mv "$STATE.tmp" "$STATE" 2>/dev/null || true
        exit 43
      fi
      ;;
    BLOCKED_QUESTIONS)
      echo "Phase is BLOCKED_QUESTIONS — questions need answers."
      if [[ ! -f "$QUESTIONS" ]]; then
        echo "ERROR: BLOCKED_QUESTIONS but $QUESTIONS not found. Resetting to TRIAGE."
        jq '.current_phase = "TRIAGE" | .last_updated = (now | strftime("%Y-%m-%dT%H:%M:%SZ"))' "$STATE" > "$STATE.tmp" && mv "$STATE.tmp" "$STATE"
        continue
      fi

      if [[ "${RALPH_MANAGED:-0}" == "1" ]]; then
        echo "MANAGED: BLOCKED_QUESTIONS — questions in $QUESTIONS"
        rm -f "$LOCKFILE"
        jq '.pid = null' "$STATE" > "$STATE.tmp" && mv "$STATE.tmp" "$STATE" 2>/dev/null || true
        exit 42
      elif [[ "${RALPH_HEADLESS:-0}" == "1" ]]; then
        # Headless: notify + poll
        "$SKILL_DIR/notify.sh" "$(pwd)" "$QUESTIONS" "$STATE"

        POLL_TIMEOUT=${RALPH_POLL_TIMEOUT:-14400}
        ELAPSED=0
        while true; do
          UNANSWERED=$(jq '[.questions[] | select(.answer == null or .answer == "")] | length' "$QUESTIONS")
          if [[ "$UNANSWERED" -eq 0 ]]; then
            echo "All questions answered."
            break
          fi

          if [[ "$ELAPSED" -ge "$POLL_TIMEOUT" ]]; then
            TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
            jq --arg ts "$TIMESTAMP" '
              .current_phase = "DONE"
              | del(.resume_phase)
              | .phase_history += [{"phase": "BLOCKED_QUESTIONS", "completed_at": $ts, "result": "blocked_timeout"}]
              | .last_updated = $ts
            ' "$STATE" > "$STATE.tmp" && mv "$STATE.tmp" "$STATE"
            echo "BLOCKED_QUESTIONS timed out after ${POLL_TIMEOUT}s. Forcing DONE."
            break 2
          fi

          # Check for escalation answers
          ESCALATIONS=$(jq -r '[.questions[] | select(.answer != null and (.answer | test("\\.escalate$")))] | length' "$QUESTIONS" 2>/dev/null || echo "0")
          if [[ "$ESCALATIONS" -gt 0 ]]; then
            echo "Escalation detected. Posting questions with context to PR..."
            PR_NUMBER=$(jq -r '.pr_number' "$STATE")

            jq -r '.questions[] | select(.answer != null and (.answer | test("\\.escalate$"))) | .id + "|" + .prompt' "$QUESTIONS" | while IFS='|' read -r QID QPROMPT; do
              ESCALATION_BODY="## Escalated Question ($QID)

$QPROMPT

_This question was escalated from the Ralph Loop because it needs maintainer input. Please reply with your answer._

---
_Escalated by Ralph Loop (headless mode)_"
              gh pr comment "$PR_NUMBER" --body "$ESCALATION_BODY" 2>/dev/null || true
              echo "Escalated question $QID to PR #$PR_NUMBER"
            done

            jq '
              .questions |= map(
                if .answer != null and (.answer | test("\\.escalate$"))
                then .answer = null
                else . end
              )
            ' "$QUESTIONS" > "$QUESTIONS.tmp" && mv "$QUESTIONS.tmp" "$QUESTIONS"
          fi

          echo "[poll] BLOCKED_QUESTIONS: $UNANSWERED unanswered, ${ELAPSED}s / ${POLL_TIMEOUT}s elapsed"
          sleep 60 &
          CHILD_PID=$!
          wait $CHILD_PID
          CHILD_PID=""
          ELAPSED=$((ELAPSED + 60))
        done
      else
        # Interactive: open editor for user to answer questions
        while true; do
          ${EDITOR:-nano} "$QUESTIONS"
          UNANSWERED=$(jq '[.questions[] | select(.answer == null or .answer == "")] | length' "$QUESTIONS")
          if [[ "$UNANSWERED" -eq 0 ]]; then
            break
          fi
          echo "WARNING: $UNANSWERED question(s) still unanswered. Re-opening editor..."
        done
      fi

      RESUME_PHASE=$(jq -r '.resume_phase // "TRIAGE"' "$STATE")
      TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
      jq --arg phase "$RESUME_PHASE" --arg ts "$TIMESTAMP" '
        .current_phase = $phase
        | del(.resume_phase)
        | .phase_history += [{"phase": "BLOCKED_QUESTIONS", "completed_at": $ts, "result": "questions_answered"}]
        | .last_updated = $ts
      ' "$STATE" > "$STATE.tmp" && mv "$STATE.tmp" "$STATE"
      echo "Questions answered. Resuming phase: $RESUME_PHASE"
      ;;
    PLAN)
      # Plan phase with review/research sub-loop.
      PLAN_FILE="$SKILL_DIR/plan-output.md"
      PLAN_STREAM="$SKILL_DIR/plan-stream.jsonl"
      GAPS_FILE="$SKILL_DIR/plan-gaps.json"
      RESEARCH_DIR="$SKILL_DIR/plan-research"
      MAX_PLAN_REVIEW_ITERS=3

      # Check if resuming from BLOCKED_QUESTIONS mid-sub-loop
      RESUME_ITER=$(jq -r '.plan_review_iteration // 0' "$STATE" 2>/dev/null || echo "0")

      if [[ "$RESUME_ITER" -gt 0 && -f "$PLAN_FILE" ]]; then
        echo "Resuming plan review sub-loop at iteration $RESUME_ITER"
        PLAN_REVIEW_ITER=$((RESUME_ITER - 1))
      else
        # Fresh start: generate initial plan
        rm -rf "$RESEARCH_DIR"
        mkdir -p "$RESEARCH_DIR"
        rm -f "$GAPS_FILE"
        PLAN_REVIEW_ITER=0

        MODEL_FLAG=""
        local_model=$(phase_model "PLAN")
        if [[ -n "$local_model" ]]; then
          MODEL_FLAG="--model $local_model"
        fi

        RUN_COUNT=$((RUN_COUNT + 1))
        plan_session_name="loop-${PROJECT_NAME}-plan-${RUN_COUNT}"

        ( claude $MODEL_FLAG -p '/pr-loop' -n "$plan_session_name" \
            --output-format stream-json --verbose 2>>"$SKILL_DIR/ralph-stderr.log" \
            > "$PLAN_STREAM" ) &
        CHILD_PID=$!
        wait $CHILD_PID
        CHILD_PID=""

        extract_result < "$PLAN_STREAM" > "$PLAN_FILE"
        jq --unbuffered -rj "$JQ_RENDER" < "$PLAN_STREAM" || true
        rm -f "$PLAN_STREAM"
      fi

      # Sub-loop: review -> research -> review -> ...
      while [[ "$PLAN_REVIEW_ITER" -lt "$MAX_PLAN_REVIEW_ITERS" ]]; do
        PLAN_REVIEW_ITER=$((PLAN_REVIEW_ITER + 1))
        echo "=== Plan Review iteration $PLAN_REVIEW_ITER / $MAX_PLAN_REVIEW_ITERS ==="

        # --- PLAN_REVIEW (read-only, capture stdout as plan-gaps.json) ---
        jq '.current_phase = "PLAN_REVIEW"' "$STATE" > "$STATE.tmp" && mv "$STATE.tmp" "$STATE"
        REVIEW_STREAM="$SKILL_DIR/plan-review-stream.jsonl"
        REVIEW_MODEL_FLAG=""
        local_model=$(phase_model "PLAN_REVIEW")
        if [[ -n "$local_model" ]]; then
          REVIEW_MODEL_FLAG="--model $local_model"
        fi

        RUN_COUNT=$((RUN_COUNT + 1))
        review_session_name="loop-${PROJECT_NAME}-plan_review-${RUN_COUNT}"

        ( claude $REVIEW_MODEL_FLAG -p '/pr-loop' -n "$review_session_name" \
            --output-format stream-json --verbose 2>>"$SKILL_DIR/ralph-stderr.log" \
            > "$REVIEW_STREAM" ) &
        CHILD_PID=$!
        wait $CHILD_PID
        CHILD_PID=""

        # Extract and validate JSON output
        REVIEW_RESULT=$(extract_result < "$REVIEW_STREAM")
        jq --unbuffered -rj "$JQ_RENDER" < "$REVIEW_STREAM" || true
        rm -f "$REVIEW_STREAM"

        if ! echo "$REVIEW_RESULT" | jq . > "$GAPS_FILE" 2>/dev/null; then
          echo "WARNING: Plan reviewer output invalid JSON. Skipping sub-loop, proceeding to CODE."
          break
        fi

        # Check if plan is solid
        PLAN_SOLID=$(jq -r '.plan_is_solid // false' "$GAPS_FILE")
        if [[ "$PLAN_SOLID" == "true" ]]; then
          echo "Plan is solid. Proceeding to CODE."
          break
        fi

        # Check for user questions -> BLOCKED_QUESTIONS
        USER_Q_COUNT=$(jq '[.user_questions // [] | .[]] | length' "$GAPS_FILE" 2>/dev/null || echo "0")
        if [[ "$USER_Q_COUNT" -gt 0 ]]; then
          echo "Plan review found $USER_Q_COUNT user question(s). Entering BLOCKED_QUESTIONS."
          jq '{
            asked_at: (now | strftime("%Y-%m-%dT%H:%M:%SZ")),
            asked_by_phase: "PLAN_REVIEW",
            questions: [.user_questions[] | {id: .id, prompt: .description, answer: null}]
          }' "$GAPS_FILE" > "$QUESTIONS"

          TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
          jq --arg ts "$TIMESTAMP" --argjson iter "$PLAN_REVIEW_ITER" '
            .current_phase = "BLOCKED_QUESTIONS"
            | .resume_phase = "PLAN"
            | .plan_review_iteration = $iter
            | .last_updated = $ts
          ' "$STATE" > "$STATE.tmp" && mv "$STATE.tmp" "$STATE"
          continue 2
        fi

        # Check for researchable gaps
        GAP_COUNT=$(jq '[.gaps // [] | .[] | select(.finding == null)] | length' "$GAPS_FILE" 2>/dev/null || echo "0")
        if [[ "$GAP_COUNT" -eq 0 ]]; then
          echo "No researchable gaps. Proceeding to CODE."
          break
        fi

        echo "Found $GAP_COUNT researchable gap(s). Running researcher..."

        # --- PLAN_RESEARCH (full access) ---
        jq '.current_phase = "PLAN_RESEARCH"' "$STATE" > "$STATE.tmp" && mv "$STATE.tmp" "$STATE"
        RESEARCH_MODEL_FLAG=""
        local_model=$(phase_model "PLAN_RESEARCH")
        if [[ -n "$local_model" ]]; then
          RESEARCH_MODEL_FLAG="--model $local_model"
        fi

        RUN_COUNT=$((RUN_COUNT + 1))
        research_session_name="loop-${PROJECT_NAME}-plan_research-${RUN_COUNT}"

        ( claude $RESEARCH_MODEL_FLAG -p '/pr-loop' -n "$research_session_name" \
            --output-format stream-json --verbose 2>>"$SKILL_DIR/ralph-stderr.log" | \
          jq --unbuffered -rj "$JQ_RENDER" ) &
        CHILD_PID=$!
        wait $CHILD_PID
        CHILD_PID=""
      done

      # Transition to CODE
      TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
      jq --arg ts "$TIMESTAMP" '
        .current_phase = "CODE"
        | del(.plan_review_iteration)
        | .phase_history += [{"phase": "PLAN", "completed_at": $ts, "result": "plan_written_and_reviewed"}]
        | .last_updated = $ts
      ' "$STATE" > "$STATE.tmp" && mv "$STATE.tmp" "$STATE"
      ;;
    PUBLISH)
      run_claude
      ;;
    *)
      run_claude
      ;;
  esac
done
