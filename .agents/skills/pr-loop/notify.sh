#!/usr/bin/env bash
# notify.sh — Pluggable notification for headless Ralph Loop.
# Posts questions to the configured channel when BLOCKED_QUESTIONS fires.
#
# Usage: ./notify.sh <worktree_path> <questions_json_path> <state_json_path>
#
# Env vars:
#   RALPH_NOTIFY  — backend to use: "github" (default), "none"
set -euo pipefail

WORKTREE_PATH="$1"
QUESTIONS_PATH="$2"
STATE_PATH="$3"

BACKEND="${RALPH_NOTIFY:-github}"
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Format questions as markdown
format_questions() {
  jq -r '
    "## Ralph Loop: Questions Need Answers\n\n" +
    "The Ralph Loop is blocked on the following questions. " +
    "Please answer by editing `questions.json` or replying to this comment.\n\n" +
    (.questions | to_entries | map(
      .value |
      "### " + .id + ". " + .prompt + "\n" +
      (if .options then
        (.options | map("- `" + .id + "`: " + .label) | join("\n")) + "\n"
      else
        "_Free text answer expected._\n"
      end) +
      "- `" + .id + ".escalate`: Reply with this to escalate — I'\''ll post the question with full context on the PR for maintainer input.\n"
    ) | join("\n")) +
    "\n---\n_Posted by Ralph Loop (headless mode) at " + "'"$TIMESTAMP"'" + "_"
  ' "$QUESTIONS_PATH"
}

case "$BACKEND" in
  github)
    PR_NUMBER=$(jq -r '.pr_number' "$STATE_PATH")
    if [[ -z "$PR_NUMBER" || "$PR_NUMBER" == "null" ]]; then
      echo "ERROR: Cannot determine PR number from $STATE_PATH"
      exit 1
    fi

    BODY=$(format_questions)
    COMMENT_URL=$(gh pr comment "$PR_NUMBER" --body "$BODY" 2>&1 | grep -o 'https://[^ ]*' || true)

    # Update state with notification info
    jq --arg url "$COMMENT_URL" --arg ts "$TIMESTAMP" '
      .last_notification = {
        "channel": "github",
        "url": $url,
        "asked_at": $ts
      }
    ' "$STATE_PATH" > "$STATE_PATH.tmp" && mv "$STATE_PATH.tmp" "$STATE_PATH"

    echo "Posted questions to PR #$PR_NUMBER via GitHub comment."
    if [[ -n "$COMMENT_URL" ]]; then
      echo "Comment URL: $COMMENT_URL"
    fi
    ;;

  none)
    echo "Notification backend: none. Questions logged but not sent externally."
    echo "Questions pending in: $QUESTIONS_PATH"

    # Still update state for tracking
    jq --arg ts "$TIMESTAMP" '
      .last_notification = {
        "channel": "none",
        "url": null,
        "asked_at": $ts
      }
    ' "$STATE_PATH" > "$STATE_PATH.tmp" && mv "$STATE_PATH.tmp" "$STATE_PATH"
    ;;

  *)
    echo "ERROR: Unknown RALPH_NOTIFY backend: $BACKEND"
    echo "Supported: github, none"
    exit 1
    ;;
esac
