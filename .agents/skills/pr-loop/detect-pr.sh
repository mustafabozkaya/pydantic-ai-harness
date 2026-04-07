#!/usr/bin/env bash
# Detect the PR number for the current branch.
# Outputs the PR number on success, exits 1 on failure.
set -euo pipefail

BRANCH=$(git branch --show-current)

if [[ -z "$BRANCH" ]]; then
  echo "Error: Could not detect current branch (detached HEAD?)." >&2
  exit 1
fi

# Try direct lookup first
PR_NUMBER=$(gh pr view --json number -q .number 2>/dev/null || true)

if [[ -n "$PR_NUMBER" ]]; then
  echo "$PR_NUMBER"
  exit 0
fi

echo "Error: No PR found for branch '$BRANCH'." >&2
exit 1
