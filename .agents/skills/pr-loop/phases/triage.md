# TRIAGE Phase

Fetch PR data, classify comments, determine if work is needed.

## Resuming from BLOCKED_QUESTIONS

If this phase is running after a BLOCKED_QUESTIONS->TRIAGE resume (check: `questions.json` exists with all answers filled):
1. Read `questions.json` answers
2. Read existing `triage.json`
3. Reclassify each "discuss" item based on user answers: map to **do**, **dismiss**, or **done**
4. Rewrite `triage.json` with updated classifications
5. Delete `questions.json`
6. Recalculate `actionable_count` and skip to **Next Phase** logic

## Data Source

Check `.claude/skills/pr-loop/triage-input.json`:
- If it exists and was modified within the last 30 minutes -> use it (pre-fetched)
- Otherwise -> fetch fresh data using incremental logic below

### Incremental Fetch

Read `ralph-state.json` for `iteration` and `phase_history`:

**Getting the PR number**: read `pr_number` from `ralph-state.json`. If null or missing, run `.claude/skills/pr-loop/detect-pr.sh` and update the state.

**First iteration** (`iteration == 0`): fetch all comments.

```bash
PR_NUMBER=$(jq -r '.pr_number' .claude/skills/pr-loop/ralph-state.json)
.claude/skills/github-workflows/fetch-pr-comments "$PR_NUMBER" --max-body 0 > .claude/skills/pr-loop/triage-input.json
.claude/skills/github-workflows/check-pr-ci "$PR_NUMBER" >> .claude/skills/pr-loop/triage-input.json
```

**Subsequent iterations** (`iteration > 0`):
1. Keep the previous `triage.json` as baseline
2. Find the last PUBLISH phase `completed_at` from `phase_history` (fallback: `started_at` from state)
3. Calculate days since that timestamp (minimum 1)
4. Fetch only recent comments:
   ```bash
   PR_NUMBER=$(jq -r '.pr_number' .claude/skills/pr-loop/ralph-state.json)
   .claude/skills/github-workflows/fetch-pr-comments "$PR_NUMBER" --since $DAYS --max-body 0 > .claude/skills/pr-loop/triage-input.json
   .claude/skills/github-workflows/check-pr-ci "$PR_NUMBER" >> .claude/skills/pr-loop/triage-input.json
   ```
5. Merge with previous triage: carry forward **done** and **waiting** items from previous `triage.json`, reclassify only threads with new activity

Also check for CI failures:
```bash
.claude/skills/github-workflows/get-ci-failure-logs "$PR_NUMBER"
```

## Classification

For every non-author comment thread:
1. Group comments by thread (`in_reply_to_id`) -- use the final state of each thread
2. Classify each thread as: **do**, **dismiss**, **discuss**, **waiting**, or **done** (see glossary)
3. Apply reviewer priority (see glossary)
4. Include CI failures as implicit "do" items

### Handling "discuss" Items

If any threads are classified as **discuss**:
1. Write `questions.json` with one question per discuss item (use thread_id as question id prefix)
2. Include options where possible (e.g. "Follow reviewer suggestion" / "Keep current approach" / "Other")
3. Set `current_phase: "BLOCKED_QUESTIONS"`, `resume_phase: "TRIAGE"` in ralph-state.json
4. Exit -- the loop will resume TRIAGE after user answers

## Output

Write `.claude/skills/pr-loop/triage.json`:

```json
{
  "pr_number": 42,
  "triaged_at": "2026-04-06T10:00:00Z",
  "last_fetch_cutoff": null,
  "classifications": {
    "do": [{"comment_id": "...", "thread_id": "...", "path": "...", "summary": "..."}],
    "dismiss": [{"comment_id": "...", "thread_id": "...", "path": "...", "summary": "...", "draft_reply": "..."}],
    "discuss": [{"comment_id": "...", "thread_id": "...", "path": "...", "summary": "...", "question": "..."}],
    "waiting": [{"comment_id": "...", "thread_id": "...", "path": "...", "summary": "..."}],
    "done": [{"comment_id": "...", "thread_id": "...", "summary": "..."}]
  },
  "ci_failures": [{"job": "...", "summary": "...", "log_snippet": "..."}],
  "actionable_count": 5
}
```

## Next Phase

- If `actionable_count == 0` (no "do" items and no CI failures) -> set phase to **DONE**
- If any "discuss" items -> enter **BLOCKED_QUESTIONS** (see above, do not proceed to GOALS)
- Otherwise -> set phase to **GOALS**
