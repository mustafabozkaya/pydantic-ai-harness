# PUBLISH Phase

Commit, push, and handle thread resolution via `questions.json`.

## Resuming from BLOCKED_QUESTIONS

If this phase is running after a BLOCKED_QUESTIONS->PUBLISH resume (check: `publish_step` is `"confirm"` in ralph-state.json):
1. Read `questions.json` answers
2. For approved thread resolutions:
   ```bash
   .claude/skills/github-workflows/resolve-threads <thread_id_1> <thread_id_2> ...
   ```
3. For approved dismiss replies:
   ```bash
   .claude/skills/github-workflows/reply-to-thread <comment_id> "<reply text>"
   ```
   (Skip any the user declined)
4. Delete `questions.json`, clear `publish_step` from ralph-state.json
5. Proceed to **Next Phase**

## Process

### 1. Commit and Push

```bash
git add <specific changed files>
git commit -m "<message summarizing addressed items>"
git push
```

Reference the addressed items briefly (e.g. "address review: fix typing, update docs, handle edge case").

### 2. Write Questions for Thread Resolution and Dismiss Replies

Build `questions.json` with two sections:

**Thread resolutions** -- for each "do" item from `triage.json` that was addressed:
```json
{
  "id": "resolve-<thread_id>",
  "prompt": "Resolve thread <thread_id>? (<comment snippet>)",
  "options": [
    {"id": "resolve-<thread_id>.yes", "label": "Resolve"},
    {"id": "resolve-<thread_id>.no", "label": "Skip"}
  ],
  "answer": null
}
```

**Dismiss replies** -- for each "dismiss" item from `triage.json`:
```json
{
  "id": "dismiss-<thread_id>",
  "prompt": "Post dismiss reply to <thread_id>?\n\nDraft: <draft_reply>",
  "options": [
    {"id": "dismiss-<thread_id>.yes", "label": "Post reply"},
    {"id": "dismiss-<thread_id>.no", "label": "Skip"}
  ],
  "answer": null
}
```

### 3. Enter BLOCKED_QUESTIONS

Set in ralph-state.json:
- `current_phase: "BLOCKED_QUESTIONS"`
- `resume_phase: "PUBLISH"`
- `publish_step: "confirm"`

Exit -- `run-loop.sh` will open `$EDITOR` for the user to approve/deny each action.

## Rules

- Thread resolution and dismiss replies always require user confirmation via BLOCKED_QUESTIONS
- Never force-push or amend published commits
- All dismiss replies must start with "Claude here: " per project conventions

## Next Phase

Always -> **WAIT** (run-loop.sh handles conflict check + 10min sleep externally)
