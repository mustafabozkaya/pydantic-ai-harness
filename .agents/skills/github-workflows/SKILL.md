---
name: github-workflows
description: Tools for fetching GitHub PR/issue/commit comments, checking CI/conflicts, listing/resolving review threads, and gathering PR data via gh CLI.
allowed-tools: Bash(.claude/skills/github-workflows/*)
---

# github-workflows

## fetch-pr-comments

Consolidated PR comment fetcher: inline review comments, issue comments, formal reviews, and review thread IDs (via GraphQL). Includes role tagging, configurable body truncation, and time filtering.

```bash
.claude/skills/github-workflows/fetch-pr-comments <pr_number> [options]
```

**Options:**
- `--since DAYS` — only comments from last N days (default: all)
- `--latest-round` — scope to latest review round (mutually exclusive with --since)
- `--max-body N` — truncate body to N chars (default: 500, 0 = no limit)
- `--repo OWNER/REPO` — repository (default: auto-detect)
- `--author AUTHOR` — PR author for role tagging (default: auto-detect from gh auth)

**Output:** JSON with `inline_comments`, `issue_comments`, `reviews`, `threads` (comment_id -> thread_id map), `comment_counts` (`{total, by_role}`). Each comment includes a `role` field: `author`, `maintainer`, `trusted-bot` (devin-ai-integration[bot], github-actions[bot]), `bot`, or `other`. Inline comments include `diff_hunk` (the code context the reviewer is commenting on).

## gather-pr-data

Collects mechanical data for all open PRs by an author: worktree mapping, conflicts, CI status, comments with role tags and thread IDs. Calls `check-branch-conflicts`, `check-pr-ci`, and `fetch-pr-comments` per PR. Runs `git fetch origin main` up front.

```bash
.claude/skills/github-workflows/gather-pr-data [author] [repo] [bare_repo_path]
```

Defaults: auto-detect user and repo; bare_repo_path is required. Output: JSON array to stdout with per-PR objects containing `number`, `branch`, `title`, `worktree`, `conflicts`, `conflict_files`, `ci`, `ci_failures`, `last_commit`, `review_comments`, `issue_comments`, `reviews`, `threads`, `comment_counts`.

## check-branch-conflicts

Checks if a worktree branch has merge conflicts with `origin/main` using read-only `merge-tree`. Assumes `origin/main` already fetched.

```bash
.claude/skills/github-workflows/check-branch-conflicts <worktree_path>
```

Output: `{"status": "CLEAN|BEHIND|CONFLICTS", "conflict_files": [...]}`.

## check-pr-ci

Returns structured CI status for a PR.

```bash
.claude/skills/github-workflows/check-pr-ci <pr_number> [repo]
```

Output: `{"status": "ALL_PASSING|PENDING|FAILING", "failures": [...]}`. Defaults repo to auto-detect.

## get-ci-failure-logs

Fetches the "Summary of Failures" log content from the latest failed test run on a PR.

```bash
.claude/skills/github-workflows/get-ci-failure-logs <pr_number> [repo]
```

Defaults repo to auto-detect. Use after `check-pr-ci` reports `FAILING` to get the actual error output.

## fetch-comment

Fetches PR/issue/commit comments with structured JSON output. For `discussion_r` refs, includes `is_resolved` via GraphQL lookup.

```bash
.claude/skills/github-workflows/fetch-comment <owner/repo> <comment_ref>
```

**comment_ref formats:** `discussion_r{id}`, `issuecomment-{id}`, `commitcomment-{id}`.

## unresolved-threads

Lists unresolved review threads on a PR with full conversation, dates, paths, and outdated status.

```bash
.claude/skills/github-workflows/unresolved-threads <pr_number> [owner] [repo]
```

Defaults to auto-detect. Returns JSON array of unresolved threads with full comments, createdAt, path, line, and isOutdated fields.

## reply-to-thread

Replies to a PR review thread. Accepts thread node IDs (from `unresolved-threads`), comment database IDs (numeric), or GitHub comment URLs.

```bash
.claude/skills/github-workflows/reply-to-thread <thread_ref> <body> [owner/repo]
```

**thread_ref formats:** `PRRT_...` node ID, numeric comment ID, or `https://...#discussion_r<id>` URL.

## resolve-threads

Resolves review threads by their GraphQL node IDs (from `unresolved-threads`). Optionally reply before resolving with `--reply`.

```bash
.claude/skills/github-workflows/resolve-threads [--reply <body>] <thread_id> [thread_id ...]
```

Output: `[{"id": "PRT_...", "resolved": true}, ...]`.
