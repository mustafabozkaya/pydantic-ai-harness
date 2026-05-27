# Mutation Testing Results

> Generated from commit `bd268c8` on 2026-05-26. Results may become stale as code
> evolves — regenerate via `uv run mutmut run --max-children 1`.

Covers `pydantic_ai_harness/filesystem/_toolset.py` and `pydantic_ai_harness/shell/_toolset.py`.

Run with [mutmut](https://mutmut.readthedocs.io/) v3 (`uv run mutmut run --max-children 1`).

## Summary

| Metric | Value |
|---|---|
| Total mutants | 584 |
| Killed | 524 |
| Survived | 60 |
| Kill rate | **89.7%** |

## Equivalent Mutants (60 survivors)

All 60 survivors are provably equivalent — no test can distinguish them from the original.

| Category | Count | Why unkillable |
|---|---|---|
| Trampoline default params | 7 | mutmut v3 wraps functions; wrapper keeps original defaults, so mutated defaults are never observed |
| `name=None` / omitted in `add_function()` | 18 | pydantic-ai falls back to `method.__name__`, which equals the original explicit name |
| Encoding case `'utf-8'` → `'UTF-8'` | 10 | Python's codec lookup is case-insensitive |
| Encoding omit/`None` (`utf-8` is default) | 11 | Default text encoding is UTF-8 on all supported platforms |
| Unreachable `except` blocks (`pragma: no cover`) | 6 | `except ValueError/OSError` paths can't be triggered in the test environment |
| `replace()` count removed/changed | 2 | Count is pre-validated as exactly 1 before the call |
| `CancelScope(shield=True)` → `False`/`None` | 2 | Requires an outer cancellation to fire during the ~instant cleanup window |
| Dead `returncode` branch | 1 | `proc.returncode` is never `None` after `await proc.wait()` |
| `errors='replace'` mutations | 3 | Test data is valid UTF-8; the error handler is never invoked |

## Limitations

Trio-parametrized tests are excluded during mutation testing (`-k 'not trio'` in
`pyproject.toml [tool.mutmut]`) because trio segfaults in mutmut's subprocess
environment on Python 3.14 / macOS. This does not affect the kill rate — trio
tests exercise the same code paths as the asyncio tests.

## Running

```bash
uv run mutmut run --max-children 1
uv run mutmut results
uv run mutmut show <mutant-name>
```
