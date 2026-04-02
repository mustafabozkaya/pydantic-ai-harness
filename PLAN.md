# ToolBudget Capability

## Summary

Implements `ToolBudget`, an `AbstractCapability` subclass that limits how many times tools can be called per agent run.

## Design

### Core mechanism

1. **`prepare_tools`**: Before each model request, removes exhausted tools from the model's view so it stops trying to call them.
2. **`wrap_tool_execute`**: Counts each tool call and enforces budgets as a safety net (handles batch calls where the model calls the same tool multiple times in one response).
3. **`for_run`**: Returns a fresh instance per run for state isolation, so concurrent or sequential runs don't share counters.

### Configuration

- `max_total_calls: int | None` -- total call budget across all tools (None = unlimited)
- `max_per_tool: dict[str, int]` -- per-tool limits by name (unlisted tools are unlimited)
- `action: 'inform' | 'error'` -- what to do on violation:
  - `'inform'` (default): return a synthetic "Tool call budget exceeded" message as the tool result so the model can adapt
  - `'error'`: raise `ToolBudgetExceeded`

### Serialization

Tier P compatible: `- ToolBudget: {max_total_calls: 10, max_per_tool: {web_search: 3}}`

## Relationship to pydantic-ai #4359

Issue #4359 proposes _proactive budget awareness_ (injecting reminder messages like "5/10 tool calls used"). This capability takes a complementary _enforcement_ approach: hiding exhausted tools and blocking calls that exceed budgets. Both approaches are valid and can be composed.

## Files

- `src/pydantic_harness/tool_budget.py` -- `ToolBudget` capability and `ToolBudgetExceeded` exception
- `src/pydantic_harness/__init__.py` -- re-exports
- `tests/test_tool_budget.py` -- 16 tests, 100% coverage
- `pyproject.toml` -- added `anyio_mode = 'auto'` for async test support, pyright `executionEnvironments` for test files
