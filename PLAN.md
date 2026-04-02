# Approval Capability

## Problem

Agent tool calls that have real-world side effects (deleting files, sending emails,
executing commands) need human oversight. Without a structured approval mechanism,
developers either skip approval entirely or implement ad-hoc solutions that don't
compose well with the agent lifecycle.

## Solution

An `Approval` capability that uses `wrap_tool_execute` to intercept tool execution
and call a user-supplied approval callback before proceeding.

### Design

- **`tool_patterns`**: `fnmatch`-style glob patterns (e.g. `"delete_*"`, `"*"`)
  that determine which tools require approval.
- **`callback`**: Sync or async `(tool_name: str, args: dict) -> bool` function
  that the user provides to implement their approval UI.
- **`mode`**: Controls approval frequency:
  - `"always"`: Ask every time the tool is called.
  - `"once"`: Ask the first time, then auto-approve for the rest of the run.
  - `"never"`: Auto-approve all calls (useful for testing or trusted contexts).
- **Denial handling**: When denied (callback returns `False` or no callback configured),
  returns `"Tool execution was denied by user."` to the model instead of executing.
- **Per-run isolation**: `for_run()` returns a fresh instance with empty approved-tools
  set, so `mode="once"` memory doesn't leak across runs.

### Why `wrap_tool_execute` (not deferred tools)

The `DeferredToolCallsPresent` exception (harness #142) is the cleaner core primitive
for approval workflows, but it's not yet available. Using `wrap_tool_execute` provides
the same user-facing behavior without requiring core changes: the capability intercepts
execution, calls the callback inline, and either proceeds or returns a denial message.

When #142 lands, this capability can be updated to use `on_node_run_error` + deferred
tools for a more structured flow (with proper `DeferredToolRequests` support), but the
public API (`Approval(tool_patterns=..., callback=..., mode=...)`) would remain the same.

### Not spec-serializable

Because the capability takes a callable (`callback`), it returns `None` from
`get_serialization_name()` and cannot be constructed from YAML/JSON specs.

## References

- Harness #29: Approval capability
- Harness #142: DeferredToolCallsPresent exception
- Pydantic AI deferred tools / `ApprovalToolset`
- Claude Code: `permission_mode` with always/ask-once/ask-every-time
