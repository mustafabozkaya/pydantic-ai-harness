# SubAgent Capability

## Problem

When building multi-agent systems with Pydantic AI, there's no reusable capability for delegating tasks from a parent (orchestrator) agent to specialized sub-agents. Users currently have to manually wire up tool functions that call `agent.run()`, duplicate boilerplate for description injection, and handle error cases like unknown agent names.

## Solution

A `SubAgent` capability (implementing `AbstractCapability`) that:

1. Accepts a dict of named `Agent` instances
2. Provides a `delegate_task(agent_name, task)` tool to the parent agent
3. Injects sub-agent descriptions into the system prompt so the parent knows what's available
4. Forwards the parent's `deps` to sub-agents (configurable via `pass_deps`)
5. Returns sub-agent output as a string tool result
6. Raises `ModelRetry` for unknown agent names (self-correcting)

## Design decisions

- **Synchronous delegation only** (for now): the `delegate_task` tool blocks until the sub-agent finishes. This is the simplest correct behavior and matches the "Agent-as-Tool" pattern from OpenAI Agents SDK. Async background tasks (#32 scope expansion) and full handoffs (#44) are left for follow-up.
- **Descriptions from agent metadata**: falls back through `agent.description`, `agent.name`, then a default. Users can also pass explicit `descriptions` dict.
- **Not spec-serializable**: since it takes `Agent` instances, YAML/JSON serialization is not supported (`get_serialization_name()` returns `None`).
- **`str()` conversion of output**: all sub-agent outputs are converted to string for the tool result, regardless of the sub-agent's `output_type`.

## Files

- `src/pydantic_harness/subagent.py` — the `SubAgent` capability
- `src/pydantic_harness/__init__.py` — re-exports `SubAgent`
- `tests/test_subagent.py` — 19 tests covering construction, instructions, toolset, end-to-end delegation (deps forwarding, unknown agent retry, multiple agents), and imports
- `pyproject.toml` — added `pytest-asyncio` dev dependency

## References

- Issue #32: SubAgent / Agent-as-Tool capability
- Issue #44: Handoff / Agent Transfer (follow-up, blocked by this)
- Prior art: [vstorm-co/subagents-pydantic-ai](https://github.com/vstorm-co/subagents-pydantic-ai), OpenAI Agents SDK handoffs, Google ADK sub_agents, Pydantic AI's `ImageGeneration` capability subagent pattern
