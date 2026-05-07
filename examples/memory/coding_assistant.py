"""Self-Improving Coding Assistant — procedural memory via instructions injection.

Demonstrates: instructions injection as self-modifying prompt, scoping, search, delete.
"""

from __future__ import annotations

import sys

import logfire
from pydantic_ai import Agent

from pydantic_harness.memory import DictMemoryStore, Memory

logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_openai()


def main() -> None:
    """Run the coding assistant example."""
    store = DictMemoryStore()
    memory = Memory(store=store, max_instructions_memories=10)

    agent = Agent(
        'openai:gpt-4o-mini',
        capabilities=[memory],
        system_prompt=(
            'You are a coding assistant that learns from user corrections. '
            'When the user gives you a coding rule or correction, save it as a memory '
            'with scope "rules" and tags like ["python", "style"] or ["typescript", "testing"]. '
            'Use descriptive keys like "rule_python_fstrings" or "rule_ts_const". '
            'When asked to write code, search your memories for relevant rules first.'
        ),
    )

    # --- Teach rules ---
    with logfire.span('teach-rules'):
        result1 = agent.run_sync(
            'Remember these coding rules:\n'
            '1. Always use f-strings in Python, never .format() or % formatting\n'
            '2. In TypeScript, prefer const over let, never use var\n'
            '3. Always add type hints to Python function signatures'
        )
        print(f'Assistant: {result1.output}')

    rules = store.list_all()
    print(f'\nRules stored: {len(rules)}')
    for r in rules:
        print(f'  [{r.key}] {r.content} (scope={r.scope}, tags={r.tags})')

    assert len(rules) >= 3, f'Expected at least 3 rules saved, got {len(rules)}'

    # Check that search works across stored rules
    python_rules = store.search('python')
    print(f'Rules matching "python": {len(python_rules)}')
    assert len(python_rules) >= 1, 'Expected at least 1 rule matching "python"'

    # --- Verify instructions injection ---
    # Build instructions should now include the rules
    from unittest.mock import MagicMock

    from pydantic_ai._run_context import RunContext
    from pydantic_ai.usage import RunUsage

    ctx: RunContext[None] = RunContext(deps=None, model=MagicMock(), usage=RunUsage())
    instructions = memory.build_instructions(ctx)
    print(f'\nInstructions preview (first 300 chars):\n{instructions[:300]}...')

    assert 'Currently stored memories' in instructions, 'Expected memories in instructions'

    # --- Ask for code, verify rules are considered ---
    with logfire.span('apply-rules'):
        result2 = agent.run_sync(
            'Write a Python function that greets a user by name. Follow all coding rules you know.'
        )
        print(f'\nAssistant: {result2.output}')

    # The output should use f-strings and type hints (based on rules)
    output_lower = result2.output.lower()
    assert "f'" in result2.output or 'f"' in result2.output or 'f-string' in output_lower, (
        'Expected f-string usage in code output'
    )

    # --- Delete an obsolete rule ---
    with logfire.span('delete-rule'):
        result3 = agent.run_sync('Actually, the TypeScript const rule is outdated for this project. Delete it.')
        print(f'\nAssistant: {result3.output}')

    remaining = store.list_all()
    print(f'\nRules after deletion: {len(remaining)}')
    for r in remaining:
        print(f'  [{r.key}] {r.content}')

    # Should have fewer rules now
    assert len(remaining) < len(rules), 'Expected at least one rule deleted'

    print('\n--- Coding Assistant example passed! ---')


if __name__ == '__main__':
    sys.exit(main() or 0)
