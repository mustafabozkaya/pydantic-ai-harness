"""Personal Assistant — remembers user preferences across sessions.

Demonstrates: FileMemoryStore persistence, save/recall, instructions injection, tags, scoping.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import logfire
from pydantic_ai import Agent

from pydantic_harness.memory import FileMemoryStore, Memory

logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_openai()


def main() -> None:
    """Run the personal assistant example."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mem_path = Path(tmpdir) / 'preferences.json'
        store = FileMemoryStore(mem_path)
        memory = Memory(store=store)

        agent = Agent(
            'openai:gpt-4o-mini',
            capabilities=[memory],
            system_prompt=(
                'You are a helpful personal assistant. '
                'When the user tells you about their preferences, save each one as a memory '
                'with scope "user_prefs" and appropriate tags. '
                'Use descriptive keys like "preferred_name" or "theme_preference".'
            ),
        )

        # --- Session 1: user shares preferences ---
        with logfire.span('session-1-save-preferences'):
            result1 = agent.run_sync("Hi! My name is Alice, I prefer dark mode, and I'm vegetarian.")
            print(f'Assistant: {result1.output}')

        entries = store.list_all()
        print(f'\nMemories after session 1: {len(entries)}')
        for e in entries:
            print(f'  [{e.key}] {e.content} (tags={e.tags}, scope={e.scope})')

        assert len(entries) >= 2, f'Expected at least 2 memories saved, got {len(entries)}'
        all_content = ' '.join(e.content.lower() for e in entries)
        assert 'alice' in all_content or any('alice' in e.key.lower() for e in entries), 'Expected a memory about Alice'

        # --- Session 2: new agent instance loads from same file (persistence) ---
        store2 = FileMemoryStore(mem_path)
        memory2 = Memory(store=store2)
        agent2 = Agent(
            'openai:gpt-4o-mini',
            capabilities=[memory2],
            system_prompt='You are a helpful personal assistant.',
        )

        loaded_entries = store2.list_all()
        print(f'\nMemories loaded in session 2: {len(loaded_entries)}')
        assert len(loaded_entries) == len(entries), 'FileMemoryStore persistence failed'

        with logfire.span('session-2-recall-preferences'):
            result2 = agent2.run_sync('What do you know about me?')
            print(f'Assistant: {result2.output}')

        # The instructions injection should have included the memories
        assert 'alice' in result2.output.lower() or 'dark' in result2.output.lower(), (
            'Expected assistant to recall preferences from instructions injection'
        )

        # --- Session 3: update a preference ---
        with logfire.span('session-3-update-preference'):
            result3 = agent2.run_sync('Actually, I go by Ali now. Please update my name.')
            print(f'\nAssistant: {result3.output}')

        updated_entries = store2.list_all()
        print(f'\nMemories after update: {len(updated_entries)}')
        for e in updated_entries:
            print(f'  [{e.key}] {e.content} (tags={e.tags})')

        print('\n--- Personal Assistant example passed! ---')


if __name__ == '__main__':
    sys.exit(main() or 0)
