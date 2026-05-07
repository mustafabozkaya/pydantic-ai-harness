"""Study Coach — spaced repetition with TTL.

Demonstrates: TTL/expiration, save with ttl_minutes, list/search, tags.
"""

from __future__ import annotations

import sys

import logfire
from pydantic_ai import Agent

from pydantic_ai_harness.memory import DictMemoryStore, Memory

logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_openai()  # pyright: ignore[reportUnknownMemberType]


def main() -> None:
    """Run the study coach example."""
    store = DictMemoryStore()
    memory = Memory(store=store)

    agent = Agent(
        'openai:gpt-4o-mini',
        capabilities=[memory],
        system_prompt=(
            'You are a study coach that helps users learn facts. '
            'When the user provides a fact to learn, save it as a memory with '
            'tag "study" and a ttl_minutes value: use 1 for new/hard facts, '
            '60 for reviewed facts, and 1440 for mastered facts. '
            'Use descriptive keys like "biology_mitochondria" or "history_magna_carta".'
        ),
    )

    # --- Learn some facts ---
    with logfire.span('learn-facts'):
        result1 = agent.run_sync(
            'I need to learn these facts:\n'
            '1. Mitochondria are the powerhouse of the cell\n'
            '2. The Magna Carta was signed in 1215\n'
            '3. Water boils at 100 degrees Celsius at sea level'
        )
        print(f'Coach: {result1.output}')

    entries = store.list_all()
    print(f'\nFacts stored: {len(entries)}')
    for e in entries:
        print(f'  [{e.key}] {e.content} (tags={e.tags}, ttl={e.expires_at})')

    assert len(entries) >= 3, f'Expected at least 3 facts saved, got {len(entries)}'

    # Check that TTL was set on at least some entries
    entries_with_ttl = [e for e in entries if e.expires_at is not None]
    assert len(entries_with_ttl) >= 1, 'Expected at least 1 entry with TTL set'
    print(f'Entries with TTL: {len(entries_with_ttl)}')

    # Check tags
    entries_with_study_tag = [e for e in entries if 'study' in e.tags]
    assert len(entries_with_study_tag) >= 1, 'Expected at least 1 entry with "study" tag'

    # --- Search for facts ---
    with logfire.span('search-facts'):
        result2 = agent.run_sync('Search my memories for anything about biology.')
        print(f'\nCoach: {result2.output}')

    # --- List all facts ---
    with logfire.span('list-facts'):
        result3 = agent.run_sync('List all my study memories.')
        print(f'\nCoach: {result3.output}')

    print('\n--- Study Coach example passed! ---')


if __name__ == '__main__':
    sys.exit(main() or 0)
