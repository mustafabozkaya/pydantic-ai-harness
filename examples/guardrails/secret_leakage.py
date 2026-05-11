"""Secret leakage prevention using OutputGuardrail.

Demonstrates checking model output for API key patterns and blocking
responses that would leak sensitive credentials.

Usage:
    env-run .env -- uv run --group examples python examples/guardrails/secret_leakage.py
"""

from __future__ import annotations

import re

import logfire
from dotenv import load_dotenv
from pydantic_ai import Agent

from pydantic_ai_harness import OutputBlocked, OutputGuardrail

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()

SECRET_PATTERNS = [
    re.compile(r'sk-[a-zA-Z0-9]{20,}'),  # OpenAI keys
    re.compile(r'ghp_[a-zA-Z0-9]{36,}'),  # GitHub PATs
    re.compile(r'AKIA[A-Z0-9]{16}'),  # AWS access keys
    re.compile(r'xoxb-[a-zA-Z0-9\-]+'),  # Slack bot tokens
    re.compile(r'Bearer\s+[a-zA-Z0-9\-._~+/]+=*'),  # Bearer tokens
]


def check_for_secrets(text: str) -> bool:
    """Return True if the text does NOT contain secret patterns."""
    return not any(pattern.search(text) for pattern in SECRET_PATTERNS)


agent = Agent(
    'openai:gpt-5.4-mini',
    capabilities=[OutputGuardrail(guard=check_for_secrets)],
    instructions='You are a helpful assistant. Repeat back exactly what the user says.',
)


async def main() -> None:
    """Run prompts that trigger secret detection in model output."""
    # Safe output
    with logfire.span('secret leakage — safe output'):
        print('--- Safe output ---')
        result = await agent.run('Hello, world!')
        print(f'Response: {result.output}\n')

    # Output containing a fake API key
    with logfire.span('secret leakage — blocked'):
        print('--- Output with secret ---')
        try:
            await agent.run('Please repeat: my key is sk-abc123def456ghi789jkl012mno345')
        except OutputBlocked as e:
            print(f'Blocked: {e}')


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
