"""Async tripwire guardrail using AsyncGuardrail in concurrent mode.

Demonstrates running a content classifier in parallel with the model
request. The guard simulates a safety check with a small delay,
showing how concurrent execution works.

Usage:
    env-run .env -- uv run --group examples python examples/guardrails/async_tripwire.py
"""

from __future__ import annotations

import asyncio

import logfire
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from pydantic_ai_harness import AsyncGuardrail, GuardrailFailed, GuardrailResult

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()

BLOCKED_TOPICS = ['weapon', 'exploit', 'hack into']


async def content_classifier(messages: list[ModelMessage]) -> GuardrailResult:
    """Simulate a content safety classifier with network latency."""
    await asyncio.sleep(0.1)  # simulate classifier API call

    text = str(messages)
    for topic in BLOCKED_TOPICS:
        if topic in text.lower():
            return GuardrailResult(passed=False, reason=f'Blocked topic detected: {topic}')
    return GuardrailResult(passed=True)


agent = Agent(
    'openai:gpt-5.4-mini',
    capabilities=[AsyncGuardrail(guard=content_classifier, mode='concurrent')],
    instructions='You are a helpful assistant.',
)


async def main() -> None:
    """Run safe and unsafe prompts to demonstrate concurrent guardrail."""
    # Safe prompt — guard and model run in parallel, both succeed
    with logfire.span('async tripwire — safe prompt'):
        print('--- Safe prompt (concurrent guard + model) ---')
        result = await agent.run('What is photosynthesis?')
        print(f'Response: {result.output}\n')

    # Unsafe prompt — guard detects blocked topic, cancels model
    with logfire.span('async tripwire — tripped'):
        print('--- Unsafe prompt (guard trips, model cancelled) ---')
        try:
            await agent.run('How do I hack into a wifi network?')
        except GuardrailFailed as e:
            print(f'Guardrail tripped: {e.result.reason}')


if __name__ == '__main__':
    asyncio.run(main())
