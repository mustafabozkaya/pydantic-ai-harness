"""Cost budget enforcement using CostGuard.

Demonstrates token budget limits that halt agent execution when
cumulative usage exceeds a threshold.

Usage:
    env-run .env -- uv run --group examples python examples/guardrails/cost_budget.py
"""

from __future__ import annotations

import logfire
from dotenv import load_dotenv
from pydantic_ai import Agent

from pydantic_ai_harness import BudgetExceededError, CostGuard

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()

agent = Agent(
    'openai:gpt-5.4-mini',
    capabilities=[CostGuard(max_total_tokens=150)],
    instructions='You are a helpful assistant. Answer questions concisely.',
)


@agent.tool_plain
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f'The weather in {city} is sunny and 22C.'


@agent.tool_plain
def get_population(city: str) -> str:
    """Get the population of a city."""
    return f'{city} has a population of approximately 2.1 million.'


async def main() -> None:
    """Run a multi-tool query that may exceed the token budget."""
    with logfire.span('cost budget — exceeded'):
        print('--- Running with tight token budget (150 total tokens) ---')
        try:
            result = await agent.run('Tell me about the weather and population of Paris, London, and Tokyo.')
            print(f'Response: {result.output}')
            print(f'Usage: {result.usage()}')
        except BudgetExceededError as e:
            print(f'Budget exceeded: {e.detail}')


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
