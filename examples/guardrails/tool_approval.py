"""Tool approval workflow using ToolGuard.

Demonstrates blocking dangerous tools entirely and requiring interactive
approval for sensitive tools via terminal confirmation.

Usage:
    env-run .env -- uv run --group examples python examples/guardrails/tool_approval.py
"""

from __future__ import annotations

from typing import Any

import logfire
from dotenv import load_dotenv
from pydantic_ai import Agent

from pydantic_ai_harness import ToolGuard

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()


def terminal_approval(tool_name: str, args: dict[str, Any]) -> bool:
    """Prompt the user in the terminal for tool approval."""
    print(f'\n  Tool: {tool_name}')
    print(f'  Args: {args}')
    response = input('  Approve? (y/n): ').strip().lower()
    return response == 'y'


agent = Agent(
    'openai:gpt-5.4-mini',
    capabilities=[
        ToolGuard(
            blocked=['drop_database'],
            require_approval=['send_email'],
            approval_callback=terminal_approval,
        ),
    ],
    instructions='You are a helpful assistant with access to tools.',
)


@agent.tool_plain
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    return f'Email sent to {to} with subject "{subject}".'


@agent.tool_plain
def drop_database(name: str) -> str:
    """Drop a database by name."""
    return f'Database {name} dropped.'


@agent.tool_plain
def list_files(directory: str) -> str:
    """List files in a directory."""
    return f'Files in {directory}: readme.md, app.py, config.yaml'


async def main() -> None:
    """Demonstrate tool blocking and approval workflow."""
    # The model cannot see drop_database (blocked), will need approval for send_email,
    # and can freely use list_files.
    with logfire.span('tool approval — blocked + approval flow'):
        print('--- Tool approval demo ---')
        print('(drop_database is hidden, send_email requires approval, list_files is free)\n')

        result = await agent.run(
            'List files in /app, then send an email to alice@example.com summarizing what you found.'
        )
        print(f'\nResponse: {result.output}')


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
