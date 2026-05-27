"""A minimal headless coding agent built on the execution-environment capability.

This is an *example*, not part of the shipped package: the harness is a library of
capabilities, and a runnable agent is an application built on top of it (it bakes in a
model choice, a prompt, a REPL, and Logfire config that don't belong in a capability lib).

Run it (from the repo root):

    uv run python examples/cli.py --root .

Then ask it about this repo. Every model request, tool call, and truncation shows up as a
Logfire span if you've authenticated (`uv run logfire auth`); otherwise it runs fine and
just prints a one-line notice.
"""

from __future__ import annotations

import argparse
import asyncio

import logfire
from pydantic_ai import Agent, ModelMessage

from pydantic_ai_harness.environments.local import LocalEnvironment
from pydantic_ai_harness.execution_env import ExecutionEnv


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser (workspace root and model)."""
    parser = argparse.ArgumentParser(description='A minimal headless coding agent.')
    parser.add_argument('--root', default='.', help='Workspace root the agent can read (default: cwd).')
    parser.add_argument('--model', default='anthropic:claude-sonnet-4-6', help='Model string (provider:name).')
    return parser


def configure_logfire() -> None:
    """Configure Logfire and instrument Pydantic AI (no-op without a token)."""
    # send_to_logfire='if-token-present' keeps the CLI usable without auth: with a token
    # configured you get full traces, without one it's a no-op rather than an error.
    logfire.configure(send_to_logfire='if-token-present', service_name='harness-cli')
    logfire.instrument_pydantic_ai()


async def main() -> None:
    """Parse args, configure tracing, and run the interactive prompt loop."""
    args = build_parser().parse_args()
    configure_logfire()

    environment = LocalEnvironment(root=args.root)
    agent = Agent(
        model='anthropic:claude-sonnet-4-6',
        capabilities=[ExecutionEnv(environment=environment)],
        instructions='You are a coding agent that can read and write files in the workspace.',
    )

    message_history: list[ModelMessage] = []
    print("Type a question, or 'exit' to quit.")
    while True:
        try:
            prompt = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt in {'exit', 'quit'}:
            break
        if not prompt:
            continue

        result = await agent.run(prompt, message_history=message_history)
        message_history = result.all_messages()
        print(result.output)


if __name__ == '__main__':
    asyncio.run(main())
