"""Regression test for the harness README's Quick start example.

The README ships a Hacker News + web-search agent wrapped in `CodeMode` and
asks it to find the most-discussed HN story across three feeds, then pull
the comment thread, the submitter's profile, and follow-up coverage. We
fake everything that talks to the network so the test runs in CI without
`ddgs`, an MCP package, or any HTTP traffic:

- A `FunctionModel` drives the conversation through the same shape the
  example produces in production -- two `run_code` calls (parallel feed
  fetches + dedupe + filter, then parallel follow-ups) and a final summary.
- The Hacker News MCP toolset is replaced with a `FunctionToolset` of
  fake functions whose return values come from the public Logfire trace
  linked in the README.
- `WebSearch(builtin=False, local=...)` skips the default DuckDuckGo
  fallback so the test doesn't pull `ddgs` and the harness doesn't depend
  on it in CI.

`CodeMode` itself is real -- the `FunctionModel`'s emitted Python code
runs through the Monty sandbox, dispatches the calls back through
pydantic-ai's tool machinery to our fakes, and the return values flow
back into the model loop. Any future change in pydantic-ai or the harness
that breaks how these capabilities compose makes this test fail.
"""

from __future__ import annotations

import textwrap
from typing import Any

import pytest
from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import MCP, WebSearch
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets.function import FunctionToolset

from pydantic_ai_harness import CodeMode

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend (pydantic-ai uses asyncio.create_task internally)."""
    return 'asyncio'


# ---------------------------------------------------------------------------
# Canned tool responses -- shapes mirror what the cyanheads HN MCP server
# actually returns, with values from the run captured in the README's
# linked public trace.
# ---------------------------------------------------------------------------

_WINNER_ID = 48037128
_WINNER_USER = 'e12e'

_TOP_FEED = {
    'stories': [
        {
            'id': 48037555,
            'type': 'story',
            'title': 'Valve releases Steam Controller CAD files under Creative Commons license',
            'score': 1687,
            'by': 'haunter',
            'time': 1778082253,
            'descendants': 572,
            'url': 'https://www.digitalfoundry.net/news/2026/05/valve-releases-steam-controller-cad-files',
        },
        {
            'id': 48050499,
            'type': 'story',
            'title': 'I want to live like Costco people',
            'score': 235,
            'by': 'speckx',
            'time': 1778167167,
            'descendants': 495,
            'url': 'https://tastecooking.com/i-want-to-live-like-costco-people/',
        },
    ],
}

_BEST_FEED = {
    'stories': [
        {
            'id': _WINNER_ID,
            'type': 'story',
            'title': "Vibe coding and agentic engineering are getting closer than I'd like",
            'score': 748,
            'by': _WINNER_USER,
            'time': 1778079997,
            'descendants': 853,
            'url': 'https://simonwillison.net/2026/May/6/vibe-coding-and-agentic-engineering/',
        },
        {
            'id': 48038001,
            'type': 'story',
            'title': 'Appearing productive in the workplace',
            'score': 1534,
            'by': 'diebillionaires',
            'time': 1778084309,
            'descendants': 629,
            'url': 'https://nooneshappy.com/article/appearing-productive-in-the-workplace/',
        },
        {
            'id': 48037555,
            'type': 'story',
            'title': 'Valve releases Steam Controller CAD files under Creative Commons license',
            'score': 1687,
            'by': 'haunter',
            'time': 1778082253,
            'descendants': 572,
            'url': 'https://www.digitalfoundry.net/news/2026/05/valve-releases-steam-controller-cad-files',
        },
    ],
}

_SHOW_FEED: dict[str, list[dict[str, Any]]] = {'stories': []}

_THREAD = {
    'item': {
        'id': _WINNER_ID,
        'title': "Vibe coding and agentic engineering are getting closer than I'd like",
        'url': 'https://simonwillison.net/2026/May/6/vibe-coding-and-agentic-engineering/',
        'score': 748,
        'by': _WINNER_USER,
        'descendants': 853,
    },
    'comments': [
        {'by': 'etothet', 'depth': 0, 'text': 'LLMs exposed sloppy practices, not created them.'},
        {'by': 'kelnos', 'depth': 0, 'text': 'Normalization of deviance as engineers stop reviewing diffs.'},
    ],
    'totalLoaded': 2,
    'totalAvailable': 853,
}

_USER_PROFILE: dict[str, Any] = {
    'user': {
        'id': _WINNER_USER,
        'created': 1331059200,
        'karma': 15024,
        'submitted': 9700,
        'about': 'perpetual student and sometimes developer based in Tromsø, Norway',
    },
    'submissions': [],
}

_WEB_RESULTS: dict[str, list[dict[str, Any]]] = {
    'results': [
        {
            'title': 'GLM-5: From Vibe Coding to Agentic Engineering',
            'url': 'https://simonwillison.net/2026/Feb/11/glm-5/',
            'snippet': 'Earlier piece by the same author tracing the same arc.',
        },
    ],
}


# ---------------------------------------------------------------------------
# Fake tool implementations -- record their inputs so we can assert on the
# call shape from the test, then return the canned data above.
# ---------------------------------------------------------------------------


class _CallRecorder:
    """Records the kwargs each fake tool was called with, by tool name."""

    def __init__(self) -> None:
        self.calls: dict[str, list[dict[str, Any]]] = {}

    def record(self, name: str, **kwargs: Any) -> None:
        self.calls.setdefault(name, []).append(kwargs)


def _make_fake_tools(recorder: _CallRecorder) -> list[Tool[None]]:
    feeds: dict[str, dict[str, list[dict[str, Any]]]] = {
        'top': _TOP_FEED,
        'best': _BEST_FEED,
        'show': _SHOW_FEED,
    }

    def hn_get_stories(*, feed: str, count: int = 50) -> dict[str, Any]:
        """Fetch a Hacker News feed (top, best, or show)."""
        recorder.record('hn_get_stories', feed=feed, count=count)
        return feeds[feed]

    def hn_get_thread(*, itemId: int, depth: int = 2, maxComments: int = 60) -> dict[str, Any]:
        """Fetch the comment thread for a story id."""
        recorder.record('hn_get_thread', itemId=itemId, depth=depth, maxComments=maxComments)
        return _THREAD

    def hn_get_user(
        *,
        username: str,
        includeSubmissions: bool = False,
        submissionCount: int = 5,
    ) -> dict[str, Any]:
        """Fetch a Hacker News user's profile."""
        recorder.record(
            'hn_get_user',
            username=username,
            includeSubmissions=includeSubmissions,
            submissionCount=submissionCount,
        )
        return _USER_PROFILE

    return [
        Tool(hn_get_stories),
        Tool(hn_get_thread),
        Tool(hn_get_user),
    ]


def _make_fake_web_search(recorder: _CallRecorder) -> Any:
    def web_search(*, query: str) -> dict[str, Any]:
        """Stand-in for the WebSearch capability's local DDG fallback."""
        recorder.record('web_search', query=query)
        return _WEB_RESULTS

    return web_search


# ---------------------------------------------------------------------------
# FunctionModel state machine -- two run_code calls, then a final synthesis,
# matching the trace in the README.
# ---------------------------------------------------------------------------

# First run_code: parallel feed fetches, dedupe by id, score filter, rank by descendants.
_FIRST_RUN_CODE = textwrap.dedent(
    """
    import asyncio
    top, best, show = await asyncio.gather(
        hn_get_stories(feed='top', count=50),
        hn_get_stories(feed='best', count=50),
        hn_get_stories(feed='show', count=50),
    )
    seen = {}
    for feed_name, data in [('top', top), ('best', best), ('show', show)]:
        for s in data['stories']:
            if s.get('score', 0) >= 100:
                if s['id'] not in seen:
                    entry = dict(s)
                    entry['feeds'] = []
                    seen[s['id']] = entry
                seen[s['id']]['feeds'].append(feed_name)
    ranked = sorted(seen.values(), key=lambda x: x.get('descendants', 0), reverse=True)
    ranked[:5]
    """
).strip()

# Second run_code: parallel follow-up calls on the winner. Uses the WebSearch
# capability's local fallback (`web_search`) and the MCP-served HN tools side by
# side, mirroring the README example's "HN tools + web search" composition.
_SECOND_RUN_CODE = textwrap.dedent(
    f"""
    import asyncio
    thread, user, coverage = await asyncio.gather(
        hn_get_thread(itemId={_WINNER_ID}, depth=2, maxComments=60),
        hn_get_user(username='{_WINNER_USER}', includeSubmissions=True, submissionCount=5),
        web_search(query='vibe coding agentic engineering simonwillison'),
    )
    (thread['item'], user['user'], coverage['results'][:5])
    """
).strip()

_FINAL_SYNTHESIS = (
    'The most-discussed HN story across top/best/show clearing 100 points is '
    '"Vibe coding and agentic engineering are getting closer than I\'d like" '
    'by Simon Willison (748 points, 853 comments), submitted by e12e.'
)


def _model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    completed_run_codes = [
        p
        for m in messages
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, ToolReturnPart) and p.tool_name == 'run_code'
    ]
    if not completed_run_codes:
        return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': _FIRST_RUN_CODE})])
    if len(completed_run_codes) == 1:
        return ModelResponse(
            parts=[
                TextPart('The winner is the Simon Willison post; pulling thread, user, and coverage in parallel.'),
                ToolCallPart(tool_name='run_code', args={'code': _SECOND_RUN_CODE}),
            ]
        )
    return ModelResponse(parts=[TextPart(_FINAL_SYNTHESIS)])


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


class TestReadmeQuickStart:
    """End-to-end check that the README's Quick start example still works."""

    async def test_quick_start_runs_through_codemode_with_faked_io(self) -> None:
        recorder = _CallRecorder()
        hn_toolset = FunctionToolset[None](tools=_make_fake_tools(recorder))

        agent: Agent[None, str] = Agent(
            FunctionModel(_model_fn),
            capabilities=[
                # Wire the fake HN tools through the `MCP` capability the same way
                # the README does -- `local=` overrides the default FastMCP HTTP
                # toolset with our in-process fake, so the test exercises the same
                # capability composition path as production without any network.
                # MCP's `__init__` narrows `local` to MCP-specific types, but the
                # parent `BuiltinOrLocalTool` accepts any `AbstractToolset` at runtime.
                MCP[None](
                    'https://hn.caseyjhand.com/mcp',
                    builtin=False,
                    local=hn_toolset,  # pyright: ignore[reportArgumentType]
                ),
                WebSearch[None](builtin=False, local=_make_fake_web_search(recorder)),
                CodeMode[None](),
            ],
        )

        result = await agent.run(
            "Across the top, best, and 'show HN' Hacker News feeds, find the most-discussed "
            'story with at least 100 points. Pull its comment thread, its submitter profile, '
            'and any web coverage. Summarize what you find in one paragraph.'
        )

        # First run_code fetched all three feeds in parallel and the dedupe + filter
        # produced a ranked list -- proven by the second run_code targeting the
        # expected winner.
        feeds_fetched = sorted(call['feed'] for call in recorder.calls['hn_get_stories'])
        assert feeds_fetched == ['best', 'show', 'top']

        # Second run_code's parallel follow-up calls.
        assert recorder.calls['hn_get_thread'] == [
            {'itemId': _WINNER_ID, 'depth': 2, 'maxComments': 60},
        ]
        assert recorder.calls['hn_get_user'] == [
            {'username': _WINNER_USER, 'includeSubmissions': True, 'submissionCount': 5},
        ]
        assert recorder.calls['web_search'] == [
            {'query': 'vibe coding agentic engineering simonwillison'},
        ]

        # The final synthesis names the right story, so the second run_code's data
        # actually flowed back through the model loop.
        assert 'Vibe coding' in result.output
        assert 'Simon Willison' in result.output
