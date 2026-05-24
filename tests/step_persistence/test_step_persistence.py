"""Tests for the `StepPersistence` capability.

Exercises the public capability behavior through `Agent(...)`/`TestModel` and
covers the helper / store branches that are awkward to reach through a real
agent run (e.g. the path-traversal guard on `FileStepStore`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai._agent_graph import GraphAgentState  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness import (
    ContinuableSnapshot,
    FileStepStore,
    InMemoryStepStore,
    RunRecord,
    StepEvent,
    StepPersistence,
    StepStore,
    ToolEffectRecord,
    continue_run,
    fork_run,
    is_provider_valid,
)
from pydantic_ai_harness.step_persistence._store import _validate_id  # pyright: ignore[reportPrivateUsage]

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Restrict async tests to asyncio (Agent.run uses `asyncio.create_task`)."""
    return 'asyncio'


def build_run_context(deps: Any = None, *, run_id: str | None = None, run_step: int = 0) -> RunContext[Any]:
    """Fabricate a minimal `RunContext` for direct hook invocation."""
    return RunContext[Any](
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=run_step,
        run_id=run_id,
    )


def make_simple_agent(capabilities: list[Any]) -> Agent[None, str]:
    agent: Agent[None, str] = Agent(TestModel(), capabilities=capabilities)

    @agent.tool_plain
    def add(a: int, b: int) -> int:  # pyright: ignore[reportUnusedFunction]
        return a + b

    return agent


async def first_run_id(store: StepStore) -> str:
    runs = await store.list_runs()
    assert len(runs) >= 1
    return runs[0].run_id


# ---------------------------------------------------------------------------
# is_provider_valid
# ---------------------------------------------------------------------------


class TestIsProviderValid:
    def test_empty_history_is_valid(self) -> None:
        assert is_provider_valid([]) is True

    def test_matched_tool_call_is_valid(self) -> None:
        messages: list[ModelMessage] = [
            ModelResponse(parts=[ToolCallPart(tool_name='add', args={}, tool_call_id='c1')]),
            ModelRequest(parts=[ToolReturnPart(tool_name='add', content=3, tool_call_id='c1')]),
        ]
        assert is_provider_valid(messages) is True

    def test_unmatched_tool_call_is_invalid(self) -> None:
        messages: list[ModelMessage] = [
            ModelResponse(parts=[ToolCallPart(tool_name='add', args={}, tool_call_id='c1')])
        ]
        assert is_provider_valid(messages) is False

    def test_retry_prompt_resolves_a_tool_call(self) -> None:
        messages: list[ModelMessage] = [
            ModelResponse(parts=[ToolCallPart(tool_name='add', args={}, tool_call_id='c1')]),
            ModelRequest(parts=[RetryPromptPart(content='try again', tool_name='add', tool_call_id='c1')]),
        ]
        assert is_provider_valid(messages) is True

    def test_request_only_history_is_valid(self) -> None:
        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='hi')])]
        assert is_provider_valid(messages) is True

    def test_text_only_response_is_valid(self) -> None:
        messages: list[ModelMessage] = [ModelResponse(parts=[TextPart(content='hi')])]
        assert is_provider_valid(messages) is True


# ---------------------------------------------------------------------------
# continue_run / fork_run
# ---------------------------------------------------------------------------


class TestContinueAndForkRun:
    async def test_continue_run_returns_snapshot_messages(self) -> None:
        store = InMemoryStepStore()
        msgs: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='hi')])]
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=msgs))

        loaded = await continue_run(store, run_id='r1')
        assert len(loaded) == 1
        assert loaded is not msgs  # caller gets an independent list

    async def test_continue_run_raises_when_no_snapshot(self) -> None:
        store = InMemoryStepStore()
        with pytest.raises(LookupError, match="no continuable snapshot for run_id 'missing'"):
            await continue_run(store, run_id='missing')

    async def test_fork_run_delegates_to_continue_run(self) -> None:
        store = InMemoryStepStore()
        msgs: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='hi')])]
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=msgs))

        forked = await fork_run(store, run_id='r1')
        assert len(forked) == 1


# ---------------------------------------------------------------------------
# _validate_id
# ---------------------------------------------------------------------------


class TestValidateId:
    @pytest.mark.parametrize('bad', ['../evil', 'a/b', '', 'a..b', '..', 'has space', 'x' * 201])
    def test_rejects_bad_ids(self, bad: str) -> None:
        with pytest.raises(ValueError, match='invalid run_id'):
            _validate_id(bad, field='run_id')

    @pytest.mark.parametrize('good', ['good', 'a.b-c_d', 'A1', 'x' * 200])
    def test_accepts_safe_ids(self, good: str) -> None:
        _validate_id(good, field='run_id')


# ---------------------------------------------------------------------------
# InMemoryStepStore
# ---------------------------------------------------------------------------


class TestInMemoryStepStore:
    async def test_register_and_get_run(self) -> None:
        store = InMemoryStepStore()
        await store.register_run(RunRecord(run_id='r1', agent_name='a'))

        record = await store.get_run(run_id='r1')
        assert record is not None
        assert record.agent_name == 'a'
        assert await store.get_run(run_id='missing') is None

    async def test_list_runs_with_and_without_parent_filter(self) -> None:
        store = InMemoryStepStore()
        await store.register_run(RunRecord(run_id='r1', parent_run_id=None))
        await store.register_run(RunRecord(run_id='r2', parent_run_id='r1'))
        await store.register_run(RunRecord(run_id='r3', parent_run_id='r1'))
        await store.register_run(RunRecord(run_id='r4', parent_run_id='other'))

        assert {r.run_id for r in await store.list_runs()} == {'r1', 'r2', 'r3', 'r4'}
        children = await store.list_runs(parent_run_id='r1')
        assert {r.run_id for r in children} == {'r2', 'r3'}

    async def test_append_and_list_events(self) -> None:
        store = InMemoryStepStore()
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0))
        await store.append_event(StepEvent(run_id='r1', kind='run_completed', step_index=1))

        events = await store.list_events(run_id='r1')
        assert [e.kind for e in events] == ['run_started', 'run_completed']
        assert await store.list_events(run_id='missing') == []

    async def test_latest_snapshot_returns_last_appended(self) -> None:
        store = InMemoryStepStore()
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=[]))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=2, messages=[]))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=1, messages=[]))

        latest = await store.latest_snapshot(run_id='r1')
        assert latest is not None
        # InMemoryStepStore returns the last *appended* snapshot.
        assert latest.step_index == 1
        assert await store.latest_snapshot(run_id='missing') is None

    async def test_tool_effects_started_then_completed(self) -> None:
        store = InMemoryStepStore()
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='c1', tool_name='add', run_id='r1', status='started')
        )
        unresolved = await store.list_unresolved_tool_effects(run_id='r1')
        assert [r.tool_call_id for r in unresolved] == ['c1']

        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='c1', tool_name='add', run_id='r1', status='completed')
        )
        assert await store.list_unresolved_tool_effects(run_id='r1') == []

        # mix completed and another started; only the started one is unresolved.
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='c2', tool_name='add', run_id='r1', status='started')
        )
        unresolved = await store.list_unresolved_tool_effects(run_id='r1')
        assert [r.tool_call_id for r in unresolved] == ['c2']

    async def test_get_tool_effect_returns_latest_or_none(self) -> None:
        store = InMemoryStepStore()
        assert await store.get_tool_effect(tool_call_id='missing') is None

        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='c1', tool_name='add', run_id='r1', status='started')
        )
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='c1', tool_name='add', run_id='r1', status='completed')
        )

        record = await store.get_tool_effect(tool_call_id='c1')
        assert record is not None
        assert record.status == 'completed'


# ---------------------------------------------------------------------------
# FileStepStore
# ---------------------------------------------------------------------------


class TestFileStepStore:
    async def test_runs_round_trip(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        await store.register_run(RunRecord(run_id='r1', parent_run_id='p1', agent_name='a', metadata={'k': 'v'}))

        record = await store.get_run(run_id='r1')
        assert record is not None
        assert record.parent_run_id == 'p1'
        assert record.agent_name == 'a'
        assert record.metadata == {'k': 'v'}
        assert await store.get_run(run_id='missing') is None

    async def test_list_runs_returns_empty_when_root_missing(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path / 'does-not-exist')
        assert await store.list_runs() == []

    async def test_list_runs_skips_directories_without_run_json(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        (tmp_path / 'orphan').mkdir()
        await store.register_run(RunRecord(run_id='real', agent_name='a'))

        runs = await store.list_runs()
        assert [r.run_id for r in runs] == ['real']

    async def test_list_runs_filters_by_parent(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        await store.register_run(RunRecord(run_id='r1', parent_run_id=None))
        await store.register_run(RunRecord(run_id='r2', parent_run_id='r1'))

        children = await store.list_runs(parent_run_id='r1')
        assert [r.run_id for r in children] == ['r2']

    async def test_events_round_trip_skips_blank_lines(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0))
        await store.append_event(
            StepEvent(
                run_id='r1',
                kind='tool_call_started',
                step_index=1,
                tool_call_id='c1',
                tool_name='add',
                metadata={'k': 'v'},
            )
        )

        # Inject a blank line to exercise the strip() branch on read.
        events_file = tmp_path / 'r1' / 'events.jsonl'
        events_file.write_text(events_file.read_text(encoding='utf-8') + '\n', encoding='utf-8')

        events = await store.list_events(run_id='r1')
        assert [e.kind for e in events] == ['run_started', 'tool_call_started']
        assert events[1].metadata == {'k': 'v'}
        assert events[1].tool_call_id == 'c1'

    async def test_list_events_empty_when_file_missing(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        assert await store.list_events(run_id='nonexistent') == []

    async def test_snapshot_round_trip(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='hi')]),
            ModelResponse(parts=[TextPart(content='ok')]),
        ]
        await store.save_snapshot(
            ContinuableSnapshot(
                run_id='r1',
                step_index=0,
                messages=messages,
                conversation_id='c1',
                parent_run_id='p1',
                agent_name='a',
            )
        )

        snap = await store.latest_snapshot(run_id='r1')
        assert snap is not None
        assert snap.step_index == 0
        assert snap.conversation_id == 'c1'
        assert snap.parent_run_id == 'p1'
        assert snap.agent_name == 'a'
        assert len(snap.messages) == 2

    async def test_latest_snapshot_picks_highest_step_index(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        for step in (0, 2, 1):
            await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=step, messages=[]))

        # Drop a non-integer filename to exercise the ValueError branch.
        (tmp_path / 'r1' / 'snapshots' / 'not-a-number.json').write_text('{}', encoding='utf-8')

        snap = await store.latest_snapshot(run_id='r1')
        assert snap is not None
        assert snap.step_index == 2

    async def test_latest_snapshot_returns_none_when_dir_missing(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        assert await store.latest_snapshot(run_id='nope') is None

    async def test_latest_snapshot_returns_none_when_dir_empty(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        await store.register_run(RunRecord(run_id='r1'))  # creates snapshots/ but no files
        assert await store.latest_snapshot(run_id='r1') is None

    async def test_tool_effects_round_trip(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='c1', tool_name='add', run_id='r1', status='started')
        )
        await store.record_tool_effect(
            ToolEffectRecord(
                tool_call_id='c2',
                tool_name='mul',
                run_id='r1',
                status='completed',
                idempotency_key='k',
                effect_summary='ok',
            )
        )

        # Blank line to exercise the strip branch on read.
        path = tmp_path / 'r1' / 'tool_effects.jsonl'
        path.write_text(path.read_text(encoding='utf-8') + '\n', encoding='utf-8')

        unresolved = await store.list_unresolved_tool_effects(run_id='r1')
        assert [r.tool_call_id for r in unresolved] == ['c1']

    async def test_list_unresolved_tool_effects_empty_when_file_missing(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        assert await store.list_unresolved_tool_effects(run_id='nonexistent') == []

    async def test_get_tool_effect_finds_record_across_runs(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='c1', tool_name='add', run_id='runA', status='started')
        )
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='c1', tool_name='add', run_id='runA', status='completed')
        )
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='c2', tool_name='mul', run_id='runB', status='started')
        )
        # Blank line + non-tool-effects directory at root to exercise both inner branches.
        (tmp_path / 'runA' / 'tool_effects.jsonl').write_text(
            (tmp_path / 'runA' / 'tool_effects.jsonl').read_text(encoding='utf-8') + '\n',
            encoding='utf-8',
        )
        (tmp_path / 'empty-run').mkdir()

        record = await store.get_tool_effect(tool_call_id='c1')
        assert record is not None
        assert record.status == 'completed'
        assert record.run_id == 'runA'

        other = await store.get_tool_effect(tool_call_id='c2')
        assert other is not None and other.status == 'started'

        assert await store.get_tool_effect(tool_call_id='missing') is None

    async def test_get_tool_effect_returns_none_when_root_missing(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path / 'absent')
        assert await store.get_tool_effect(tool_call_id='anything') is None

    async def test_register_run_rejects_bad_run_id(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        with pytest.raises(ValueError, match='invalid run_id'):
            await store.register_run(RunRecord(run_id='../evil'))

    async def test_event_deserialization_rejects_unknown_kind(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        run_dir = tmp_path / 'r1'
        run_dir.mkdir()
        (run_dir / 'events.jsonl').write_text(
            json.dumps(
                {
                    'run_id': 'r1',
                    'kind': 'made_up',
                    'step_index': 0,
                    'timestamp': '2024-01-01T00:00:00+00:00',
                }
            )
            + '\n',
            encoding='utf-8',
        )
        with pytest.raises(ValueError, match='unknown event kind'):
            await store.list_events(run_id='r1')

    async def test_event_deserialization_rejects_wrong_types(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        run_dir = tmp_path / 'r1'
        run_dir.mkdir()
        (run_dir / 'events.jsonl').write_text(
            json.dumps(
                {
                    'run_id': 1,  # wrong type
                    'kind': 'run_started',
                    'step_index': 0,
                    'timestamp': '2024-01-01T00:00:00+00:00',
                }
            )
            + '\n',
            encoding='utf-8',
        )
        with pytest.raises(ValueError, match='event payload has wrong types'):
            await store.list_events(run_id='r1')

    async def test_event_deserialization_rejects_non_string_optional(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        run_dir = tmp_path / 'r1'
        run_dir.mkdir()
        (run_dir / 'events.jsonl').write_text(
            json.dumps(
                {
                    'run_id': 'r1',
                    'kind': 'run_started',
                    'step_index': 0,
                    'timestamp': '2024-01-01T00:00:00+00:00',
                    'agent_name': 5,  # neither None nor str
                }
            )
            + '\n',
            encoding='utf-8',
        )
        with pytest.raises(ValueError, match='expected str'):
            await store.list_events(run_id='r1')

    async def test_run_deserialization_rejects_wrong_types(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        run_dir = tmp_path / 'r1'
        run_dir.mkdir()
        (run_dir / 'run.json').write_text(json.dumps({'run_id': 1, 'started_at': 'x'}), encoding='utf-8')
        with pytest.raises(ValueError, match='run record has wrong types'):
            await store.get_run(run_id='r1')

    async def test_tool_effect_deserialization_rejects_wrong_types(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        run_dir = tmp_path / 'r1'
        run_dir.mkdir()
        (run_dir / 'tool_effects.jsonl').write_text(
            json.dumps({'tool_call_id': 1, 'tool_name': 'add', 'run_id': 'r1', 'status': 'started', 'started_at': 'x'})
            + '\n',
            encoding='utf-8',
        )
        with pytest.raises(ValueError, match='tool effect record has wrong types'):
            await store.list_unresolved_tool_effects(run_id='r1')

    async def test_tool_effect_deserialization_rejects_unknown_status(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        run_dir = tmp_path / 'r1'
        run_dir.mkdir()
        (run_dir / 'tool_effects.jsonl').write_text(
            json.dumps(
                {
                    'tool_call_id': 'c1',
                    'tool_name': 'add',
                    'run_id': 'r1',
                    'status': 'pending',
                    'started_at': '2024-01-01T00:00:00+00:00',
                }
            )
            + '\n',
            encoding='utf-8',
        )
        with pytest.raises(ValueError, match='unknown tool effect status'):
            await store.list_unresolved_tool_effects(run_id='r1')

    async def test_snapshot_deserialization_rejects_wrong_types(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        snap_dir = tmp_path / 'r1' / 'snapshots'
        snap_dir.mkdir(parents=True)
        (snap_dir / '0.json').write_text(
            json.dumps({'step_index': 'wrong', 'timestamp': 'x', 'messages': []}),
            encoding='utf-8',
        )
        with pytest.raises(ValueError, match='snapshot has wrong types'):
            await store.latest_snapshot(run_id='r1')


# ---------------------------------------------------------------------------
# Capability behavior via Agent + TestModel
# ---------------------------------------------------------------------------


class TestStepPersistenceCapability:
    async def test_basic_run_records_lifecycle_and_snapshot(self) -> None:
        store = InMemoryStepStore()
        agent = make_simple_agent([StepPersistence(store=store, agent_name='librarian')])

        result = await agent.run('add 1 and 2')

        rid = await first_run_id(store)
        events = await store.list_events(run_id=rid)
        kinds = [e.kind for e in events]
        assert kinds[0] == 'run_started'
        assert kinds[-1] == 'run_completed'
        assert 'tool_call_started' in kinds
        assert 'tool_call_completed' in kinds

        # RunRecord lineage was registered.
        record = await store.get_run(run_id=rid)
        assert record is not None
        assert record.agent_name == 'librarian'
        assert record.parent_run_id is None

        # Snapshot is provider-valid and round-trips through `continue_run`.
        snap = await store.latest_snapshot(run_id=rid)
        assert snap is not None
        assert is_provider_valid(snap.messages) is True
        assert len(snap.messages) == len(result.all_messages())

        # All events tagged with the same run_id and agent_name.
        assert {e.run_id for e in events} == {rid}
        assert {e.agent_name for e in events} == {'librarian'}

    async def test_continue_from_prepends_prior_snapshot(self) -> None:
        store = InMemoryStepStore()
        agent1 = make_simple_agent([StepPersistence(store=store, agent_name='a')])
        await agent1.run('add 1 and 2')
        first_rid = await first_run_id(store)
        snap = await store.latest_snapshot(run_id=first_rid)
        assert snap is not None
        prior_len = len(snap.messages)

        agent2 = make_simple_agent([StepPersistence(store=store, agent_name='a', continue_from=first_rid)])
        result2 = await agent2.run('add 3 and 4')

        # The prior snapshot's messages are present at the head of the new run's history.
        msgs = result2.all_messages()
        assert len(msgs) > prior_len
        for prior_msg, replayed in zip(snap.messages, msgs[:prior_len]):
            assert type(prior_msg) is type(replayed)

    async def test_continue_from_with_no_snapshot_is_a_no_op(self) -> None:
        store = InMemoryStepStore()
        agent = make_simple_agent([StepPersistence(store=store, continue_from='nope')])

        result = await agent.run('add 1 and 2')

        rid = await first_run_id(store)
        # The run still ran cleanly; the missing snapshot is silently ignored.
        assert (await store.latest_snapshot(run_id=rid)) is not None
        assert len(result.all_messages()) > 0

    async def test_tool_effect_records_started_then_completed(self) -> None:
        store = InMemoryStepStore()
        agent = make_simple_agent([StepPersistence(store=store)])
        await agent.run('add 1 and 2')

        rid = await first_run_id(store)
        assert await store.list_unresolved_tool_effects(run_id=rid) == []
        effect = await store.get_tool_effect(tool_call_id='pyd_ai_tool_call_id__add')
        assert effect is not None
        assert effect.status == 'completed'
        assert effect.tool_name == 'add'
        assert effect.started_at is not None
        assert effect.ended_at is not None

    async def test_tool_failure_records_failed_status_and_event(self) -> None:
        store = InMemoryStepStore()
        agent: Agent[None, str] = Agent(TestModel(), capabilities=[StepPersistence(store=store)])

        @agent.tool_plain
        def boom() -> int:  # pyright: ignore[reportUnusedFunction]
            raise ValueError('kaboom')

        with pytest.raises(ValueError, match='kaboom'):
            await agent.run('boom please')

        rid = await first_run_id(store)
        events = await store.list_events(run_id=rid)
        kinds = [e.kind for e in events]
        assert 'tool_call_started' in kinds
        assert 'tool_call_failed' in kinds
        assert 'run_failed' in kinds
        # The failure event records the exception repr.
        failed_event = next(e for e in events if e.kind == 'tool_call_failed')
        assert failed_event.error is not None and 'kaboom' in failed_event.error

        effect = await store.get_tool_effect(tool_call_id='pyd_ai_tool_call_id__boom')
        assert effect is not None
        assert effect.status == 'failed'
        assert effect.effect_summary is not None and 'kaboom' in effect.effect_summary

    async def test_explicit_run_id_wins_over_ctx_run_id(self) -> None:
        store = InMemoryStepStore()
        agent = make_simple_agent(
            [StepPersistence(store=store, run_id='librarian-001', agent_name='librarian')],
        )

        await agent.run('add 1 and 2')

        # Caller-supplied run_id is the persisted identity, not the auto-generated ctx.run_id.
        record = await store.get_run(run_id='librarian-001')
        assert record is not None
        assert record.agent_name == 'librarian'
        events = await store.list_events(run_id='librarian-001')
        assert {e.run_id for e in events} == {'librarian-001'}

    async def test_parent_run_id_lineage(self) -> None:
        store = InMemoryStepStore()
        orchestrator = make_simple_agent([StepPersistence(store=store, agent_name='orch')])
        await orchestrator.run('orchestrate')
        orch_rid = await first_run_id(store)

        delegate = make_simple_agent([StepPersistence(store=store, agent_name='delegate', parent_run_id=orch_rid)])
        await delegate.run('delegate work')

        children = await store.list_runs(parent_run_id=orch_rid)
        assert len(children) == 1
        assert children[0].agent_name == 'delegate'
        assert children[0].parent_run_id == orch_rid

        # The delegate's events also carry the parent_run_id.
        child_events = await store.list_events(run_id=children[0].run_id)
        assert {e.parent_run_id for e in child_events} == {orch_rid}

    async def test_from_spec_memory_backend(self) -> None:
        cap = StepPersistence.from_spec()
        assert isinstance(cap.store, InMemoryStepStore)

    async def test_from_spec_explicit_memory_backend_with_kwargs(self) -> None:
        cap = StepPersistence.from_spec(backend='memory', agent_name='a')
        assert isinstance(cap.store, InMemoryStepStore)
        assert cap.agent_name == 'a'

    async def test_from_spec_file_backend(self, tmp_path: Path) -> None:
        cap = StepPersistence.from_spec(backend='file', directory=tmp_path)
        assert isinstance(cap.store, FileStepStore)

    async def test_from_spec_file_backend_default_directory(self) -> None:
        cap = StepPersistence.from_spec(backend='file')
        assert isinstance(cap.store, FileStepStore)


# ---------------------------------------------------------------------------
# Headline acceptance test
# ---------------------------------------------------------------------------


class TestCrashMidToolCallContract:
    """The signature acceptance test from the PR comment.

    A run killed after a tool starts but before its return is persisted must
    leave a visible event trail without exposing the killed point as a
    valid `message_history` continuation. The latest snapshot must be older
    than the in-flight call, and that snapshot must be provider-valid.
    """

    async def test_visible_trail_no_false_continuation_point(self) -> None:
        store = InMemoryStepStore()
        cap: StepPersistence[None] = StepPersistence(store=store, agent_name='delegate')
        agent: Agent[None, str] = Agent(TestModel(), capabilities=[cap])

        @agent.tool_plain
        def add(a: int, b: int) -> int:  # pyright: ignore[reportUnusedFunction]
            return a + b

        # 1) Drive a full successful run so a provider-valid snapshot exists.
        result = await agent.run('add 1 and 2')
        rid = await first_run_id(store)
        snap_before_crash = await store.latest_snapshot(run_id=rid)
        assert snap_before_crash is not None
        assert is_provider_valid(snap_before_crash.messages) is True
        snap_step = snap_before_crash.step_index

        # 2) Simulate a crash mid-tool-call by calling `before_tool_execute`
        # directly with a synthesised ToolCallPart and never firing
        # `after_tool_execute` / `on_tool_execute_error`.
        crash_ctx = build_run_context(deps=None, run_id=rid, run_step=snap_step + 1)
        crash_call = ToolCallPart(tool_name='add', args={'a': 9, 'b': 9}, tool_call_id='crash-call-1')
        tool_def = ToolDefinition(name='add', description='Add two numbers.')
        await cap.before_tool_execute(crash_ctx, call=crash_call, tool_def=tool_def, args={'a': 9, 'b': 9})

        # 3) Assert the event log shows the started call with no terminal update.
        events = await store.list_events(run_id=rid)
        started = [e for e in events if e.kind == 'tool_call_started' and e.tool_call_id == 'crash-call-1']
        completed = [e for e in events if e.kind == 'tool_call_completed' and e.tool_call_id == 'crash-call-1']
        failed = [e for e in events if e.kind == 'tool_call_failed' and e.tool_call_id == 'crash-call-1']
        assert len(started) == 1
        assert completed == []
        assert failed == []

        # 4) The unresolved-effect ledger surfaces the in-flight tool call.
        unresolved = await store.list_unresolved_tool_effects(run_id=rid)
        crash_records = [r for r in unresolved if r.tool_call_id == 'crash-call-1']
        assert len(crash_records) == 1
        assert crash_records[0].status == 'started'

        # 5) Resume point is the snapshot from step 1 — older than the crash —
        # and is still provider-valid.
        snap_after_crash = await store.latest_snapshot(run_id=rid)
        assert snap_after_crash is not None
        assert snap_after_crash.step_index == snap_step
        assert is_provider_valid(snap_after_crash.messages) is True
        # And the snapshot is consistent with what the prior successful run produced.
        assert len(snap_after_crash.messages) == len(result.all_messages())


# ---------------------------------------------------------------------------
# Hook-level branches awkward to reach through Agent
# ---------------------------------------------------------------------------


class TestCapabilityHookBranches:
    async def test_effective_run_id_falls_back_to_capability_field(self) -> None:
        """When `ctx.run_id` is missing, the capability uses its own `run_id`."""
        store = InMemoryStepStore()
        cap: StepPersistence[None] = StepPersistence(store=store, run_id='configured', agent_name='a')
        ctx_no_run_id = build_run_context(deps=None, run_id=None)
        await cap.before_run(ctx_no_run_id)

        record = await store.get_run(run_id='configured')
        assert record is not None
        events = await store.list_events(run_id='configured')
        assert [e.kind for e in events] == ['run_started']

    async def test_after_run_skips_snapshot_when_history_not_provider_valid(self) -> None:
        """`after_run` only persists a snapshot when the history is provider-valid."""
        store = InMemoryStepStore()
        cap: StepPersistence[None] = StepPersistence(store=store)
        ctx = build_run_context(deps=None, run_id='r1')

        unmatched: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='hi')]),
            ModelResponse(parts=[ToolCallPart(tool_name='add', args={}, tool_call_id='orphan')]),
        ]
        result: AgentRunResult[str] = AgentRunResult(
            output='out',
            _state=GraphAgentState(message_history=unmatched, run_id='r1'),
        )

        await cap.after_run(ctx, result=result)

        assert await store.latest_snapshot(run_id='r1') is None
        events = await store.list_events(run_id='r1')
        assert [e.kind for e in events] == ['run_completed']

    async def test_on_model_request_error_records_event_and_reraises(self) -> None:
        store = InMemoryStepStore()
        cap: StepPersistence[None] = StepPersistence(store=store)
        ctx = build_run_context(deps=None, run_id='r1')
        request_context = ModelRequestContext(
            model=ctx.model,
            messages=[],
            model_settings=None,
            model_request_parameters=ModelRequestParameters(),
        )
        boom = RuntimeError('nope')

        with pytest.raises(RuntimeError, match='nope'):
            await cap.on_model_request_error(ctx, request_context=request_context, error=boom)

        events = await store.list_events(run_id='r1')
        assert [e.kind for e in events] == ['model_request_failed']
        assert events[0].error is not None and 'nope' in events[0].error

    async def test_run_record_load_with_missing_metadata(self, tmp_path: Path) -> None:
        """`_str_str_dict(None)` returns `{}` when metadata is absent in storage."""
        store = FileStepStore(tmp_path)
        run_dir = tmp_path / 'r1'
        run_dir.mkdir()
        (run_dir / 'run.json').write_text(
            json.dumps({'run_id': 'r1', 'started_at': '2024-01-01T00:00:00+00:00'}),
            encoding='utf-8',
        )

        record = await store.get_run(run_id='r1')
        assert record is not None
        assert record.metadata == {}
