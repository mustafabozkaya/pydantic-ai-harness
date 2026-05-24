"""Storage backends for step events, snapshots, and tool-effect records."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import anyio.to_thread
from pydantic import TypeAdapter
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

from pydantic_ai_harness.step_persistence._types import (
    ContinuableSnapshot,
    EventKind,
    RunRecord,
    StepEvent,
    ToolEffectRecord,
    ToolEffectStatus,
)

_VALID_ID_RE = re.compile(r'^[A-Za-z0-9_.-]{1,200}$')

_EVENT_KIND_TABLE: dict[str, EventKind] = {
    'run_started': 'run_started',
    'run_completed': 'run_completed',
    'run_failed': 'run_failed',
    'model_request_started': 'model_request_started',
    'model_request_completed': 'model_request_completed',
    'model_request_failed': 'model_request_failed',
    'tool_call_started': 'tool_call_started',
    'tool_call_completed': 'tool_call_completed',
    'tool_call_failed': 'tool_call_failed',
}

_TOOL_STATUS_TABLE: dict[str, ToolEffectStatus] = {
    'started': 'started',
    'completed': 'completed',
    'failed': 'failed',
}


def _validate_id(value: str, *, field: str) -> None:
    r"""Reject identifiers that would let a caller escape the store directory.

    `FileStepStore` interpolates `run_id` into a path. Allowing `..`, `/`,
    `\`, empty input, or oversized values would enable path traversal.
    """
    if not _VALID_ID_RE.fullmatch(value) or '..' in value:
        raise ValueError(f'invalid {field}: {value!r}')


@runtime_checkable
class StepStore(Protocol):
    """Async protocol for step-persistence backends.

    All methods are async so the same protocol covers in-memory stores and
    file/database stores that would otherwise block the event loop.
    """

    async def register_run(self, record: RunRecord) -> None: ...  # pragma: no cover

    async def get_run(self, *, run_id: str) -> RunRecord | None: ...  # pragma: no cover

    async def list_runs(
        self,
        *,
        parent_run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[RunRecord]:
        """Return matching `RunRecord`s sorted by `started_at` ascending.

        Both filters are optional and AND-combine when both are set. The
        chronological ordering is part of the protocol — callers may pick
        the most recent run with `[-1]`.
        """
        ...  # pragma: no cover

    async def append_event(self, event: StepEvent) -> None: ...  # pragma: no cover

    async def list_events(self, *, run_id: str) -> list[StepEvent]: ...  # pragma: no cover

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None: ...  # pragma: no cover

    async def latest_snapshot(self, *, run_id: str) -> ContinuableSnapshot | None: ...  # pragma: no cover

    async def record_tool_effect(self, record: ToolEffectRecord) -> None: ...  # pragma: no cover

    async def get_tool_effect(
        self, *, run_id: str, tool_call_id: str
    ) -> ToolEffectRecord | None: ...  # pragma: no cover

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]: ...  # pragma: no cover


class InMemoryStepStore:
    """Process-local store, suitable for tests and single-process orchestrators."""

    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._events: dict[str, list[StepEvent]] = defaultdict(list)
        self._snapshots: dict[str, list[ContinuableSnapshot]] = defaultdict(list)
        self._tool_effects: dict[tuple[str, str], ToolEffectRecord] = {}

    async def register_run(self, record: RunRecord) -> None:
        self._runs[record.run_id] = record

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    async def list_runs(
        self,
        *,
        parent_run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[RunRecord]:
        records = [
            r
            for r in self._runs.values()
            if (parent_run_id is None or r.parent_run_id == parent_run_id)
            and (conversation_id is None or r.conversation_id == conversation_id)
        ]
        return sorted(records, key=lambda r: r.started_at)

    async def append_event(self, event: StepEvent) -> None:
        self._events[event.run_id].append(event)

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        return list(self._events.get(run_id, ()))

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        self._snapshots[snapshot.run_id].append(snapshot)

    async def latest_snapshot(self, *, run_id: str) -> ContinuableSnapshot | None:
        snaps = self._snapshots.get(run_id)
        if not snaps:
            return None
        return snaps[-1]

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        self._tool_effects[(record.run_id, record.tool_call_id)] = record

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        return self._tool_effects.get((run_id, tool_call_id))

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        return [r for r in self._tool_effects.values() if r.run_id == run_id and r.status == 'started']


def _opt_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f'expected str|None, got {type(value).__name__}')
    return value


_STR_STR_DICT_ADAPTER: TypeAdapter[dict[str, str]] = TypeAdapter(dict[str, str])
_OBJECT_DICT_ADAPTER: TypeAdapter[dict[str, object]] = TypeAdapter(dict[str, object])


def _str_str_dict(value: object) -> dict[str, str]:
    if value is None:
        return {}
    return _STR_STR_DICT_ADAPTER.validate_python(value)


def _load_json_object(text: str) -> dict[str, object]:
    return _OBJECT_DICT_ADAPTER.validate_json(text)


def _event_to_dict(event: StepEvent) -> dict[str, object]:
    return {
        'run_id': event.run_id,
        'kind': event.kind,
        'step_index': event.step_index,
        'timestamp': event.timestamp.isoformat(),
        'conversation_id': event.conversation_id,
        'parent_run_id': event.parent_run_id,
        'agent_name': event.agent_name,
        'tool_call_id': event.tool_call_id,
        'tool_name': event.tool_name,
        'error': event.error,
        'metadata': dict(event.metadata),
    }


def _event_from_dict(data: dict[str, object]) -> StepEvent:
    run_id = data['run_id']
    kind_raw = data['kind']
    step_raw = data['step_index']
    timestamp_raw = data['timestamp']
    if not (
        isinstance(run_id, str)
        and isinstance(kind_raw, str)
        and isinstance(step_raw, int)
        and isinstance(timestamp_raw, str)
    ):
        raise ValueError('event payload has wrong types')
    kind = _EVENT_KIND_TABLE.get(kind_raw)
    if kind is None:
        raise ValueError(f'unknown event kind: {kind_raw!r}')
    return StepEvent(
        run_id=run_id,
        kind=kind,
        step_index=step_raw,
        timestamp=datetime.fromisoformat(timestamp_raw),
        conversation_id=_opt_str(data.get('conversation_id')),
        parent_run_id=_opt_str(data.get('parent_run_id')),
        agent_name=_opt_str(data.get('agent_name')),
        tool_call_id=_opt_str(data.get('tool_call_id')),
        tool_name=_opt_str(data.get('tool_name')),
        error=_opt_str(data.get('error')),
        metadata=_str_str_dict(data.get('metadata')),
    )


def _run_to_dict(record: RunRecord) -> dict[str, object]:
    return {
        'run_id': record.run_id,
        'conversation_id': record.conversation_id,
        'parent_run_id': record.parent_run_id,
        'agent_name': record.agent_name,
        'metadata': dict(record.metadata),
        'started_at': record.started_at.isoformat(),
    }


def _run_from_dict(data: dict[str, object]) -> RunRecord:
    run_id = data['run_id']
    started_at_raw = data['started_at']
    if not (isinstance(run_id, str) and isinstance(started_at_raw, str)):
        raise ValueError('run record has wrong types')
    return RunRecord(
        run_id=run_id,
        conversation_id=_opt_str(data.get('conversation_id')),
        parent_run_id=_opt_str(data.get('parent_run_id')),
        agent_name=_opt_str(data.get('agent_name')),
        metadata=_str_str_dict(data.get('metadata')),
        started_at=datetime.fromisoformat(started_at_raw),
    )


def _tool_effect_to_dict(record: ToolEffectRecord) -> dict[str, object]:
    return {
        'tool_call_id': record.tool_call_id,
        'tool_name': record.tool_name,
        'run_id': record.run_id,
        'status': record.status,
        'started_at': record.started_at.isoformat(),
        'ended_at': record.ended_at.isoformat() if record.ended_at is not None else None,
        'idempotency_key': record.idempotency_key,
        'effect_summary': record.effect_summary,
    }


def _tool_effect_from_dict(data: dict[str, object]) -> ToolEffectRecord:
    tool_call_id = data['tool_call_id']
    tool_name = data['tool_name']
    run_id = data['run_id']
    status_raw = data['status']
    started_at_raw = data['started_at']
    if not (
        isinstance(tool_call_id, str)
        and isinstance(tool_name, str)
        and isinstance(run_id, str)
        and isinstance(status_raw, str)
        and isinstance(started_at_raw, str)
    ):
        raise ValueError('tool effect record has wrong types')
    status = _TOOL_STATUS_TABLE.get(status_raw)
    if status is None:
        raise ValueError(f'unknown tool effect status: {status_raw!r}')
    ended_at_raw = data.get('ended_at')
    ended_at = datetime.fromisoformat(ended_at_raw) if isinstance(ended_at_raw, str) else None
    return ToolEffectRecord(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        run_id=run_id,
        status=status,
        started_at=datetime.fromisoformat(started_at_raw),
        ended_at=ended_at,
        idempotency_key=_opt_str(data.get('idempotency_key')),
        effect_summary=_opt_str(data.get('effect_summary')),
    )


class FileStepStore:
    """Filesystem-backed store using JSONL for events/tool effects and JSON for snapshots.

    Layout under the configured root directory:

    ```text
    <root>/
      <run_id>/
        run.json
        events.jsonl
        tool_effects.jsonl
        snapshots/
          <step_index>.json
    ```

    `run_id` is validated against `[A-Za-z0-9_.-]{1,200}` (with `..` rejected)
    to prevent path traversal. Blocking I/O is dispatched to a worker thread
    via `anyio.to_thread`, so capability hooks do not stall the event loop.
    """

    def __init__(self, directory: str | Path) -> None:
        self._root = Path(directory)

    def _run_dir(self, run_id: str) -> Path:
        _validate_id(run_id, field='run_id')
        return self._root / run_id

    async def register_run(self, record: RunRecord) -> None:
        await anyio.to_thread.run_sync(self._sync_register_run, record)

    def _sync_register_run(self, record: RunRecord) -> None:
        run_dir = self._run_dir(record.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / 'snapshots').mkdir(exist_ok=True)
        (run_dir / 'run.json').write_text(json.dumps(_run_to_dict(record)), encoding='utf-8')

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        return await anyio.to_thread.run_sync(self._sync_get_run, run_id)

    def _sync_get_run(self, run_id: str) -> RunRecord | None:
        path = self._run_dir(run_id) / 'run.json'
        if not path.exists():
            return None
        return _run_from_dict(_load_json_object(path.read_text(encoding='utf-8')))

    async def list_runs(
        self,
        *,
        parent_run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[RunRecord]:
        return await anyio.to_thread.run_sync(self._sync_list_runs, parent_run_id, conversation_id)

    def _sync_list_runs(
        self,
        parent_run_id: str | None,
        conversation_id: str | None,
    ) -> list[RunRecord]:
        if not self._root.exists():
            return []
        records: list[RunRecord] = []
        for sub in self._root.iterdir():
            run_file = sub / 'run.json'
            if not run_file.exists():
                continue
            record = _run_from_dict(_load_json_object(run_file.read_text(encoding='utf-8')))
            if parent_run_id is not None and record.parent_run_id != parent_run_id:
                continue
            if conversation_id is not None and record.conversation_id != conversation_id:
                continue
            records.append(record)
        return sorted(records, key=lambda r: r.started_at)

    async def append_event(self, event: StepEvent) -> None:
        await anyio.to_thread.run_sync(self._sync_append_event, event)

    def _sync_append_event(self, event: StepEvent) -> None:
        run_dir = self._run_dir(event.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(_event_to_dict(event))
        with (run_dir / 'events.jsonl').open('a', encoding='utf-8') as fp:
            fp.write(line + '\n')

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        return await anyio.to_thread.run_sync(self._sync_list_events, run_id)

    def _sync_list_events(self, run_id: str) -> list[StepEvent]:
        path = self._run_dir(run_id) / 'events.jsonl'
        if not path.exists():
            return []
        events: list[StepEvent] = []
        for raw in path.read_text(encoding='utf-8').splitlines():
            if raw.strip():
                events.append(_event_from_dict(_load_json_object(raw)))
        return events

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        await anyio.to_thread.run_sync(self._sync_save_snapshot, snapshot)

    def _sync_save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        run_dir = self._run_dir(snapshot.run_id)
        snap_dir = run_dir / 'snapshots'
        snap_dir.mkdir(parents=True, exist_ok=True)
        messages_json = json.loads(ModelMessagesTypeAdapter.dump_json(snapshot.messages).decode('utf-8'))
        payload: dict[str, object] = {
            'run_id': snapshot.run_id,
            'step_index': snapshot.step_index,
            'conversation_id': snapshot.conversation_id,
            'parent_run_id': snapshot.parent_run_id,
            'agent_name': snapshot.agent_name,
            'timestamp': snapshot.timestamp.isoformat(),
            'messages': messages_json,
        }
        seq = self._next_snapshot_seq(snap_dir)
        (snap_dir / f'{seq}.json').write_text(json.dumps(payload), encoding='utf-8')

    @staticmethod
    def _next_snapshot_seq(snap_dir: Path) -> int:
        """Append-only monotonic counter per run directory.

        Using the snapshot's `step_index` as the filename would collide when
        the same `run_id` is reused across `Agent.run` calls — `ctx.run_step`
        resets to 0 each call. The physical sequence is independent of
        `step_index`, which lives inside the JSON payload.
        """
        max_seq = -1
        for path in snap_dir.glob('*.json'):
            try:
                seq = int(path.stem)
            except ValueError:
                continue
            if seq > max_seq:
                max_seq = seq
        return max_seq + 1

    async def latest_snapshot(self, *, run_id: str) -> ContinuableSnapshot | None:
        return await anyio.to_thread.run_sync(self._sync_latest_snapshot, run_id)

    def _sync_latest_snapshot(self, run_id: str) -> ContinuableSnapshot | None:
        snap_dir = self._run_dir(run_id) / 'snapshots'
        if not snap_dir.exists():
            return None
        candidates: list[tuple[int, Path]] = []
        for path in snap_dir.glob('*.json'):
            try:
                candidates.append((int(path.stem), path))
            except ValueError:
                continue
        if not candidates:
            return None
        _, latest_path = max(candidates, key=lambda c: c[0])
        data = _load_json_object(latest_path.read_text(encoding='utf-8'))
        messages: list[ModelMessage] = ModelMessagesTypeAdapter.validate_python(data['messages'])
        timestamp_raw = data['timestamp']
        step_raw = data['step_index']
        if not (isinstance(timestamp_raw, str) and isinstance(step_raw, int)):
            raise ValueError('snapshot has wrong types')
        return ContinuableSnapshot(
            run_id=run_id,
            step_index=step_raw,
            messages=messages,
            conversation_id=_opt_str(data.get('conversation_id')),
            parent_run_id=_opt_str(data.get('parent_run_id')),
            agent_name=_opt_str(data.get('agent_name')),
            timestamp=datetime.fromisoformat(timestamp_raw),
        )

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        await anyio.to_thread.run_sync(self._sync_record_tool_effect, record)

    def _sync_record_tool_effect(self, record: ToolEffectRecord) -> None:
        run_dir = self._run_dir(record.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(_tool_effect_to_dict(record))
        with (run_dir / 'tool_effects.jsonl').open('a', encoding='utf-8') as fp:
            fp.write(line + '\n')

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        return await anyio.to_thread.run_sync(self._sync_get_tool_effect, run_id, tool_call_id)

    def _sync_get_tool_effect(self, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        path = self._run_dir(run_id) / 'tool_effects.jsonl'
        if not path.exists():
            return None
        latest: ToolEffectRecord | None = None
        for raw in path.read_text(encoding='utf-8').splitlines():
            if not raw.strip():
                continue
            record = _tool_effect_from_dict(_load_json_object(raw))
            if record.tool_call_id == tool_call_id:
                latest = record
        return latest

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        return await anyio.to_thread.run_sync(self._sync_list_unresolved_tool_effects, run_id)

    def _sync_list_unresolved_tool_effects(self, run_id: str) -> list[ToolEffectRecord]:
        path = self._run_dir(run_id) / 'tool_effects.jsonl'
        if not path.exists():
            return []
        latest_by_call: dict[str, ToolEffectRecord] = {}
        for raw in path.read_text(encoding='utf-8').splitlines():
            if not raw.strip():
                continue
            record = _tool_effect_from_dict(_load_json_object(raw))
            latest_by_call[record.tool_call_id] = record
        return [r for r in latest_by_call.values() if r.status == 'started']
