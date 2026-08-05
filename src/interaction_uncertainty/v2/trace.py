"""Append-only, hash-chained policy trace events."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..firewall import PolicyFirewall
from .task import TaskKey, canonical_digest
from .validation import require_exact_keys, strict_integer, strict_string

_TRACE_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "event_id",
        "parent_event_id",
        "event_type",
        "episode_id",
        "step_index",
        "task_key",
        "timestamp_utc",
        "payload",
        "content_digest",
    }
)


def _validate_utc_timestamp(value: str) -> None:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("TraceEvent.timestamp_utc must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("TraceEvent.timestamp_utc must include an explicit UTC offset")


def _reject_nonstandard_json_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant {value!r}")


@dataclass(frozen=True)
class TraceEvent:
    schema_version: str
    event_id: str
    parent_event_id: str | None
    event_type: str
    episode_id: str
    step_index: int
    task_key: TaskKey
    timestamp_utc: str
    payload: Mapping[str, object]
    content_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != "interaction-uncertainty.trace-event.v2":
            raise ValueError("unsupported trace schema")
        if re.fullmatch(r"[0-9a-f]{64}", self.event_id) is None:
            raise ValueError("trace event_id must be a lowercase SHA-256 digest")
        if self.parent_event_id is not None and re.fullmatch(
            r"[0-9a-f]{64}", self.parent_event_id
        ) is None:
            raise ValueError("trace parent_event_id must be null or a SHA-256 digest")
        if re.fullmatch(r"[0-9a-f]{64}", self.content_digest) is None:
            raise ValueError("trace content_digest must be a lowercase SHA-256 digest")
        if not isinstance(self.task_key, TaskKey):
            raise TypeError("trace task_key must be a TaskKey")
        if not isinstance(self.payload, Mapping):
            raise TypeError("trace payload must be a mapping")
        if (
            not isinstance(self.event_type, str)
            or not isinstance(self.episode_id, str)
            or not self.event_type.strip()
            or not self.episode_id.strip()
            or isinstance(self.step_index, bool)
            or not isinstance(self.step_index, int)
            or self.step_index < 0
        ):
            raise ValueError("invalid trace event identity")
        _validate_utc_timestamp(self.timestamp_utc)
        PolicyFirewall().validate_recursive(self.payload, location="TraceEvent.payload")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "parent_event_id": self.parent_event_id,
            "event_type": self.event_type,
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "task_key": self.task_key.to_dict(),
            "timestamp_utc": self.timestamp_utc,
            "payload": dict(self.payload),
            "content_digest": self.content_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TraceEvent:
        if not isinstance(payload, Mapping):
            raise TypeError("TraceEvent must be a mapping")
        require_exact_keys(payload, required=_TRACE_EVENT_FIELDS, location="TraceEvent")

        parent_event_id = payload["parent_event_id"]
        if parent_event_id is not None:
            parent_event_id = strict_string(
                parent_event_id, location="TraceEvent.parent_event_id"
            )

        task_key_payload = payload["task_key"]
        if not isinstance(task_key_payload, Mapping):
            raise TypeError("TraceEvent.task_key must be a mapping")

        timestamp_utc = strict_string(
            payload["timestamp_utc"], location="TraceEvent.timestamp_utc"
        )
        _validate_utc_timestamp(timestamp_utc)

        event_payload = payload["payload"]
        if not isinstance(event_payload, Mapping):
            raise TypeError("TraceEvent.payload must be a mapping")
        PolicyFirewall().validate_recursive(event_payload, location="TraceEvent.payload")

        return cls(
            schema_version=strict_string(
                payload["schema_version"], location="TraceEvent.schema_version"
            ),
            event_id=strict_string(payload["event_id"], location="TraceEvent.event_id"),
            parent_event_id=parent_event_id,
            event_type=strict_string(
                payload["event_type"], location="TraceEvent.event_type"
            ),
            episode_id=strict_string(
                payload["episode_id"], location="TraceEvent.episode_id"
            ),
            step_index=strict_integer(
                payload["step_index"], location="TraceEvent.step_index", minimum=0
            ),
            task_key=TaskKey.from_dict(task_key_payload),
            timestamp_utc=timestamp_utc,
            payload=event_payload,
            content_digest=strict_string(
                payload["content_digest"], location="TraceEvent.content_digest"
            ),
        )


def _event_identity_content(event: TraceEvent) -> dict[str, object]:
    """All immutable fields whose digest is linked by the next event."""

    return {
        "schema_version": event.schema_version,
        "parent_event_id": event.parent_event_id,
        "event_type": event.event_type,
        "episode_id": event.episode_id,
        "step_index": event.step_index,
        "task_key": event.task_key.to_dict(),
        "timestamp_utc": event.timestamp_utc,
        "payload": dict(event.payload),
    }


def _serialized_event_content(event: TraceEvent) -> dict[str, object]:
    """All immutable serialized fields protected by ``content_digest``."""

    return {
        "schema_version": event.schema_version,
        "event_id": event.event_id,
        "parent_event_id": event.parent_event_id,
        "event_type": event.event_type,
        "episode_id": event.episode_id,
        "step_index": event.step_index,
        "task_key": event.task_key.to_dict(),
        "timestamp_utc": event.timestamp_utc,
        "payload": dict(event.payload),
    }


@runtime_checkable
class TraceSink(Protocol):
    def append(self, event: TraceEvent) -> None: ...


@dataclass
class InMemoryTraceSink:
    events: list[TraceEvent] = field(default_factory=list)

    def append(self, event: TraceEvent) -> None:
        expected_parent = None if not self.events else self.events[-1].event_id
        if event.parent_event_id != expected_parent:
            raise ValueError("trace sink received an event with the wrong parent")
        self.events.append(event)


@dataclass
class JSONLTraceSink:
    path: Path
    _initialized: bool = field(default=False, init=False)
    _last_event_id: str | None = field(default=None, init=False)

    def append(self, event: TraceEvent) -> None:
        if event.parent_event_id != self._last_event_id:
            raise ValueError("trace sink received an event with the wrong parent")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(
            event.to_dict(), sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        mode = "a" if self._initialized else "x"
        with self.path.open(mode, encoding="utf-8") as handle:
            handle.write(serialized + "\n")
        self._initialized = True
        self._last_event_id = event.event_id


def load_trace_jsonl(path: Path) -> list[TraceEvent]:
    """Load a non-empty JSONL trace with line-local parsing errors."""

    events: list[TraceEvent] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"trace JSONL contains an empty line at line {line_number}")
            try:
                raw = json.loads(line, parse_constant=_reject_nonstandard_json_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
                raise ValueError(
                    f"invalid JSON in trace JSONL at line {line_number}: {detail}"
                ) from exc
            if not isinstance(raw, Mapping):
                raise TypeError(
                    f"trace JSONL line {line_number} must contain one JSON object"
                )
            try:
                event = TraceEvent.from_dict(raw)
            except (TypeError, ValueError) as exc:
                error_type = TypeError if isinstance(exc, TypeError) else ValueError
                raise error_type(
                    f"invalid trace event at line {line_number}: {exc}"
                ) from exc
            events.append(event)
    if not events:
        raise ValueError("trace JSONL file is empty")
    return events


def make_trace_event(
    *,
    parent_event_id: str | None,
    event_type: str,
    episode_id: str,
    step_index: int,
    task_key: TaskKey,
    payload: Mapping[str, object],
) -> TraceEvent:
    if not event_type.strip() or not episode_id.strip() or step_index < 0:
        raise ValueError("invalid trace event identity")
    PolicyFirewall().validate_recursive(payload, location=f"trace.{event_type}")
    timestamp = datetime.now(timezone.utc).isoformat()
    schema_version = "interaction-uncertainty.trace-event.v2"
    identity_content = {
        "schema_version": schema_version,
        "parent_event_id": parent_event_id,
        "event_type": event_type,
        "episode_id": episode_id,
        "step_index": step_index,
        "task_key": task_key.to_dict(),
        "timestamp_utc": timestamp,
        "payload": dict(payload),
    }
    event_id = canonical_digest(identity_content)
    serialized_content = {
        "schema_version": schema_version,
        "event_id": event_id,
        "parent_event_id": parent_event_id,
        "event_type": event_type,
        "episode_id": episode_id,
        "step_index": step_index,
        "task_key": task_key.to_dict(),
        "timestamp_utc": timestamp,
        "payload": dict(payload),
    }
    return TraceEvent(
        schema_version=schema_version,
        event_id=event_id,
        parent_event_id=parent_event_id,
        event_type=event_type,
        episode_id=episode_id,
        step_index=step_index,
        task_key=task_key,
        timestamp_utc=timestamp,
        payload=payload,
        content_digest=canonical_digest(serialized_content),
    )


def verify_trace_chain(events: Sequence[TraceEvent]) -> None:
    """Fail closed if event identity, content, or parent linkage was changed."""

    previous_id: str | None = None
    seen: set[str] = set()
    for index, event in enumerate(events):
        if event.schema_version != "interaction-uncertainty.trace-event.v2":
            raise ValueError(f"unsupported trace schema at index {index}")
        if event.parent_event_id != previous_id:
            raise ValueError(f"broken trace parent link at index {index}")
        expected_event_id = canonical_digest(_event_identity_content(event))
        if event.event_id != expected_event_id:
            raise ValueError(f"trace event identity mismatch at index {index}")
        expected_content_digest = canonical_digest(_serialized_event_content(event))
        if event.content_digest != expected_content_digest:
            raise ValueError(f"trace content digest mismatch at index {index}")
        if event.event_id in seen:
            raise ValueError(f"duplicate trace event identity at index {index}")
        seen.add(event.event_id)
        previous_id = event.event_id
