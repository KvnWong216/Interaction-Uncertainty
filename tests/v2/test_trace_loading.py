from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from interaction_uncertainty.cli import main
from interaction_uncertainty.v2.trace import (
    JSONLTraceSink,
    TraceEvent,
    load_trace_jsonl,
    make_trace_event,
    verify_trace_chain,
)
from tests.v2.test_v2_pipeline import task


def _write_trace(path: Path) -> list[TraceEvent]:
    key = task().key
    first = make_trace_event(
        parent_event_id=None,
        event_type="EvidenceReceived",
        episode_id="trace-load-test",
        step_index=0,
        task_key=key,
        payload={"status": "observed"},
    )
    second = make_trace_event(
        parent_event_id=first.event_id,
        event_type="PlanSelected",
        episode_id="trace-load-test",
        step_index=0,
        task_key=key,
        payload={"candidate_id": "open::container"},
    )
    sink = JSONLTraceSink(path)
    sink.append(first)
    sink.append(second)
    return [first, second]


def _rewrite_event(
    path: Path, index: int, mutate: Callable[[dict[str, object]], None]
) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    mutate(rows[index])
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_load_trace_jsonl_and_verify_trace_cli(tmp_path: Path, capsys) -> None:
    path = tmp_path / "trace.jsonl"
    expected = _write_trace(path)

    loaded = load_trace_jsonl(path)
    assert loaded == expected
    verify_trace_chain(loaded)

    assert main(["verify-trace", "--trace", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "event_count": 2,
        "head_event_id": expected[-1].event_id,
    }


@pytest.mark.parametrize(
    ("mutate", "error"),
    (
        (lambda row: row["payload"].update({"tampered": True}), "identity mismatch"),
        (
            lambda row: row.__setitem__("timestamp_utc", "1970-01-01T00:00:00+00:00"),
            "identity mismatch",
        ),
        (lambda row: row.__setitem__("parent_event_id", "0" * 64), "broken trace parent"),
        (lambda row: row.__setitem__("content_digest", "0" * 64), "content digest"),
    ),
)
def test_loaded_trace_detects_tampering(
    tmp_path: Path,
    mutate: Callable[[dict[str, object]], None],
    error: str,
) -> None:
    path = tmp_path / "trace.jsonl"
    _write_trace(path)
    _rewrite_event(path, 1, mutate)

    with pytest.raises(ValueError, match=error):
        verify_trace_chain(load_trace_jsonl(path))


@pytest.mark.parametrize(
    ("contents", "error_type", "error"),
    (
        ('{"event_id":\n', ValueError, "invalid JSON.*line 1"),
        ('{"payload": NaN}\n', ValueError, "non-standard JSON constant"),
        ("", ValueError, "file is empty"),
        ("\n", ValueError, "empty line at line 1"),
        ("[]\n", TypeError, "line 1.*JSON object"),
    ),
)
def test_load_trace_jsonl_rejects_invalid_files(
    tmp_path: Path,
    contents: str,
    error_type: type[Exception],
    error: str,
) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(error_type, match=error):
        load_trace_jsonl(path)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("step_index", "0", "step_index must be an integer"),
        ("event_type", 7, "event_type must be a string"),
        ("parent_event_id", 7, "parent_event_id must be a string"),
        ("task_key", [], "task_key must be a mapping"),
        ("payload", [], "payload must be a mapping"),
        ("timestamp_utc", "2026-08-04T12:00:00-07:00", "explicit UTC offset"),
    ),
)
def test_trace_event_from_dict_rejects_type_and_timestamp_errors(
    tmp_path: Path, field: str, value: object, error: str
) -> None:
    path = tmp_path / "trace.jsonl"
    event = _write_trace(path)[0].to_dict()
    event[field] = value

    with pytest.raises((TypeError, ValueError), match=error):
        TraceEvent.from_dict(event)


def test_trace_event_from_dict_runs_policy_firewall(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    event = _write_trace(path)[0].to_dict()
    event["payload"] = {"oracle_action": "grasp"}

    with pytest.raises(ValueError, match="forbidden key"):
        TraceEvent.from_dict(event)
