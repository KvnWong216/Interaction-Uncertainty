"""Command-line entry points for v0.2 contracts and remote model services."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def _read_json(path: str) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_task(task_path: str) -> dict[str, object]:
    from collections.abc import Mapping

    from .v2.task import TaskSpec

    raw = _read_json(task_path)
    if not isinstance(raw, Mapping):
        raise TypeError("task JSON must contain one object")
    task = TaskSpec.from_dict(raw)
    return task.to_dict()


def verify_trace(trace_path: str) -> dict[str, object]:
    from .v2.trace import load_trace_jsonl, verify_trace_chain

    events = load_trace_jsonl(Path(trace_path))
    verify_trace_chain(events)
    return {
        "event_count": len(events),
        "head_event_id": events[-1].event_id,
    }


def plan_remote(
    *,
    task_path: str,
    observation_path: str,
    episode_id: str,
    evidence_endpoint: str,
    effect_endpoint: str,
    output_path: str | None,
    trace_path: str | None,
) -> dict[str, object]:
    """Run one real bridge step against versioned evidence/effect services."""

    from collections.abc import Mapping

    from .observation import PolicyObservation
    from .v2.belief import EvidentialBeliefFilter
    from .v2.controller import EpisodeController
    from .v2.remote import RemoteActionOutcomeCritic, RemoteEvidenceModel
    from .v2.task import TaskSpec
    from .v2.trace import InMemoryTraceSink, JSONLTraceSink

    task_raw = _read_json(task_path)
    observation_raw = _read_json(observation_path)
    if not isinstance(task_raw, Mapping) or not isinstance(observation_raw, Mapping):
        raise TypeError("task and observation files must each contain one JSON object")
    task = TaskSpec.from_dict(task_raw)
    observation = PolicyObservation.from_dict(observation_raw)
    trace_sink = (
        InMemoryTraceSink()
        if trace_path is None
        else JSONLTraceSink(Path(trace_path))
    )
    controller = EpisodeController(
        task=task,
        episode_id=episode_id,
        evidence_model=RemoteEvidenceModel(evidence_endpoint),
        belief_filter=EvidentialBeliefFilter(),
        outcome_critic=RemoteActionOutcomeCritic(effect_endpoint),
        trace_sink=trace_sink,
    )
    result = controller.observe_and_plan(observation).to_dict()
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if output_path is None:
        print(serialized)
    else:
        Path(output_path).write_text(serialized + "\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="interaction-uncertainty")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-task", help="validate a v0.2 TaskSpec JSON")
    validate.add_argument("--task", required=True, help="path to TaskSpec JSON")
    verify = subparsers.add_parser(
        "verify-trace", help="load and verify a v0.2 hash-chained trace"
    )
    verify.add_argument("--trace", required=True, help="path to trace JSONL")
    remote = subparsers.add_parser(
        "plan-remote", help="plan one v0.2 step with external evidence/effect services"
    )
    remote.add_argument("--task", required=True)
    remote.add_argument("--observation", required=True)
    remote.add_argument("--episode-id", required=True)
    remote.add_argument("--evidence-endpoint", required=True)
    remote.add_argument("--effect-endpoint", required=True)
    remote.add_argument("--output")
    remote.add_argument("--trace")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-task":
        print(json.dumps(validate_task(args.task), indent=2, sort_keys=True))
        return 0
    if args.command == "verify-trace":
        print(json.dumps(verify_trace(args.trace), indent=2, sort_keys=True))
        return 0
    if args.command == "plan-remote":
        plan_remote(
            task_path=args.task,
            observation_path=args.observation,
            episode_id=args.episode_id,
            evidence_endpoint=args.evidence_endpoint,
            effect_endpoint=args.effect_endpoint,
            output_path=args.output,
            trace_path=args.trace,
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
