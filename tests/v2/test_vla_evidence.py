"""The VLA-backed evidence model, and the anchor discipline it enforces."""

from __future__ import annotations

import pytest

from interaction_uncertainty.observation import PolicyContext
from interaction_uncertainty.v2.evidence import ModelStamp
from interaction_uncertainty.v2.task import TaskSpec, TerminalDecision
from interaction_uncertainty.v2.vla_bridge import ActionChunkSample, HypothesisAnchor
from interaction_uncertainty.v2.vla_evidence import (
    EVALUATOR_ONLY_ROLES,
    HypothesisSite,
    VlaActionEvidenceModel,
    sites_from_spec,
)

from .test_v2_pipeline import observation

STAMP = ModelStamp("pi05-libero-test", "a" * 64, "identity")

# motion_scale is 6.0, so a chunk travelling 3.0 is decisive without pinning
# the ceiling, and one travelling 12.0 saturates it.
UNSATURATED = 3.0
SATURATING = 12.0


def task() -> TaskSpec:
    return TaskSpec.create(
        task_id="butter-drawer",
        final_goal_prompt="Place the butter in the wicker basket.",
        ontology_id="target-access-v1",
        ontology_version="1.0.0",
        hypotheses=("IN_DRAWER", "ON_SURFACE", "TARGET_ABSENT"),
        terminal_decisions=(TerminalDecision.DIRECT_ACT, TerminalDecision.NOT_FOUND),
        loss_matrix=((0.0, 0.0, 1.0), (1.0, 1.0, 0.0)),
    )


def chunk_towards(direction: tuple[float, float, float], distance: float) -> ActionChunkSample:
    norm = sum(value**2 for value in direction) ** 0.5
    step = [value / norm * distance for value in direction]
    return ActionChunkSample.from_chunk([[*step, 0.0, 0.0, 0.0, -1.0]])


def sites() -> tuple[HypothesisSite, ...]:
    return (
        HypothesisSite("IN_DRAWER", HypothesisAnchor("IN_DRAWER", (1.0, 0.0, 0.0))),
        HypothesisSite("ON_SURFACE", HypothesisAnchor("ON_SURFACE", (-1.0, 0.0, 0.0))),
        HypothesisSite("TARGET_ABSENT"),
    )


def model(samples: list[ActionChunkSample]) -> VlaActionEvidenceModel:
    return VlaActionEvidenceModel(
        sampler=lambda _context: samples,
        eef_position=lambda _context: (0.0, 0.0, 0.0),
        sites=sites(),
        model_stamp=STAMP,
    )


def context(spec: TaskSpec) -> PolicyContext:
    return PolicyContext("ep", 0, spec.final_goal_prompt, observation("f", "public://closed"))


def test_evaluator_private_anchors_never_reach_the_controller() -> None:
    """The hidden target's pose is the one thing the controller must not get."""

    resolved: list[str] = []
    declared = [
        {"label": "IN_DRAWER", "role": "occluder", "ref": "wooden_cabinet_1"},
        {"label": "ON_SURFACE", "role": "placement", "ref": "basket_1"},
        {"label": "butter", "role": "task_target", "ref": "butter_1"},
    ]

    def resolve(ref: str) -> tuple[float, float, float]:
        resolved.append(ref)
        return (1.0, 0.0, 0.0)

    built = sites_from_spec(
        ("IN_DRAWER", "ON_SURFACE", "TARGET_ABSENT"), declared, resolve=resolve
    )

    assert "task_target" in EVALUATOR_ONLY_ROLES
    # Never resolved, so its position was never even computed, let alone used.
    assert "butter_1" not in resolved
    assert [site.label for site in built] == ["IN_DRAWER", "ON_SURFACE", "TARGET_ABSENT"]
    assert built[-1].anchor is None


def test_unmatched_hypotheses_become_abstract_rather_than_raising() -> None:
    built = sites_from_spec(
        ("IN_DRAWER", "TARGET_ABSENT"),
        [{"label": "IN_DRAWER", "role": "occluder", "ref": "cab"}],
        resolve=lambda _ref: (1.0, 0.0, 0.0),
    )
    assert built[1].label == "TARGET_ABSENT"
    assert not built[1].is_grounded


def test_abstract_hypothesis_receives_no_action_evidence() -> None:
    spec = task()
    packet = model([chunk_towards((1.0, 0.0, 0.0), UNSATURATED)]).infer(context(spec), spec)
    assert packet.hypothesis_labels == spec.hypotheses
    # TARGET_ABSENT is last and structurally unreachable by a reaching motion.
    assert packet.hypothesis_evidence[2] == 0.0
    assert packet.hypothesis_evidence[0] > packet.hypothesis_evidence[1]


def test_evidence_follows_the_direction_the_policy_actually_moves() -> None:
    spec = task()
    towards_drawer = model([chunk_towards((1.0, 0.0, 0.0), UNSATURATED)]).infer(
        context(spec), spec
    )
    towards_surface = model([chunk_towards((-1.0, 0.0, 0.0), UNSATURATED)]).infer(
        context(spec), spec
    )
    assert towards_drawer.hypothesis_evidence[0] > towards_drawer.hypothesis_evidence[1]
    assert towards_surface.hypothesis_evidence[1] > towards_surface.hypothesis_evidence[0]


def test_a_policy_that_does_not_move_produces_no_evidence() -> None:
    """Vacuity has to survive a stationary policy, or it measures nothing."""

    spec = task()
    packet = model([chunk_towards((1.0, 0.0, 0.0), 0.0)]).infer(context(spec), spec)
    assert sum(packet.hypothesis_evidence) == 0.0
    assert packet.sufficiency_evidence == (0.0, 0.0)


def test_saturation_is_reported_so_a_flat_curve_can_be_diagnosed() -> None:
    spec = task()
    saturated = model([chunk_towards((1.0, 0.0, 0.0), SATURATING)])
    saturated.infer(context(spec), spec)
    assert saturated.last_saturated_fraction == 1.0

    modest = model([chunk_towards((1.0, 0.0, 0.0), UNSATURATED)])
    modest.infer(context(spec), spec)
    assert modest.last_saturated_fraction == 0.0


def test_fewer_than_two_grounded_sites_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="at least two hypotheses must be grounded"):
        VlaActionEvidenceModel(
            sampler=lambda _context: [],
            eef_position=lambda _context: (0.0, 0.0, 0.0),
            sites=(
                HypothesisSite("IN_DRAWER", HypothesisAnchor("IN_DRAWER", (1.0, 0.0, 0.0))),
                HypothesisSite("TARGET_ABSENT"),
            ),
            model_stamp=STAMP,
        )


def test_ontology_mismatch_is_refused() -> None:
    spec = task()
    mismatched = VlaActionEvidenceModel(
        sampler=lambda _context: [chunk_towards((1.0, 0.0, 0.0), UNSATURATED)],
        eef_position=lambda _context: (0.0, 0.0, 0.0),
        sites=(
            HypothesisSite("WRONG_A", HypothesisAnchor("WRONG_A", (1.0, 0.0, 0.0))),
            HypothesisSite("WRONG_B", HypothesisAnchor("WRONG_B", (-1.0, 0.0, 0.0))),
        ),
        model_stamp=STAMP,
    )
    with pytest.raises(ValueError, match="does not match TaskSpec"):
        mismatched.infer(context(spec), spec)


def test_packet_authenticates_against_its_context() -> None:
    spec = task()
    packet = model([chunk_towards((1.0, 0.0, 0.0), UNSATURATED)]).infer(context(spec), spec)
    packet.validate_request(context(spec), spec)
