"""Behavioural contract for reading a belief out of VLA action samples."""

from __future__ import annotations

import numpy as np
import pytest

from interaction_uncertainty.uncertainty import (
    subjective_logic_dissonance,
    summarize_task_uncertainty,
)
from interaction_uncertainty.v2.primitives import (
    COARSE_ACTION_SPACE,
    COARSE_BY_PRIMITIVE,
    PRIMITIVES_BY_COARSE,
    CoarsePrimitive,
    PrimitiveKind,
    to_coarse,
)
from interaction_uncertainty.v2.vla_bridge import (
    ActionChunkSample,
    AttributionConfig,
    HypothesisAnchor,
    action_induced_belief,
    attribute_samples,
)

EEF = (0.0, 0.0, 0.0)
ANCHORS = (
    HypothesisAnchor("target_in_drawer", (1.0, 0.0, 0.0)),
    HypothesisAnchor("basket", (-1.0, 0.0, 0.0)),
    HypothesisAnchor("distractor", (0.0, 1.0, 0.0)),
)


def _samples(direction: tuple[float, float, float], *, count: int, scale: float):
    unit = np.asarray(direction, dtype=np.float64)
    unit = unit / np.linalg.norm(unit)
    return [ActionChunkSample(tuple((unit * scale).tolist())) for _ in range(count)]


def _belief(samples, **kwargs):
    return action_induced_belief(
        prompt="Place the butter in the wicker basket.",
        samples=samples,
        eef_position=EEF,
        anchors=ANCHORS,
        **kwargs,
    )


def test_stationary_samples_report_maximum_vacuity() -> None:
    """A policy that commits to nothing must not look informed."""

    belief = _belief([ActionChunkSample((0.0, 0.0, 0.0)) for _ in range(16)])
    report = summarize_task_uncertainty(belief)
    assert report.vacuity == pytest.approx(1.0)
    assert report.normalized_predictive_entropy == pytest.approx(1.0)
    assert belief.sufficiency.mean == pytest.approx(0.5)


def test_decisive_agreement_collapses_vacuity_and_dissonance() -> None:
    belief = _belief(_samples((1.0, 0.0, 0.0), count=16, scale=0.2))
    report = summarize_task_uncertainty(belief)
    assert report.vacuity < 0.2
    assert report.dissonance < 0.1
    assert belief.hypotheses.probability("target_in_drawer") > 0.8
    assert belief.sufficiency.mean > 0.9


def test_split_commitment_raises_dissonance_without_raising_vacuity() -> None:
    """Conflict and absence of evidence are different failures.

    Eight samples driving at one anchor and eight at another produce plenty of
    evidence, so vacuity stays low; what rises is dissonance.
    """

    samples = _samples((1.0, 0.0, 0.0), count=8, scale=0.2) + _samples(
        (0.0, 1.0, 0.0), count=8, scale=0.2
    )
    belief = _belief(samples)
    report = summarize_task_uncertainty(belief)
    assert report.vacuity < 0.2
    assert report.dissonance > 0.5
    assert subjective_logic_dissonance(belief.hypotheses) == report.dissonance


def test_evidence_total_is_not_the_sample_count() -> None:
    """The property that makes vacuity informative at fixed sample budget.

    Were evidence simply a per-sample count, ``S = N + K`` would be constant and
    vacuity would be identical for a committed and a dithering policy.
    """

    decisive = _belief(_samples((1.0, 0.0, 0.0), count=16, scale=0.2))
    timid = _belief(_samples((1.0, 0.0, 0.0), count=16, scale=0.002))
    assert decisive.hypotheses.strength > timid.hypotheses.strength
    assert summarize_task_uncertainty(timid).vacuity > summarize_task_uncertainty(
        decisive
    ).vacuity


def test_motion_supporting_no_hypothesis_yields_no_evidence() -> None:
    """Decisive travel away from every anchor is vacuity, not support."""

    away = _samples((0.0, -1.0, -1.0), count=12, scale=0.4)
    belief = _belief(away, config=AttributionConfig(alignment_floor=0.1))
    assert summarize_task_uncertainty(belief).vacuity == pytest.approx(1.0)


def test_decisiveness_saturates_at_motion_scale() -> None:
    config = AttributionConfig(motion_scale=0.05)
    attributions = attribute_samples(
        _samples((1.0, 0.0, 0.0), count=1, scale=0.5),
        eef_position=EEF,
        anchors=ANCHORS,
        config=config,
    )
    assert attributions[0].decisiveness == pytest.approx(1.0)
    assert attributions[0].best_label == "target_in_drawer"


def test_from_chunk_sums_translation_and_averages_gripper() -> None:
    chunk = np.zeros((10, 7), dtype=np.float64)
    chunk[:, 0] = 0.01
    chunk[:, 5] = 0.02
    chunk[:, 6] = -1.0
    sample = ActionChunkSample.from_chunk(chunk)
    assert sample.translation_delta == pytest.approx((0.1, 0.0, 0.0))
    assert sample.rotation_norm == pytest.approx(0.2)
    assert sample.gripper_command == pytest.approx(-1.0)


def test_from_chunk_rejects_malformed_chunks() -> None:
    with pytest.raises(ValueError):
        ActionChunkSample.from_chunk(np.zeros((10, 3)))
    with pytest.raises(ValueError):
        ActionChunkSample.from_chunk(np.full((10, 7), np.nan))


def test_bridge_rejects_degenerate_hypothesis_sets() -> None:
    with pytest.raises(ValueError):
        attribute_samples(
            _samples((1.0, 0.0, 0.0), count=1, scale=0.1),
            eef_position=EEF,
            anchors=ANCHORS[:1],
        )
    with pytest.raises(ValueError):
        attribute_samples(
            _samples((1.0, 0.0, 0.0), count=1, scale=0.1),
            eef_position=EEF,
            anchors=(HypothesisAnchor("a", (1.0, 0.0, 0.0)), HypothesisAnchor("a", (0.0, 1.0, 0.0))),
        )
    with pytest.raises(ValueError):
        # An anchor sitting on the end effector has no defined direction.
        attribute_samples(
            _samples((1.0, 0.0, 0.0), count=1, scale=0.1),
            eef_position=EEF,
            anchors=(HypothesisAnchor("a", EEF), HypothesisAnchor("b", (0.0, 1.0, 0.0))),
        )


def test_bridge_requires_a_prompt_and_samples() -> None:
    with pytest.raises(ValueError):
        _belief([])
    with pytest.raises(ValueError):
        action_induced_belief(
            prompt="   ",
            samples=_samples((1.0, 0.0, 0.0), count=1, scale=0.1),
            eef_position=EEF,
            anchors=ANCHORS,
        )


def test_coarse_action_space_is_a_total_projection() -> None:
    """Every execution primitive must land in ``A``, and ``A`` must be covered."""

    assert set(COARSE_BY_PRIMITIVE) == set(PrimitiveKind)
    assert set(COARSE_ACTION_SPACE) == set(CoarsePrimitive)
    covered = {kind for kinds in PRIMITIVES_BY_COARSE.values() for kind in kinds}
    assert covered == set(PrimitiveKind)
    assert to_coarse(PrimitiveKind.PULL_DRAWER) is CoarsePrimitive.REMOVE_OCCLUDER
    assert to_coarse(PrimitiveKind.STOP_NOT_FOUND) is CoarsePrimitive.NOT_FOUND
    with pytest.raises(TypeError):
        to_coarse("PULL_DRAWER")  # type: ignore[arg-type]


def test_container_opening_folds_into_remove_occluder() -> None:
    """Recorded so the design decision is caught if it is ever reversed."""

    assert PRIMITIVES_BY_COARSE[CoarsePrimitive.REMOVE_OCCLUDER] == (
        PrimitiveKind.OPEN_CONTAINER,
        PrimitiveKind.PULL_DRAWER,
        PrimitiveKind.UNCOVER,
        PrimitiveKind.CLEAR_OCCLUDER,
        PrimitiveKind.PUSH_ASIDE,
    )
