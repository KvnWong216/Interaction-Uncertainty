from __future__ import annotations

from dataclasses import replace

import pytest

from interaction_uncertainty.beliefs import DeficitKind
from interaction_uncertainty.observation import PolicyContext, PolicyObservation, VisualAnchor
from interaction_uncertainty.v2.effects import (
    CandidateEffectForecast,
    ExecutionStatus,
    ObservationOutcomeModel,
)
from interaction_uncertainty.v2.evidence import ModelStamp
from interaction_uncertainty.v2.needs import InformationNeed
from interaction_uncertainty.v2.primitives import (
    CandidateSet,
    NeedDrivenPrimitiveProposer,
    NoParameters,
    OpenParameters,
    PrimitiveCall,
    PrimitiveKind,
    validate_candidate_set,
)
from interaction_uncertainty.v2.trace import JSONLTraceSink, make_trace_event
from tests.v2.test_p0_contracts import _controller
from tests.v2.test_v2_pipeline import (
    ContentDependentEvidenceModel,
    observation,
    task,
)


def test_probability_contract_does_not_use_numpy_relative_tolerance() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        task().validate_probabilities((0.25, 0.25, 0.500009))


def test_nearly_normalized_observation_columns_are_rejected() -> None:
    spec = task()
    outcomes = (
        ObservationOutcomeModel(
            "same_0",
            (0.5000045, 0.4999955, 0.5),
            ExecutionStatus.SUCCEEDED,
        ),
        ObservationOutcomeModel(
            "same_1",
            (0.5000045, 0.4999955, 0.5),
            ExecutionStatus.SUCCEEDED,
        ),
    )
    with pytest.raises(ValueError, match="exhaustive"):
        CandidateEffectForecast(
            schema_version="interaction-uncertainty.effect.v2",
            task_key=spec.key,
            candidate_id="near-normalized",
            hypothesis_labels=spec.hypotheses,
            transition_matrix=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            outcomes=outcomes,
            critic_uncertainty=0.0,
            model_stamp=ModelStamp("strict-z", "5" * 64, "identity"),
            rng_seed=0,
        )


def test_counterfactual_rollout_rejects_nan_before_ranking() -> None:
    plan = _controller().observe_and_plan(observation("closed", "public://closed"))
    with pytest.raises(ValueError, match="expected_posterior_risk"):
        replace(plan.rollouts[0], expected_posterior_risk=float("nan"))


def test_direct_action_requires_public_grounding_and_effect_forecast() -> None:
    closed = _controller().observe_and_plan(observation("closed", "public://closed"))
    assert PrimitiveKind.DIRECT_ACT not in {
        candidate.kind for candidate in closed.candidates.candidates
    }

    visible = _controller().observe_and_plan(
        observation("visible", "public://open-visible-orange-juice")
    )
    direct = next(
        candidate
        for candidate in visible.candidates.candidates
        if candidate.kind is PrimitiveKind.DIRECT_ACT
    )
    assert direct.source_anchor is not None
    assert direct.requires_effect_forecast
    assert direct.candidate_id in {forecast.candidate_id for forecast in visible.forecasts}
    assert visible.decision.selected.kind is PrimitiveKind.DIRECT_ACT
    direct_rollout = next(
        rollout
        for rollout in visible.rollouts
        if rollout.candidate_id == direct.candidate_id
    )
    assert direct_rollout.decision_rule == "FIXED_DIRECT_ACT"
    assert direct_rollout.decision_commitment_penalty >= 0.0
    assert direct_rollout.total_task_risk_reduction == pytest.approx(
        direct_rollout.physical_progress_value
        + direct_rollout.conditional_information_value
        - direct_rollout.decision_commitment_penalty
    )


def test_candidate_budget_keeps_high_priority_need_not_lexical_first() -> None:
    spec = task()
    low_anchor = VisualAnchor(
        "a-low",
        "budget-frame",
        "deployable_detector",
        affordances=("openable",),
    )
    high_anchor = VisualAnchor(
        "z-high",
        "budget-frame",
        "deployable_detector",
        affordances=("openable",),
    )
    public = PolicyObservation(
        "budget-frame",
        ("public://budget",),
        anchors=(low_anchor, high_anchor),
    )

    def need(identifier: str, anchor: str, priority: float) -> InformationNeed:
        return InformationNeed(
            need_id=identifier,
            task_key=spec.key,
            proposition_id=identifier,
            anchor_token=anchor,
            deficit_kind=DeficitKind.OCCLUDED_REGION,
            probability=1.0,
            decision_relevance=1.0,
            max_task_risk_reduction=priority,
            sufficiency_shortfall=1.0,
            priority=priority,
            source_deficit_ids=(identifier,),
        )

    candidates = NeedDrivenPrimitiveProposer(max_candidates=2).propose(
        task_key=spec.key,
        observation=public,
        needs=(need("low", "a-low", 0.001), need("high", "z-high", 100.0)),
    )
    selected_information = next(
        candidate
        for candidate in candidates.candidates
        if candidate.kind is PrimitiveKind.OPEN_CONTAINER
    )
    assert selected_information.source_anchor == high_anchor


def test_generic_candidate_validator_rejects_missing_affordance() -> None:
    spec = task()
    anchor = VisualAnchor(
        "not-openable",
        "validator-frame",
        "deployable_detector",
        affordances=("graspable",),
    )
    public = PolicyObservation(
        "validator-frame",
        ("public://validator",),
        anchors=(anchor,),
    )
    need = InformationNeed(
        need_id="need-0",
        task_key=spec.key,
        proposition_id="deficit-0",
        anchor_token=anchor.token,
        deficit_kind=DeficitKind.OCCLUDED_REGION,
        probability=1.0,
        decision_relevance=1.0,
        max_task_risk_reduction=1.0,
        sufficiency_shortfall=1.0,
        priority=1.0,
        source_deficit_ids=("deficit-0",),
    )
    invalid = PrimitiveCall(
        candidate_id="invalid-open",
        task_key=spec.key,
        kind=PrimitiveKind.OPEN_CONTAINER,
        source_anchor=anchor,
        destination_anchor=None,
        parameters=OpenParameters(),
        addresses_need_ids=(need.need_id,),
        proposal_score=1.0,
        proposer_id="external",
        stop_condition="reobserve",
    )
    stop = PrimitiveCall(
        candidate_id="terminal::not_found",
        task_key=spec.key,
        kind=PrimitiveKind.STOP_NOT_FOUND,
        source_anchor=None,
        destination_anchor=None,
        parameters=NoParameters(),
        addresses_need_ids=(),
        proposal_score=0.0,
        proposer_id="external",
        stop_condition="terminal",
    )
    candidate_set = CandidateSet(spec.key, public.frame_id, (stop, invalid))
    with pytest.raises(ValueError, match="affordances"):
        validate_candidate_set(candidate_set, public, (need,))


def test_execution_id_authenticates_timeout_and_selected_command() -> None:
    plan = _controller().observe_and_plan(observation("closed", "public://closed"))
    with pytest.raises(ValueError, match="exact command"):
        replace(
            plan.execution_request,
            timeout_s=plan.execution_request.timeout_s + 1.0,
        )


def test_jsonl_trace_sink_refuses_to_append_a_new_root_to_existing_file(
    tmp_path,
) -> None:
    spec = task()
    event = make_trace_event(
        parent_event_id=None,
        event_type="Test",
        episode_id="trace-exclusive",
        step_index=0,
        task_key=spec.key,
        payload={"status": "ok"},
    )
    path = tmp_path / "trace.jsonl"
    JSONLTraceSink(path).append(event)
    with pytest.raises(FileExistsError):
        JSONLTraceSink(path).append(event)


def test_task_and_observation_runtime_reject_schema_invalid_coercions() -> None:
    payload = task().to_dict()
    payload["key"]["task_id"] = 123  # type: ignore[index]
    with pytest.raises(TypeError, match="string"):
        type(task()).from_dict(payload)

    observation_payload = observation("frame", "public://closed").to_dict()
    observation_payload["image_refs"] = "abc"
    with pytest.raises(TypeError, match="array"):
        PolicyObservation.from_dict(observation_payload)


def test_evidence_packet_rejects_overflowing_finite_pseudo_evidence() -> None:
    spec = task()
    context = PolicyContext(
        "overflow",
        0,
        spec.final_goal_prompt,
        observation("overflow", "public://closed"),
    )
    packet = ContentDependentEvidenceModel().infer(context, spec)
    with pytest.raises(ValueError, match="strength must be finite"):
        replace(
            packet,
            hypothesis_evidence=(1e308, 1e308, 1e308),
            sufficiency_evidence=(1e308, 1e308),
            content_digest="",
        )
