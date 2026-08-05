"""P0 contracts for the production-oriented v0.2 control loop.

These tests intentionally exercise public interfaces only.  They verify
cross-module invariants that must hold regardless of the learned evidence or
effect backend used in a future experiment.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from interaction_uncertainty.observation import PolicyContext
from interaction_uncertainty.v2.belief import EvidentialBeliefFilter
from interaction_uncertainty.v2.controller import ControllerState, EpisodeController
from interaction_uncertainty.v2.effects import (
    CandidateEffectForecast,
    ExecutionStatus,
    ObservationOutcomeModel,
    rollout_forecast,
)
from interaction_uncertainty.v2.evidence import ModelStamp, observation_digest
from interaction_uncertainty.v2.execution import ExecutionReport, ExecutionResult
from interaction_uncertainty.v2.needs import BayesRiskNeedExtractor
from interaction_uncertainty.v2.planning import PlanningWeights, rank_primitives
from interaction_uncertainty.v2.primitives import (
    CandidateSet,
    NeedDrivenPrimitiveProposer,
    PrimitiveKind,
)
from interaction_uncertainty.v2.trace import InMemoryTraceSink, verify_trace_chain
from tests.v2.test_v2_pipeline import (
    AnalyticObservationCritic,
    ContentDependentEvidenceModel,
    observation,
    task,
)


def _weights() -> PlanningWeights:
    return PlanningWeights(
        action_cost=0.01,
        physical_risk=0.01,
        disturbance=0.0,
        critic_uncertainty=0.01,
    )


def _controller(*, trace_sink: InMemoryTraceSink | None = None) -> EpisodeController:
    kwargs: dict[str, object] = {}
    if trace_sink is not None:
        kwargs["trace_sink"] = trace_sink
    return EpisodeController(
        task=task(),
        episode_id="p0-contract-episode",
        evidence_model=ContentDependentEvidenceModel(),
        belief_filter=EvidentialBeliefFilter(),
        outcome_critic=AnalyticObservationCritic(),
        planning_weights=_weights(),
        **kwargs,
    )


@pytest.mark.parametrize(
    ("candidate_id_override", "primitive_kind_override", "error"),
    (
        ("terminal::direct", None, "changed the selected candidate"),
        (None, PrimitiveKind.ROTATE_TO_LABEL.value, "changed the selected primitive family"),
    ),
)
def test_execution_report_cannot_change_selected_candidate_or_family(
    candidate_id_override: str | None,
    primitive_kind_override: str | None,
    error: str,
) -> None:
    """The executor refines a selected skill; it may not silently re-plan."""

    controller = _controller()
    plan = controller.observe_and_plan(observation("closed", "public://closed"))
    selected = plan.decision.selected
    assert selected.kind is PrimitiveKind.OPEN_CONTAINER

    report = ExecutionReport(
        schema_version="interaction-uncertainty.execution-report.v2",
        execution_id=plan.execution_request.execution_id,
        candidate_id=candidate_id_override or selected.candidate_id,
        primitive_kind=primitive_kind_override or selected.kind.value,
        result=ExecutionResult.SUCCEEDED,
        executed_steps=8,
        termination_reason="executor_claimed_success",
        next_public_observation_required=True,
        executor_id="adversarial-contract-executor",
    )

    with pytest.raises(ValueError, match=error):
        controller.accept_execution_report(report)

    # A rejected report must not advance or terminate the episode state.
    assert controller.state is ControllerState.WAITING_FOR_EXECUTION_REPORT


def test_candidate_input_order_does_not_change_planning_ranking() -> None:
    """Ranking is keyed by candidate identity, never by tuple position."""

    spec = task()
    context = PolicyContext(
        "order-contract",
        0,
        spec.final_goal_prompt,
        observation("closed", "public://closed"),
    )
    evidence = ContentDependentEvidenceModel().infer(context, spec)
    belief = EvidentialBeliefFilter().update(
        task=spec,
        episode_id=context.episode_id,
        step_index=context.step_index,
        evidence=evidence,
    )
    needs = BayesRiskNeedExtractor().extract(spec, belief)
    candidates = NeedDrivenPrimitiveProposer().propose(
        task_key=spec.key,
        observation=context.observation,
        needs=needs,
    )
    forecasts = AnalyticObservationCritic().forecast(
        context=context,
        task=spec,
        belief=belief,
        needs=needs,
        candidates=candidates,
    )
    forecast_by_id = {item.candidate_id: item for item in forecasts}
    rollouts = tuple(
        rollout_forecast(
            task=spec,
            belief=belief,
            candidate=candidate,
            forecast=forecast_by_id[candidate.candidate_id],
        )
        for candidate in candidates.candidates
        if not candidate.is_terminal
    )
    reference = rank_primitives(
        task=spec,
        belief=belief,
        candidates=candidates,
        rollouts=rollouts,
        weights=_weights(),
    )

    reversed_candidates = CandidateSet(
        task_key=candidates.task_key,
        observation_frame_id=candidates.observation_frame_id,
        candidates=tuple(reversed(candidates.candidates)),
    )
    permuted = rank_primitives(
        task=spec,
        belief=belief,
        candidates=reversed_candidates,
        rollouts=tuple(reversed(rollouts)),
        weights=_weights(),
    )

    assert permuted.selected.candidate_id == reference.selected.candidate_id
    assert permuted.runner_up_id == reference.runner_up_id
    assert permuted.margin == pytest.approx(reference.margin)
    assert [item.candidate_id for item in permuted.ranking] == [
        item.candidate_id for item in reference.ranking
    ]
    np.testing.assert_allclose(
        [item.objective for item in permuted.ranking],
        [item.objective for item in reference.ranking],
    )


@pytest.mark.parametrize(
    "invalid_likelihoods",
    (
        # The first post-state column sums to 0.9, not one.
        ((0.7, 0.2, 0.2), (0.2, 0.8, 0.8)),
        # The first post-state column sums to 1.1, not one.
        ((0.9, 0.2, 0.2), (0.2, 0.8, 0.8)),
    ),
)
def test_effect_forecast_rejects_non_normalized_observation_model_columns(
    invalid_likelihoods: tuple[tuple[float, ...], ...],
) -> None:
    """For every post-state, likelihoods over all outcomes must sum to one."""

    spec = task()
    outcomes = tuple(
        ObservationOutcomeModel(
            outcome_id=f"outcome_{index}",
            likelihood_by_post_state=likelihood,
            execution_status=ExecutionStatus.SUCCEEDED,
        )
        for index, likelihood in enumerate(invalid_likelihoods)
    )

    with pytest.raises(ValueError, match="exhaustive"):
        CandidateEffectForecast(
            schema_version="interaction-uncertainty.effect.v2",
            task_key=spec.key,
            candidate_id="open_container::track_fridge_0",
            hypothesis_labels=spec.hypotheses,
            transition_matrix=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            outcomes=outcomes,
            critic_uncertainty=0.0,
            model_stamp=ModelStamp("critic", "2" * 64, "identity"),
            rng_seed=0,
        )


def test_policy_trace_is_hash_chained_and_detects_payload_tampering() -> None:
    sink = InMemoryTraceSink()
    controller = _controller(trace_sink=sink)
    controller.observe_and_plan(observation("closed", "public://closed"))

    assert len(sink.events) >= 2
    verify_trace_chain(sink.events)

    index = len(sink.events) // 2
    tampered = list(sink.events)
    tampered[index] = replace(tampered[index], payload={"tampered": True})
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_trace_chain(tampered)

    timestamp_tampered = list(sink.events)
    timestamp_tampered[index] = replace(
        timestamp_tampered[index], timestamp_utc="1970-01-01T00:00:00+00:00"
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_trace_chain(timestamp_tampered)


def test_actual_reobservation_replaces_optimistic_predicted_posterior() -> None:
    """Counterfactual posteriors rank actions but never become observations."""

    controller = _controller()
    first = controller.observe_and_plan(observation("closed", "public://closed"))
    assert first.decision.selected.kind is PrimitiveKind.OPEN_CONTAINER
    open_rollout = next(
        rollout
        for rollout in first.rollouts
        if rollout.candidate_id == first.decision.selected.candidate_id
    )
    predicted = np.asarray(open_rollout.predictive_probabilities)
    assert predicted[0] > 0.8  # The critic optimistically predicts TARGET_VISIBLE.

    controller.accept_execution_report(
        ExecutionReport(
            schema_version="interaction-uncertainty.execution-report.v2",
            execution_id=first.execution_request.execution_id,
            candidate_id=first.decision.selected.candidate_id,
            primitive_kind=first.decision.selected.kind.value,
            result=ExecutionResult.SUCCEEDED,
            executed_steps=12,
            termination_reason="container_opened_but_target_still_not_visible",
            next_public_observation_required=True,
            executor_id="contract-executor",
        )
    )

    actual_observation = observation("closed-after", "public://still-closed")
    second = controller.observe_and_plan(actual_observation)
    actual = second.belief.probabilities

    # The public frame still says TARGET_HIDDEN.  The controller must use this
    # evidence even though its pre-action transition prediction was optimistic.
    assert actual[1] > 0.8
    assert not np.allclose(actual, predicted)
    expected_context = PolicyContext(
        controller.episode_id,
        1,
        task().final_goal_prompt,
        actual_observation,
    )
    assert second.belief.observation_digest == observation_digest(expected_context)
    assert controller.state is ControllerState.WAITING_FOR_EXECUTION_REPORT
