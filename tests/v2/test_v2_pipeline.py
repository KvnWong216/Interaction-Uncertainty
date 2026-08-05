from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from interaction_uncertainty.beliefs import BetaBelief, DeficitKind, EvidenceDeficit
from interaction_uncertainty.observation import PolicyContext, PolicyObservation, VisualAnchor
from interaction_uncertainty.v2.belief import EvidentialBeliefFilter, FilterMode
from interaction_uncertainty.v2.controller import ControllerState, EpisodeController
from interaction_uncertainty.v2.effects import (
    CandidateEffectForecast,
    ExecutionStatus,
    ObservationOutcomeModel,
    rollout_forecast,
)
from interaction_uncertainty.v2.evidence import (
    EvidencePacket,
    ModelStamp,
    make_evidence_event_id,
    observation_digest,
)
from interaction_uncertainty.v2.execution import ExecutionReport, ExecutionResult
from interaction_uncertainty.v2.libero import (
    LiberoPublicObservationAdapter,
    decode_embedded_public_image,
)
from interaction_uncertainty.v2.needs import BayesRiskNeedExtractor
from interaction_uncertainty.v2.planning import PlanningWeights, rank_primitives
from interaction_uncertainty.v2.primitives import (
    NeedDrivenPrimitiveProposer,
    PrimitiveKind,
)
from interaction_uncertainty.v2.task import TaskSpec, TerminalDecision


def task() -> TaskSpec:
    return TaskSpec.create(
        task_id="orange-juice",
        final_goal_prompt="Place the orange juice in the wicker basket.",
        ontology_id="target-access-v1",
        ontology_version="1.0.0",
        hypotheses=("TARGET_VISIBLE", "TARGET_HIDDEN", "TARGET_ABSENT"),
        terminal_decisions=(TerminalDecision.DIRECT_ACT, TerminalDecision.NOT_FOUND),
        loss_matrix=((0.0, 1.0, 1.0), (1.0, 1.0, 0.0)),
    )


def fridge_anchor(frame_id: str) -> VisualAnchor:
    return VisualAnchor(
        token="track_fridge_0",
        frame_id=frame_id,
        source="deployable_detector",
        region_xyxy_normalized=(0.3, 0.1, 0.9, 0.95),
        affordances=("openable",),
        attributes=("opaque_container",),
    )


def observation(frame_id: str, image_ref: str) -> PolicyObservation:
    anchors = [fridge_anchor(frame_id)]
    if "open-visible" in image_ref:
        anchors.append(
            VisualAnchor(
                token="track_target_0",
                frame_id=frame_id,
                source="deployable_detector",
                region_xyxy_normalized=(0.42, 0.35, 0.58, 0.72),
                affordances=("graspable", "task_target"),
                attributes=("prompt_matched_target",),
            )
        )
    return PolicyObservation(
        frame_id=frame_id,
        image_refs=(image_ref,),
        anchors=tuple(anchors),
    )


@dataclass
class ContentDependentEvidenceModel:
    """Contract fake: depends on public image content token, never frame ID."""

    stamp: ModelStamp = ModelStamp("tiny-public-model", "0" * 64, "identity")

    def infer(self, context: PolicyContext, task_spec: TaskSpec) -> EvidencePacket:
        visible = "open-visible" in context.observation.image_refs[0]
        hypothesis_evidence = (30.0, 0.0, 0.0) if visible else (0.0, 20.0, 0.0)
        deficits = (
            ()
            if visible
            else (
                EvidenceDeficit(
                    deficit_id="closed_container_contents",
                    kind=DeficitKind.OCCLUDED_REGION,
                    anchor_token="track_fridge_0",
                    probability=BetaBelief(15.0, 1.0),
                    prompt_relevance=1.0,
                ),
            )
        )
        return EvidencePacket(
            schema_version="interaction-uncertainty.evidence.v2",
            event_id=make_evidence_event_id(
                task=task_spec, context=context, model_stamp=self.stamp
            ),
            task_key=task_spec.key,
            observation_digest=observation_digest(context),
            hypothesis_labels=task_spec.hypotheses,
            hypothesis_evidence=hypothesis_evidence,
            sufficiency_evidence=(20.0, 0.0) if visible else (0.0, 15.0),
            deficits=deficits,
            model_stamp=self.stamp,
            correlation_group="wrist-rgb",
        )


@dataclass
class AnalyticObservationCritic:
    stamp: ModelStamp = ModelStamp("analytic-test-critic", "1" * 64, "identity")

    def forecast(self, *, context, task, belief, needs, candidates):
        del context, belief
        need_ids = tuple(item.need_id for item in needs)
        forecasts = []
        for candidate in candidates.candidates:
            if not candidate.requires_effect_forecast:
                continue
            if candidate.kind is PrimitiveKind.DIRECT_ACT:
                forecasts.append(
                    CandidateEffectForecast(
                        schema_version="interaction-uncertainty.effect.v2",
                        task_key=task.key,
                        candidate_id=candidate.candidate_id,
                        hypothesis_labels=task.hypotheses,
                        transition_matrix=(
                            (1.0, 0.0, 0.0),
                            (0.0, 1.0, 0.0),
                            (0.0, 0.0, 1.0),
                        ),
                        outcomes=(
                            ObservationOutcomeModel(
                                outcome_id="direct_execution",
                                likelihood_by_post_state=(1.0, 1.0, 1.0),
                                execution_status=ExecutionStatus.SUCCEEDED,
                                sufficiency_evidence=(20.0, 0.0),
                                action_cost=0.05,
                            ),
                        ),
                        critic_uncertainty=0.01,
                        model_stamp=self.stamp,
                        rng_seed=0,
                    )
                )
                continue
            assert candidate.kind is PrimitiveKind.OPEN_CONTAINER
            forecasts.append(
                CandidateEffectForecast(
                    schema_version="interaction-uncertainty.effect.v2",
                    task_key=task.key,
                    candidate_id=candidate.candidate_id,
                    hypothesis_labels=task.hypotheses,
                    # Hidden target becomes visible with probability 0.9; absent stays absent.
                    transition_matrix=(
                        (1.0, 0.0, 0.0),
                        (0.9, 0.1, 0.0),
                        (0.0, 0.0, 1.0),
                    ),
                    outcomes=(
                        ObservationOutcomeModel(
                            "target_like_evidence",
                            (0.855, 0.095, 0.095),
                            ExecutionStatus.SUCCEEDED,
                            resolves_need_ids=need_ids,
                            action_cost=0.05,
                            physical_risk=0.01,
                        ),
                        ObservationOutcomeModel(
                            "no_target_like_evidence",
                            (0.095, 0.855, 0.855),
                            ExecutionStatus.SUCCEEDED,
                            action_cost=0.05,
                            physical_risk=0.01,
                        ),
                        ObservationOutcomeModel(
                            "execution_failed",
                            (0.05, 0.05, 0.05),
                            ExecutionStatus.FAILED,
                            action_cost=0.05,
                            physical_risk=0.05,
                        ),
                    ),
                    critic_uncertainty=0.02,
                    model_stamp=self.stamp,
                    rng_seed=0,
                )
            )
        return tuple(forecasts)


def test_task_spec_binds_prompt_and_typed_terminal_decisions() -> None:
    spec = task()
    assert spec.risk_of(TerminalDecision.DIRECT_ACT, (1.0, 0.0, 0.0)) == 0.0
    with pytest.raises(ValueError, match="prompt_digest"):
        TaskSpec(
            key=spec.key.__class__("x", "wrong", "o", "v"),
            final_goal_prompt=spec.final_goal_prompt,
            hypotheses=spec.hypotheses,
            terminal_decisions=spec.terminal_decisions,
            loss_matrix=spec.loss_matrix,
        )


def test_evidence_is_content_dependent_not_frame_lookup() -> None:
    spec = task()
    model = ContentDependentEvidenceModel()
    first = PolicyContext("ep", 0, spec.final_goal_prompt, observation("frame_a", "public://closed"))
    renamed = PolicyContext(
        "ep", 0, spec.final_goal_prompt, observation("frame_b", "public://closed")
    )
    changed = PolicyContext(
        "ep", 0, spec.final_goal_prompt, observation("frame_a", "public://open-visible")
    )
    assert model.infer(first, spec).hypothesis_evidence == model.infer(
        renamed, spec
    ).hypothesis_evidence
    assert observation_digest(first) == observation_digest(renamed)
    assert model.infer(first, spec).hypothesis_evidence != model.infer(
        changed, spec
    ).hypothesis_evidence
    assert observation_digest(first) != observation_digest(changed)


def test_filter_rejects_duplicate_evidence_and_discounts_correlated_frames() -> None:
    spec = task()
    model = ContentDependentEvidenceModel()
    context = PolicyContext(
        "ep", 0, spec.final_goal_prompt, observation("frame_a", "public://closed")
    )
    packet = model.infer(context, spec)
    filter_model = EvidentialBeliefFilter(
        mode=FilterMode.DISCOUNTED_EVIDENCE,
        retention=1.0,
        same_group_discount=0.25,
    )
    state = filter_model.update(
        task=spec, episode_id="ep", step_index=0, evidence=packet
    )
    with pytest.raises(ValueError, match="duplicate evidence"):
        filter_model.update(
            task=spec,
            episode_id="ep",
            step_index=1,
            evidence=packet,
            previous=state,
        )


def test_need_to_candidate_generation_is_localized_and_recall_oriented() -> None:
    spec = task()
    context = PolicyContext(
        "ep", 0, spec.final_goal_prompt, observation("frame_a", "public://closed")
    )
    packet = ContentDependentEvidenceModel().infer(context, spec)
    belief = EvidentialBeliefFilter().update(
        task=spec, episode_id="ep", step_index=0, evidence=packet
    )
    needs = BayesRiskNeedExtractor().extract(spec, belief)
    candidates = NeedDrivenPrimitiveProposer().propose(
        task_key=spec.key,
        observation=context.observation,
        needs=needs,
    )
    assert needs[0].deficit_kind is DeficitKind.OCCLUDED_REGION
    assert {item.kind for item in candidates.candidates} == {
        PrimitiveKind.STOP_NOT_FOUND,
        PrimitiveKind.OPEN_CONTAINER,
    }
    open_call = next(
        item for item in candidates.candidates if item.kind is PrimitiveKind.OPEN_CONTAINER
    )
    assert open_call.addresses_need_ids == (needs[0].need_id,)
    assert open_call.source_anchor == context.observation.anchors[0]


def test_observation_model_rollout_is_bayes_consistent_and_ranks_open() -> None:
    spec = task()
    context = PolicyContext(
        "ep", 0, spec.final_goal_prompt, observation("frame_a", "public://closed")
    )
    belief = EvidentialBeliefFilter().update(
        task=spec,
        episode_id="ep",
        step_index=0,
        evidence=ContentDependentEvidenceModel().infer(context, spec),
    )
    needs = BayesRiskNeedExtractor().extract(spec, belief)
    candidates = NeedDrivenPrimitiveProposer().propose(
        task_key=spec.key, observation=context.observation, needs=needs
    )
    forecast = AnalyticObservationCritic().forecast(
        context=context, task=spec, belief=belief, needs=needs, candidates=candidates
    )[0]
    open_call = candidates.by_id(forecast.candidate_id)
    rollout = rollout_forecast(
        task=spec, belief=belief, candidate=open_call, forecast=forecast
    )
    assert rollout.posterior_martingale_l1 == pytest.approx(0.0, abs=1e-12)
    assert rollout.conditional_information_value >= -1e-12
    decision = rank_primitives(
        task=spec,
        belief=belief,
        candidates=candidates,
        rollouts=(rollout,),
        weights=PlanningWeights(
            action_cost=0.01,
            physical_risk=0.01,
            disturbance=0.0,
            critic_uncertainty=0.01,
        ),
    )
    assert decision.selected.kind is PrimitiveKind.OPEN_CONTAINER


def test_controller_reinfers_from_actual_observation_after_execution() -> None:
    spec = task()
    controller = EpisodeController(
        task=spec,
        episode_id="episode-1",
        evidence_model=ContentDependentEvidenceModel(),
        belief_filter=EvidentialBeliefFilter(),
        outcome_critic=AnalyticObservationCritic(),
        planning_weights=PlanningWeights(
            action_cost=0.01,
            physical_risk=0.01,
            disturbance=0.0,
            critic_uncertainty=0.01,
        ),
    )
    first = controller.observe_and_plan(observation("closed", "public://closed"))
    assert first.decision.selected.kind is PrimitiveKind.OPEN_CONTAINER
    controller.accept_execution_report(
        ExecutionReport(
            schema_version="interaction-uncertainty.execution-report.v2",
            execution_id=first.execution_request.execution_id,
            candidate_id=first.decision.selected.candidate_id,
            primitive_kind=first.decision.selected.kind.value,
            result=ExecutionResult.SUCCEEDED,
            executed_steps=12,
            termination_reason="container_opened",
            next_public_observation_required=True,
            executor_id="test-executor",
        )
    )
    assert controller.state is ControllerState.WAITING_FOR_OBSERVATION
    second = controller.observe_and_plan(
        observation("open", "public://open-visible-orange-juice")
    )
    assert second.belief.posterior.hypotheses.probability("TARGET_VISIBLE") > 0.8
    assert second.decision.selected.kind is PrimitiveKind.DIRECT_ACT


def test_libero_adapter_hashes_pixels_and_drops_private_simulator_state() -> None:
    adapter = LiberoPublicObservationAdapter()
    image_a = np.zeros((8, 8, 3), dtype=np.uint8)
    image_b = image_a.copy()
    image_b[0, 0, 0] = 1
    private = {
        "robot0_eye_in_hand_image": image_a,
        "robot0_eef_pos": np.asarray([0.1, 0.2, 0.3]),
        "object-state": np.asarray([99.0]),
        "semantic_id": 42,
    }
    obs_a = adapter.adapt(
        raw_observation=private,
        frame_id="same",
        prompt=task().final_goal_prompt,
    )
    private["robot0_eye_in_hand_image"] = image_b
    obs_b = adapter.adapt(
        raw_observation=private,
        frame_id="same",
        prompt=task().final_goal_prompt,
    )
    assert obs_a.image_refs != obs_b.image_refs
    np.testing.assert_array_equal(decode_embedded_public_image(obs_a.image_refs[0]), image_a)
    np.testing.assert_array_equal(decode_embedded_public_image(obs_b.image_refs[0]), image_b)
    assert "semantic_id" not in str(obs_a.to_dict())
    assert "object-state" not in str(obs_a.to_dict())


def test_embedded_public_image_rejects_tampering() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    reference = LiberoPublicObservationAdapter().adapt(
        raw_observation={"robot0_eye_in_hand_image": image},
        frame_id="wrist-0",
        prompt=task().final_goal_prompt,
    ).image_refs[0]
    payload_start = reference.index(",") + 1
    tamper_at = payload_start + 24
    replacement = "A" if reference[tamper_at] != "A" else "B"
    tampered = reference[:tamper_at] + replacement + reference[tamper_at + 1 :]
    with pytest.raises(ValueError, match="content-digest|valid non-pickled"):
        decode_embedded_public_image(tampered)
