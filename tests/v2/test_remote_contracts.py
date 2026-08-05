from __future__ import annotations

from collections.abc import Mapping

import pytest

from interaction_uncertainty.beliefs import BetaBelief, DeficitKind, EvidenceDeficit
from interaction_uncertainty.firewall import PrivilegeViolation
from interaction_uncertainty.observation import PolicyContext
from interaction_uncertainty.v2.belief import EvidentialBeliefFilter
from interaction_uncertainty.v2.evidence import (
    EvidencePacket,
    ModelStamp,
    make_evidence_event_id,
    observation_digest,
)
from interaction_uncertainty.v2.needs import BayesRiskNeedExtractor
from interaction_uncertainty.v2.primitives import NeedDrivenPrimitiveProposer
from interaction_uncertainty.v2.remote import (
    RemoteActionOutcomeCritic,
    RemoteEvidenceModel,
    RemotePolicyBackend,
)
from tests.v2.test_p0_contracts import _controller
from tests.v2.test_v2_pipeline import observation, task


def context() -> PolicyContext:
    spec = task()
    return PolicyContext(
        "remote-v2",
        0,
        spec.final_goal_prompt,
        observation("frame", "public://closed"),
    )


def evidence_packet() -> EvidencePacket:
    spec = task()
    ctx = context()
    stamp = ModelStamp("remote-evidence", "3" * 64, "cal-v1")
    return EvidencePacket(
        schema_version="interaction-uncertainty.evidence.v2",
        event_id=make_evidence_event_id(task=spec, context=ctx, model_stamp=stamp),
        task_key=spec.key,
        observation_digest=observation_digest(ctx),
        hypothesis_labels=spec.hypotheses,
        hypothesis_evidence=(0.0, 10.0, 0.0),
        sufficiency_evidence=(0.0, 8.0),
        deficits=(
            EvidenceDeficit(
                deficit_id="closed",
                kind=DeficitKind.OCCLUDED_REGION,
                anchor_token="track_fridge_0",
                probability=BetaBelief(8.0, 1.0),
                prompt_relevance=1.0,
            ),
        ),
        model_stamp=stamp,
        correlation_group="wrist",
    )


def test_remote_evidence_requires_echoed_content_and_task_contract() -> None:
    packet = evidence_packet()
    captured: dict[str, object] = {}

    def transport(endpoint, payload, timeout_s):
        captured.update(endpoint=endpoint, payload=payload, timeout_s=timeout_s)
        return {
            "schema_version": "interaction-uncertainty.evidence-response.v2",
            "evidence": packet.to_dict(),
        }

    model = RemoteEvidenceModel("https://model.invalid/evidence", transport=transport)
    actual = model.infer(context(), task())
    assert actual == packet
    request = captured["payload"]
    assert isinstance(request, Mapping)
    assert request["observation_digest"] == packet.observation_digest


def test_remote_evidence_rejects_privileged_camel_case_response() -> None:
    def transport(endpoint, payload, timeout_s):
        del endpoint, payload, timeout_s
        return {
            "schema_version": "interaction-uncertainty.evidence-response.v2",
            "semanticId": 42,
            "evidence": evidence_packet().to_dict(),
        }

    model = RemoteEvidenceModel("https://model.invalid/evidence", transport=transport)
    with pytest.raises(PrivilegeViolation):
        model.infer(context(), task())


def test_remote_effect_model_requires_complete_pomdp_observation_model() -> None:
    spec = task()
    ctx = context()
    packet = evidence_packet()
    belief = EvidentialBeliefFilter().update(
        task=spec, episode_id="remote-v2", step_index=0, evidence=packet
    )
    needs = BayesRiskNeedExtractor().extract(spec, belief)
    candidates = NeedDrivenPrimitiveProposer().propose(
        task_key=spec.key,
        observation=ctx.observation,
        needs=needs,
    )
    interactive = next(item for item in candidates.candidates if not item.is_terminal)
    stamp = ModelStamp("remote-critic", "4" * 64, "cal-v1")

    response = {
        "schema_version": "interaction-uncertainty.effect-response.v2",
        "task_key": spec.key.to_dict(),
        "observation_digest": observation_digest(ctx),
        "forecasts": [
            {
                    "schema_version": "interaction-uncertainty.effect.v2",
                    "task_key": spec.key.to_dict(),
                    "candidate_id": interactive.candidate_id,
                    "hypothesis_labels": list(spec.hypotheses),
                    "transition_matrix": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    "outcomes": [
                        {
                            "outcome_id": "seen",
                            "likelihood_by_post_state": [0.8, 0.2, 0.2],
                            "execution_status": "SUCCEEDED",
                            "sufficiency_evidence": [8.0, 0.0],
                            "resolves_need_ids": [needs[0].need_id],
                            "action_cost": 0.05,
                            "physical_risk": 0.01,
                            "disturbance": 0.0,
                        },
                        {
                            "outcome_id": "not_seen",
                            "likelihood_by_post_state": [0.2, 0.8, 0.8],
                            "execution_status": "SUCCEEDED",
                            "sufficiency_evidence": [0.0, 4.0],
                            "resolves_need_ids": [],
                            "action_cost": 0.05,
                            "physical_risk": 0.01,
                            "disturbance": 0.0,
                        },
                    ],
                    "critic_uncertainty": 0.1,
                    "model_stamp": stamp.to_dict(),
                    "rng_seed": 7,
            }
        ],
    }

    def transport(endpoint, payload, timeout_s):
        del endpoint, payload, timeout_s
        return response

    critic = RemoteActionOutcomeCritic(
        "https://model.invalid/effects", transport=transport
    )
    forecasts = critic.forecast(
        context=ctx,
        task=spec,
        belief=belief,
        needs=needs,
        candidates=candidates,
    )
    assert forecasts[0].candidate_id == interactive.candidate_id
    assert forecasts[0].model_stamp == stamp

    response["forecasts"][0]["rng_seed"] = 7.0
    with pytest.raises(TypeError, match="rng_seed must be an integer"):
        critic.forecast(
            context=ctx,
            task=spec,
            belief=belief,
            needs=needs,
            candidates=candidates,
        )

    response["forecasts"][0]["rng_seed"] = 7
    response["observation_digest"] = 123
    with pytest.raises(TypeError, match="observation_digest must be a string"):
        critic.forecast(
            context=ctx,
            task=spec,
            belief=belief,
            needs=needs,
            candidates=candidates,
        )


def test_remote_policy_backend_echoes_selected_primitive_and_pinned_embodiment() -> None:
    plan = _controller().observe_and_plan(observation("closed", "public://closed"))
    request = plan.execution_request
    backend_hash = "6" * 64
    response = {
        "schema_version": "interaction-uncertainty.policy-response.v2",
        "execution_id": request.execution_id,
        "candidate_id": request.selected_primitive.candidate_id,
        "primitive_kind": request.selected_primitive.kind.value,
        "action_chunk": {
            "actions": [[0.0] * 7, [0.1] * 7],
            "action_space": "libero_delta_ee_gripper_v1",
            "normalization_stats_id": "libero-10-no-noops-v1",
            "backend_id": "openvla-oft-test",
            "backend_sha256": backend_hash,
            "rng_seed": 11,
        },
    }
    captured: dict[str, object] = {}

    def transport(endpoint, payload, timeout_s):
        captured.update(endpoint=endpoint, payload=payload, timeout_s=timeout_s)
        return response

    backend = RemotePolicyBackend(
        endpoint="https://model.invalid/policy",
        expected_action_space="libero_delta_ee_gripper_v1",
        expected_action_dimension=7,
        expected_normalization_stats_id="libero-10-no-noops-v1",
        expected_backend_id="openvla-oft-test",
        expected_backend_sha256=backend_hash,
        transport=transport,
    )
    chunk = backend.generate(request)
    assert chunk.horizon == 2
    assert chunk.action_dimension == 7
    sent = captured["payload"]
    assert isinstance(sent, Mapping)
    assert sent["schema_version"] == "interaction-uncertainty.policy-request.v2"

    response["candidate_id"] = "silently-replanned-candidate"
    with pytest.raises(ValueError, match="changed the selected candidate"):
        backend.generate(request)

    response["candidate_id"] = request.selected_primitive.candidate_id
    response["action_chunk"]["actions"][0][0] = True  # type: ignore[index]
    with pytest.raises(TypeError, match="not bool"):
        backend.generate(request)
