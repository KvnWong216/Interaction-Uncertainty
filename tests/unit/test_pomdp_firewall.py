from __future__ import annotations

import numpy as np
import pytest

from interaction_uncertainty.beliefs import pomdp_belief_update
from interaction_uncertainty.firewall import PolicyFirewall, PrivilegeViolation
from interaction_uncertainty.observation import (
    PolicyObservation,
    VisualAnchor,
    ensure_final_goal_prompt,
)


def test_finite_state_pomdp_update_matches_hand_calculation() -> None:
    posterior = pomdp_belief_update(
        prior=[0.6, 0.4],
        transition=[[0.8, 0.2], [0.1, 0.9]],
        observation_likelihood=[0.2, 0.9],
    )

    # Predictive belief is [0.52, 0.48], then normalize [0.104, 0.432].
    np.testing.assert_allclose(posterior, [0.104 / 0.536, 0.432 / 0.536])
    assert posterior.sum() == pytest.approx(1.0)


def test_pomdp_update_rejects_impossible_observation() -> None:
    with pytest.raises(ValueError, match="zero probability"):
        pomdp_belief_update(
            prior=[0.5, 0.5],
            transition=[[1.0, 0.0], [0.0, 1.0]],
            observation_likelihood=[0.0, 0.0],
        )


def test_pomdp_update_rejects_nonfinite_likelihood() -> None:
    with pytest.raises(ValueError, match="likelihoods"):
        pomdp_belief_update(
            prior=[0.5, 0.5],
            transition=[[1.0, 0.0], [0.0, 1.0]],
            observation_likelihood=[float("nan"), 0.5],
        )


def test_pomdp_update_rejects_boolean_probability_coercion() -> None:
    with pytest.raises(TypeError, match="boolean"):
        pomdp_belief_update(
            prior=[True, False],
            transition=[[1.0, 0.0], [0.0, 1.0]],
            observation_likelihood=[0.5, 0.5],
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"nested": {"semantic_id": 7}},
        {"nested": {"semanticId": 7}},
        {"nested": {"SemanticID": 7}},
        {"nested": {"target-bbox": [0, 0, 1, 1]}},
        {"nested": {"simulator state": "private"}},
        {"nested": [{"target_mask": [0, 1]}]},
        {"state": {"object_qpos": [0.0] * 7}},
        {"message": "Use the mujoco_joint identifier."},
        {"oracle_action": "OPEN_CONTAINER"},
    ],
)
def test_firewall_rejects_recursive_privileged_payloads(payload: object) -> None:
    with pytest.raises(PrivilegeViolation):
        PolicyFirewall().validate_recursive(payload)


def test_firewall_accepts_policy_visible_anchor_and_observation() -> None:
    anchor = VisualAnchor(
        token="visual_fixture_0",
        frame_id="wrist_t000",
        source="deployable_detector",
        region_xyxy_normalized=(0.1, 0.2, 0.8, 0.9),
        affordances=("openable",),
    )
    observation = PolicyObservation(
        frame_id="wrist_t000",
        image_refs=("memory://wrist_t000",),
        anchors=(anchor,),
    )

    PolicyFirewall().validate_observation(observation)


def test_firewall_rejects_simulator_grounding_source() -> None:
    anchor = VisualAnchor(
        token="visual_fixture_0",
        frame_id="wrist_t000",
        source="simulator",
    )

    with pytest.raises(PrivilegeViolation, match="not a deployable/public source"):
        PolicyFirewall().validate_anchor(anchor)


def test_final_goal_prompt_guard_allows_goal_but_rejects_exploration_leakage() -> None:
    ensure_final_goal_prompt("Place the orange juice in the wicker basket.")

    with pytest.raises(ValueError, match="leaks"):
        ensure_final_goal_prompt(
            "Explore the refrigerator, then place the orange juice in the basket."
        )
