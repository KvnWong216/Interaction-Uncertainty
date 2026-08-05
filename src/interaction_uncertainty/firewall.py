"""Recursive privilege firewall for policy-facing payloads."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .observation import PolicyObservation, VisualAnchor


class PrivilegeViolation(ValueError):
    """Raised when simulator-private or oracle information reaches policy data."""


DEFAULT_FORBIDDEN_KEYS = frozenset(
    {
        "semantic_id",
        "instance_id",
        "instance_segmentation",
        "semantic_segmentation",
        "target_mask",
        "target_bbox",
        "target_bbox_xyxy",
        "target_instance",
        "target_pose",
        "target_location",
        "object_pose",
        "object_state",
        "object_list",
        "container_membership",
        "container_contents",
        "object_qpos",
        "qpos",
        "qvel",
        "fixture_joint_state",
        "simulator_state",
        "simulator_id",
        "private_state",
        "success_predicate",
        "collision_truth",
        "oracle_action",
        "oracle_endpoint",
        "oracle_utility",
        "ground_truth_hypothesis",
        "information_sufficiency_truth",
        "bddl_path",
        "asset_filename",
        "mujoco_joint",
        "geom_id",
        "body_id",
        "joint_id",
    }
)

DEFAULT_FORBIDDEN_PATTERNS = (
    re.compile(r"\b(?:mujoco|robosuite)[_:/.]", re.IGNORECASE),
    re.compile(r"\b(?:qpos|qvel|geom_id|body_id|joint_id)\b", re.IGNORECASE),
    re.compile(r"\boracle[_ -]", re.IGNORECASE),
    re.compile(r"\bground[_ -]?truth[_ -]", re.IGNORECASE),
    re.compile(
        r"\b(?:semanticId|instanceId|targetPose|targetLocation|objectPose|"
        r"objectState|containerMembership|successPredicate|privateState)\s*[:=]",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:semantic[_ -]?id|instance[_ -]?id|target[_ -]?pose|"
        r"target[_ -]?location|object[_ -]?pose|object[_ -]?state|"
        r"container[_ -]?membership|success[_ -]?predicate|private[_ -]?state)"
        r"\s*[:=]",
        re.IGNORECASE,
    ),
)

ALLOWED_ANCHOR_SOURCES = frozenset(
    {"policy_vla", "deployable_detector", "public_user_reference", "scripted_public_demo"}
)


def _canonical_key(value: object) -> str:
    """Normalize snake/kebab/camel/Pascal spellings before denylist matching."""

    text = str(value)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return text.strip("_").lower()


@dataclass(frozen=True)
class PolicyFirewall:
    forbidden_keys: frozenset[str] = DEFAULT_FORBIDDEN_KEYS
    allowed_anchor_sources: frozenset[str] = ALLOWED_ANCHOR_SOURCES

    def validate_recursive(self, payload: object, *, location: str = "payload") -> None:
        if isinstance(payload, Mapping):
            forbidden = {_canonical_key(key) for key in self.forbidden_keys}
            compact_forbidden = {key.replace("_", "") for key in forbidden}
            for key, value in payload.items():
                normalized_key = _canonical_key(key)
                if (
                    normalized_key in forbidden
                    or normalized_key.replace("_", "") in compact_forbidden
                ):
                    raise PrivilegeViolation(f"forbidden key at {location}: {key!r}")
                self.validate_recursive(value, location=f"{location}.{key}")
            return
        if isinstance(payload, Sequence) and not isinstance(payload, str | bytes | bytearray):
            for index, value in enumerate(payload):
                self.validate_recursive(value, location=f"{location}[{index}]")
            return
        if isinstance(payload, str):
            for pattern in DEFAULT_FORBIDDEN_PATTERNS:
                if pattern.search(payload):
                    raise PrivilegeViolation(
                        f"forbidden simulator/oracle string at {location}: {payload!r}"
                    )

    def validate_anchor(self, anchor: VisualAnchor) -> None:
        if anchor.source not in self.allowed_anchor_sources:
            raise PrivilegeViolation(
                f"anchor source {anchor.source!r} is not a deployable/public source"
            )
        self.validate_recursive(anchor.to_dict(), location=f"anchor[{anchor.token}]")

    def validate_observation(self, observation: PolicyObservation) -> None:
        self.validate_recursive(observation.to_dict(), location="observation")
        for anchor in observation.anchors:
            self.validate_anchor(anchor)


def assert_policy_safe(payload: object) -> None:
    PolicyFirewall().validate_recursive(payload)
