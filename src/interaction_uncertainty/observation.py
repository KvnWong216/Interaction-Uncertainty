"""Strongly typed, deployable policy observations.

Simulator-private data belongs in benchmark generators and evaluators, never in
these objects.  Visual anchors must be generated from policy-visible sensing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np


def _strict_string(value: object, location: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{location} must be a string")
    if not allow_empty and not value:
        raise ValueError(f"{location} must be non-empty")
    return value


def _strict_array(value: object, location: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise TypeError(f"{location} must be an array")
    return value


def _strict_number(value: object, location: str) -> float:
    if isinstance(value, bool | np.bool_) or isinstance(
        value, str | bytes | bytearray
    ):
        raise TypeError(f"{location} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{location} must be a numeric scalar") from exc
    if not isfinite(result):
        raise ValueError(f"{location} must be finite")
    return result


@dataclass(frozen=True)
class VisualAnchor:
    """A temporary visual token produced from the current public observation."""

    token: str
    frame_id: str
    source: str
    region_xyxy_normalized: tuple[float, float, float, float] | None = None
    affordances: tuple[str, ...] = ()
    attributes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.token, str)
            or not isinstance(self.frame_id, str)
            or not isinstance(self.source, str)
            or not self.token
            or not self.frame_id
            or not self.source
        ):
            raise ValueError("token, frame_id, and source must be non-empty")
        if any(not isinstance(value, str) or not value for value in self.affordances):
            raise TypeError("anchor affordances must be non-empty strings")
        if any(not isinstance(value, str) or not value for value in self.attributes):
            raise TypeError("anchor attributes must be non-empty strings")
        if self.region_xyxy_normalized is not None:
            if any(
                isinstance(value, bool | np.bool_)
                for value in self.region_xyxy_normalized
            ):
                raise TypeError("normalized region coordinates must be numeric")
            x1, y1, x2, y2 = (float(v) for v in self.region_xyxy_normalized)
            if not all(isfinite(v) and 0.0 <= v <= 1.0 for v in (x1, y1, x2, y2)):
                raise ValueError("normalized region coordinates must lie in [0, 1]")
            if not x1 < x2 or not y1 < y2:
                raise ValueError("region must have positive area")
            object.__setattr__(self, "region_xyxy_normalized", (x1, y1, x2, y2))
        object.__setattr__(self, "affordances", tuple(sorted(set(self.affordances))))
        object.__setattr__(self, "attributes", tuple(sorted(set(self.attributes))))

    def to_dict(self) -> dict[str, object]:
        return {
            "token": self.token,
            "frame_id": self.frame_id,
            "source": self.source,
            "region_xyxy_normalized": (
                None
                if self.region_xyxy_normalized is None
                else list(self.region_xyxy_normalized)
            ),
            "affordances": list(self.affordances),
            "attributes": list(self.attributes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> VisualAnchor:
        expected = {
            "token",
            "frame_id",
            "source",
            "region_xyxy_normalized",
            "affordances",
            "attributes",
        }
        if set(payload) != expected:
            raise ValueError(
                "VisualAnchor fields do not match the strict public-observation contract"
            )
        region = payload.get("region_xyxy_normalized")
        raw_affordances = _strict_array(
            payload["affordances"], "VisualAnchor.affordances"
        )
        raw_attributes = _strict_array(
            payload["attributes"], "VisualAnchor.attributes"
        )
        raw_region = (
            None
            if region is None
            else _strict_array(region, "VisualAnchor.region_xyxy_normalized")
        )
        return cls(
            token=_strict_string(payload["token"], "VisualAnchor.token"),
            frame_id=_strict_string(payload["frame_id"], "VisualAnchor.frame_id"),
            source=_strict_string(payload["source"], "VisualAnchor.source"),
            region_xyxy_normalized=(
                None
                if raw_region is None
                else tuple(
                    _strict_number(value, f"VisualAnchor.region_xyxy_normalized[{index}]")
                    for index, value in enumerate(raw_region)
                )
            ),
            affordances=tuple(
                _strict_string(value, f"VisualAnchor.affordances[{index}]")
                for index, value in enumerate(raw_affordances)
            ),
            attributes=tuple(
                _strict_string(value, f"VisualAnchor.attributes[{index}]")
                for index, value in enumerate(raw_attributes)
            ),
        )


@dataclass(frozen=True)
class PolicyObservation:
    """Public observation packet shared by belief, proposal, and VLA modules."""

    frame_id: str
    image_refs: tuple[str, ...]
    proprioception: tuple[float, ...] = ()
    anchors: tuple[VisualAnchor, ...] = ()
    action_history: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str) or not self.frame_id or not self.image_refs:
            raise ValueError("frame_id and at least one image reference are required")
        if any(not isinstance(ref, str) or not ref for ref in self.image_refs):
            raise ValueError("image references must be non-empty")
        if any(not isinstance(item, str) for item in self.action_history):
            raise TypeError("action_history entries must be strings")
        if any(anchor.frame_id != self.frame_id for anchor in self.anchors):
            raise ValueError("all visual anchors must belong to the current frame")
        tokens = [anchor.token for anchor in self.anchors]
        if len(tokens) != len(set(tokens)):
            raise ValueError("visual anchor tokens must be unique within a frame")
        if any(isinstance(value, bool | np.bool_) for value in self.proprioception):
            raise TypeError("proprioception must be numeric, not boolean")
        proprio = tuple(float(v) for v in self.proprioception)
        if any(not isfinite(v) for v in proprio):
            raise ValueError("proprioception must be finite")
        object.__setattr__(self, "proprioception", proprio)

    def anchor(self, token: str) -> VisualAnchor:
        for anchor in self.anchors:
            if anchor.token == token:
                return anchor
        raise KeyError(token)

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "image_refs": list(self.image_refs),
            "proprioception": list(self.proprioception),
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "action_history": list(self.action_history),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PolicyObservation:
        expected = {
            "frame_id",
            "image_refs",
            "proprioception",
            "anchors",
            "action_history",
        }
        if set(payload) != expected:
            raise ValueError(
                "PolicyObservation fields do not match the strict public-observation contract"
            )
        raw_refs = _strict_array(payload["image_refs"], "PolicyObservation.image_refs")
        raw_proprio = _strict_array(
            payload["proprioception"], "PolicyObservation.proprioception"
        )
        raw_anchors = _strict_array(payload["anchors"], "PolicyObservation.anchors")
        raw_history = _strict_array(
            payload["action_history"], "PolicyObservation.action_history"
        )
        if any(not isinstance(item, Mapping) for item in raw_anchors):
            raise TypeError("PolicyObservation.anchors entries must be objects")
        return cls(
            frame_id=_strict_string(payload["frame_id"], "PolicyObservation.frame_id"),
            image_refs=tuple(
                _strict_string(value, f"PolicyObservation.image_refs[{index}]")
                for index, value in enumerate(raw_refs)
            ),
            proprioception=tuple(
                _strict_number(value, f"PolicyObservation.proprioception[{index}]")
                for index, value in enumerate(raw_proprio)
            ),
            anchors=tuple(
                VisualAnchor.from_dict(item)
                for item in raw_anchors  # type: ignore[arg-type]
            ),
            action_history=tuple(
                _strict_string(
                    value,
                    f"PolicyObservation.action_history[{index}]",
                    allow_empty=True,
                )
                for index, value in enumerate(raw_history)
            ),
        )


@dataclass(frozen=True)
class PolicyContext:
    episode_id: str
    step_index: int
    prompt: str
    observation: PolicyObservation

    def __post_init__(self) -> None:
        if not self.episode_id or not self.prompt.strip():
            raise ValueError("episode_id and final-goal prompt must be non-empty")
        if (
            isinstance(self.step_index, bool)
            or not isinstance(self.step_index, int)
            or self.step_index < 0
        ):
            raise ValueError("step_index must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "prompt": self.prompt,
            "observation": self.observation.to_dict(),
        }


def ensure_final_goal_prompt(prompt: str, forbidden_directives: Sequence[str] = ()) -> None:
    """Reject explicit exploration-control leakage in benchmark prompts.

    This is a conservative lexical guard for fixtures, not a semantic proof.
    """

    if not isinstance(prompt, str) or not prompt.strip():
        raise TypeError("final goal prompt must be a non-empty string")
    if any(not isinstance(item, str) for item in forbidden_directives):
        raise TypeError("forbidden_directives must contain strings")
    normalized = " ".join(prompt.lower().split())
    default_forbidden = (
        "look for",
        "explore",
        "inspect first",
        "open the fridge first",
        "rotate it",
        "remove the occluder",
    )
    for phrase in tuple(default_forbidden) + tuple(forbidden_directives):
        if phrase.lower() in normalized:
            raise ValueError(f"prompt leaks an exploration directive: {phrase!r}")
