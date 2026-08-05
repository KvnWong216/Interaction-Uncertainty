"""Deployable evidence packets and model protocols.

An evidence model consumes public observation content plus the final-goal
prompt.  It returns calibrated pseudo-evidence and localized deficits, never an
action label or simulator-private identifier.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Protocol, runtime_checkable

from ..beliefs import BetaBelief, DeficitKind, EvidenceDeficit
from ..observation import PolicyContext
from .task import TaskKey, TaskSpec, canonical_digest
from .validation import (
    finite_number,
    require_exact_keys,
    strict_array,
    strict_string,
)


@dataclass(frozen=True)
class ModelStamp:
    model_id: str
    model_sha256: str
    calibration_id: str

    def __post_init__(self) -> None:
        if not self.model_id.strip() or not self.model_sha256.strip():
            raise ValueError("model_id and model_sha256 must be non-empty")
        if re.fullmatch(r"[0-9a-f]{64}", self.model_sha256) is None:
            raise ValueError("model_sha256 must be a lowercase SHA-256 digest")
        if not self.calibration_id.strip():
            raise ValueError("calibration_id must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {
            "model_id": self.model_id,
            "model_sha256": self.model_sha256,
            "calibration_id": self.calibration_id,
        }


def observation_digest(context: PolicyContext) -> str:
    """Hash public content/history while excluding arbitrary frame naming.

    Deployable adapters should use content-addressed image references.  The
    frame identifier is intentionally excluded so renaming an identical frame
    cannot manufacture an independent evidence event.
    """

    observation = context.observation
    anchors = []
    for anchor in observation.anchors:
        item = anchor.to_dict()
        item.pop("frame_id", None)
        anchors.append(item)
    return canonical_digest(
        {
            "image_refs": list(observation.image_refs),
            "proprioception": list(observation.proprioception),
            "anchors": anchors,
            "action_history": list(observation.action_history),
        }
    )


def evidence_content_digest(packet: EvidencePacket) -> str:
    """Authenticate model outputs separately from deterministic request identity."""

    return canonical_digest(
        {
            "schema_version": packet.schema_version,
            "event_id": packet.event_id,
            "task_key": packet.task_key.to_dict(),
            "observation_digest": packet.observation_digest,
            "hypothesis_labels": list(packet.hypothesis_labels),
            "hypothesis_evidence": list(packet.hypothesis_evidence),
            "sufficiency_evidence": list(packet.sufficiency_evidence),
            "deficits": [item.to_dict() for item in packet.deficits],
            "model_stamp": packet.model_stamp.to_dict(),
            "correlation_group": packet.correlation_group,
        }
    )


@dataclass(frozen=True)
class EvidencePacket:
    """One model event used by the online filter and counterfactual rollout."""

    schema_version: str
    event_id: str
    task_key: TaskKey
    observation_digest: str
    hypothesis_labels: tuple[str, ...]
    hypothesis_evidence: tuple[float, ...]
    sufficiency_evidence: tuple[float, float]
    deficits: tuple[EvidenceDeficit, ...]
    model_stamp: ModelStamp
    correlation_group: str
    content_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != "interaction-uncertainty.evidence.v2":
            raise ValueError("unsupported evidence schema_version")
        if re.fullmatch(r"[0-9a-f]{64}", self.event_id) is None:
            raise ValueError("event_id must be a lowercase SHA-256 digest")
        if re.fullmatch(r"[0-9a-f]{64}", self.observation_digest) is None:
            raise ValueError("observation_digest must be a lowercase SHA-256 digest")
        if not self.correlation_group.strip():
            raise ValueError("correlation_group must be non-empty")
        if len(self.hypothesis_labels) < 2:
            raise ValueError("at least two hypothesis labels are required")
        if len(self.hypothesis_labels) != len(self.hypothesis_evidence):
            raise ValueError("hypothesis evidence must align with labels")
        values = tuple(self.hypothesis_evidence) + tuple(self.sufficiency_evidence)
        if any(isinstance(value, bool) for value in values):
            raise TypeError("evidence must be numeric, not boolean")
        if any(not isfinite(float(value)) or float(value) < 0.0 for value in values):
            raise ValueError("evidence must be finite and non-negative")
        if not isfinite(sum(float(value) for value in self.hypothesis_evidence)):
            raise ValueError("hypothesis evidence strength must be finite")
        if not isfinite(sum(float(value) for value in self.sufficiency_evidence)):
            raise ValueError("sufficiency evidence strength must be finite")
        deficit_ids = [item.deficit_id for item in self.deficits]
        if len(deficit_ids) != len(set(deficit_ids)):
            raise ValueError("evidence deficit IDs must be unique")
        expected_content_digest = evidence_content_digest(self)
        if self.content_digest:
            if self.content_digest != expected_content_digest:
                raise ValueError("evidence content_digest does not authenticate the packet")
        else:
            object.__setattr__(self, "content_digest", expected_content_digest)

    def validate_request(self, context: PolicyContext, task: TaskSpec) -> None:
        if self.task_key != task.key:
            raise ValueError("evidence packet belongs to a different TaskKey")
        if self.hypothesis_labels != task.hypotheses:
            raise ValueError("evidence ontology does not match TaskSpec")
        if self.observation_digest != observation_digest(context):
            raise ValueError("evidence packet does not match public observation content")
        expected_event_id = make_evidence_event_id(
            task=task,
            context=context,
            model_stamp=self.model_stamp,
        )
        if self.event_id != expected_event_id:
            raise ValueError("evidence event_id is not content-addressed")
        visible = {anchor.token for anchor in context.observation.anchors}
        unknown = {item.anchor_token for item in self.deficits} - visible
        if unknown:
            raise ValueError(f"evidence deficits use unknown public anchors: {sorted(unknown)}")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "task_key": self.task_key.to_dict(),
            "observation_digest": self.observation_digest,
            "hypothesis_labels": list(self.hypothesis_labels),
            "hypothesis_evidence": list(self.hypothesis_evidence),
            "sufficiency_evidence": list(self.sufficiency_evidence),
            "deficits": [item.to_dict() for item in self.deficits],
            "model_stamp": self.model_stamp.to_dict(),
            "correlation_group": self.correlation_group,
            "content_digest": self.content_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> EvidencePacket:
        require_exact_keys(
            payload,
            required=frozenset(
                {
                    "schema_version",
                    "event_id",
                    "task_key",
                    "observation_digest",
                    "hypothesis_labels",
                    "hypothesis_evidence",
                    "sufficiency_evidence",
                    "deficits",
                    "model_stamp",
                    "correlation_group",
                    "content_digest",
                }
            ),
            location="EvidencePacket",
        )
        key = payload["task_key"]
        stamp = payload["model_stamp"]
        if not isinstance(key, Mapping) or not isinstance(stamp, Mapping):
            raise TypeError("task_key and model_stamp must be mappings")
        require_exact_keys(
            key,
            required=frozenset(
                {"task_id", "prompt_digest", "ontology_id", "ontology_version"}
            ),
            location="EvidencePacket.task_key",
        )
        require_exact_keys(
            stamp,
            required=frozenset({"model_id", "model_sha256", "calibration_id"}),
            location="EvidencePacket.model_stamp",
        )
        raw_sufficiency = strict_array(
            payload["sufficiency_evidence"],
            location="EvidencePacket.sufficiency_evidence",
        )
        sufficiency = tuple(
            finite_number(
                value,
                location=f"EvidencePacket.sufficiency_evidence[{index}]",
                minimum=0.0,
            )
            for index, value in enumerate(raw_sufficiency)
        )
        if len(sufficiency) != 2:
            raise ValueError("sufficiency_evidence must contain two values")
        raw_deficits = strict_array(
            payload["deficits"], location="EvidencePacket.deficits"
        )
        deficits: list[EvidenceDeficit] = []
        for index, item in enumerate(raw_deficits):
            if not isinstance(item, Mapping):
                raise TypeError(f"EvidencePacket.deficits[{index}] must be an object")
            require_exact_keys(
                item,
                required=frozenset(
                    {
                        "deficit_id",
                        "kind",
                        "anchor_token",
                        "probability",
                        "prompt_relevance",
                        "rationale",
                    }
                ),
                location=f"EvidencePacket.deficits[{index}]",
            )
            probability = item["probability"]
            if not isinstance(probability, Mapping):
                raise TypeError(
                    f"EvidencePacket.deficits[{index}].probability must be an object"
                )
            require_exact_keys(
                probability,
                required=frozenset({"alpha", "beta", "prior_weight", "base_rate"}),
                location=f"EvidencePacket.deficits[{index}].probability",
            )
            deficits.append(
                EvidenceDeficit(
                    deficit_id=strict_string(
                        item["deficit_id"],
                        location=f"EvidencePacket.deficits[{index}].deficit_id",
                    ),
                    kind=DeficitKind(
                        strict_string(
                            item["kind"],
                            location=f"EvidencePacket.deficits[{index}].kind",
                        )
                    ),
                    anchor_token=strict_string(
                        item["anchor_token"],
                        location=f"EvidencePacket.deficits[{index}].anchor_token",
                    ),
                    probability=BetaBelief(
                        alpha=finite_number(
                            probability["alpha"],
                            location=(
                                f"EvidencePacket.deficits[{index}].probability.alpha"
                            ),
                            minimum=0.0,
                            exclusive_minimum=True,
                        ),
                        beta=finite_number(
                            probability["beta"],
                            location=f"EvidencePacket.deficits[{index}].probability.beta",
                            minimum=0.0,
                            exclusive_minimum=True,
                        ),
                        prior_weight=finite_number(
                            probability["prior_weight"],
                            location=(
                                f"EvidencePacket.deficits[{index}].probability.prior_weight"
                            ),
                            minimum=0.0,
                            exclusive_minimum=True,
                        ),
                        base_rate=finite_number(
                            probability["base_rate"],
                            location=(
                                f"EvidencePacket.deficits[{index}].probability.base_rate"
                            ),
                            minimum=0.0,
                            maximum=1.0,
                        ),
                    ),
                    prompt_relevance=finite_number(
                        item["prompt_relevance"],
                        location=f"EvidencePacket.deficits[{index}].prompt_relevance",
                        minimum=0.0,
                        maximum=1.0,
                    ),
                    rationale=strict_string(
                        item["rationale"],
                        location=f"EvidencePacket.deficits[{index}].rationale",
                        allow_empty=True,
                    ),
                )
            )
        raw_labels = strict_array(
            payload["hypothesis_labels"], location="EvidencePacket.hypothesis_labels"
        )
        raw_hypothesis_evidence = strict_array(
            payload["hypothesis_evidence"],
            location="EvidencePacket.hypothesis_evidence",
        )
        return cls(
            schema_version=strict_string(
                payload["schema_version"], location="EvidencePacket.schema_version"
            ),
            event_id=strict_string(
                payload["event_id"], location="EvidencePacket.event_id"
            ),
            task_key=TaskKey(
                task_id=strict_string(
                    key["task_id"], location="EvidencePacket.task_key.task_id"
                ),
                prompt_digest=strict_string(
                    key["prompt_digest"],
                    location="EvidencePacket.task_key.prompt_digest",
                ),
                ontology_id=strict_string(
                    key["ontology_id"], location="EvidencePacket.task_key.ontology_id"
                ),
                ontology_version=strict_string(
                    key["ontology_version"],
                    location="EvidencePacket.task_key.ontology_version",
                ),
            ),
            observation_digest=strict_string(
                payload["observation_digest"],
                location="EvidencePacket.observation_digest",
            ),
            hypothesis_labels=tuple(
                strict_string(
                    value, location=f"EvidencePacket.hypothesis_labels[{index}]"
                )
                for index, value in enumerate(raw_labels)
            ),
            hypothesis_evidence=tuple(
                finite_number(
                    value,
                    location=f"EvidencePacket.hypothesis_evidence[{index}]",
                    minimum=0.0,
                )
                for index, value in enumerate(raw_hypothesis_evidence)
            ),
            sufficiency_evidence=(sufficiency[0], sufficiency[1]),
            deficits=tuple(deficits),
            model_stamp=ModelStamp(
                model_id=strict_string(
                    stamp["model_id"], location="EvidencePacket.model_stamp.model_id"
                ),
                model_sha256=strict_string(
                    stamp["model_sha256"],
                    location="EvidencePacket.model_stamp.model_sha256",
                ),
                calibration_id=strict_string(
                    stamp["calibration_id"],
                    location="EvidencePacket.model_stamp.calibration_id",
                ),
            ),
            correlation_group=strict_string(
                payload["correlation_group"],
                location="EvidencePacket.correlation_group",
            ),
            content_digest=strict_string(
                payload["content_digest"], location="EvidencePacket.content_digest"
            ),
        )


@runtime_checkable
class PromptEvidenceModel(Protocol):
    def infer(self, context: PolicyContext, task: TaskSpec) -> EvidencePacket: ...


def make_evidence_event_id(
    *, task: TaskSpec, context: PolicyContext, model_stamp: ModelStamp
) -> str:
    return canonical_digest(
        {
            "task_key": task.key.to_dict(),
            "observation_digest": observation_digest(context),
            "model_stamp": model_stamp.to_dict(),
        }
    )


def softplus_evidence(values: Sequence[float], temperature: float = 1.0) -> tuple[float, ...]:
    """Stable softplus link for an evidential neural head's raw logits."""

    import numpy as np

    if not isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    logits = np.asarray(values, dtype=np.float64) / temperature
    if logits.ndim != 1 or not np.all(np.isfinite(logits)):
        raise ValueError("raw evidence logits must be a finite vector")
    linked = np.logaddexp(0.0, logits)
    return tuple(float(value) for value in linked)
