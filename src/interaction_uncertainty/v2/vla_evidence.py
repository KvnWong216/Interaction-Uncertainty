"""Evidence model that reads a frozen VLA's action samples as observations.

This is the seam that lets the controller and the baseline share one policy.
:mod:`.vla_bridge` turns action chunks into a belief; this module wraps that
into the :class:`~.evidence.EvidencePacket` contract the online filter expects,
so the same frozen checkpoint both *supplies* the belief and *executes* the
primitive chosen from it. Nothing else about the policy changes between the
baseline arm and the controller arm, which is what makes the comparison a
controlled one.

Anchor discipline
-----------------
An anchor carries a 3D position. Handing the controller the position of an
object it cannot see would let it plan against simulator state, and every
result downstream would be an artefact of that leak rather than a measurement
of the method. The benchmark marks such anchors ``role: task_target``, and this
module refuses them: hypotheses the policy could not have localised are
represented as *abstract* sites that receive no action evidence at all, exactly
like the structurally-zero ``NOT_FOUND`` primitive.

A hypothesis with no anchor is not a hypothesis with zero probability. It keeps
its prior and is driven only by the loss matrix and by evidence *against* the
grounded alternatives -- which is precisely how "I have searched everywhere and
it is not here" is supposed to become believable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..beliefs import EvidenceDeficit
from ..observation import PolicyContext
from .evidence import (
    EvidencePacket,
    ModelStamp,
    make_evidence_event_id,
    observation_digest,
)
from .task import TaskSpec
from .vla_bridge import (
    ActionChunkSample,
    AttributionConfig,
    HypothesisAnchor,
    attribute_samples,
)

__all__ = [
    "EVALUATOR_ONLY_ROLES",
    "HypothesisSite",
    "VlaActionEvidenceModel",
    "sites_from_spec",
]

EVALUATOR_ONLY_ROLES = frozenset({"task_target"})
"""Anchor roles the controller must never receive.

``task_target`` resolves to the pose of the object the task is about, which on
every scene worth measuring is the object that starts hidden.
"""

SCHEMA_VERSION = "interaction-uncertainty.evidence.v2"


@dataclass(frozen=True)
class HypothesisSite:
    """One hypothesis, optionally grounded at a position the policy can see."""

    label: str
    anchor: HypothesisAnchor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")
        if self.anchor is not None and not isinstance(self.anchor, HypothesisAnchor):
            raise TypeError("anchor must be a HypothesisAnchor or None")

    @property
    def is_grounded(self) -> bool:
        return self.anchor is not None


def sites_from_spec(
    hypotheses: Sequence[str],
    declared_anchors: Sequence[dict],
    *,
    resolve: Callable[[str], Sequence[float]],
) -> tuple[HypothesisSite, ...]:
    """Build controller-visible sites from a benchmark task entry.

    ``declared_anchors`` is the benchmark's ``hypothesis_anchors`` list.
    ``resolve`` maps an anchor ``ref`` to a position; it is only ever called for
    anchors the controller is allowed to see, so an evaluator-private resolver
    can be passed without leaking through this function.

    Any hypothesis without a matching permitted anchor becomes abstract rather
    than raising: that is the intended representation for "somewhere I have not
    looked" and for "not present at all".
    """

    permitted = {
        str(item["label"]): item
        for item in declared_anchors
        if str(item.get("role", "")) not in EVALUATOR_ONLY_ROLES
    }
    sites: list[HypothesisSite] = []
    for label in hypotheses:
        entry = permitted.get(label)
        if entry is None:
            sites.append(HypothesisSite(label=label))
            continue
        position = resolve(str(entry["ref"]))
        sites.append(
            HypothesisSite(
                label=label,
                anchor=HypothesisAnchor(label=label, position=tuple(position)),
            )
        )
    return tuple(sites)


@dataclass
class VlaActionEvidenceModel:
    """Turn action chunks sampled on the current frame into an evidence packet.

    ``sampler`` is called once per planning step and must return independent
    action chunks for the *current* observation. Independence is what makes the
    spread across samples a measurement rather than a repeat of one number: with
    openpi's server each ``infer`` call splits the PRNG, so repeated queries on
    a frozen observation already give independent draws.
    """

    sampler: Callable[[PolicyContext], Sequence[ActionChunkSample]]
    eef_position: Callable[[PolicyContext], Sequence[float]]
    sites: tuple[HypothesisSite, ...]
    model_stamp: ModelStamp
    config: AttributionConfig = field(default_factory=AttributionConfig)
    correlation_group: str = "vla_action_samples"
    last_saturated_fraction: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if len(self.sites) < 2:
            raise ValueError("at least two hypothesis sites are required")
        labels = [site.label for site in self.sites]
        if len(set(labels)) != len(labels):
            raise ValueError("hypothesis site labels must be unique")
        if sum(site.is_grounded for site in self.sites) < 2:
            raise ValueError(
                "at least two hypotheses must be grounded; attribution is a "
                "softmax over competing directions, so with fewer than two "
                "anchors the action samples cannot discriminate at all and the "
                "belief would be a function of the prior alone"
            )

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(site.label for site in self.sites)

    def infer(self, context: PolicyContext, task: TaskSpec) -> EvidencePacket:
        if task.hypotheses != self.labels:
            raise ValueError(
                "evidence ontology does not match TaskSpec: "
                f"{self.labels} vs {task.hypotheses}"
            )
        samples = tuple(self.sampler(context))
        if not samples:
            raise RuntimeError("sampler returned no action chunks")

        grounded = [site for site in self.sites if site.is_grounded]
        attributions = attribute_samples(
            samples,
            eef_position=self.eef_position(context),
            anchors=[site.anchor for site in grounded],
            config=self.config,
        )

        # Scatter the grounded evidence back into full hypothesis order; the
        # abstract sites keep the zero they were initialised with.
        evidence = np.zeros(len(self.sites), dtype=np.float64)
        index_of = {site.label: index for index, site in enumerate(self.sites)}
        columns = [index_of[site.label] for site in grounded]
        agreement = 0.0
        committed = 0.0
        saturated = 0
        for attribution in attributions:
            # decisiveness is min(1, travel / motion_scale), so it pins at 1.0
            # exactly when the chunk outruns the scale. If every sample pins,
            # total evidence collapses to the sample count and vacuity stops
            # measuring the observation -- the reading is then an artefact of
            # motion_scale, and the caller has to be able to see that.
            if attribution.decisiveness >= 1.0:
                saturated += 1
            if attribution.decisiveness <= 0.0:
                continue
            weights = np.asarray(attribution.weights, dtype=np.float64)
            evidence[columns] += attribution.decisiveness * weights
            agreement += attribution.decisiveness * float(weights.max())
            committed += attribution.decisiveness

        # EvidencePacket has no field for this, and inventing one would change a
        # schema the filter authenticates by digest. Exposing it here keeps the
        # diagnostic reachable by the runner's trace without touching the wire
        # contract.
        self.last_saturated_fraction = saturated / len(attributions)

        deficits: tuple[EvidenceDeficit, ...] = ()
        return EvidencePacket(
            schema_version=SCHEMA_VERSION,
            event_id=make_evidence_event_id(
                task=task, context=context, model_stamp=self.model_stamp
            ),
            task_key=task.key,
            observation_digest=observation_digest(context),
            hypothesis_labels=self.labels,
            hypothesis_evidence=tuple(evidence.tolist()),
            sufficiency_evidence=(agreement, max(0.0, committed - agreement)),
            deficits=deficits,
            model_stamp=self.model_stamp,
            correlation_group=self.correlation_group,
        )
