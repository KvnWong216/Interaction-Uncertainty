"""Read an evidential belief out of a monolithic VLA's action samples.

A feed-forward vision-language-action policy exposes no belief.  It emits a
continuous action chunk and nothing else.  That makes it impossible to ask the
question this project cares about -- *does the policy know that it does not
know?* -- using the policy's own outputs.

This module supplies the missing measurement device.  Repeatedly querying a
flow-matching policy on a **single frozen observation** returns a different
action chunk each time, because the sampler draws fresh noise per call.  The
spread of those chunks over task-relevant anchors is an observable proxy for
what the policy has committed to.  Converting that spread into subjective-logic
evidence makes every quantity in :mod:`interaction_uncertainty.uncertainty` --
vacuity, dissonance, predictive entropy, Dirichlet mutual information --
computable for a baseline that was never designed to report any of them.

Three properties of the construction matter when interpreting results.

*The evidence total is not the sample count.*  Each sample contributes evidence
in proportion to how decisively it moves.  A policy that dithers accumulates
little evidence and therefore reports high vacuity.  A policy that drives
confidently at one anchor accumulates near-unit evidence per sample and reports
low vacuity.  Were evidence simply the sample count, ``S = N + K`` would be
constant and vacuity would carry no information about the observation at all.

*Anchors are evaluator-private.*  Anchor world positions come from simulator
state.  They are used to *read* the policy, never to inform it.  The policy
observation is untouched by anything in this module.  This is the same boundary
the experiment contract draws around segmentation masks and object poses.

*The resulting belief is a diagnostic, not the policy's own confidence.*  It
describes where the policy's action distribution points.  A low-vacuity reading
means the policy is committed -- not that it is correct, and not that it has
represented evidence internally.  The central hypothesis of this project is
precisely that a monolithic VLA reports low vacuity while holding no evidence:
confident commitment under an empty belief.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np

from ..beliefs import BetaBelief, DirichletBelief, EvidenceDeficit, TaskBelief

__all__ = [
    "ActionChunkSample",
    "AttributionConfig",
    "HypothesisAnchor",
    "SampleAttribution",
    "action_induced_belief",
    "attribute_samples",
]


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be a real number, not boolean")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real numeric scalar") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return result


def _vector3(value: Sequence[float], name: str) -> tuple[float, float, float]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(
        value, Sequence | np.ndarray
    ):
        raise TypeError(f"{name} must be a length-3 sequence")
    items = tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))
    if len(items) != 3:
        raise ValueError(f"{name} must have exactly three components")
    return items  # type: ignore[return-value]


@dataclass(frozen=True)
class HypothesisAnchor:
    """A task-relevant location the policy could be driving toward.

    ``position`` is an evaluator-private world coordinate.  It never enters a
    policy observation; it exists so the evaluator can decide which hypothesis a
    sampled action chunk supports.
    """

    label: str
    position: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("anchor label must be a non-empty string")
        object.__setattr__(self, "position", _vector3(self.position, "position"))


@dataclass(frozen=True)
class ActionChunkSample:
    """One action chunk drawn from the policy on a frozen observation.

    ``translation_delta`` is the chunk's summed end-effector translation
    command, i.e. the net displacement the chunk asks for.  ``rotation_delta``
    is the summed axis-angle command and ``gripper_command`` the mean gripper
    channel; both are recorded for behavioural decoding rather than for the
    hypothesis belief.
    """

    translation_delta: tuple[float, float, float]
    rotation_delta: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gripper_command: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "translation_delta",
            _vector3(self.translation_delta, "translation_delta"),
        )
        object.__setattr__(
            self, "rotation_delta", _vector3(self.rotation_delta, "rotation_delta")
        )
        object.__setattr__(
            self, "gripper_command", _finite(self.gripper_command, "gripper_command")
        )

    @property
    def translation_norm(self) -> float:
        return float(np.linalg.norm(np.asarray(self.translation_delta, dtype=np.float64)))

    @property
    def rotation_norm(self) -> float:
        return float(np.linalg.norm(np.asarray(self.rotation_delta, dtype=np.float64)))

    @classmethod
    def from_chunk(cls, chunk: Sequence[Sequence[float]]) -> ActionChunkSample:
        """Summarize a ``(horizon, 7)`` OSC_POSE delta chunk.

        LIBERO's Panda controller consumes ``[dx, dy, dz, drx, dry, drz,
        gripper]`` per step, so summing the first six channels over the chunk
        gives the net motion the chunk requests, and averaging the seventh gives
        its grasp intent.
        """

        array = np.asarray(chunk, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] < 7:
            raise ValueError("action chunk must have shape [horizon, >=7]")
        if not np.all(np.isfinite(array)):
            raise ValueError("action chunk must be finite")
        return cls(
            translation_delta=tuple(array[:, 0:3].sum(axis=0).tolist()),  # type: ignore[arg-type]
            rotation_delta=tuple(array[:, 3:6].sum(axis=0).tolist()),  # type: ignore[arg-type]
            gripper_command=float(array[:, 6].mean()),
        )


@dataclass(frozen=True)
class AttributionConfig:
    """Thresholds converting motion geometry into evidence.

    ``motion_scale`` is the net chunk translation at which a sample counts as
    fully committed; below it a sample contributes proportionally less evidence
    and therefore raises vacuity.  ``temperature`` controls how sharply
    alignment differences separate anchors: a small value makes attribution
    nearly hard-assigned, a large value spreads a sample across anchors and
    raises dissonance.  ``alignment_floor`` discards motion pointing away from
    every anchor so that wandering does not manufacture evidence.

    **``motion_scale`` must be calibrated to the policy's action units, and the
    units are usually not metres.**  A controller such as LIBERO's ``OSC_POSE``
    consumes commands normalized to ``[-1, 1]`` per step, so a ten-step chunk
    sums to something of order one to ten rather than of order a centimetre.
    The default here is the measured median net translation of ``pi05_libero``
    on the Interactive-Perception scenes (samples spanned 1.1 to 9.3, median
    6.0).

    Getting this wrong is quiet rather than loud.  If every sample exceeds the
    scale, decisiveness saturates at one for all of them, ``S`` becomes ``K+N``
    exactly, and vacuity is a constant that no longer depends on the
    observation -- the quantity still plots, it just measures nothing.  Callers
    should watch the saturated fraction reported alongside each probe.
    """

    motion_scale: float = 6.0
    temperature: float = 0.25
    alignment_floor: float = 0.0

    def __post_init__(self) -> None:
        motion_scale = _finite(self.motion_scale, "motion_scale")
        temperature = _finite(self.temperature, "temperature")
        alignment_floor = _finite(self.alignment_floor, "alignment_floor")
        if motion_scale <= 0.0:
            raise ValueError("motion_scale must be > 0")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        if not -1.0 <= alignment_floor < 1.0:
            raise ValueError("alignment_floor must lie in [-1, 1)")
        object.__setattr__(self, "motion_scale", motion_scale)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "alignment_floor", alignment_floor)


@dataclass(frozen=True)
class SampleAttribution:
    """How one sample distributed its evidence across the hypotheses."""

    weights: tuple[float, ...]
    decisiveness: float
    best_label: str | None
    alignment_margin: float

    def to_dict(self) -> dict[str, object]:
        return {
            "weights": list(self.weights),
            "decisiveness": self.decisiveness,
            "best_label": self.best_label,
            "alignment_margin": self.alignment_margin,
        }


def _vacuous_attribution(count: int) -> SampleAttribution:
    return SampleAttribution(
        weights=tuple(0.0 for _ in range(count)),
        decisiveness=0.0,
        best_label=None,
        alignment_margin=0.0,
    )


def attribute_samples(
    samples: Sequence[ActionChunkSample],
    *,
    eef_position: Sequence[float],
    anchors: Sequence[HypothesisAnchor],
    config: AttributionConfig | None = None,
) -> tuple[SampleAttribution, ...]:
    """Attribute each sampled chunk across the hypothesis anchors.

    A sample's evidence contribution is its ``decisiveness`` -- how far the
    chunk asks the end effector to travel, saturating at ``motion_scale`` -- and
    its distribution across anchors is a softmax over directional alignment.  A
    stationary chunk contributes nothing, which is what makes total evidence,
    and therefore vacuity, sensitive to the observation.
    """

    config = config or AttributionConfig()
    if len(anchors) < 2:
        raise ValueError("at least two hypothesis anchors are required")
    if any(not isinstance(anchor, HypothesisAnchor) for anchor in anchors):
        raise TypeError("anchors must contain HypothesisAnchor objects")
    labels = [anchor.label for anchor in anchors]
    if len(set(labels)) != len(labels):
        raise ValueError("anchor labels must be unique")

    origin = np.asarray(_vector3(eef_position, "eef_position"), dtype=np.float64)
    targets = np.asarray([anchor.position for anchor in anchors], dtype=np.float64)
    directions = targets - origin[None, :]
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms <= 1e-9):
        raise ValueError("an anchor coincides with the end-effector position")
    directions = directions / norms[:, None]

    attributions: list[SampleAttribution] = []
    for sample in samples:
        delta = np.asarray(sample.translation_delta, dtype=np.float64)
        magnitude = float(np.linalg.norm(delta))
        if magnitude <= 1e-12:
            attributions.append(_vacuous_attribution(len(anchors)))
            continue

        alignment = directions @ (delta / magnitude)
        admissible = alignment >= config.alignment_floor
        if not bool(np.any(admissible)):
            # Decisive travel that supports no hypothesis. Recording zero
            # evidence keeps this legible as vacuity instead of laundering it
            # into a spurious hypothesis.
            attributions.append(_vacuous_attribution(len(anchors)))
            continue

        scores = np.where(admissible, alignment, -np.inf) / config.temperature
        scores = scores - float(np.max(scores[np.isfinite(scores)]))
        exponentiated = np.where(np.isfinite(scores), np.exp(scores), 0.0)
        weights = exponentiated / float(exponentiated.sum())

        ordered = np.sort(alignment[admissible])[::-1]
        margin = float(ordered[0] - ordered[1]) if ordered.size >= 2 else float(ordered[0])
        attributions.append(
            SampleAttribution(
                weights=tuple(weights.tolist()),
                decisiveness=float(min(1.0, magnitude / config.motion_scale)),
                best_label=labels[int(np.argmax(weights))],
                alignment_margin=margin,
            )
        )
    return tuple(attributions)


def action_induced_belief(
    *,
    prompt: str,
    samples: Sequence[ActionChunkSample],
    eef_position: Sequence[float],
    anchors: Sequence[HypothesisAnchor],
    config: AttributionConfig | None = None,
    deficits: Sequence[EvidenceDeficit] = (),
    provenance: str = "vla_action_samples",
    tags: Sequence[str] = (),
) -> TaskBelief:
    """Build a :class:`TaskBelief` from action chunks sampled on one frame.

    The returned belief feeds
    :func:`interaction_uncertainty.uncertainty.summarize_task_uncertainty`
    unchanged, so a baseline VLA and this project's own belief filter are scored
    by identical code.

    ``sufficiency`` is a *behavioural* proxy: it records how strongly the
    samples agree with one another once they have committed to moving.  It says
    nothing about whether the observation actually contains the information the
    task needs, and must not be read as the policy's own sufficiency estimate.
    """

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not samples:
        raise ValueError("at least one action sample is required")
    if any(not isinstance(item, ActionChunkSample) for item in samples):
        raise TypeError("samples must contain ActionChunkSample objects")

    attributions = attribute_samples(
        samples, eef_position=eef_position, anchors=anchors, config=config
    )
    labels = tuple(anchor.label for anchor in anchors)

    evidence = np.zeros(len(anchors), dtype=np.float64)
    agreement = 0.0
    committed = 0.0
    for attribution in attributions:
        if attribution.decisiveness <= 0.0:
            continue
        weights = np.asarray(attribution.weights, dtype=np.float64)
        evidence += attribution.decisiveness * weights
        agreement += attribution.decisiveness * float(weights.max())
        committed += attribution.decisiveness

    hypotheses = DirichletBelief.from_evidence(labels, tuple(evidence.tolist()))

    # BetaBelief defaults are W=2, a=0.5, so alpha=beta=1 is the vacuous prior.
    # An episode with no committed motion therefore returns Beta(1, 1) rather
    # than a fabricated sufficiency reading.
    sufficiency = BetaBelief(
        alpha=1.0 + agreement,
        beta=1.0 + max(0.0, committed - agreement),
    )

    return TaskBelief(
        prompt=prompt,
        hypotheses=hypotheses,
        sufficiency=sufficiency,
        deficits=tuple(deficits),
        provenance=provenance,
        tags=tuple(tags),
    )
