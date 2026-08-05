"""Belief representations and exact finite-state Bayes filtering.

Notation follows standard POMDP and subjective-logic conventions:

* ``b_t`` is a belief over latent task hypotheses.
* ``alpha`` is a Dirichlet concentration vector and ``S=sum(alpha)``.
* ``Beta(alpha, beta)`` is the two-hypothesis special case.

The classes represent probabilities and evidence only.  They do not claim
that a neural network is calibrated; calibration must be evaluated separately.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite, log

import numpy as np


def _strict_string(value: object, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value


def _strict_array(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise TypeError(f"{name} must be an array")
    return value


def _require_exact_fields(
    payload: Mapping[object, object], fields: frozenset[str], name: str
) -> None:
    if any(not isinstance(key, str) for key in payload):
        raise TypeError(f"{name} keys must be strings")
    if set(payload) != fields:
        raise ValueError(f"{name} fields do not match the strict contract")


def _contains_boolean(value: object) -> bool:
    if isinstance(value, bool | np.bool_):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.bool_):
            return True
        return value.dtype == object and any(
            _contains_boolean(item) for item in value.reshape(-1)
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return any(_contains_boolean(item) for item in value)
    return False


def _positive_finite(value: float, name: str) -> float:
    if isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be a real number, not boolean")
    if isinstance(value, str | bytes | bytearray):
        raise TypeError(f"{name} must be a real number, not {type(value).__name__}")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real numeric scalar") from exc
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0; got {value!r}")
    return value


def _nonnegative_finite(value: float, name: str) -> float:
    if isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be a real number, not boolean")
    if isinstance(value, str | bytes | bytearray):
        raise TypeError(f"{name} must be a real number, not {type(value).__name__}")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real numeric scalar") from exc
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0; got {value!r}")
    return value


def _probability(value: float, name: str) -> float:
    if isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be a real number, not boolean")
    if isinstance(value, str | bytes | bytearray):
        raise TypeError(f"{name} must be a real number, not {type(value).__name__}")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real numeric scalar") from exc
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]; got {value!r}")
    return value


@dataclass(frozen=True)
class DirichletBelief:
    """A finite categorical belief represented by Dirichlet concentrations.

    ``prior_weight`` is the non-informative prior mass ``W`` used when
    interpreting concentrations as subjective-logic evidence.  With uniform
    base rates and ``W=K``, ``alpha_k=e_k+1`` and vacuity is ``K/S``.
    """

    labels: tuple[str, ...]
    alpha: tuple[float, ...]
    base_rate: tuple[float, ...] | None = None
    prior_weight: float | None = None

    def __post_init__(self) -> None:
        labels = tuple(
            _strict_string(label, f"labels[{index}]")
            for index, label in enumerate(self.labels)
        )
        alpha = tuple(_positive_finite(v, "alpha") for v in self.alpha)
        if len(labels) < 2:
            raise ValueError("DirichletBelief requires at least two hypotheses")
        if len(labels) != len(alpha):
            raise ValueError("labels and alpha must have identical lengths")
        if len(set(labels)) != len(labels) or any(not label for label in labels):
            raise ValueError("hypothesis labels must be non-empty and unique")

        if self.base_rate is None:
            base_rate = tuple(1.0 / len(labels) for _ in labels)
        else:
            base_rate = tuple(_probability(v, "base_rate") for v in self.base_rate)
            if len(base_rate) != len(labels):
                raise ValueError("base_rate and labels must have identical lengths")
            if not np.isclose(sum(base_rate), 1.0, rtol=0.0, atol=1e-9):
                raise ValueError("base_rate must sum to one")

        prior_weight = (
            float(len(labels))
            if self.prior_weight is None
            else _positive_finite(self.prior_weight, "prior_weight")
        )
        strength = float(sum(alpha))
        if not isfinite(strength):
            raise ValueError("Dirichlet strength must be finite")
        if strength + 1e-12 < prior_weight:
            raise ValueError("Dirichlet strength cannot be below prior_weight")
        prior_alpha = tuple(prior_weight * rate for rate in base_rate)
        if any(value + 1e-12 < prior for value, prior in zip(alpha, prior_alpha, strict=True)):
            raise ValueError(
                "evidential Dirichlet requires alpha_k >= prior_weight * base_rate_k"
            )

        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "base_rate", base_rate)
        object.__setattr__(self, "prior_weight", prior_weight)

    @classmethod
    def from_evidence(
        cls,
        labels: Sequence[str],
        evidence: Sequence[float],
        *,
        base_rate: Sequence[float] | None = None,
        prior_weight: float | None = None,
    ) -> DirichletBelief:
        labels_tuple = tuple(labels)
        if any(not isinstance(label, str) for label in labels_tuple):
            raise TypeError("labels must contain strings")
        if base_rate is None:
            base_tuple = tuple(1.0 / len(labels_tuple) for _ in labels_tuple)
        else:
            base_tuple = tuple(_probability(v, "base_rate") for v in base_rate)
        weight = (
            float(len(labels_tuple))
            if prior_weight is None
            else _positive_finite(prior_weight, "prior_weight")
        )
        evidence_tuple = tuple(
            _nonnegative_finite(v, "evidence") for v in evidence
        )
        if len(evidence_tuple) != len(labels_tuple):
            raise ValueError("evidence and labels must have identical lengths")
        if any(not isfinite(v) or v < 0.0 for v in evidence_tuple):
            raise ValueError("evidence must be finite and non-negative")
        alpha = tuple(e + weight * a for e, a in zip(evidence_tuple, base_tuple, strict=True))
        return cls(labels_tuple, alpha, base_tuple, weight)

    @property
    def strength(self) -> float:
        return float(sum(self.alpha))

    @property
    def evidence_strength(self) -> float:
        return self.strength - float(self.prior_weight)

    @property
    def evidence_vector(self) -> np.ndarray:
        """Non-negative subjective-logic evidence ``e=alpha-W*a``."""

        return np.asarray(self.alpha, dtype=np.float64) - float(
            self.prior_weight
        ) * np.asarray(self.base_rate, dtype=np.float64)

    @property
    def mean_vector(self) -> np.ndarray:
        return np.asarray(self.alpha, dtype=np.float64) / self.strength

    @property
    def mean(self) -> dict[str, float]:
        return dict(zip(self.labels, self.mean_vector.tolist(), strict=True))

    @property
    def vacuity(self) -> float:
        """Subjective-logic uncertainty mass ``u=W/S``.

        This measures lack of accumulated evidence under the chosen evidential
        parameterization; it is not automatically a calibrated epistemic error.
        """

        return min(1.0, float(self.prior_weight) / self.strength)

    @property
    def predictive_entropy(self) -> float:
        probabilities = self.mean_vector
        return float(-np.sum(probabilities * np.log(probabilities)))

    @property
    def normalized_predictive_entropy(self) -> float:
        return self.predictive_entropy / log(len(self.labels))

    def probability(self, label: str) -> float:
        try:
            index = self.labels.index(label)
        except ValueError as exc:
            raise KeyError(label) from exc
        return float(self.mean_vector[index])

    def to_dict(self) -> dict[str, object]:
        return {
            "labels": list(self.labels),
            "alpha": list(self.alpha),
            "base_rate": list(self.base_rate or ()),
            "prior_weight": self.prior_weight,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> DirichletBelief:
        if not isinstance(payload, Mapping):
            raise TypeError("DirichletBelief must be a mapping")
        _require_exact_fields(
            payload,
            frozenset({"labels", "alpha", "base_rate", "prior_weight"}),
            "DirichletBelief",
        )
        raw_labels = _strict_array(payload["labels"], "DirichletBelief.labels")
        raw_alpha = _strict_array(payload["alpha"], "DirichletBelief.alpha")
        raw_base_rate = _strict_array(
            payload["base_rate"], "DirichletBelief.base_rate"
        )
        return cls(
            labels=tuple(
                _strict_string(value, f"DirichletBelief.labels[{index}]")
                for index, value in enumerate(raw_labels)
            ),
            alpha=tuple(
                _positive_finite(value, f"DirichletBelief.alpha[{index}]")
                for index, value in enumerate(raw_alpha)
            ),
            base_rate=tuple(
                _probability(value, f"DirichletBelief.base_rate[{index}]")
                for index, value in enumerate(raw_base_rate)
            ),
            prior_weight=_positive_finite(
                payload["prior_weight"], "DirichletBelief.prior_weight"
            ),
        )


@dataclass(frozen=True)
class BetaBelief:
    """Beta belief for a binary proposition, such as information sufficiency."""

    alpha: float
    beta: float
    prior_weight: float = 2.0
    base_rate: float = 0.5

    def __post_init__(self) -> None:
        alpha = _positive_finite(self.alpha, "alpha")
        beta = _positive_finite(self.beta, "beta")
        prior_weight = _positive_finite(self.prior_weight, "prior_weight")
        base_rate = _probability(self.base_rate, "base_rate")
        strength = alpha + beta
        if not isfinite(strength):
            raise ValueError("Beta strength must be finite")
        if strength + 1e-12 < prior_weight:
            raise ValueError("Beta strength cannot be below prior_weight")
        prior_alpha = prior_weight * base_rate
        prior_beta = prior_weight * (1.0 - base_rate)
        if alpha + 1e-12 < prior_alpha or beta + 1e-12 < prior_beta:
            raise ValueError(
                "evidential Beta requires alpha,beta >= W*(a,1-a)"
            )
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "prior_weight", prior_weight)
        object.__setattr__(self, "base_rate", base_rate)

    @property
    def strength(self) -> float:
        return self.alpha + self.beta

    @property
    def mean(self) -> float:
        return self.alpha / self.strength

    @property
    def variance(self) -> float:
        return (self.alpha * self.beta) / (
            self.strength * self.strength * (self.strength + 1.0)
        )

    @property
    def vacuity(self) -> float:
        return min(1.0, self.prior_weight / self.strength)

    def to_dict(self) -> dict[str, float]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "prior_weight": self.prior_weight,
            "base_rate": self.base_rate,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> BetaBelief:
        if not isinstance(payload, Mapping):
            raise TypeError("BetaBelief must be a mapping")
        _require_exact_fields(
            payload,
            frozenset({"alpha", "beta", "prior_weight", "base_rate"}),
            "BetaBelief",
        )
        return cls(
            alpha=_positive_finite(payload["alpha"], "BetaBelief.alpha"),
            beta=_positive_finite(payload["beta"], "BetaBelief.beta"),
            prior_weight=_positive_finite(
                payload["prior_weight"], "BetaBelief.prior_weight"
            ),
            base_rate=_probability(payload["base_rate"], "BetaBelief.base_rate"),
        )


class DeficitKind(str, Enum):
    """Observed evidence deficits, deliberately not action labels."""

    IDENTITY_AMBIGUITY = "IDENTITY_AMBIGUITY"
    PRESENCE_UNCERTAINTY = "PRESENCE_UNCERTAINTY"
    OCCLUDED_REGION = "OCCLUDED_REGION"
    UNOBSERVED_SURFACE = "UNOBSERVED_SURFACE"
    LOW_RESOLUTION = "LOW_RESOLUTION"
    SEARCH_COVERAGE_GAP = "SEARCH_COVERAGE_GAP"
    ACCESS_BLOCKED = "ACCESS_BLOCKED"


@dataclass(frozen=True)
class EvidenceDeficit:
    """Prompt-relevant uncertainty assigned to a deployable visual anchor."""

    deficit_id: str
    kind: DeficitKind
    anchor_token: str
    probability: BetaBelief
    prompt_relevance: float
    rationale: str = ""

    def __post_init__(self) -> None:
        _strict_string(self.deficit_id, "deficit_id")
        _strict_string(self.anchor_token, "anchor_token")
        if not isinstance(self.kind, DeficitKind):
            raise TypeError("kind must be a DeficitKind")
        if not isinstance(self.probability, BetaBelief):
            raise TypeError("probability must be a BetaBelief")
        _strict_string(self.rationale, "rationale", allow_empty=True)
        object.__setattr__(
            self, "prompt_relevance", _probability(self.prompt_relevance, "prompt_relevance")
        )

    @property
    def responsibility(self) -> float:
        """Expected prompt-weighted responsibility for the evidence deficit."""

        return self.probability.mean * self.prompt_relevance

    def to_dict(self) -> dict[str, object]:
        return {
            "deficit_id": self.deficit_id,
            "kind": self.kind.value,
            "anchor_token": self.anchor_token,
            "probability": self.probability.to_dict(),
            "prompt_relevance": self.prompt_relevance,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> EvidenceDeficit:
        if not isinstance(payload, Mapping):
            raise TypeError("EvidenceDeficit must be a mapping")
        _require_exact_fields(
            payload,
            frozenset(
                {
                    "deficit_id",
                    "kind",
                    "anchor_token",
                    "probability",
                    "prompt_relevance",
                    "rationale",
                }
            ),
            "EvidenceDeficit",
        )
        probability = payload["probability"]
        if not isinstance(probability, Mapping):
            raise TypeError("EvidenceDeficit.probability must be a mapping")
        return cls(
            deficit_id=_strict_string(
                payload["deficit_id"], "EvidenceDeficit.deficit_id"
            ),
            kind=DeficitKind(
                _strict_string(payload["kind"], "EvidenceDeficit.kind")
            ),
            anchor_token=_strict_string(
                payload["anchor_token"], "EvidenceDeficit.anchor_token"
            ),
            probability=BetaBelief.from_dict(probability),
            prompt_relevance=_probability(
                payload["prompt_relevance"], "EvidenceDeficit.prompt_relevance"
            ),
            rationale=_strict_string(
                payload["rationale"], "EvidenceDeficit.rationale", allow_empty=True
            ),
        )


@dataclass(frozen=True)
class TaskBelief:
    """Prompt-conditioned task belief consumed by the action bridge."""

    prompt: str
    hypotheses: DirichletBelief
    sufficiency: BetaBelief
    deficits: tuple[EvidenceDeficit, ...] = ()
    provenance: str = "policy_model"
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _strict_string(self.prompt, "prompt")
        _strict_string(self.provenance, "provenance")
        if not isinstance(self.hypotheses, DirichletBelief):
            raise TypeError("hypotheses must be a DirichletBelief")
        if not isinstance(self.sufficiency, BetaBelief):
            raise TypeError("sufficiency must be a BetaBelief")
        if any(not isinstance(item, EvidenceDeficit) for item in self.deficits):
            raise TypeError("deficits must contain EvidenceDeficit objects")
        if any(not isinstance(tag, str) or not tag for tag in self.tags):
            raise TypeError("tags must be non-empty strings")
        if len({d.deficit_id for d in self.deficits}) != len(self.deficits):
            raise ValueError("deficit IDs must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "hypotheses": self.hypotheses.to_dict(),
            "sufficiency": self.sufficiency.to_dict(),
            "deficits": [deficit.to_dict() for deficit in self.deficits],
            "provenance": self.provenance,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TaskBelief:
        if not isinstance(payload, Mapping):
            raise TypeError("TaskBelief must be a mapping")
        _require_exact_fields(
            payload,
            frozenset(
                {"prompt", "hypotheses", "sufficiency", "deficits", "provenance", "tags"}
            ),
            "TaskBelief",
        )
        hypotheses = payload["hypotheses"]
        sufficiency = payload["sufficiency"]
        raw_deficits = _strict_array(payload["deficits"], "TaskBelief.deficits")
        raw_tags = _strict_array(payload["tags"], "TaskBelief.tags")
        if not isinstance(hypotheses, Mapping) or not isinstance(sufficiency, Mapping):
            raise TypeError("TaskBelief hypotheses and sufficiency must be mappings")
        if any(not isinstance(item, Mapping) for item in raw_deficits):
            raise TypeError("TaskBelief.deficits entries must be mappings")
        return cls(
            prompt=_strict_string(payload["prompt"], "TaskBelief.prompt"),
            hypotheses=DirichletBelief.from_dict(hypotheses),
            sufficiency=BetaBelief.from_dict(sufficiency),
            deficits=tuple(
                EvidenceDeficit.from_dict(item)
                for item in raw_deficits  # type: ignore[arg-type]
            ),
            provenance=_strict_string(
                payload["provenance"], "TaskBelief.provenance"
            ),
            tags=tuple(
                _strict_string(value, f"TaskBelief.tags[{index}]")
                for index, value in enumerate(raw_tags)
            ),
        )


def pomdp_belief_update(
    prior: Sequence[float],
    transition: Sequence[Sequence[float]],
    observation_likelihood: Sequence[float],
) -> np.ndarray:
    """Perform the finite-state POMDP update ``b' ∝ O(o|s',a) T_a^T b``.

    Rows of ``transition`` index the old state and columns index the new
    state.  ``observation_likelihood[j]`` is ``P(o | s'_j, a)``.
    """

    if _contains_boolean(prior) or _contains_boolean(transition) or _contains_boolean(
        observation_likelihood
    ):
        raise TypeError("POMDP probabilities must be numeric, not boolean")
    belief = np.asarray(prior, dtype=np.float64)
    transition_matrix = np.asarray(transition, dtype=np.float64)
    likelihood = np.asarray(observation_likelihood, dtype=np.float64)

    if belief.ndim != 1:
        raise ValueError("prior must be one-dimensional")
    if transition_matrix.shape != (belief.size, belief.size):
        raise ValueError("transition must have shape [num_states, num_states]")
    if likelihood.shape != belief.shape:
        raise ValueError("observation_likelihood must match prior shape")
    if (
        not np.all(np.isfinite(belief))
        or np.any(belief < 0.0)
        or not np.isclose(belief.sum(), 1.0, rtol=0.0, atol=1e-9)
    ):
        raise ValueError("prior must be a non-negative distribution summing to one")
    if (
        not np.all(np.isfinite(transition_matrix))
        or np.any(transition_matrix < 0.0)
        or not np.allclose(
            transition_matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-9
        )
    ):
        raise ValueError("each transition row must be a distribution")
    if (
        not np.all(np.isfinite(likelihood))
        or np.any(likelihood < 0.0)
        or np.any(likelihood > 1.0)
    ):
        raise ValueError("observation likelihoods must be in [0, 1]")

    predictive = transition_matrix.T @ belief
    unnormalized = likelihood * predictive
    normalizer = float(unnormalized.sum())
    if normalizer <= 0.0:
        raise ValueError("observation has zero probability under the predictive belief")
    return unnormalized / normalizer


def mixture_belief(
    beliefs: Iterable[DirichletBelief], weights: Iterable[float]
) -> DirichletBelief:
    """Moment-match a weighted mixture at the categorical-mean level.

    This helper is intentionally explicit about being an approximation: a
    mixture of Dirichlets is generally not itself a Dirichlet.
    """

    belief_list = list(beliefs)
    weight_values = list(weights)
    if _contains_boolean(weight_values):
        raise TypeError("weights must be numeric, not boolean")
    weight_array = np.asarray(weight_values, dtype=np.float64)
    if not belief_list:
        raise ValueError("at least one belief is required")
    if weight_array.shape != (len(belief_list),) or np.any(weight_array < 0.0):
        raise ValueError("weights must be non-negative and match beliefs")
    if not np.isclose(weight_array.sum(), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("weights must sum to one")
    first = belief_list[0]
    if any(belief.labels != first.labels for belief in belief_list):
        raise ValueError("all beliefs must use the same hypothesis labels")
    means = np.stack([belief.mean_vector for belief in belief_list], axis=0)
    mean = weight_array @ means
    strength = float(weight_array @ np.asarray([b.strength for b in belief_list]))
    alpha = tuple((mean * strength).tolist())
    return DirichletBelief(first.labels, alpha, first.base_rate, first.prior_weight)
