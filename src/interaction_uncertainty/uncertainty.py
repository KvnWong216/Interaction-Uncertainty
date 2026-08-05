"""Uncertainty quantities used by the bridge.

Predictive entropy, expected categorical entropy, and mutual information are
kept separate.  A single scalar cannot identify an exploratory action; these
quantities summarize a structured belief and are only inputs to action-effect
comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log
from typing import TYPE_CHECKING

import numpy as np

from .beliefs import DirichletBelief, TaskBelief

if TYPE_CHECKING:
    from .v2.effects import CounterfactualRollout


def _digamma_positive(value: float) -> float:
    """Accurate digamma approximation for strictly positive real inputs.

    Recurrence moves the input to the asymptotic regime.  The expansion is
    adequate for evidential concentrations and avoids making SciPy a core
    dependency.  A reflection branch is included for completeness, although
    belief concentrations are always positive.
    """

    if isinstance(value, bool | np.bool_):
        raise TypeError("digamma input must be numeric, not boolean")
    x = float(value)
    if not isfinite(x) or x <= 0.0:
        raise ValueError("digamma input must be finite and positive")
    if x < 1e-8:
        return -0.5772156649015329 - 1.0 / x
    result = 0.0
    while x < 8.0:
        result -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    result += (
        log(x)
        - 0.5 * inv
        - inv2
        * (
            1.0 / 12.0
            - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0 - inv2 * 1.0 / 240.0))
        )
    )
    return result


def categorical_entropy(probabilities: np.ndarray, *, normalize: bool = False) -> float:
    if isinstance(normalize, np.bool_) or not isinstance(normalize, bool):
        raise TypeError("normalize must be a boolean")
    raw = np.asarray(probabilities)
    if np.issubdtype(raw.dtype, np.bool_) or (
        raw.dtype == object
        and any(isinstance(item, bool | np.bool_) for item in raw.reshape(-1))
    ):
        raise TypeError("probabilities must be numeric, not boolean")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 1 or probabilities.size < 2:
        raise ValueError("probabilities must be a vector with at least two entries")
    if not np.all(np.isfinite(probabilities)) or np.any(
        probabilities < 0.0
    ) or not np.isclose(
        probabilities.sum(), 1.0, rtol=0.0, atol=1e-9
    ):
        raise ValueError("probabilities must be non-negative and sum to one")
    positive = probabilities[probabilities > 0.0]
    value = float(-np.sum(positive * np.log(positive)))
    return value / log(probabilities.size) if normalize else value


def expected_categorical_entropy(belief: DirichletBelief) -> float:
    """Compute ``E_{p~Dir(alpha)}[H(Categorical(p))]`` in closed form."""

    strength = belief.strength
    total = 0.0
    for alpha_k in belief.alpha:
        mean_k = alpha_k / strength
        total -= mean_k * (_digamma_positive(alpha_k + 1.0) - _digamma_positive(strength + 1.0))
    return float(total)


def dirichlet_mutual_information(belief: DirichletBelief) -> float:
    """Second-order MI ``H(E[p]) - E[H(Categorical(p))]``.

    This is often interpreted as distributional/knowledge uncertainty for a
    Dirichlet predictive model.  Its interpretation is only as good as the
    learned second-order distribution and its calibration.
    """

    value = belief.predictive_entropy - expected_categorical_entropy(belief)
    return max(0.0, float(value))


def subjective_logic_dissonance(belief: DirichletBelief) -> float:
    """Dissonance among non-zero subjective-logic belief masses.

    Dissonance captures conflict between supported hypotheses, while vacuity
    captures lack of evidence.  The implementation follows the standard
    pairwise balance construction used in subjective-logic uncertainty work.
    """

    masses = belief.evidence_vector / belief.strength
    if masses.sum() <= 0.0:
        return 0.0

    dissonance = 0.0
    for i, mass_i in enumerate(masses):
        if mass_i <= 0.0:
            continue
        denominator = float(masses.sum() - mass_i)
        if denominator <= 0.0:
            continue
        balanced = 0.0
        for j, mass_j in enumerate(masses):
            if i == j or mass_j <= 0.0:
                continue
            balance = 1.0 - abs(mass_i - mass_j) / (mass_i + mass_j)
            balanced += mass_j * balance
        dissonance += mass_i * balanced / denominator
    return float(np.clip(dissonance, 0.0, 1.0))


@dataclass(frozen=True)
class UncertaintyReport:
    predictive_entropy: float
    normalized_predictive_entropy: float
    expected_data_entropy: float
    dirichlet_mutual_information: float
    vacuity: float
    dissonance: float
    sufficiency_mean: float
    sufficiency_variance: float
    sufficiency_vacuity: float
    max_deficit_responsibility: float

    def to_dict(self) -> dict[str, float]:
        return {
            "predictive_entropy": self.predictive_entropy,
            "normalized_predictive_entropy": self.normalized_predictive_entropy,
            "expected_data_entropy": self.expected_data_entropy,
            "dirichlet_mutual_information": self.dirichlet_mutual_information,
            "vacuity": self.vacuity,
            "dissonance": self.dissonance,
            "sufficiency_mean": self.sufficiency_mean,
            "sufficiency_variance": self.sufficiency_variance,
            "sufficiency_vacuity": self.sufficiency_vacuity,
            "max_deficit_responsibility": self.max_deficit_responsibility,
        }


def summarize_task_uncertainty(belief: TaskBelief) -> UncertaintyReport:
    hypotheses = belief.hypotheses
    return UncertaintyReport(
        predictive_entropy=hypotheses.predictive_entropy,
        normalized_predictive_entropy=hypotheses.normalized_predictive_entropy,
        expected_data_entropy=expected_categorical_entropy(hypotheses),
        dirichlet_mutual_information=dirichlet_mutual_information(hypotheses),
        vacuity=hypotheses.vacuity,
        dissonance=subjective_logic_dissonance(hypotheses),
        sufficiency_mean=belief.sufficiency.mean,
        sufficiency_variance=belief.sufficiency.variance,
        sufficiency_vacuity=belief.sufficiency.vacuity,
        max_deficit_responsibility=max(
            (deficit.responsibility for deficit in belief.deficits), default=0.0
        ),
    )


def expected_predictive_entropy_after_action(
    rollout: CounterfactualRollout,
) -> float:
    """Expected hypothesis entropy after executing and re-observing an action.

    This is an action-conditioned diagnostic.  It is not the decision objective:
    entropy can decrease for hypotheses that do not affect the prompt-specific
    terminal decision.
    """

    return float(
        sum(
            branch.probability
            * categorical_entropy(np.asarray(branch.posterior_probabilities))
            for branch in rollout.outcomes
        )
    )


def action_expected_information_gain(rollout: CounterfactualRollout) -> float:
    """Conditional EIG across the action's explicitly represented outcomes.

    The reference prior is the *post-transition, pre-observation* predictive
    belief.  This avoids conflating physical world change with information
    acquired from the subsequent observation.
    """

    prior_entropy = categorical_entropy(np.asarray(rollout.predictive_probabilities))
    return max(0.0, prior_entropy - expected_predictive_entropy_after_action(rollout))
