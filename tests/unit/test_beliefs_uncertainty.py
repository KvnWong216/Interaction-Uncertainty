from __future__ import annotations

import math

import numpy as np
import pytest

from interaction_uncertainty.beliefs import BetaBelief, DirichletBelief
from interaction_uncertainty.uncertainty import (
    dirichlet_mutual_information,
    expected_categorical_entropy,
    subjective_logic_dissonance,
)


def test_beta_exact_mean_variance_and_vacuity() -> None:
    belief = BetaBelief(alpha=3.0, beta=2.0)

    assert belief.mean == pytest.approx(0.6)
    assert belief.variance == pytest.approx(0.04)
    assert belief.vacuity == pytest.approx(0.4)
    assert BetaBelief.from_dict(belief.to_dict()) == belief


@pytest.mark.parametrize(
    "alpha,beta",
    [(0.0, 1.0), (1.0, 0.0), (math.inf, 1.0), (1.0, math.nan)],
)
def test_beta_rejects_nonpositive_or_nonfinite_concentrations(
    alpha: float, beta: float
) -> None:
    with pytest.raises(ValueError):
        BetaBelief(alpha=alpha, beta=beta)


def test_beliefs_reject_finite_components_with_overflowing_total_strength() -> None:
    with pytest.raises(ValueError, match="strength must be finite"):
        BetaBelief(alpha=1e308, beta=1e308, prior_weight=2.0)
    with pytest.raises(ValueError, match="strength must be finite"):
        DirichletBelief(
            labels=("a", "b"),
            alpha=(1e308, 1e308),
            prior_weight=2.0,
        )


def test_dirichlet_mean_entropy_and_unit_prior_vacuity() -> None:
    belief = DirichletBelief(labels=("a", "b", "c"), alpha=(2.0, 3.0, 5.0))

    np.testing.assert_allclose(belief.mean_vector, [0.2, 0.3, 0.5])
    assert belief.strength == pytest.approx(10.0)
    assert belief.vacuity == pytest.approx(0.3)
    assert 0.0 <= belief.normalized_predictive_entropy <= 1.0
    assert DirichletBelief.from_dict(belief.to_dict()) == belief


def test_dirichlet_rejects_duplicate_labels_and_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="unique"):
        DirichletBelief(labels=("same", "same"), alpha=(1.0, 1.0))
    with pytest.raises(ValueError, match="identical lengths"):
        DirichletBelief(labels=("a", "b"), alpha=(1.0, 1.0, 1.0))


def test_belief_json_boundaries_do_not_coerce_strings_booleans_or_unknown_fields() -> None:
    beta_payload = BetaBelief(2.0, 3.0).to_dict()
    beta_payload["alpha"] = "2.0"
    with pytest.raises(TypeError, match="real number"):
        BetaBelief.from_dict(beta_payload)

    dirichlet_payload = DirichletBelief(("a", "b"), (1.0, 1.0)).to_dict()
    dirichlet_payload["unknown"] = 1
    with pytest.raises(ValueError, match="strict contract"):
        DirichletBelief.from_dict(dirichlet_payload)

    with pytest.raises(TypeError, match="boolean"):
        DirichletBelief.from_evidence(("a", "b"), (True, 0.0))


def test_evidential_beliefs_reject_negative_subjective_logic_evidence() -> None:
    with pytest.raises(ValueError, match="alpha_k"):
        DirichletBelief(
            labels=("a", "b"),
            alpha=(0.1, 1.9),
            base_rate=(0.5, 0.5),
            prior_weight=2.0,
        )
    with pytest.raises(ValueError, match="alpha,beta"):
        BetaBelief(alpha=0.1, beta=1.9, prior_weight=2.0)


def test_dirichlet_mutual_information_matches_uniform_beta_closed_form() -> None:
    # For p ~ Beta(1, 1), H(E[p])=log(2) and E[H(Bernoulli(p))]=1/2 nat.
    belief = DirichletBelief(labels=("yes", "no"), alpha=(1.0, 1.0))

    assert expected_categorical_entropy(belief) == pytest.approx(0.5, abs=2e-8)
    assert dirichlet_mutual_information(belief) == pytest.approx(
        math.log(2.0) - 0.5, abs=2e-8
    )


def test_mutual_information_falls_as_symmetric_evidence_grows() -> None:
    vacuous = DirichletBelief(labels=("yes", "no"), alpha=(1.0, 1.0))
    concentrated = DirichletBelief(labels=("yes", "no"), alpha=(50.0, 50.0))

    assert vacuous.mean == concentrated.mean
    assert dirichlet_mutual_information(vacuous) > dirichlet_mutual_information(
        concentrated
    )
    assert vacuous.vacuity > concentrated.vacuity


def test_dissonance_distinguishes_conflict_from_one_sided_evidence() -> None:
    conflict = DirichletBelief(labels=("yes", "no"), alpha=(6.0, 6.0))
    one_sided = DirichletBelief(labels=("yes", "no"), alpha=(11.0, 1.0))
    vacuous = DirichletBelief(labels=("yes", "no"), alpha=(1.0, 1.0))

    assert subjective_logic_dissonance(conflict) == pytest.approx(10.0 / 12.0)
    assert subjective_logic_dissonance(one_sided) == pytest.approx(0.0)
    assert subjective_logic_dissonance(vacuous) == pytest.approx(0.0)
