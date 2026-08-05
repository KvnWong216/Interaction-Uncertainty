from __future__ import annotations

import math

import pytest

from interaction_uncertainty.beliefs import BetaBelief, DirichletBelief
from interaction_uncertainty.metrics import (
    multiclass_brier_score,
    negative_log_likelihood,
    top_label_calibration,
)
from interaction_uncertainty.updates import (
    project_prompt_target_beta,
    update_beta,
    update_dirichlet,
)


def test_prompt_target_projection_is_exact_dirichlet_subset_aggregation() -> None:
    categorical = DirichletBelief(
        labels=("chocolate", "vanilla", "strawberry"),
        alpha=(2.0, 3.0, 5.0),
    )

    target = project_prompt_target_beta(categorical, ("chocolate", "strawberry"))

    # If pi ~ Dir(2, 3, 5), pi_chocolate + pi_strawberry ~ Beta(7, 3).
    assert target.alpha == pytest.approx(7.0)
    assert target.beta == pytest.approx(3.0)
    assert target.mean == pytest.approx(0.7)
    assert target.prior_weight == pytest.approx(categorical.prior_weight)


def test_prompt_target_projection_preserves_asymmetric_subset_base_rate() -> None:
    categorical = DirichletBelief.from_evidence(
        labels=("a", "b", "c"),
        evidence=(0.0, 0.0, 0.0),
        base_rate=(0.1, 0.45, 0.45),
        prior_weight=3.0,
    )

    target = project_prompt_target_beta(categorical, ("a",))

    assert target.alpha == pytest.approx(0.3)
    assert target.beta == pytest.approx(2.7)
    assert target.base_rate == pytest.approx(0.1)
    assert target.vacuity == pytest.approx(1.0)


def test_prompt_target_projection_rejects_nonproper_or_unknown_subsets() -> None:
    belief = DirichletBelief(labels=("target", "other"), alpha=(2.0, 2.0))

    with pytest.raises(ValueError, match="non-empty proper subset"):
        project_prompt_target_beta(belief, ())
    with pytest.raises(ValueError, match="non-empty proper subset"):
        project_prompt_target_beta(belief, belief.labels)
    with pytest.raises(KeyError, match="unknown"):
        project_prompt_target_beta(belief, ("unknown",))


def test_integer_counts_are_exact_conjugate_updates() -> None:
    beta = update_beta(BetaBelief(2.0, 3.0), positive_count=4, negative_count=1)
    categorical = update_dirichlet(
        DirichletBelief(labels=("a", "b", "c"), alpha=(1.0, 1.0, 1.0)),
        class_counts={"a": 2, "c": 3},
    )

    assert beta == BetaBelief(6.0, 4.0)
    assert categorical.alpha == pytest.approx((3.0, 1.0, 4.0))


def test_fractional_pseudo_evidence_requires_explicit_opt_in() -> None:
    prior_beta = BetaBelief(1.0, 1.0)
    prior_dirichlet = DirichletBelief(labels=("a", "b"), alpha=(1.0, 1.0))

    with pytest.raises(ValueError, match="allow_fractional=True"):
        update_beta(prior_beta, positive_count=0.25)
    with pytest.raises(ValueError, match="allow_fractional=True"):
        update_dirichlet(prior_dirichlet, class_counts={"a": 0.25})

    beta = update_beta(
        prior_beta,
        positive_count=0.25,
        negative_count=0.75,
        allow_fractional=True,
    )
    categorical = update_dirichlet(
        prior_dirichlet,
        class_counts={"a": 0.25, "b": 0.75},
        allow_fractional=True,
    )

    assert (beta.alpha, beta.beta) == pytest.approx((1.25, 1.75))
    assert categorical.alpha == pytest.approx((1.25, 1.75))


def test_brier_and_nll_match_hand_calculation() -> None:
    probabilities = ((0.8, 0.2), (0.4, 0.6))
    labels = (0, 1)

    assert multiclass_brier_score(probabilities, labels) == pytest.approx(0.2)
    assert negative_log_likelihood(probabilities, labels) == pytest.approx(
        -0.5 * (math.log(0.8) + math.log(0.6))
    )


def test_top_label_ece_retains_auditable_bin_statistics() -> None:
    report = top_label_calibration(
        probabilities=((0.8, 0.2), (0.4, 0.6)),
        labels=(0, 1),
        num_bins=2,
    )

    assert report.expected_calibration_error == pytest.approx(0.3)
    assert report.maximum_calibration_error == pytest.approx(0.3)
    assert len(report.bins) == 2
    assert report.bins[0].count == 0
    assert report.bins[1].count == 2
    assert report.bins[1].mean_confidence == pytest.approx(0.7)
    assert report.bins[1].accuracy == pytest.approx(1.0)


@pytest.mark.parametrize(
    "probabilities,labels",
    [
        (((0.8, 0.3),), (0,)),
        (((-0.1, 1.1),), (1,)),
        (((0.5, 0.5),), (2,)),
    ],
)
def test_proper_scores_reject_invalid_prediction_packets(
    probabilities: tuple[tuple[float, ...], ...],
    labels: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        multiclass_brier_score(probabilities, labels)


@pytest.mark.parametrize(
    "probabilities,labels",
    [
        ((), ()),
        (((0.5, 0.5),), (0.9,)),
        (((0.5, 0.5),), (float("nan"),)),
    ],
)
def test_proper_scores_reject_empty_or_fractional_truth(
    probabilities: tuple[tuple[float, ...], ...],
    labels: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError):
        multiclass_brier_score(probabilities, labels)  # type: ignore[arg-type]


def test_probability_metrics_and_prompt_projection_reject_boolean_coercion() -> None:
    with pytest.raises(TypeError, match="boolean"):
        multiclass_brier_score(((True, False),), (0,))
    with pytest.raises(TypeError, match="boolean"):
        multiclass_brier_score(((0.5, 0.5),), (True,))
    with pytest.raises(TypeError, match="strings"):
        project_prompt_target_beta(
            DirichletBelief(("a", "b"), (1.0, 1.0)),
            (1,),  # type: ignore[arg-type]
        )
