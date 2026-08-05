"""Exact evidence updates and prompt projections.

Integer count updates are ordinary conjugate Beta/Dirichlet updates.  Fractional
counts are useful as neural pseudo-evidence, but are only enabled explicitly so
that callers do not accidentally describe them as exact Bayesian observations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite

import numpy as np

from .beliefs import BetaBelief, DirichletBelief


def _count(value: float, name: str, *, allow_fractional: bool) -> float:
    if isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be numeric, not boolean")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    if not allow_fractional and not result.is_integer():
        raise ValueError(
            f"{name} is fractional; pass allow_fractional=True to mark it as pseudo-evidence"
        )
    return result


def update_beta(
    prior: BetaBelief,
    *,
    positive_count: float = 0.0,
    negative_count: float = 0.0,
    allow_fractional: bool = False,
) -> BetaBelief:
    """Conjugately update a binary proposition with observed counts."""

    positive = _count(positive_count, "positive_count", allow_fractional=allow_fractional)
    negative = _count(negative_count, "negative_count", allow_fractional=allow_fractional)
    return BetaBelief(
        prior.alpha + positive,
        prior.beta + negative,
        prior_weight=prior.prior_weight,
        base_rate=prior.base_rate,
    )


def update_dirichlet(
    prior: DirichletBelief,
    *,
    class_counts: Mapping[str, float],
    allow_fractional: bool = False,
) -> DirichletBelief:
    """Conjugately update a categorical proposition with class counts."""

    unknown = set(class_counts) - set(prior.labels)
    if unknown:
        raise KeyError(f"unknown hypotheses in class_counts: {sorted(unknown)}")
    counts = {
        label: _count(
            class_counts.get(label, 0.0),
            f"class_counts[{label!r}]",
            allow_fractional=allow_fractional,
        )
        for label in prior.labels
    }
    return DirichletBelief(
        labels=prior.labels,
        alpha=tuple(
            alpha + counts[label]
            for label, alpha in zip(prior.labels, prior.alpha, strict=True)
        ),
        base_rate=prior.base_rate,
        prior_weight=prior.prior_weight,
    )


def project_prompt_target_beta(
    belief: DirichletBelief,
    target_labels: Sequence[str],
) -> BetaBelief:
    """Exactly aggregate a Dirichlet into a prompt-target Beta belief.

    If ``pi ~ Dir(alpha)`` and the prompt denotes a hard subset ``T`` of the
    closed hypothesis ontology, then ``sum_{k in T} pi_k`` follows
    ``Beta(sum_T alpha_k, sum_not_T alpha_k)``.  Arbitrary soft language weights
    do *not* generally preserve a Beta distribution and are rejected here.
    """

    if any(not isinstance(label, str) for label in target_labels):
        raise TypeError("target_labels must contain strings")
    targets = frozenset(target_labels)
    if not targets or targets == frozenset(belief.labels):
        raise ValueError("target_labels must be a non-empty proper subset")
    unknown = targets - set(belief.labels)
    if unknown:
        raise KeyError(f"unknown prompt target labels: {sorted(unknown)}")
    positive = sum(
        alpha
        for label, alpha in zip(belief.labels, belief.alpha, strict=True)
        if label in targets
    )
    negative = belief.strength - positive
    target_base_rate = sum(
        rate
        for label, rate in zip(belief.labels, belief.base_rate, strict=True)
        if label in targets
    )
    return BetaBelief(
        positive,
        negative,
        prior_weight=float(belief.prior_weight),
        base_rate=target_base_rate,
    )
