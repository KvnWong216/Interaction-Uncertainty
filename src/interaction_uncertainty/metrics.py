"""Proper scores and calibration diagnostics for benchmark evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _validated_predictions(
    probabilities: Sequence[Sequence[float]], labels: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    raw_probs = np.asarray(probabilities)
    raw_truth = np.asarray(labels)
    if np.issubdtype(raw_probs.dtype, np.bool_) or (
        raw_probs.dtype == object
        and any(isinstance(item, bool | np.bool_) for item in raw_probs.reshape(-1))
    ):
        raise TypeError("probabilities must be numeric, not boolean")
    if np.issubdtype(raw_truth.dtype, np.bool_) or (
        raw_truth.dtype == object
        and any(isinstance(item, bool | np.bool_) for item in raw_truth.reshape(-1))
    ):
        raise TypeError("labels must be integer class indices, not boolean")
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError("probabilities must have shape [samples, classes]")
    if probs.shape[0] == 0:
        raise ValueError("at least one prediction is required")
    if raw_truth.shape != (probs.shape[0],):
        raise ValueError("labels must have one entry per probability row")
    try:
        numeric_truth = raw_truth.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels must be finite integer class indices") from exc
    if not np.all(np.isfinite(numeric_truth)) or not np.all(
        numeric_truth == np.floor(numeric_truth)
    ):
        raise ValueError("labels must be finite integer class indices")
    truth = numeric_truth.astype(np.int64)
    if not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(probs.sum(axis=1), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("each probability row must sum to one")
    if np.any(truth < 0) or np.any(truth >= probs.shape[1]):
        raise ValueError("labels must index a probability class")
    return probs, truth


def multiclass_brier_score(
    probabilities: Sequence[Sequence[float]], labels: Sequence[int]
) -> float:
    """Mean squared distance to the one-hot outcome (lower is better)."""

    probs, truth = _validated_predictions(probabilities, labels)
    targets = np.zeros_like(probs)
    targets[np.arange(truth.size), truth] = 1.0
    return float(np.mean(np.sum((probs - targets) ** 2, axis=1)))


def negative_log_likelihood(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    epsilon: float = 1e-12,
) -> float:
    """Categorical log score averaged over examples (lower is better)."""

    probs, truth = _validated_predictions(probabilities, labels)
    if isinstance(epsilon, bool | np.bool_) or not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie in (0, 1)")
    selected = np.clip(probs[np.arange(truth.size), truth], epsilon, 1.0)
    return float(-np.mean(np.log(selected)))


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float


@dataclass(frozen=True)
class CalibrationReport:
    expected_calibration_error: float
    maximum_calibration_error: float
    bins: tuple[CalibrationBin, ...]


def top_label_calibration(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    num_bins: int = 10,
) -> CalibrationReport:
    """Top-label ECE/MCE using equal-width confidence bins.

    ECE is a diagnostic, not a proper score.  Report it alongside NLL/Brier and
    retain all bin statistics so results are auditable rather than a lone scalar.
    """

    probs, truth = _validated_predictions(probabilities, labels)
    if isinstance(num_bins, bool | np.bool_) or not isinstance(
        num_bins, int | np.integer
    ):
        raise TypeError("num_bins must be an integer")
    if num_bins < 1:
        raise ValueError("num_bins must be positive")
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    correct = prediction == truth
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    bins: list[CalibrationBin] = []
    weighted_gap = 0.0
    maximum_gap = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        if index == num_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        count = int(mask.sum())
        mean_confidence = float(confidence[mask].mean()) if count else 0.0
        accuracy = float(correct[mask].mean()) if count else 0.0
        gap = abs(mean_confidence - accuracy) if count else 0.0
        weighted_gap += count / len(confidence) * gap
        maximum_gap = max(maximum_gap, gap)
        bins.append(
            CalibrationBin(
                lower=float(lower),
                upper=float(upper),
                count=count,
                mean_confidence=mean_confidence,
                accuracy=accuracy,
            )
        )
    return CalibrationReport(float(weighted_gap), float(maximum_gap), tuple(bins))
