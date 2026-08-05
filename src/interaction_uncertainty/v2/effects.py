"""Action-conditioned transition/observation forecasts and exact rollouts.

The critic predicts a finite POMDP model ``T(s'|s,a)`` and
``Z(y|s',a)``.  Branch probabilities and posteriors are derived by Bayes rule,
so a model cannot create information by independently inventing posterior
beliefs.  This is the clean-room implementation of the action-as-measurement
idea used in uncertainty-aware interactive perception.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

import numpy as np

from .belief import BeliefState
from .evidence import ModelStamp
from .primitives import CandidateSet, PrimitiveCall
from .task import TaskKey, TaskSpec


class ExecutionStatus(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be numeric, not boolean")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric scalar") from exc
    if not isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _nonnegative(value: object, name: str) -> float:
    parsed = _finite_scalar(value, name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


@dataclass(frozen=True)
class ObservationOutcomeModel:
    outcome_id: str
    likelihood_by_post_state: tuple[float, ...]
    execution_status: ExecutionStatus
    sufficiency_evidence: tuple[float, float] = (0.0, 0.0)
    resolves_need_ids: tuple[str, ...] = ()
    action_cost: float = 0.0
    physical_risk: float = 0.0
    disturbance: float = 0.0

    def __post_init__(self) -> None:
        if not self.outcome_id.strip() or not self.likelihood_by_post_state:
            raise ValueError("outcome_id and likelihood vector are required")
        if not isinstance(self.execution_status, ExecutionStatus):
            raise TypeError("execution_status must be an ExecutionStatus")
        if any(
            isinstance(value, bool | np.bool_)
            for value in self.likelihood_by_post_state
        ):
            raise TypeError("observation likelihoods must be numeric, not boolean")
        likelihood = np.asarray(self.likelihood_by_post_state, dtype=np.float64)
        if (
            not np.all(np.isfinite(likelihood))
            or np.any(likelihood < 0.0)
            or np.any(likelihood > 1.0)
        ):
            raise ValueError("observation likelihoods must lie in [0, 1]")
        if any(not item.strip() for item in self.resolves_need_ids) or len(
            set(self.resolves_need_ids)
        ) != len(self.resolves_need_ids):
            raise ValueError("resolves_need_ids must be non-empty strings and unique")
        if len(self.sufficiency_evidence) != 2 or any(
            not isfinite(float(value)) or float(value) < 0.0
            for value in self.sufficiency_evidence
        ):
            raise ValueError("sufficiency evidence must contain two non-negative values")
        if any(isinstance(value, bool | np.bool_) for value in self.sufficiency_evidence):
            raise TypeError("sufficiency evidence must be numeric, not boolean")
        if not isfinite(sum(self.sufficiency_evidence)):
            raise ValueError("sufficiency evidence strength must be finite")
        object.__setattr__(self, "action_cost", _nonnegative(self.action_cost, "action_cost"))
        object.__setattr__(
            self, "physical_risk", _nonnegative(self.physical_risk, "physical_risk")
        )
        object.__setattr__(
            self, "disturbance", _nonnegative(self.disturbance, "disturbance")
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "outcome_id": self.outcome_id,
            "likelihood_by_post_state": list(self.likelihood_by_post_state),
            "execution_status": self.execution_status.value,
            "sufficiency_evidence": list(self.sufficiency_evidence),
            "resolves_need_ids": list(self.resolves_need_ids),
            "action_cost": self.action_cost,
            "physical_risk": self.physical_risk,
            "disturbance": self.disturbance,
        }


@dataclass(frozen=True)
class CandidateEffectForecast:
    schema_version: str
    task_key: TaskKey
    candidate_id: str
    hypothesis_labels: tuple[str, ...]
    transition_matrix: tuple[tuple[float, ...], ...]
    outcomes: tuple[ObservationOutcomeModel, ...]
    critic_uncertainty: float
    model_stamp: ModelStamp
    rng_seed: int

    def __post_init__(self) -> None:
        if self.schema_version != "interaction-uncertainty.effect.v2":
            raise ValueError("unsupported effect schema_version")
        if not self.candidate_id.strip() or not self.outcomes:
            raise ValueError("candidate_id and at least one outcome are required")
        size = len(self.hypothesis_labels)
        if size < 2 or any(not label.strip() for label in self.hypothesis_labels):
            raise ValueError("at least two non-empty hypothesis labels are required")
        if len(set(self.hypothesis_labels)) != size:
            raise ValueError("hypothesis labels must be unique")
        if any(
            isinstance(value, bool | np.bool_)
            for row in self.transition_matrix
            for value in row
        ):
            raise TypeError("transition probabilities must be numeric, not boolean")
        transition = np.asarray(self.transition_matrix, dtype=np.float64)
        if transition.shape != (size, size):
            raise ValueError("transition_matrix must have shape [K, K]")
        if (
            not np.all(np.isfinite(transition))
            or np.any(transition < 0.0)
            or not np.allclose(
                transition.sum(axis=1), 1.0, rtol=0.0, atol=1e-8
            )
        ):
            raise ValueError("every transition row must be a probability distribution")
        ids = [outcome.outcome_id for outcome in self.outcomes]
        if len(ids) != len(set(ids)):
            raise ValueError("outcome IDs must be unique")
        observation = np.asarray(
            [outcome.likelihood_by_post_state for outcome in self.outcomes],
            dtype=np.float64,
        )
        if observation.shape != (len(self.outcomes), size):
            raise ValueError("every outcome likelihood must align with hypotheses")
        if not np.allclose(
            observation.sum(axis=0), 1.0, rtol=0.0, atol=1e-8
        ):
            raise ValueError("outcomes must be exhaustive for every post-action state")
        critic_uncertainty = _finite_scalar(
            self.critic_uncertainty, "critic_uncertainty"
        )
        if not 0.0 <= critic_uncertainty <= 1.0:
            raise ValueError("critic_uncertainty must lie in [0, 1]")
        if (
            isinstance(self.rng_seed, bool)
            or not isinstance(self.rng_seed, int)
            or self.rng_seed < 0
        ):
            raise ValueError("rng_seed must be a non-negative integer")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "task_key": self.task_key.to_dict(),
            "candidate_id": self.candidate_id,
            "hypothesis_labels": list(self.hypothesis_labels),
            "transition_matrix": [list(row) for row in self.transition_matrix],
            "outcomes": [item.to_dict() for item in self.outcomes],
            "critic_uncertainty": self.critic_uncertainty,
            "model_stamp": self.model_stamp.to_dict(),
            "rng_seed": self.rng_seed,
        }


def validate_forecast_coverage(
    candidates: CandidateSet, forecasts: tuple[CandidateEffectForecast, ...]
) -> None:
    expected = {
        item.candidate_id
        for item in candidates.candidates
        if item.requires_effect_forecast
    }
    observed = {item.candidate_id for item in forecasts}
    if expected != observed:
        raise ValueError(
            f"effect coverage mismatch; missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )
    if len(observed) != len(forecasts):
        raise ValueError("effect forecasts must have unique candidate IDs")
    for forecast in forecasts:
        if forecast.task_key != candidates.task_key:
            raise ValueError("effect forecast belongs to a different task")


@dataclass(frozen=True)
class CounterfactualOutcome:
    outcome_id: str
    probability: float
    posterior_probabilities: tuple[float, ...]
    terminal_bayes_risk: float
    predicted_sufficiency_mean: float
    action_cost: float
    physical_risk: float
    disturbance: float
    execution_status: ExecutionStatus
    resolves_need_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.outcome_id.strip():
            raise ValueError("counterfactual outcome_id must be non-empty")
        probability = _finite_scalar(self.probability, "probability")
        if any(
            isinstance(value, bool | np.bool_)
            for value in self.posterior_probabilities
        ):
            raise TypeError("counterfactual posterior must be numeric, not boolean")
        posterior = np.asarray(self.posterior_probabilities, dtype=np.float64)
        if not isfinite(probability) or not 0.0 < probability <= 1.0:
            raise ValueError("counterfactual outcome probability must lie in (0, 1]")
        if (
            posterior.ndim != 1
            or posterior.size < 2
            or not np.all(np.isfinite(posterior))
            or np.any(posterior < 0.0)
            or not np.isclose(posterior.sum(), 1.0, rtol=0.0, atol=1e-9)
        ):
            raise ValueError("counterfactual posterior must be a finite distribution")
        for name in ("terminal_bayes_risk", "action_cost", "physical_risk", "disturbance"):
            _nonnegative(getattr(self, name), name)
        predicted_sufficiency = _finite_scalar(
            self.predicted_sufficiency_mean, "predicted_sufficiency_mean"
        )
        if not 0.0 <= predicted_sufficiency <= 1.0:
            raise ValueError("predicted_sufficiency_mean must lie in [0, 1]")
        if not isinstance(self.execution_status, ExecutionStatus):
            raise TypeError("execution_status must be an ExecutionStatus")
        if any(not item.strip() for item in self.resolves_need_ids) or len(
            set(self.resolves_need_ids)
        ) != len(self.resolves_need_ids):
            raise ValueError("resolved need IDs must be non-empty and unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "outcome_id": self.outcome_id,
            "probability": self.probability,
            "posterior_probabilities": list(self.posterior_probabilities),
            "terminal_bayes_risk": self.terminal_bayes_risk,
            "predicted_sufficiency_mean": self.predicted_sufficiency_mean,
            "action_cost": self.action_cost,
            "physical_risk": self.physical_risk,
            "disturbance": self.disturbance,
            "execution_status": self.execution_status.value,
            "resolves_need_ids": list(self.resolves_need_ids),
        }


@dataclass(frozen=True)
class CounterfactualRollout:
    candidate_id: str
    decision_rule: str
    current_bayes_risk: float
    current_decision_risk: float
    decision_commitment_penalty: float
    transition_predictive_risk: float
    predictive_probabilities: tuple[float, ...]
    outcomes: tuple[CounterfactualOutcome, ...]
    expected_posterior_risk: float
    physical_progress_value: float
    conditional_information_value: float
    total_task_risk_reduction: float
    critic_uncertainty: float

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or not self.outcomes:
            raise ValueError("rollout candidate_id and outcomes must be non-empty")
        if self.decision_rule not in {"BAYES_AFTER_OBSERVATION", "FIXED_DIRECT_ACT"}:
            raise ValueError("unsupported rollout decision_rule")
        if any(
            isinstance(value, bool | np.bool_)
            for value in self.predictive_probabilities
        ):
            raise TypeError("rollout predictive probabilities must be numeric")
        predictive = np.asarray(self.predictive_probabilities, dtype=np.float64)
        if (
            predictive.ndim != 1
            or predictive.size < 2
            or not np.all(np.isfinite(predictive))
            or np.any(predictive < 0.0)
            or not np.isclose(predictive.sum(), 1.0, rtol=0.0, atol=1e-9)
        ):
            raise ValueError("rollout predictive probabilities must form a distribution")
        if any(
            len(item.posterior_probabilities) != predictive.size for item in self.outcomes
        ):
            raise ValueError("rollout posterior dimensions must match predictive belief")
        if len({item.outcome_id for item in self.outcomes}) != len(self.outcomes):
            raise ValueError("rollout outcome IDs must be unique")
        if not np.isclose(
            sum(item.probability for item in self.outcomes),
            1.0,
            rtol=0.0,
            atol=1e-8,
        ):
            raise ValueError("rollout outcome probabilities must sum to one")
        for name in (
            "current_bayes_risk",
            "current_decision_risk",
            "decision_commitment_penalty",
            "transition_predictive_risk",
            "expected_posterior_risk",
        ):
            _nonnegative(getattr(self, name), name)
        for name in (
            "physical_progress_value",
            "conditional_information_value",
            "total_task_risk_reduction",
        ):
            _finite_scalar(getattr(self, name), name)
        critic_uncertainty = _finite_scalar(
            self.critic_uncertainty, "critic_uncertainty"
        )
        if not 0.0 <= critic_uncertainty <= 1.0:
            raise ValueError("critic_uncertainty must lie in [0, 1]")
        expected_risk = sum(
            item.probability * item.terminal_bayes_risk for item in self.outcomes
        )
        if not np.isclose(
            expected_risk, self.expected_posterior_risk, rtol=0.0, atol=1e-9
        ):
            raise ValueError("rollout expected risk does not match its outcome mixture")
        if self.posterior_martingale_l1 > 1e-8:
            raise ValueError("rollout violates the posterior martingale identity")
        expected_commitment_penalty = self.current_decision_risk - self.current_bayes_risk
        if not np.isclose(
            expected_commitment_penalty,
            self.decision_commitment_penalty,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError("rollout decision commitment penalty is inconsistent")
        decomposed_value = (
            self.physical_progress_value
            + self.conditional_information_value
            - self.decision_commitment_penalty
        )
        if not np.isclose(
            decomposed_value,
            self.total_task_risk_reduction,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError("rollout task-risk decomposition is inconsistent")

    @property
    def posterior_martingale_l1(self) -> float:
        mixture = sum(
            item.probability * np.asarray(item.posterior_probabilities, dtype=np.float64)
            for item in self.outcomes
        )
        return float(
            np.abs(mixture - np.asarray(self.predictive_probabilities, dtype=np.float64)).sum()
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "decision_rule": self.decision_rule,
            "current_bayes_risk": self.current_bayes_risk,
            "current_decision_risk": self.current_decision_risk,
            "decision_commitment_penalty": self.decision_commitment_penalty,
            "transition_predictive_risk": self.transition_predictive_risk,
            "predictive_probabilities": list(self.predictive_probabilities),
            "outcomes": [item.to_dict() for item in self.outcomes],
            "expected_posterior_risk": self.expected_posterior_risk,
            "physical_progress_value": self.physical_progress_value,
            "conditional_information_value": self.conditional_information_value,
            "total_task_risk_reduction": self.total_task_risk_reduction,
            "critic_uncertainty": self.critic_uncertainty,
            "posterior_martingale_l1": self.posterior_martingale_l1,
        }


def rollout_forecast(
    *,
    task: TaskSpec,
    belief: BeliefState,
    candidate: PrimitiveCall,
    forecast: CandidateEffectForecast,
) -> CounterfactualRollout:
    if not candidate.requires_effect_forecast:
        raise ValueError("STOP_NOT_FOUND does not use an action-effect forecast")
    if candidate.candidate_id != forecast.candidate_id:
        raise ValueError("candidate and effect forecast IDs do not match")
    if task.key != belief.task_key or task.key != forecast.task_key:
        raise ValueError("task, belief, and forecast keys do not match")
    if forecast.hypothesis_labels != task.hypotheses:
        raise ValueError("effect forecast ontology does not match TaskSpec")
    addressed = set(candidate.addresses_need_ids)
    unexpected_resolutions = {
        need_id
        for outcome in forecast.outcomes
        for need_id in outcome.resolves_need_ids
        if need_id not in addressed
    }
    if unexpected_resolutions:
        raise ValueError(
            "effect forecast claims needs not addressed by candidate: "
            f"{sorted(unexpected_resolutions)}"
        )

    current = belief.probabilities
    transition = np.asarray(forecast.transition_matrix, dtype=np.float64)
    predictive = transition.T @ current
    _, current_risk = task.bayes_risk(current)
    fixed_decision = candidate.terminal_decision
    if fixed_decision is None:
        current_decision_risk = current_risk
        _, predictive_risk = task.bayes_risk(predictive)
        decision_rule = "BAYES_AFTER_OBSERVATION"
    else:
        current_decision_risk = task.risk_of(fixed_decision, current)
        predictive_risk = task.risk_of(fixed_decision, predictive)
        decision_rule = "FIXED_DIRECT_ACT"
    outcomes: list[CounterfactualOutcome] = []
    for modeled in forecast.outcomes:
        likelihood = np.asarray(modeled.likelihood_by_post_state, dtype=np.float64)
        probability = float(likelihood @ predictive)
        if probability <= 0.0:
            continue
        posterior = likelihood * predictive / probability
        if fixed_decision is None:
            _, posterior_risk = task.bayes_risk(posterior)
        else:
            posterior_risk = task.risk_of(fixed_decision, posterior)
        sufficiency_belief = belief.posterior.sufficiency
        prior_weight = sufficiency_belief.prior_weight
        base_rate = sufficiency_belief.base_rate
        sufficiency_alpha = (
            prior_weight * base_rate + modeled.sufficiency_evidence[0]
        )
        sufficiency_beta = (
            prior_weight * (1.0 - base_rate) + modeled.sufficiency_evidence[1]
        )
        outcomes.append(
            CounterfactualOutcome(
                outcome_id=modeled.outcome_id,
                probability=probability,
                posterior_probabilities=tuple(float(v) for v in posterior),
                terminal_bayes_risk=posterior_risk,
                predicted_sufficiency_mean=(
                    sufficiency_alpha / (sufficiency_alpha + sufficiency_beta)
                ),
                action_cost=modeled.action_cost,
                physical_risk=modeled.physical_risk,
                disturbance=modeled.disturbance,
                execution_status=modeled.execution_status,
                resolves_need_ids=modeled.resolves_need_ids,
            )
        )
    if not np.isclose(
        sum(item.probability for item in outcomes), 1.0, rtol=0.0, atol=1e-8
    ):
        raise AssertionError("derived outcome probabilities must sum to one")
    expected_posterior_risk = sum(
        item.probability * item.terminal_bayes_risk for item in outcomes
    )
    physical_value = current_decision_risk - predictive_risk
    information_value = predictive_risk - expected_posterior_risk
    return CounterfactualRollout(
        candidate_id=candidate.candidate_id,
        decision_rule=decision_rule,
        current_bayes_risk=current_risk,
        current_decision_risk=current_decision_risk,
        decision_commitment_penalty=current_decision_risk - current_risk,
        transition_predictive_risk=predictive_risk,
        predictive_probabilities=tuple(float(v) for v in predictive),
        outcomes=tuple(outcomes),
        expected_posterior_risk=expected_posterior_risk,
        physical_progress_value=physical_value,
        conditional_information_value=information_value,
        total_task_risk_reduction=current_risk - expected_posterior_risk,
        critic_uncertainty=forecast.critic_uncertainty,
    )
