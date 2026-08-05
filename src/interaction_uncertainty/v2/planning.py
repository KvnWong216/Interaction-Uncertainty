"""One-step task-risk reranking over terminal and information primitives."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite

from .belief import BeliefState
from .effects import CounterfactualRollout
from .primitives import CandidateSet, PrimitiveCall, PrimitiveKind
from .task import TaskSpec
from .validation import finite_number


@dataclass(frozen=True)
class PlanningWeights:
    action_cost: float = 0.25
    physical_risk: float = 0.5
    disturbance: float = 0.25
    critic_uncertainty: float = 0.25
    information_insufficiency: float = 0.25

    def __post_init__(self) -> None:
        for name in (
            "action_cost",
            "physical_risk",
            "disturbance",
            "critic_uncertainty",
            "information_insufficiency",
        ):
            value = finite_number(getattr(self, name), location=name, minimum=0.0)
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class CandidateValue:
    candidate_id: str
    expected_task_risk: float
    expected_action_cost: float
    expected_physical_risk: float
    expected_disturbance: float
    critic_uncertainty_penalty: float
    expected_information_insufficiency: float
    objective: float
    current_bayes_risk: float
    current_decision_risk: float
    decision_commitment_penalty: float
    physical_progress_value: float
    conditional_information_value: float
    total_task_risk_reduction: float

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")
        nonnegative = (
            "expected_task_risk",
            "expected_action_cost",
            "expected_physical_risk",
            "expected_disturbance",
            "critic_uncertainty_penalty",
            "objective",
            "current_bayes_risk",
            "current_decision_risk",
            "decision_commitment_penalty",
        )
        for name in nonnegative:
            value = finite_number(getattr(self, name), location=name, minimum=0.0)
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        insufficiency = finite_number(
            self.expected_information_insufficiency,
            location="expected_information_insufficiency",
            minimum=0.0,
            maximum=1.0,
        )
        if not 0.0 <= insufficiency <= 1.0:
            raise ValueError("expected_information_insufficiency must lie in [0, 1]")
        for name in (
            "physical_progress_value",
            "conditional_information_value",
            "total_task_risk_reduction",
        ):
            finite_number(getattr(self, name), location=name)
        if not isclose(
            self.current_decision_risk - self.current_bayes_risk,
            self.decision_commitment_penalty,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("candidate commitment penalty is inconsistent")
        if not isclose(
            self.physical_progress_value
            + self.conditional_information_value
            - self.decision_commitment_penalty,
            self.total_task_risk_reduction,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("candidate task-risk decomposition is inconsistent")

    def to_dict(self) -> dict[str, float | str]:
        return {
            "candidate_id": self.candidate_id,
            "expected_task_risk": self.expected_task_risk,
            "expected_action_cost": self.expected_action_cost,
            "expected_physical_risk": self.expected_physical_risk,
            "expected_disturbance": self.expected_disturbance,
            "critic_uncertainty_penalty": self.critic_uncertainty_penalty,
            "expected_information_insufficiency": self.expected_information_insufficiency,
            "objective": self.objective,
            "current_bayes_risk": self.current_bayes_risk,
            "current_decision_risk": self.current_decision_risk,
            "decision_commitment_penalty": self.decision_commitment_penalty,
            "physical_progress_value": self.physical_progress_value,
            "conditional_information_value": self.conditional_information_value,
            "total_task_risk_reduction": self.total_task_risk_reduction,
        }


@dataclass(frozen=True)
class PrimitiveDecision:
    selected: PrimitiveCall
    ranking: tuple[CandidateValue, ...]
    runner_up_id: str | None
    margin: float
    unstable_tie: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "selected": self.selected.to_dict(),
            "ranking": [item.to_dict() for item in self.ranking],
            "runner_up_id": self.runner_up_id,
            "margin": self.margin,
            "unstable_tie": self.unstable_tie,
        }


def _terminal_value(
    task: TaskSpec,
    belief: BeliefState,
    candidate: PrimitiveCall,
    weights: PlanningWeights,
) -> CandidateValue:
    decision = candidate.terminal_decision
    if decision is None:
        raise ValueError("terminal candidate is missing its typed decision")
    probabilities = belief.probabilities
    _, current_risk = task.bayes_risk(probabilities)
    terminal_risk = task.risk_of(decision, probabilities)
    insufficiency = 1.0 - belief.posterior.sufficiency.mean
    return CandidateValue(
        candidate_id=candidate.candidate_id,
        expected_task_risk=terminal_risk,
        expected_action_cost=0.0,
        expected_physical_risk=0.0,
        expected_disturbance=0.0,
        critic_uncertainty_penalty=0.0,
        expected_information_insufficiency=insufficiency,
        objective=terminal_risk + weights.information_insufficiency * insufficiency,
        current_bayes_risk=current_risk,
        current_decision_risk=terminal_risk,
        decision_commitment_penalty=terminal_risk - current_risk,
        physical_progress_value=0.0,
        conditional_information_value=0.0,
        total_task_risk_reduction=current_risk - terminal_risk,
    )


def _rollout_value(
    rollout: CounterfactualRollout, weights: PlanningWeights
) -> CandidateValue:
    expected_cost = sum(item.probability * item.action_cost for item in rollout.outcomes)
    expected_risk = sum(item.probability * item.physical_risk for item in rollout.outcomes)
    expected_disturbance = sum(
        item.probability * item.disturbance for item in rollout.outcomes
    )
    critic_penalty = weights.critic_uncertainty * rollout.critic_uncertainty
    expected_insufficiency = sum(
        item.probability * (1.0 - item.predicted_sufficiency_mean)
        for item in rollout.outcomes
    )
    objective = (
        rollout.expected_posterior_risk
        + weights.action_cost * expected_cost
        + weights.physical_risk * expected_risk
        + weights.disturbance * expected_disturbance
        + critic_penalty
        + weights.information_insufficiency * expected_insufficiency
    )
    return CandidateValue(
        candidate_id=rollout.candidate_id,
        expected_task_risk=rollout.expected_posterior_risk,
        expected_action_cost=expected_cost,
        expected_physical_risk=expected_risk,
        expected_disturbance=expected_disturbance,
        critic_uncertainty_penalty=critic_penalty,
        expected_information_insufficiency=expected_insufficiency,
        objective=objective,
        current_bayes_risk=rollout.current_bayes_risk,
        current_decision_risk=rollout.current_decision_risk,
        decision_commitment_penalty=rollout.decision_commitment_penalty,
        physical_progress_value=rollout.physical_progress_value,
        conditional_information_value=rollout.conditional_information_value,
        total_task_risk_reduction=rollout.total_task_risk_reduction,
    )


def rank_primitives(
    *,
    task: TaskSpec,
    belief: BeliefState,
    candidates: CandidateSet,
    rollouts: tuple[CounterfactualRollout, ...],
    weights: PlanningWeights | None = None,
    tie_tolerance: float = 1e-8,
) -> PrimitiveDecision:
    if task.key != belief.task_key or task.key != candidates.task_key:
        raise ValueError("task, belief, and candidate set do not match")
    tie_tolerance = finite_number(
        tie_tolerance, location="tie_tolerance", minimum=0.0
    )
    if not isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("tie_tolerance must be finite and non-negative")
    weights = PlanningWeights() if weights is None else weights
    rollout_by_id = {item.candidate_id: item for item in rollouts}
    expected_forecasts = {
        item.candidate_id
        for item in candidates.candidates
        if item.requires_effect_forecast
    }
    if set(rollout_by_id) != expected_forecasts or len(rollout_by_id) != len(rollouts):
        raise ValueError("counterfactual rollout coverage does not match candidates")
    values: list[CandidateValue] = []
    by_id = {item.candidate_id: item for item in candidates.candidates}
    for candidate in candidates.candidates:
        values.append(
            _terminal_value(task, belief, candidate, weights)
            if candidate.kind is PrimitiveKind.STOP_NOT_FOUND
            else _rollout_value(rollout_by_id[candidate.candidate_id], weights)
        )
    ranking = tuple(sorted(values, key=lambda item: (item.objective, item.candidate_id)))
    runner_up = ranking[1] if len(ranking) > 1 else None
    margin = 0.0 if runner_up is None else runner_up.objective - ranking[0].objective
    return PrimitiveDecision(
        selected=by_id[ranking[0].candidate_id],
        ranking=ranking,
        runner_up_id=None if runner_up is None else runner_up.candidate_id,
        margin=margin,
        unstable_tie=runner_up is not None and margin <= tie_tolerance,
    )
