"""Prompt-specific conversion from belief uncertainty to information needs."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..beliefs import DeficitKind
from .belief import BeliefState
from .task import TaskKey, TaskSpec
from .validation import finite_number


@dataclass(frozen=True)
class InformationNeed:
    """Localized missing evidence, deliberately free of action labels."""

    need_id: str
    task_key: TaskKey
    proposition_id: str
    anchor_token: str
    deficit_kind: DeficitKind
    probability: float
    decision_relevance: float
    max_task_risk_reduction: float
    sufficiency_shortfall: float
    priority: float
    source_deficit_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        strings = (self.need_id, self.proposition_id, self.anchor_token)
        if any(not isinstance(value, str) for value in strings):
            raise TypeError("need identifiers and anchor_token must be strings")
        if not all(value.strip() for value in strings):
            raise ValueError("need identifiers and anchor_token must be non-empty")
        if not isinstance(self.task_key, TaskKey):
            raise TypeError("task_key must be a TaskKey")
        if not isinstance(self.deficit_kind, DeficitKind):
            raise TypeError("deficit_kind must be a DeficitKind")
        for name in (
            "probability",
            "decision_relevance",
            "max_task_risk_reduction",
            "sufficiency_shortfall",
            "priority",
        ):
            value = finite_number(getattr(self, name), location=name, minimum=0.0)
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.probability > 1.0 or self.decision_relevance > 1.0:
            raise ValueError("probability and decision_relevance must lie in [0, 1]")
        if self.sufficiency_shortfall > 1.0:
            raise ValueError("sufficiency_shortfall must lie in [0, 1]")
        if any(
            not isinstance(identifier, str) or not identifier.strip()
            for identifier in self.source_deficit_ids
        ):
            raise TypeError("source_deficit_ids must contain non-empty strings")
        if not self.source_deficit_ids or len(set(self.source_deficit_ids)) != len(
            self.source_deficit_ids
        ):
            raise ValueError("source_deficit_ids must be non-empty and unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "need_id": self.need_id,
            "task_key": self.task_key.to_dict(),
            "proposition_id": self.proposition_id,
            "anchor_token": self.anchor_token,
            "deficit_kind": self.deficit_kind.value,
            "probability": self.probability,
            "decision_relevance": self.decision_relevance,
            "max_task_risk_reduction": self.max_task_risk_reduction,
            "sufficiency_shortfall": self.sufficiency_shortfall,
            "priority": self.priority,
            "source_deficit_ids": list(self.source_deficit_ids),
        }


@dataclass(frozen=True)
class BayesRiskNeedExtractor:
    """Rank localized deficits by prompt-specific value of perfect information."""

    sufficiency_weight: float = 0.25
    minimum_priority: float = 0.0

    def __post_init__(self) -> None:
        sufficiency_weight = finite_number(
            self.sufficiency_weight,
            location="sufficiency_weight",
            minimum=0.0,
        )
        minimum_priority = finite_number(
            self.minimum_priority,
            location="minimum_priority",
            minimum=0.0,
        )
        if (
            not isfinite(sufficiency_weight)
            or not isfinite(minimum_priority)
            or sufficiency_weight < 0.0
            or minimum_priority < 0.0
        ):
            raise ValueError("need-extractor weights must be non-negative")

    def extract(
        self, task: TaskSpec, belief: BeliefState
    ) -> tuple[InformationNeed, ...]:
        if belief.task_key != task.key:
            raise ValueError("belief and task do not match")
        probabilities = belief.posterior.hypotheses.mean_vector
        _, current_risk = task.bayes_risk(probabilities)
        perfect_risk = task.perfect_information_risk(probabilities)
        max_risk_reduction = max(0.0, current_risk - perfect_risk)
        shortfall = max(0.0, 1.0 - belief.posterior.sufficiency.mean)
        needs: list[InformationNeed] = []
        for deficit in belief.posterior.deficits:
            probability = deficit.probability.mean
            relevance = deficit.prompt_relevance
            priority = probability * relevance * (
                max_risk_reduction + self.sufficiency_weight * shortfall
            )
            if priority < self.minimum_priority:
                continue
            needs.append(
                InformationNeed(
                    need_id=f"need::{deficit.deficit_id}",
                    task_key=task.key,
                    proposition_id=deficit.deficit_id,
                    anchor_token=deficit.anchor_token,
                    deficit_kind=deficit.kind,
                    probability=probability,
                    decision_relevance=relevance,
                    max_task_risk_reduction=max_risk_reduction,
                    sufficiency_shortfall=shortfall,
                    priority=priority,
                    source_deficit_ids=(deficit.deficit_id,),
                )
            )
        return tuple(sorted(needs, key=lambda item: (-item.priority, item.need_id)))
