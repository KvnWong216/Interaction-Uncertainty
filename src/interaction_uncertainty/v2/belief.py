"""Stateful evidential filtering with explicit correlation and deduplication."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from math import isfinite

import numpy as np

from ..beliefs import BetaBelief, DirichletBelief, TaskBelief
from .evidence import EvidencePacket
from .task import TaskKey, TaskSpec, canonical_digest
from .validation import finite_number


class FilterMode(str, Enum):
    REPLACE = "REPLACE"
    DISCOUNTED_EVIDENCE = "DISCOUNTED_EVIDENCE"


@dataclass(frozen=True)
class BeliefState:
    task_key: TaskKey
    episode_id: str
    step_index: int
    posterior: TaskBelief
    observation_digest: str
    accepted_event_ids: tuple[str, ...]
    correlation_group: str
    filter_mode: FilterMode
    history_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.task_key, TaskKey):
            raise TypeError("task_key must be a TaskKey")
        if not isinstance(self.posterior, TaskBelief):
            raise TypeError("posterior must be a TaskBelief")
        if not isinstance(self.filter_mode, FilterMode):
            raise TypeError("filter_mode must be a FilterMode")
        if (
            not isinstance(self.episode_id, str)
            or isinstance(self.step_index, bool)
            or not isinstance(self.step_index, int)
            or self.step_index < 0
            or not self.episode_id.strip()
        ):
            raise ValueError("episode_id must be non-empty and step_index non-negative")
        if not isinstance(self.observation_digest, str) or re.fullmatch(
            r"[0-9a-f]{64}", self.observation_digest
        ) is None:
            raise ValueError("observation_digest must be a lowercase SHA-256 digest")
        if not isinstance(self.history_digest, str) or re.fullmatch(
            r"[0-9a-f]{64}", self.history_digest
        ) is None:
            raise ValueError("history_digest must be a lowercase SHA-256 digest")
        if any(
            not isinstance(event_id, str)
            or re.fullmatch(r"[0-9a-f]{64}", event_id) is None
            for event_id in self.accepted_event_ids
        ):
            raise ValueError("accepted_event_ids must contain SHA-256 digests")
        if not isinstance(self.correlation_group, str) or not self.correlation_group.strip():
            raise ValueError("correlation_group must be non-empty")
        if self.posterior.prompt == "":
            raise ValueError("posterior prompt and observation_digest must be non-empty")
        if len(set(self.accepted_event_ids)) != len(self.accepted_event_ids):
            raise ValueError("accepted evidence event IDs must be unique")

    @property
    def probabilities(self) -> np.ndarray:
        return self.posterior.hypotheses.mean_vector

    def to_dict(self) -> dict[str, object]:
        return {
            "task_key": self.task_key.to_dict(),
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "posterior": self.posterior.to_dict(),
            "observation_digest": self.observation_digest,
            "accepted_event_ids": list(self.accepted_event_ids),
            "correlation_group": self.correlation_group,
            "filter_mode": self.filter_mode.value,
            "history_digest": self.history_digest,
        }


@dataclass(frozen=True)
class EvidentialBeliefFilter:
    """Convert model evidence into belief state without naive frame counting.

    ``REPLACE`` is the production-safe default for highly correlated video
    frames.  ``DISCOUNTED_EVIDENCE`` is an explicit heuristic for sequential
    neural pseudo-evidence; it is not advertised as exact conjugate Bayes.
    """

    mode: FilterMode = FilterMode.REPLACE
    prior_weight: float | None = None
    sufficiency_prior_weight: float = 2.0
    retention: float = 0.0
    same_group_discount: float = 0.25

    def __post_init__(self) -> None:
        if not isinstance(self.mode, FilterMode):
            raise TypeError("mode must be a FilterMode")
        for name in (
            "prior_weight",
            "sufficiency_prior_weight",
            "retention",
            "same_group_discount",
        ):
            value = getattr(self, name)
            if isinstance(value, bool | np.bool_):
                raise TypeError(f"{name} must be numeric, not boolean")
        if self.prior_weight is not None:
            finite_number(
                self.prior_weight,
                location="prior_weight",
                minimum=0.0,
                exclusive_minimum=True,
            )
        if self.prior_weight is not None and (
            not isfinite(self.prior_weight) or self.prior_weight <= 0.0
        ):
            raise ValueError("prior_weight must be finite and positive")
        finite_number(
            self.sufficiency_prior_weight,
            location="sufficiency_prior_weight",
            minimum=0.0,
            exclusive_minimum=True,
        )
        finite_number(self.retention, location="retention", minimum=0.0, maximum=1.0)
        finite_number(
            self.same_group_discount,
            location="same_group_discount",
            minimum=0.0,
            maximum=1.0,
        )
        if not isfinite(self.sufficiency_prior_weight) or self.sufficiency_prior_weight <= 0:
            raise ValueError("sufficiency_prior_weight must be finite and positive")
        if not 0.0 <= self.retention <= 1.0:
            raise ValueError("retention must lie in [0, 1]")
        if not 0.0 <= self.same_group_discount <= 1.0:
            raise ValueError("same_group_discount must lie in [0, 1]")

    def _weight(self, task: TaskSpec) -> float:
        return float(len(task.hypotheses)) if self.prior_weight is None else self.prior_weight

    def update(
        self,
        *,
        task: TaskSpec,
        episode_id: str,
        step_index: int,
        evidence: EvidencePacket,
        previous: BeliefState | None = None,
    ) -> BeliefState:
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id must be a non-empty string")
        if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
            raise ValueError("step_index must be a non-negative integer")
        if not isinstance(evidence, EvidencePacket):
            raise TypeError("evidence must be an EvidencePacket")
        if evidence.task_key != task.key or evidence.hypothesis_labels != task.hypotheses:
            raise ValueError("evidence and TaskSpec do not match")
        if previous is not None:
            if previous.task_key != task.key or previous.episode_id != episode_id:
                raise ValueError("cannot carry belief across tasks or episodes")
            if evidence.event_id in previous.accepted_event_ids:
                raise ValueError("duplicate evidence event; pseudo-count replay is forbidden")
            if step_index <= previous.step_index:
                raise ValueError("belief updates must have strictly increasing step_index")

        new_hypothesis_evidence = np.asarray(evidence.hypothesis_evidence, dtype=np.float64)
        new_sufficiency_evidence = np.asarray(evidence.sufficiency_evidence, dtype=np.float64)
        if self.mode is FilterMode.REPLACE or previous is None:
            combined_hypothesis_evidence = new_hypothesis_evidence
            combined_sufficiency_evidence = new_sufficiency_evidence
        else:
            prior = previous.posterior.hypotheses
            prior_base = np.asarray(prior.base_rate, dtype=np.float64)
            old_hypothesis_evidence = (
                np.asarray(prior.alpha, dtype=np.float64)
                - float(prior.prior_weight) * prior_base
            )
            old_sufficiency_evidence = np.asarray(
                [
                    previous.posterior.sufficiency.alpha
                    - previous.posterior.sufficiency.prior_weight
                    * previous.posterior.sufficiency.base_rate,
                    previous.posterior.sufficiency.beta
                    - previous.posterior.sufficiency.prior_weight
                    * (1.0 - previous.posterior.sufficiency.base_rate),
                ],
                dtype=np.float64,
            )
            novelty = (
                self.same_group_discount
                if evidence.correlation_group == previous.correlation_group
                else 1.0
            )
            combined_hypothesis_evidence = (
                self.retention * old_hypothesis_evidence
                + novelty * new_hypothesis_evidence
            )
            combined_sufficiency_evidence = (
                self.retention * old_sufficiency_evidence
                + novelty * new_sufficiency_evidence
            )

        hypothesis_belief = DirichletBelief.from_evidence(
            task.hypotheses,
            combined_hypothesis_evidence,
            base_rate=task.base_rate,
            prior_weight=self._weight(task),
        )
        sufficiency_base_rate = 0.5
        sufficiency_prior = self.sufficiency_prior_weight * sufficiency_base_rate
        sufficiency = BetaBelief(
            sufficiency_prior + float(combined_sufficiency_evidence[0]),
            self.sufficiency_prior_weight * (1.0 - sufficiency_base_rate)
            + float(combined_sufficiency_evidence[1]),
            prior_weight=self.sufficiency_prior_weight,
            base_rate=sufficiency_base_rate,
        )
        posterior = TaskBelief(
            prompt=task.final_goal_prompt,
            hypotheses=hypothesis_belief,
            sufficiency=sufficiency,
            deficits=evidence.deficits,
            provenance=evidence.model_stamp.model_id,
            tags=(
                f"model_sha256:{evidence.model_stamp.model_sha256}",
                f"calibration:{evidence.model_stamp.calibration_id}",
                f"evidence_event:{evidence.event_id}",
            ),
        )
        accepted = (
            (evidence.event_id,)
            if previous is None
            else previous.accepted_event_ids + (evidence.event_id,)
        )
        history_digest = canonical_digest(
            {
                "previous": None if previous is None else previous.history_digest,
                "event_id": evidence.event_id,
                "posterior": posterior.to_dict(),
                "mode": self.mode.value,
            }
        )
        return BeliefState(
            task_key=task.key,
            episode_id=episode_id,
            step_index=step_index,
            posterior=posterior,
            observation_digest=evidence.observation_digest,
            accepted_event_ids=accepted,
            correlation_group=evidence.correlation_group,
            filter_mode=self.mode,
            history_digest=history_digest,
        )
