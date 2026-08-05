"""Versioned, prompt-conditioned task contracts for the v0.2 bridge.

The prompt defines which latent hypotheses matter and the loss of terminal
decisions.  It does not modify the physical transition model and must not
contain an exploration recipe.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import isfinite

import numpy as np

from ..observation import ensure_final_goal_prompt
from .validation import (
    finite_number,
    require_exact_keys,
    strict_array,
    strict_string,
)


def canonical_digest(payload: object) -> str:
    """Return a stable SHA-256 digest for a JSON-serializable contract."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class TerminalDecision(str, Enum):
    """Terminal semantic decisions; primitive bindings are not free strings."""

    DIRECT_ACT = "DIRECT_ACT"
    NOT_FOUND = "NOT_FOUND"


@dataclass(frozen=True)
class TaskKey:
    task_id: str
    prompt_digest: str
    ontology_id: str
    ontology_version: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str)
            for value in (
                self.task_id,
                self.prompt_digest,
                self.ontology_id,
                self.ontology_version,
            )
        ):
            raise TypeError("TaskKey fields must be strings")
        if not all(
            value.strip()
            for value in (
                self.task_id,
                self.prompt_digest,
                self.ontology_id,
                self.ontology_version,
            )
        ):
            raise ValueError("TaskKey fields must be non-empty")
        if re.fullmatch(r"[0-9a-f]{64}", self.prompt_digest) is None:
            raise ValueError("prompt_digest must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, str]:
        return {
            "task_id": self.task_id,
            "prompt_digest": self.prompt_digest,
            "ontology_id": self.ontology_id,
            "ontology_version": self.ontology_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TaskKey:
        if not isinstance(payload, Mapping):
            raise TypeError("TaskKey must be a mapping")
        require_exact_keys(
            payload,
            required=frozenset(
                {"task_id", "prompt_digest", "ontology_id", "ontology_version"}
            ),
            location="TaskKey",
        )
        return cls(
            task_id=strict_string(payload["task_id"], location="TaskKey.task_id"),
            prompt_digest=strict_string(
                payload["prompt_digest"], location="TaskKey.prompt_digest"
            ),
            ontology_id=strict_string(
                payload["ontology_id"], location="TaskKey.ontology_id"
            ),
            ontology_version=strict_string(
                payload["ontology_version"], location="TaskKey.ontology_version"
            ),
        )


@dataclass(frozen=True)
class TaskSpec:
    """Finite prompt-specific decision problem used by evidence and routing."""

    key: TaskKey
    final_goal_prompt: str
    hypotheses: tuple[str, ...]
    terminal_decisions: tuple[TerminalDecision, ...]
    loss_matrix: tuple[tuple[float, ...], ...]
    base_rate: tuple[float, ...] | None = None
    required_attributes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.key, TaskKey):
            raise TypeError("key must be a TaskKey")
        ensure_final_goal_prompt(self.final_goal_prompt)
        expected_prompt_digest = canonical_digest(self.final_goal_prompt)
        if self.key.prompt_digest != expected_prompt_digest:
            raise ValueError("TaskKey prompt_digest does not match final_goal_prompt")
        if len(self.hypotheses) < 2 or len(set(self.hypotheses)) != len(self.hypotheses):
            raise ValueError("hypotheses must contain at least two unique labels")
        if any(not isinstance(label, str) for label in self.hypotheses):
            raise TypeError("hypothesis labels must be strings")
        if any(not label.strip() for label in self.hypotheses):
            raise ValueError("hypothesis labels must be non-empty")
        if not self.terminal_decisions:
            raise ValueError("at least one terminal decision is required")
        if len(set(self.terminal_decisions)) != len(self.terminal_decisions):
            raise ValueError("terminal decisions must be unique")
        if any(
            not isinstance(decision, TerminalDecision)
            for decision in self.terminal_decisions
        ):
            raise TypeError("terminal decisions must be TerminalDecision values")
        if self.terminal_decisions != (
            TerminalDecision.DIRECT_ACT,
            TerminalDecision.NOT_FOUND,
        ):
            raise ValueError(
                "v0.2 terminal decisions must be ordered as DIRECT_ACT, NOT_FOUND"
            )
        if any(
            isinstance(value, bool | np.bool_ | str | bytes | bytearray)
            for row in self.loss_matrix
            for value in row
        ):
            raise TypeError("loss_matrix values must be numeric scalars")
        matrix = np.asarray(self.loss_matrix, dtype=np.float64)
        if matrix.shape != (len(self.terminal_decisions), len(self.hypotheses)):
            raise ValueError("loss_matrix must have shape [decisions, hypotheses]")
        if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
            raise ValueError("loss_matrix must be finite and non-negative")
        if self.base_rate is None:
            base_rate = tuple(1.0 / len(self.hypotheses) for _ in self.hypotheses)
        else:
            if any(
                isinstance(value, bool | np.bool_ | str | bytes | bytearray)
                for value in self.base_rate
            ):
                raise TypeError("base_rate values must be numeric scalars")
            base_rate = tuple(float(value) for value in self.base_rate)
            if len(base_rate) != len(self.hypotheses):
                raise ValueError("base_rate must align with hypotheses")
            if any(not isfinite(value) or value <= 0.0 for value in base_rate):
                raise ValueError("base_rate must be finite and strictly positive")
            if not np.isclose(sum(base_rate), 1.0, rtol=0.0, atol=1e-9):
                raise ValueError("base_rate must sum to one")
        object.__setattr__(self, "base_rate", base_rate)
        if any(not isinstance(item, str) for item in self.required_attributes):
            raise TypeError("required_attributes must contain strings")
        attributes = tuple(self.required_attributes)
        if any(not item.strip() for item in attributes) or len(set(attributes)) != len(
            attributes
        ):
            raise ValueError("required_attributes must be non-empty and unique")
        object.__setattr__(self, "required_attributes", attributes)

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        final_goal_prompt: str,
        ontology_id: str,
        ontology_version: str,
        hypotheses: Sequence[str],
        terminal_decisions: Sequence[TerminalDecision],
        loss_matrix: Sequence[Sequence[float]],
        base_rate: Sequence[float] | None = None,
        required_attributes: Sequence[str] = (),
    ) -> TaskSpec:
        return cls(
            key=TaskKey(
                task_id=task_id,
                prompt_digest=canonical_digest(final_goal_prompt),
                ontology_id=ontology_id,
                ontology_version=ontology_version,
            ),
            final_goal_prompt=final_goal_prompt,
            hypotheses=tuple(hypotheses),
            terminal_decisions=tuple(terminal_decisions),
            loss_matrix=tuple(
                tuple(
                    finite_number(v, location=f"loss_matrix[{i}][{j}]", minimum=0.0)
                    for j, v in enumerate(row)
                )
                for i, row in enumerate(loss_matrix)
            ),
            base_rate=(
                None
                if base_rate is None
                else tuple(
                    finite_number(
                        v,
                        location=f"base_rate[{i}]",
                        minimum=0.0,
                        maximum=1.0,
                        exclusive_minimum=True,
                    )
                    for i, v in enumerate(base_rate)
                )
            ),
            required_attributes=tuple(required_attributes),
        )

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(self.loss_matrix, dtype=np.float64)

    def validate_probabilities(self, probabilities: Sequence[float]) -> np.ndarray:
        vector = np.asarray(probabilities, dtype=np.float64)
        if any(isinstance(value, bool | np.bool_) for value in probabilities):
            raise TypeError("probabilities must be numeric, not boolean")
        if vector.shape != (len(self.hypotheses),):
            raise ValueError("probability vector must align with task hypotheses")
        if not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
            raise ValueError("probabilities must be finite and non-negative")
        if not np.isclose(vector.sum(), 1.0, rtol=0.0, atol=1e-9):
            raise ValueError("probabilities must sum to one")
        return vector

    def decision_risks(self, probabilities: Sequence[float]) -> dict[TerminalDecision, float]:
        vector = self.validate_probabilities(probabilities)
        risks = self.matrix @ vector
        return dict(zip(self.terminal_decisions, risks.tolist(), strict=True))

    def risk_of(self, decision: TerminalDecision, probabilities: Sequence[float]) -> float:
        try:
            return self.decision_risks(probabilities)[decision]
        except KeyError as exc:
            raise KeyError(f"decision {decision.value!r} is not part of this task") from exc

    def bayes_risk(
        self, probabilities: Sequence[float]
    ) -> tuple[TerminalDecision, float]:
        risks = self.decision_risks(probabilities)
        decision = min(risks, key=lambda item: (risks[item], item.value))
        return decision, risks[decision]

    def perfect_information_risk(self, probabilities: Sequence[float]) -> float:
        """Expected risk if the hypothesis were revealed before committing."""

        vector = self.validate_probabilities(probabilities)
        return float(vector @ self.matrix.min(axis=0))

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key.to_dict(),
            "final_goal_prompt": self.final_goal_prompt,
            "hypotheses": list(self.hypotheses),
            "terminal_decisions": [item.value for item in self.terminal_decisions],
            "loss_matrix": [list(row) for row in self.loss_matrix],
            "base_rate": list(self.base_rate or ()),
            "required_attributes": list(self.required_attributes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TaskSpec:
        require_exact_keys(
            payload,
            required=frozenset(
                {
                    "key",
                    "final_goal_prompt",
                    "hypotheses",
                    "terminal_decisions",
                    "loss_matrix",
                    "base_rate",
                    "required_attributes",
                }
            ),
            location="TaskSpec",
        )
        key_payload = payload["key"]
        if not isinstance(key_payload, Mapping):
            raise TypeError("task key must be a mapping")
        raw_hypotheses = strict_array(
            payload["hypotheses"], location="TaskSpec.hypotheses"
        )
        raw_decisions = strict_array(
            payload["terminal_decisions"], location="TaskSpec.terminal_decisions"
        )
        raw_loss_matrix = strict_array(
            payload["loss_matrix"], location="TaskSpec.loss_matrix"
        )
        loss_rows = tuple(
            strict_array(row, location=f"TaskSpec.loss_matrix[{index}]")
            for index, row in enumerate(raw_loss_matrix)
        )
        raw_base_rate = strict_array(
            payload["base_rate"], location="TaskSpec.base_rate"
        )
        raw_attributes = strict_array(
            payload["required_attributes"], location="TaskSpec.required_attributes"
        )
        return cls(
            key=TaskKey.from_dict(key_payload),
            final_goal_prompt=strict_string(
                payload["final_goal_prompt"], location="TaskSpec.final_goal_prompt"
            ),
            hypotheses=tuple(
                strict_string(value, location=f"TaskSpec.hypotheses[{index}]")
                for index, value in enumerate(raw_hypotheses)
            ),
            terminal_decisions=tuple(
                TerminalDecision(
                    strict_string(
                        value, location=f"TaskSpec.terminal_decisions[{index}]"
                    )
                )
                for index, value in enumerate(raw_decisions)
            ),
            loss_matrix=tuple(
                tuple(
                    finite_number(
                        v,
                        location=f"TaskSpec.loss_matrix[{i}][{j}]",
                        minimum=0.0,
                    )
                    for j, v in enumerate(row)
                )
                for i, row in enumerate(loss_rows)
            ),
            base_rate=tuple(
                finite_number(
                    v,
                    location=f"TaskSpec.base_rate[{i}]",
                    minimum=0.0,
                    maximum=1.0,
                    exclusive_minimum=True,
                )
                for i, v in enumerate(raw_base_rate)
            )
            or None,
            required_attributes=tuple(
                strict_string(
                    value, location=f"TaskSpec.required_attributes[{index}]"
                )
                for index, value in enumerate(raw_attributes)
            ),
        )
