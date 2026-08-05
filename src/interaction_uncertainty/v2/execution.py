"""Typed VLA/skill execution boundary.

The uncertainty bridge selects the high-level primitive.  A policy backend may
refine grounding and emit continuous action chunks, but it may not silently
replace the selected primitive family.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Protocol, runtime_checkable

import numpy as np

from ..firewall import PolicyFirewall
from ..observation import PolicyContext
from .belief import BeliefState
from .planning import PrimitiveDecision
from .primitives import PrimitiveCall, PrimitiveKind
from .task import TaskKey, canonical_digest
from .validation import (
    finite_number,
    require_exact_keys,
    strict_array,
    strict_integer,
    strict_string,
)


@dataclass(frozen=True)
class VLAExecutionRequest:
    schema_version: str
    execution_id: str
    task_key: TaskKey
    final_goal_prompt: str
    context: PolicyContext
    selected_primitive: PrimitiveCall
    belief_summary: Mapping[str, object]
    maximum_skill_steps: int
    timeout_s: float
    must_reobserve_after_nonterminal: bool

    def __post_init__(self) -> None:
        if self.schema_version != "interaction-uncertainty.execution-request.v2":
            raise ValueError("unsupported execution request schema")
        if not isinstance(self.execution_id, str) or re.fullmatch(
            r"[0-9a-f]{64}", self.execution_id
        ) is None:
            raise ValueError("execution_id must be a lowercase SHA-256 digest")
        if not isinstance(self.context, PolicyContext):
            raise TypeError("context must be a PolicyContext")
        if not isinstance(self.selected_primitive, PrimitiveCall):
            raise TypeError("selected_primitive must be a PrimitiveCall")
        if not isinstance(self.belief_summary, Mapping):
            raise TypeError("belief_summary must be a mapping")
        if self.task_key != self.selected_primitive.task_key:
            raise ValueError("selected primitive belongs to another task")
        if self.task_key.prompt_digest != canonical_digest(self.final_goal_prompt):
            raise ValueError("task key does not bind the execution prompt")
        if self.context.prompt != self.final_goal_prompt:
            raise ValueError("context prompt and final goal prompt must match")
        if (
            isinstance(self.maximum_skill_steps, bool)
            or not isinstance(self.maximum_skill_steps, int)
            or self.maximum_skill_steps <= 0
        ):
            raise ValueError("maximum_skill_steps must be a positive integer")
        if isinstance(self.timeout_s, bool | np.bool_) or not isfinite(
            self.timeout_s
        ) or self.timeout_s <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        if not isinstance(self.must_reobserve_after_nonterminal, bool):
            raise TypeError("must_reobserve_after_nonterminal must be a boolean")
        expected_reobservation = not self.selected_primitive.is_terminal
        if self.must_reobserve_after_nonterminal is not expected_reobservation:
            raise ValueError(
                "re-observation flag must be true exactly for nonterminal primitives"
            )
        if self.execution_id != canonical_digest(self.identity_payload()):
            raise ValueError("execution_id does not authenticate the exact command")

    def identity_payload(self) -> dict[str, object]:
        """All command fields except the digest that authenticates them."""

        return {
            "schema_version": self.schema_version,
            "task_key": self.task_key.to_dict(),
            "final_goal_prompt": self.final_goal_prompt,
            "context": self.context.to_dict(),
            "selected_primitive": self.selected_primitive.to_dict(),
            "belief_summary": dict(self.belief_summary),
            "maximum_skill_steps": self.maximum_skill_steps,
            "timeout_s": self.timeout_s,
            "must_reobserve_after_nonterminal": self.must_reobserve_after_nonterminal,
        }

    def to_dict(self) -> dict[str, object]:
        payload = {"execution_id": self.execution_id, **self.identity_payload()}
        PolicyFirewall().validate_recursive(payload, location="v2_execution_request")
        return payload


class ExecutionResult(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"
    TIMED_OUT = "TIMED_OUT"


@dataclass(frozen=True)
class ExecutionReport:
    schema_version: str
    execution_id: str
    candidate_id: str
    primitive_kind: str
    result: ExecutionResult
    executed_steps: int
    termination_reason: str
    next_public_observation_required: bool
    executor_id: str

    def __post_init__(self) -> None:
        if self.schema_version != "interaction-uncertainty.execution-report.v2":
            raise ValueError("unsupported execution report schema")
        if not isinstance(self.execution_id, str) or re.fullmatch(
            r"[0-9a-f]{64}", self.execution_id
        ) is None:
            raise ValueError("execution report ID must be a lowercase SHA-256 digest")
        strings = (
            self.execution_id,
            self.candidate_id,
            self.primitive_kind,
            self.termination_reason,
            self.executor_id,
        )
        if any(not isinstance(value, str) for value in strings):
            raise TypeError("execution report text fields must be strings")
        if not all(value.strip() for value in strings):
            raise ValueError("execution report strings must be non-empty")
        if not isinstance(self.result, ExecutionResult):
            raise TypeError("result must be an ExecutionResult")
        if not isinstance(self.next_public_observation_required, bool):
            raise TypeError("next_public_observation_required must be a boolean")
        if isinstance(self.executed_steps, bool) or self.executed_steps < 0:
            raise ValueError("executed_steps must be a non-negative integer")
        if not isinstance(self.executed_steps, int):
            raise ValueError("executed_steps must be a non-negative integer")
        try:
            PrimitiveKind(self.primitive_kind)
        except ValueError as exc:
            raise ValueError("primitive_kind is not a registered primitive") from exc

    def to_dict(self) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "execution_id": self.execution_id,
            "candidate_id": self.candidate_id,
            "primitive_kind": self.primitive_kind,
            "result": self.result.value,
            "executed_steps": self.executed_steps,
            "termination_reason": self.termination_reason,
            "next_public_observation_required": self.next_public_observation_required,
            "executor_id": self.executor_id,
        }
        PolicyFirewall().validate_recursive(payload, location="v2_execution_report")
        return payload


@dataclass(frozen=True)
class ActionChunk:
    actions: tuple[tuple[float, ...], ...]
    action_space: str
    normalization_stats_id: str
    backend_id: str
    backend_sha256: str
    rng_seed: int

    def __post_init__(self) -> None:
        text_fields = (
            self.action_space,
            self.normalization_stats_id,
            self.backend_id,
        )
        if (
            not self.actions
            or any(not isinstance(value, str) for value in text_fields)
            or not all(value.strip() for value in text_fields)
        ):
            raise ValueError("action chunk and normalization metadata are required")
        if not isinstance(self.backend_sha256, str) or re.fullmatch(
            r"[0-9a-f]{64}", self.backend_sha256
        ) is None:
            raise ValueError("backend_sha256 must be a lowercase SHA-256 digest")
        if (
            isinstance(self.rng_seed, bool | np.bool_)
            or not isinstance(self.rng_seed, int | np.integer)
            or self.rng_seed < 0
        ):
            raise ValueError("rng_seed must be a non-negative integer")
        dimension = len(self.actions[0])
        if dimension == 0 or any(len(action) != dimension for action in self.actions):
            raise ValueError("action chunk must have a fixed positive action dimension")
        if any(
            isinstance(value, bool | np.bool_)
            for action in self.actions
            for value in action
        ):
            raise TypeError("continuous actions must be numeric, not boolean")
        if any(not isfinite(value) for action in self.actions for value in action):
            raise ValueError("continuous actions must be finite")

    @property
    def horizon(self) -> int:
        return len(self.actions)

    @property
    def action_dimension(self) -> int:
        return len(self.actions[0])

    def to_dict(self) -> dict[str, object]:
        return {
            "actions": [list(action) for action in self.actions],
            "action_space": self.action_space,
            "normalization_stats_id": self.normalization_stats_id,
            "backend_id": self.backend_id,
            "backend_sha256": self.backend_sha256,
            "rng_seed": int(self.rng_seed),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ActionChunk:
        require_exact_keys(
            payload,
            required=frozenset(
                {
                    "actions",
                    "action_space",
                    "normalization_stats_id",
                    "backend_id",
                    "backend_sha256",
                    "rng_seed",
                }
            ),
            location="ActionChunk",
        )
        raw_actions = strict_array(payload["actions"], location="ActionChunk.actions")
        actions: list[tuple[float, ...]] = []
        for i, row in enumerate(raw_actions):
            raw_row = strict_array(row, location=f"ActionChunk.actions[{i}]")
            actions.append(
                tuple(
                    finite_number(
                        value, location=f"ActionChunk.actions[{i}][{j}]"
                    )
                    for j, value in enumerate(raw_row)
                )
            )
        return cls(
            actions=tuple(actions),
            action_space=strict_string(
                payload["action_space"], location="ActionChunk.action_space"
            ),
            normalization_stats_id=strict_string(
                payload["normalization_stats_id"],
                location="ActionChunk.normalization_stats_id",
            ),
            backend_id=strict_string(
                payload["backend_id"], location="ActionChunk.backend_id"
            ),
            backend_sha256=strict_string(
                payload["backend_sha256"], location="ActionChunk.backend_sha256"
            ),
            rng_seed=strict_integer(
                payload["rng_seed"], location="ActionChunk.rng_seed", minimum=0
            ),
        )


@runtime_checkable
class PolicyBackend(Protocol):
    def generate(self, request: VLAExecutionRequest) -> ActionChunk: ...


def make_execution_request(
    *,
    context: PolicyContext,
    belief: BeliefState,
    decision: PrimitiveDecision,
    maximum_skill_steps: int = 64,
    timeout_s: float = 60.0,
) -> VLAExecutionRequest:
    belief_summary = {
        "hypotheses": belief.posterior.hypotheses.mean,
        "sufficiency_mean": belief.posterior.sufficiency.mean,
        "sufficiency_vacuity": belief.posterior.sufficiency.vacuity,
        "history_digest": belief.history_digest,
        "decision_margin": decision.margin,
    }
    identity_payload = {
        "schema_version": "interaction-uncertainty.execution-request.v2",
        "task_key": belief.task_key.to_dict(),
        "final_goal_prompt": context.prompt,
        "context": context.to_dict(),
        "selected_primitive": decision.selected.to_dict(),
        "belief_summary": belief_summary,
        "maximum_skill_steps": maximum_skill_steps,
        "timeout_s": timeout_s,
        "must_reobserve_after_nonterminal": not decision.selected.is_terminal,
    }
    return VLAExecutionRequest(
        schema_version="interaction-uncertainty.execution-request.v2",
        execution_id=canonical_digest(identity_payload),
        task_key=belief.task_key,
        final_goal_prompt=context.prompt,
        context=context,
        selected_primitive=decision.selected,
        belief_summary=belief_summary,
        maximum_skill_steps=maximum_skill_steps,
        timeout_s=timeout_s,
        must_reobserve_after_nonterminal=not decision.selected.is_terminal,
    )
