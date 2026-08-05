"""Strict JSON adapters for GPU evidence and action-outcome services."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite
from urllib.request import Request, urlopen

from ..firewall import PolicyFirewall
from ..observation import PolicyContext
from .belief import BeliefState
from .effects import (
    CandidateEffectForecast,
    ExecutionStatus,
    ObservationOutcomeModel,
    validate_forecast_coverage,
)
from .evidence import EvidencePacket, PromptEvidenceModel, observation_digest
from .execution import ActionChunk, VLAExecutionRequest
from .needs import InformationNeed
from .primitives import CandidateSet, PrimitiveKind
from .task import TaskKey, TaskSpec
from .validation import (
    finite_number,
    require_exact_keys,
    strict_array,
    strict_integer,
    strict_string,
)

JSONMapping = Mapping[str, object]
JSONTransport = Callable[[str, JSONMapping, float], JSONMapping]
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


def _reject_nonstandard_json_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant {value!r} is forbidden")


def urllib_json_transport(endpoint: str, payload: JSONMapping, timeout_s: float) -> JSONMapping:
    body = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
    request = Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_s) as response:  # noqa: S310
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("remote JSON response exceeds the 8 MiB safety limit")
    decoded = json.loads(
        raw.decode("utf-8"), parse_constant=_reject_nonstandard_json_constant
    )
    if not isinstance(decoded, Mapping):
        raise TypeError("remote service must return a JSON object")
    return decoded


def _validate_endpoint(endpoint: str, timeout_s: float) -> None:
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise TypeError("remote endpoint must be a non-empty string")
    if not endpoint.startswith(("http://", "https://")):
        raise ValueError("remote endpoint must use http:// or https://")
    parsed_timeout = finite_number(
        timeout_s, location="timeout_s", minimum=0.0, exclusive_minimum=True
    )
    if not isfinite(parsed_timeout) or parsed_timeout <= 0.0:
        raise ValueError("timeout_s must be finite and positive")


@dataclass
class RemoteEvidenceModel(PromptEvidenceModel):
    endpoint: str
    transport: JSONTransport = urllib_json_transport
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        _validate_endpoint(self.endpoint, self.timeout_s)

    def infer(self, context: PolicyContext, task: TaskSpec) -> EvidencePacket:
        payload = {
            "schema_version": "interaction-uncertainty.evidence-request.v2",
            "task": task.to_dict(),
            "context": context.to_dict(),
            "observation_digest": observation_digest(context),
        }
        firewall = PolicyFirewall()
        firewall.validate_recursive(payload, location="remote_evidence_request")
        response = self.transport(self.endpoint, payload, self.timeout_s)
        firewall.validate_recursive(response, location="remote_evidence_response")
        require_exact_keys(
            response,
            required=frozenset({"schema_version", "evidence"}),
            location="evidence response",
        )
        if response.get("schema_version") != "interaction-uncertainty.evidence-response.v2":
            raise ValueError("unsupported or missing evidence response schema_version")
        packet_payload = response.get("evidence")
        if not isinstance(packet_payload, Mapping):
            raise TypeError("evidence response must contain one evidence object")
        packet = EvidencePacket.from_dict(packet_payload)
        packet.validate_request(context, task)
        return packet


def _task_key(payload: object) -> TaskKey:
    if not isinstance(payload, Mapping):
        raise TypeError("task_key must be a mapping")
    require_exact_keys(
        payload,
        required=frozenset(
            {"task_id", "prompt_digest", "ontology_id", "ontology_version"}
        ),
        location="task_key",
    )
    return TaskKey(
        task_id=strict_string(payload["task_id"], location="task_key.task_id"),
        prompt_digest=strict_string(
            payload["prompt_digest"], location="task_key.prompt_digest"
        ),
        ontology_id=strict_string(
            payload["ontology_id"], location="task_key.ontology_id"
        ),
        ontology_version=strict_string(
            payload["ontology_version"], location="task_key.ontology_version"
        ),
    )


def _forecast(payload: Mapping[str, object]) -> CandidateEffectForecast:
    from .evidence import ModelStamp

    require_exact_keys(
        payload,
        required=frozenset(
            {
                "schema_version",
                "task_key",
                "candidate_id",
                "hypothesis_labels",
                "transition_matrix",
                "outcomes",
                "critic_uncertainty",
                "model_stamp",
                "rng_seed",
            }
        ),
        location="effect forecast",
    )
    stamp = payload["model_stamp"]
    raw_outcomes = payload["outcomes"]
    if not isinstance(stamp, Mapping):
        raise TypeError("effect model_stamp must be a mapping")
    require_exact_keys(
        stamp,
        required=frozenset({"model_id", "model_sha256", "calibration_id"}),
        location="effect forecast.model_stamp",
    )
    raw_outcomes = strict_array(raw_outcomes, location="effect forecast.outcomes")
    outcomes: list[ObservationOutcomeModel] = []
    for raw in raw_outcomes:
        if not isinstance(raw, Mapping):
            raise TypeError("each effect outcome must be an object")
        require_exact_keys(
            raw,
            required=frozenset(
                {
                    "outcome_id",
                    "likelihood_by_post_state",
                    "execution_status",
                    "sufficiency_evidence",
                    "resolves_need_ids",
                    "action_cost",
                    "physical_risk",
                    "disturbance",
                }
            ),
            location="effect outcome",
        )
        raw_sufficiency = strict_array(
            raw["sufficiency_evidence"], location="effect outcome.sufficiency_evidence"
        )
        sufficiency = tuple(
            finite_number(
                value,
                location=f"effect outcome.sufficiency_evidence[{index}]",
                minimum=0.0,
            )
            for index, value in enumerate(raw_sufficiency)
        )
        if len(sufficiency) != 2:
            raise ValueError("outcome sufficiency_evidence must contain two values")
        outcomes.append(
            ObservationOutcomeModel(
                outcome_id=strict_string(
                    raw["outcome_id"], location="effect outcome.outcome_id"
                ),
                likelihood_by_post_state=tuple(
                    finite_number(
                        value,
                        location=f"effect outcome.likelihood_by_post_state[{index}]",
                        minimum=0.0,
                        maximum=1.0,
                    )
                    for index, value in enumerate(
                        strict_array(
                            raw["likelihood_by_post_state"],
                            location="effect outcome.likelihood_by_post_state",
                        )
                    )
                ),
                execution_status=ExecutionStatus(
                    strict_string(
                        raw["execution_status"],
                        location="effect outcome.execution_status",
                    )
                ),
                sufficiency_evidence=(sufficiency[0], sufficiency[1]),
                resolves_need_ids=tuple(
                    strict_string(
                        value,
                        location=f"effect outcome.resolves_need_ids[{index}]",
                    )
                    for index, value in enumerate(
                        strict_array(
                            raw["resolves_need_ids"],
                            location="effect outcome.resolves_need_ids",
                        )
                    )
                ),
                action_cost=finite_number(
                    raw["action_cost"], location="effect outcome.action_cost", minimum=0.0
                ),
                physical_risk=finite_number(
                    raw["physical_risk"],
                    location="effect outcome.physical_risk",
                    minimum=0.0,
                ),
                disturbance=finite_number(
                    raw["disturbance"],
                    location="effect outcome.disturbance",
                    minimum=0.0,
                ),
            )
        )
    raw_labels = strict_array(
        payload["hypothesis_labels"], location="effect forecast.hypothesis_labels"
    )
    raw_transition = strict_array(
        payload["transition_matrix"], location="effect forecast.transition_matrix"
    )
    return CandidateEffectForecast(
        schema_version=strict_string(
            payload["schema_version"], location="effect forecast.schema_version"
        ),
        task_key=_task_key(payload["task_key"]),
        candidate_id=strict_string(
            payload["candidate_id"], location="effect forecast.candidate_id"
        ),
        hypothesis_labels=tuple(
            strict_string(
                value, location=f"effect forecast.hypothesis_labels[{index}]"
            )
            for index, value in enumerate(raw_labels)
        ),
        transition_matrix=tuple(
            tuple(
                finite_number(
                    value,
                    location=f"effect forecast.transition_matrix[{i}][{j}]",
                    minimum=0.0,
                    maximum=1.0,
                )
                for j, value in enumerate(
                    strict_array(
                        row, location=f"effect forecast.transition_matrix[{i}]"
                    )
                )
            )
            for i, row in enumerate(raw_transition)
        ),
        outcomes=tuple(outcomes),
        critic_uncertainty=finite_number(
            payload["critic_uncertainty"],
            location="effect forecast.critic_uncertainty",
            minimum=0.0,
            maximum=1.0,
        ),
        model_stamp=ModelStamp(
            model_id=strict_string(
                stamp["model_id"], location="effect forecast.model_stamp.model_id"
            ),
            model_sha256=strict_string(
                stamp["model_sha256"],
                location="effect forecast.model_stamp.model_sha256",
            ),
            calibration_id=strict_string(
                stamp["calibration_id"],
                location="effect forecast.model_stamp.calibration_id",
            ),
        ),
        rng_seed=strict_integer(
            payload["rng_seed"], location="effect forecast.rng_seed", minimum=0
        ),
    )


@dataclass
class RemoteActionOutcomeCritic:
    endpoint: str
    transport: JSONTransport = urllib_json_transport
    timeout_s: float = 45.0
    max_outcomes_per_candidate: int = 32

    def __post_init__(self) -> None:
        _validate_endpoint(self.endpoint, self.timeout_s)
        if (
            isinstance(self.max_outcomes_per_candidate, bool)
            or not isinstance(self.max_outcomes_per_candidate, int)
            or not 1 <= self.max_outcomes_per_candidate <= 256
        ):
            raise ValueError("max_outcomes_per_candidate must lie in [1, 256]")

    def forecast(
        self,
        *,
        context: PolicyContext,
        task: TaskSpec,
        belief: BeliefState,
        needs: tuple[InformationNeed, ...],
        candidates: CandidateSet,
    ) -> tuple[CandidateEffectForecast, ...]:
        payload = {
            "schema_version": "interaction-uncertainty.effect-request.v2",
            "task": task.to_dict(),
            "context": context.to_dict(),
            "belief": belief.to_dict(),
            "needs": [item.to_dict() for item in needs],
            "candidates": candidates.to_dict(),
            "observation_digest": observation_digest(context),
        }
        firewall = PolicyFirewall()
        firewall.validate_recursive(payload, location="remote_effect_request")
        response = self.transport(self.endpoint, payload, self.timeout_s)
        firewall.validate_recursive(response, location="remote_effect_response")
        require_exact_keys(
            response,
            required=frozenset(
                {"schema_version", "task_key", "observation_digest", "forecasts"}
            ),
            location="effect response",
        )
        if response.get("schema_version") != "interaction-uncertainty.effect-response.v2":
            raise ValueError("unsupported or missing effect response schema_version")
        response_digest = strict_string(
            response["observation_digest"],
            location="effect response.observation_digest",
        )
        if response_digest != observation_digest(context):
            raise ValueError("effect response is stale or belongs to another observation")
        if _task_key(response.get("task_key")) != task.key:
            raise ValueError("effect response belongs to another task")
        raw = strict_array(response["forecasts"], location="effect response.forecasts")
        forecasts = tuple(
            _forecast(item)
            for item in raw
            if isinstance(item, Mapping)
        )
        if len(forecasts) != len(raw):
            raise TypeError("every effect forecast must be a JSON object")
        if any(len(item.outcomes) > self.max_outcomes_per_candidate for item in forecasts):
            raise ValueError("effect service returned too many outcomes")
        validate_forecast_coverage(candidates, forecasts)
        return forecasts


@dataclass
class RemotePolicyBackend:
    """Strict process boundary for OpenVLA/Octo/ASA-style action services.

    The remote backend realizes an already selected primitive.  Echo fields and
    pinned embodiment metadata prevent it from silently replacing the high-level
    decision or returning an action vector under the wrong normalization.
    """

    endpoint: str
    expected_action_space: str
    expected_action_dimension: int
    expected_normalization_stats_id: str
    expected_backend_id: str
    expected_backend_sha256: str
    transport: JSONTransport = urllib_json_transport
    timeout_s: float = 60.0
    max_action_horizon: int = 64

    def __post_init__(self) -> None:
        _validate_endpoint(self.endpoint, self.timeout_s)
        for name in (
            "expected_action_space",
            "expected_normalization_stats_id",
            "expected_backend_id",
            "expected_backend_sha256",
        ):
            strict_string(getattr(self, name), location=name)
        if re.fullmatch(r"[0-9a-f]{64}", self.expected_backend_sha256) is None:
            raise ValueError("expected_backend_sha256 must be a SHA-256 digest")
        strict_integer(
            self.expected_action_dimension,
            location="expected_action_dimension",
            minimum=1,
            maximum=4096,
        )
        strict_integer(
            self.max_action_horizon,
            location="max_action_horizon",
            minimum=1,
            maximum=4096,
        )

    def generate(self, request: VLAExecutionRequest) -> ActionChunk:
        if not isinstance(request, VLAExecutionRequest):
            raise TypeError("request must be a VLAExecutionRequest")
        if request.selected_primitive.kind is PrimitiveKind.STOP_NOT_FOUND:
            raise ValueError("STOP_NOT_FOUND does not require a continuous action chunk")
        payload = {
            "schema_version": "interaction-uncertainty.policy-request.v2",
            "execution_request": request.to_dict(),
        }
        firewall = PolicyFirewall()
        firewall.validate_recursive(payload, location="remote_policy_request")
        response = self.transport(self.endpoint, payload, self.timeout_s)
        firewall.validate_recursive(response, location="remote_policy_response")
        require_exact_keys(
            response,
            required=frozenset(
                {
                    "schema_version",
                    "execution_id",
                    "candidate_id",
                    "primitive_kind",
                    "action_chunk",
                }
            ),
            location="policy response",
        )
        if response["schema_version"] != "interaction-uncertainty.policy-response.v2":
            raise ValueError("unsupported or missing policy response schema_version")
        execution_id = strict_string(
            response["execution_id"], location="policy response.execution_id"
        )
        candidate_id = strict_string(
            response["candidate_id"], location="policy response.candidate_id"
        )
        primitive_kind = strict_string(
            response["primitive_kind"], location="policy response.primitive_kind"
        )
        if execution_id != request.execution_id:
            raise ValueError("policy response belongs to another execution request")
        if candidate_id != request.selected_primitive.candidate_id:
            raise ValueError("policy backend silently changed the selected candidate")
        if primitive_kind != request.selected_primitive.kind.value:
            raise ValueError("policy backend silently changed the primitive family")
        raw_chunk = response["action_chunk"]
        if not isinstance(raw_chunk, Mapping):
            raise TypeError("policy response action_chunk must be an object")
        chunk = ActionChunk.from_dict(raw_chunk)
        if chunk.action_dimension != self.expected_action_dimension:
            raise ValueError("policy backend returned the wrong action dimension")
        if chunk.horizon > self.max_action_horizon:
            raise ValueError("policy backend returned an excessive action horizon")
        if chunk.action_space != self.expected_action_space:
            raise ValueError("policy backend returned the wrong action space")
        if chunk.normalization_stats_id != self.expected_normalization_stats_id:
            raise ValueError("policy backend returned unpinned normalization statistics")
        if chunk.backend_id != self.expected_backend_id:
            raise ValueError("policy backend identity does not match deployment config")
        if chunk.backend_sha256 != self.expected_backend_sha256:
            raise ValueError("policy backend checkpoint hash does not match deployment config")
        return chunk
