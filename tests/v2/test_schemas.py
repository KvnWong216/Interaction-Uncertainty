from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from interaction_uncertainty.observation import PolicyContext
from interaction_uncertainty.v2.belief import EvidentialBeliefFilter
from interaction_uncertainty.v2.controller import EpisodeController
from interaction_uncertainty.v2.evidence import observation_digest
from interaction_uncertainty.v2.execution import ActionChunk, ExecutionReport, ExecutionResult
from interaction_uncertainty.v2.needs import BayesRiskNeedExtractor
from interaction_uncertainty.v2.planning import PlanningWeights
from interaction_uncertainty.v2.primitives import (
    NeedDrivenPrimitiveProposer,
    PrimitiveKind,
)
from interaction_uncertainty.v2.trace import InMemoryTraceSink
from tests.v2.test_v2_pipeline import (
    AnalyticObservationCritic,
    ContentDependentEvidenceModel,
    observation,
    task,
)

ROOT = Path(__file__).resolve().parents[2]


def _schema(name: str) -> dict[str, object]:
    return json.loads((ROOT / "schemas" / "v2" / name).read_text(encoding="utf-8"))


def _validator(name: str) -> Draft202012Validator:
    schema = _schema(name)
    Draft202012Validator.check_schema(schema)
    registry = Registry().with_resources(
        [
            (
                candidate["$id"],
                Resource.from_contents(candidate),
            )
            for path in (ROOT / "schemas" / "v2").glob("*.json")
            for candidate in [json.loads(path.read_text(encoding="utf-8"))]
        ]
    )
    return Draft202012Validator(
        schema,
        registry=registry,
        format_checker=FormatChecker(),
    )


@pytest.fixture
def boundary_examples() -> dict[str, dict[str, object]]:
    spec = task()
    public_observation = observation("schema-frame", "public://closed")
    sink = InMemoryTraceSink()
    controller = EpisodeController(
        task=spec,
        episode_id="schema-boundary-episode",
        evidence_model=ContentDependentEvidenceModel(),
        belief_filter=EvidentialBeliefFilter(),
        outcome_critic=AnalyticObservationCritic(),
        planning_weights=PlanningWeights(
            action_cost=0.01,
            physical_risk=0.01,
            disturbance=0.0,
            critic_uncertainty=0.01,
        ),
        trace_sink=sink,
    )
    plan = controller.observe_and_plan(public_observation)
    evidence = ContentDependentEvidenceModel().infer(plan.context, spec)
    report = ExecutionReport(
        schema_version="interaction-uncertainty.execution-report.v2",
        execution_id=plan.execution_request.execution_id,
        candidate_id=plan.decision.selected.candidate_id,
        primitive_kind=plan.decision.selected.kind.value,
        result=ExecutionResult.SUCCEEDED,
        executed_steps=8,
        termination_reason="schema_contract_fixture",
        next_public_observation_required=True,
        executor_id="schema-test-executor",
    )
    action_chunk = ActionChunk(
        actions=((0.0,) * 7, (0.1,) * 7),
        action_space="libero_delta_ee_gripper_v1",
        normalization_stats_id="libero-10-no-noops-v1",
        backend_id="schema-policy-backend",
        backend_sha256="7" * 64,
        rng_seed=0,
    )
    digest = observation_digest(plan.context)
    return {
        "task-spec.schema.json": spec.to_dict(),
        "policy-context.schema.json": plan.context.to_dict(),
        "belief-state.schema.json": plan.belief.to_dict(),
        "information-need.schema.json": plan.needs[0].to_dict(),
        "evidence-packet.schema.json": evidence.to_dict(),
        "evidence-request.schema.json": {
            "schema_version": "interaction-uncertainty.evidence-request.v2",
            "task": spec.to_dict(),
            "context": plan.context.to_dict(),
            "observation_digest": digest,
        },
        "evidence-response.schema.json": {
            "schema_version": "interaction-uncertainty.evidence-response.v2",
            "evidence": evidence.to_dict(),
        },
        "effect-request.schema.json": {
            "schema_version": "interaction-uncertainty.effect-request.v2",
            "task": spec.to_dict(),
            "context": plan.context.to_dict(),
            "belief": plan.belief.to_dict(),
            "needs": [item.to_dict() for item in plan.needs],
            "candidates": plan.candidates.to_dict(),
            "observation_digest": digest,
        },
        "effect-response.schema.json": {
            "schema_version": "interaction-uncertainty.effect-response.v2",
            "task_key": spec.key.to_dict(),
            "observation_digest": digest,
            "forecasts": [item.to_dict() for item in plan.forecasts],
        },
        "public-observation.schema.json": public_observation.to_dict(),
        "candidate-set.schema.json": plan.candidates.to_dict(),
        "execution-request.schema.json": plan.execution_request.to_dict(),
        "execution-report.schema.json": report.to_dict(),
        "action-chunk.schema.json": action_chunk.to_dict(),
        "policy-request.schema.json": {
            "schema_version": "interaction-uncertainty.policy-request.v2",
            "execution_request": plan.execution_request.to_dict(),
        },
        "policy-response.schema.json": {
            "schema_version": "interaction-uncertainty.policy-response.v2",
            "execution_id": plan.execution_request.execution_id,
            "candidate_id": plan.decision.selected.candidate_id,
            "primitive_kind": plan.decision.selected.kind.value,
            "action_chunk": action_chunk.to_dict(),
        },
        "trace-event.schema.json": sink.events[-1].to_dict(),
    }


def test_v2_task_evidence_and_effect_examples_satisfy_strict_schemas() -> None:
    spec = task()
    context = PolicyContext(
        "schema-test",
        0,
        spec.final_goal_prompt,
        observation("frame", "public://closed"),
    )
    evidence = ContentDependentEvidenceModel().infer(context, spec)
    belief = EvidentialBeliefFilter().update(
        task=spec,
        episode_id="schema-test",
        step_index=0,
        evidence=evidence,
    )
    needs = BayesRiskNeedExtractor().extract(spec, belief)
    candidates = NeedDrivenPrimitiveProposer().propose(
        task_key=spec.key,
        observation=context.observation,
        needs=needs,
    )
    effect = AnalyticObservationCritic().forecast(
        context=context,
        task=spec,
        belief=belief,
        needs=needs,
        candidates=candidates,
    )[0]

    _validator("task-spec.schema.json").validate(spec.to_dict())
    _validator("evidence-packet.schema.json").validate(evidence.to_dict())
    _validator("effect-forecast.schema.json").validate(effect.to_dict())


def test_runtime_boundary_objects_satisfy_strict_schemas(
    boundary_examples: dict[str, dict[str, object]],
) -> None:
    for schema_name, payload in boundary_examples.items():
        _validator(schema_name).validate(payload)


@pytest.mark.parametrize(
    "injected_field",
    ("unknown_runtime_field", "semantic_id"),
)
def test_runtime_boundary_schemas_reject_unknown_and_privileged_fields(
    boundary_examples: dict[str, dict[str, object]],
    injected_field: str,
) -> None:
    for schema_name, original in boundary_examples.items():
        payload = deepcopy(original)
        payload[injected_field] = 42
        errors = list(_validator(schema_name).iter_errors(payload))
        assert errors, f"{schema_name} accepted forbidden field {injected_field!r}"


def test_nested_boundary_objects_are_also_closed_to_unknown_fields(
    boundary_examples: dict[str, dict[str, object]],
) -> None:
    public_observation = deepcopy(
        boundary_examples["public-observation.schema.json"]
    )
    public_observation["anchors"][0]["semantic_id"] = 42  # type: ignore[index]
    assert list(
        _validator("public-observation.schema.json").iter_errors(public_observation)
    )

    candidate_set = deepcopy(boundary_examples["candidate-set.schema.json"])
    interactive = next(
        candidate
        for candidate in candidate_set["candidates"]  # type: ignore[union-attr]
        if candidate["kind"] == "OPEN_CONTAINER"
    )
    interactive["parameters"]["unknown_parameter"] = 1
    assert list(_validator("candidate-set.schema.json").iter_errors(candidate_set))

    execution_request = deepcopy(
        boundary_examples["execution-request.schema.json"]
    )
    execution_request["context"]["observation"]["oracle_action"] = "OPEN_CONTAINER"  # type: ignore[index]
    assert list(
        _validator("execution-request.schema.json").iter_errors(execution_request)
    )

    trace_event = deepcopy(boundary_examples["trace-event.schema.json"])
    trace_event["task_key"]["unknown_key"] = "not-allowed"  # type: ignore[index]
    assert list(_validator("trace-event.schema.json").iter_errors(trace_event))


def test_primitive_parameter_schema_is_conditioned_on_primitive_kind(
    boundary_examples: dict[str, dict[str, object]],
) -> None:
    candidate_set = deepcopy(boundary_examples["candidate-set.schema.json"])
    interactive = next(
        candidate
        for candidate in candidate_set["candidates"]  # type: ignore[union-attr]
        if candidate["kind"] == "OPEN_CONTAINER"
    )
    interactive["parameters"] = {
        "type": "ROTATE",
        "rotation_budget_degrees": 90.0,
        "max_attempts": 1,
    }
    assert list(_validator("candidate-set.schema.json").iter_errors(candidate_set))


def test_direct_and_stop_schema_enforce_grounding_and_reobservation_contracts(
    boundary_examples: dict[str, dict[str, object]],
) -> None:
    direct_plan = EpisodeController(
        task=task(),
        episode_id="schema-direct-episode",
        evidence_model=ContentDependentEvidenceModel(),
        belief_filter=EvidentialBeliefFilter(),
        outcome_critic=AnalyticObservationCritic(),
    ).observe_and_plan(observation("visible", "public://open-visible-orange-juice"))
    request = direct_plan.execution_request.to_dict()
    assert request["selected_primitive"]["kind"] == "DIRECT_ACT"  # type: ignore[index]
    _validator("execution-request.schema.json").validate(request)

    wrong_reobservation = deepcopy(request)
    wrong_reobservation["must_reobserve_after_nonterminal"] = True
    assert list(
        _validator("execution-request.schema.json").iter_errors(wrong_reobservation)
    )

    missing_target_affordance = deepcopy(request)
    missing_target_affordance["selected_primitive"]["source_anchor"][  # type: ignore[index]
        "affordances"
    ] = ["graspable"]
    assert list(
        _validator("execution-request.schema.json").iter_errors(
            missing_target_affordance
        )
    )

    duplicate_stop = deepcopy(boundary_examples["candidate-set.schema.json"])
    stop = next(
        candidate
        for candidate in duplicate_stop["candidates"]  # type: ignore[union-attr]
        if candidate["kind"] == "STOP_NOT_FOUND"
    )
    second_stop = deepcopy(stop)
    second_stop["candidate_id"] = "terminal::not_found_duplicate"
    duplicate_stop["candidates"].append(second_stop)  # type: ignore[union-attr]
    assert list(_validator("candidate-set.schema.json").iter_errors(duplicate_stop))


def test_schema_enums_match_runtime_enums() -> None:
    primitive_values = {item.value for item in PrimitiveKind}
    result_values = {item.value for item in ExecutionResult}

    candidate_schema = _schema("candidate-set.schema.json")
    candidate_kind = candidate_schema["$defs"]["primitiveCall"]["properties"]["kind"]  # type: ignore[index]
    assert set(candidate_kind["enum"]) == primitive_values

    request_schema = _schema("execution-request.schema.json")
    request_kind = request_schema["$defs"]["primitiveCall"]["properties"]["kind"]  # type: ignore[index]
    assert set(request_kind["enum"]) == primitive_values

    report_schema = _schema("execution-report.schema.json")
    assert set(report_schema["properties"]["primitive_kind"]["enum"]) == primitive_values  # type: ignore[index]
    assert set(report_schema["properties"]["result"]["enum"]) == result_values  # type: ignore[index]

    response_schema = _schema("policy-response.schema.json")
    assert set(response_schema["properties"]["primitive_kind"]["enum"]) == primitive_values  # type: ignore[index]


def test_task_schema_rejects_unknown_privileged_field() -> None:
    payload = task().to_dict()
    payload["semantic_id"] = 42
    errors = list(_validator("task-spec.schema.json").iter_errors(payload))
    assert errors
