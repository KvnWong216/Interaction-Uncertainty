"""Stateful uncertainty-to-action episode controller."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..firewall import PolicyFirewall
from ..observation import PolicyContext, PolicyObservation, ensure_final_goal_prompt
from .belief import BeliefState, EvidentialBeliefFilter
from .effects import (
    CandidateEffectForecast,
    CounterfactualRollout,
    rollout_forecast,
    validate_forecast_coverage,
)
from .evidence import PromptEvidenceModel
from .execution import ExecutionReport, VLAExecutionRequest, make_execution_request
from .needs import BayesRiskNeedExtractor, InformationNeed
from .planning import PlanningWeights, PrimitiveDecision, rank_primitives
from .primitives import CandidateSet, NeedDrivenPrimitiveProposer, validate_candidate_set
from .protocols import ActionOutcomeCritic
from .task import TaskSpec
from .trace import InMemoryTraceSink, TraceSink, make_trace_event


class ControllerState(str, Enum):
    WAITING_FOR_OBSERVATION = "WAITING_FOR_OBSERVATION"
    WAITING_FOR_EXECUTION_REPORT = "WAITING_FOR_EXECUTION_REPORT"
    TERMINATED = "TERMINATED"


@dataclass(frozen=True)
class PlanResult:
    context: PolicyContext
    belief: BeliefState
    needs: tuple[InformationNeed, ...]
    candidates: CandidateSet
    forecasts: tuple[CandidateEffectForecast, ...]
    rollouts: tuple[CounterfactualRollout, ...]
    decision: PrimitiveDecision
    execution_request: VLAExecutionRequest

    def to_dict(self) -> dict[str, object]:
        return {
            "context": self.context.to_dict(),
            "belief": self.belief.to_dict(),
            "needs": [item.to_dict() for item in self.needs],
            "candidates": self.candidates.to_dict(),
            "forecasts": [item.to_dict() for item in self.forecasts],
            "rollouts": [item.to_dict() for item in self.rollouts],
            "decision": self.decision.to_dict(),
            "execution_request": self.execution_request.to_dict(),
        }


@dataclass
class EpisodeController:
    """Own one task/episode belief and enforce actual re-observation."""

    task: TaskSpec
    episode_id: str
    evidence_model: PromptEvidenceModel
    belief_filter: EvidentialBeliefFilter
    outcome_critic: ActionOutcomeCritic
    need_extractor: BayesRiskNeedExtractor = field(default_factory=BayesRiskNeedExtractor)
    proposer: NeedDrivenPrimitiveProposer = field(default_factory=NeedDrivenPrimitiveProposer)
    planning_weights: PlanningWeights = field(default_factory=PlanningWeights)
    trace_sink: TraceSink = field(default_factory=InMemoryTraceSink)
    maximum_skill_steps: int = 64
    execution_timeout_s: float = 60.0

    _state: ControllerState = field(
        default=ControllerState.WAITING_FOR_OBSERVATION, init=False
    )
    _belief: BeliefState | None = field(default=None, init=False)
    _last_plan: PlanResult | None = field(default=None, init=False)
    _parent_event_id: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        ensure_final_goal_prompt(self.task.final_goal_prompt)
        if not self.episode_id.strip():
            raise ValueError("episode_id must be non-empty")

    @property
    def state(self) -> ControllerState:
        return self._state

    @property
    def belief(self) -> BeliefState | None:
        return self._belief

    def _trace(self, event_type: str, step_index: int, payload: dict[str, object]) -> None:
        event = make_trace_event(
            parent_event_id=self._parent_event_id,
            event_type=event_type,
            episode_id=self.episode_id,
            step_index=step_index,
            task_key=self.task.key,
            payload=payload,
        )
        self.trace_sink.append(event)
        self._parent_event_id = event.event_id

    def observe_and_plan(self, observation: PolicyObservation) -> PlanResult:
        if self._state is not ControllerState.WAITING_FOR_OBSERVATION:
            raise RuntimeError(f"controller cannot accept an observation in state {self._state}")
        step_index = 0 if self._belief is None else self._belief.step_index + 1
        context = PolicyContext(
            episode_id=self.episode_id,
            step_index=step_index,
            prompt=self.task.final_goal_prompt,
            observation=observation,
        )
        PolicyFirewall().validate_observation(observation)
        self._trace("ObservationReceived", step_index, context.to_dict())

        evidence = self.evidence_model.infer(context, self.task)
        evidence.validate_request(context, self.task)
        PolicyFirewall().validate_recursive(evidence.to_dict(), location="v2_evidence")
        self._trace("EvidenceProduced", step_index, evidence.to_dict())
        belief = self.belief_filter.update(
            task=self.task,
            episode_id=self.episode_id,
            step_index=step_index,
            evidence=evidence,
            previous=self._belief,
        )
        self._trace("BeliefUpdated", step_index, belief.to_dict())

        needs = self.need_extractor.extract(self.task, belief)
        self._trace(
            "InformationNeedsExtracted",
            step_index,
            {"needs": [item.to_dict() for item in needs]},
        )
        candidates = self.proposer.propose(
            task_key=self.task.key,
            observation=observation,
            needs=needs,
        )
        validate_candidate_set(candidates, observation, needs)
        self._trace("CandidatesProposed", step_index, candidates.to_dict())

        forecasts = self.outcome_critic.forecast(
            context=context,
            task=self.task,
            belief=belief,
            needs=needs,
            candidates=candidates,
        )
        validate_forecast_coverage(candidates, forecasts)
        self._trace(
            "EffectsForecast",
            step_index,
            {"forecasts": [item.to_dict() for item in forecasts]},
        )
        forecast_by_id = {item.candidate_id: item for item in forecasts}
        rollouts = tuple(
            rollout_forecast(
                task=self.task,
                belief=belief,
                candidate=candidate,
                forecast=forecast_by_id[candidate.candidate_id],
            )
            for candidate in candidates.candidates
            if candidate.requires_effect_forecast
        )
        decision = rank_primitives(
            task=self.task,
            belief=belief,
            candidates=candidates,
            rollouts=rollouts,
            weights=self.planning_weights,
        )
        self._trace("RankingComputed", step_index, decision.to_dict())
        request = make_execution_request(
            context=context,
            belief=belief,
            decision=decision,
            maximum_skill_steps=self.maximum_skill_steps,
            timeout_s=self.execution_timeout_s,
        )
        self._trace("CommandIssued", step_index, request.to_dict())
        result = PlanResult(
            context=context,
            belief=belief,
            needs=needs,
            candidates=candidates,
            forecasts=forecasts,
            rollouts=rollouts,
            decision=decision,
            execution_request=request,
        )
        self._belief = belief
        self._last_plan = result
        self._state = ControllerState.WAITING_FOR_EXECUTION_REPORT
        return result

    def accept_execution_report(self, report: ExecutionReport) -> None:
        if self._state is not ControllerState.WAITING_FOR_EXECUTION_REPORT:
            raise RuntimeError("controller is not waiting for an execution report")
        assert self._last_plan is not None
        request = self._last_plan.execution_request
        selected = self._last_plan.decision.selected
        if report.execution_id != request.execution_id:
            raise ValueError("execution report ID does not match the issued command")
        if report.candidate_id != selected.candidate_id:
            raise ValueError("executor silently changed the selected candidate")
        if report.primitive_kind != selected.kind.value:
            raise ValueError("executor silently changed the selected primitive family")
        if report.next_public_observation_required != (not selected.is_terminal):
            raise ValueError("execution report violates the re-observation contract")
        self._trace(
            "ExecutionFinished",
            self._last_plan.context.step_index,
            report.to_dict(),
        )
        self._state = (
            ControllerState.TERMINATED
            if selected.is_terminal
            else ControllerState.WAITING_FOR_OBSERVATION
        )
