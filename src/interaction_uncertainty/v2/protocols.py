"""Typed production extension points for the v0.2 pipeline."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..observation import PolicyContext, PolicyObservation
from .belief import BeliefState
from .effects import CandidateEffectForecast
from .needs import InformationNeed
from .primitives import CandidateSet
from .task import TaskKey, TaskSpec


@runtime_checkable
class PrimitiveProposer(Protocol):
    def propose(
        self,
        *,
        task_key: TaskKey,
        observation: PolicyObservation,
        needs: tuple[InformationNeed, ...],
    ) -> CandidateSet: ...


@runtime_checkable
class ActionOutcomeCritic(Protocol):
    def forecast(
        self,
        *,
        context: PolicyContext,
        task: TaskSpec,
        belief: BeliefState,
        needs: tuple[InformationNeed, ...],
        candidates: CandidateSet,
    ) -> tuple[CandidateEffectForecast, ...]: ...
