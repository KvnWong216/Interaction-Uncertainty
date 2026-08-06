"""Typed primitive calls and uncertainty-conditioned candidate generation.

The proposer is recall-oriented: an information need may produce several
physically compatible candidates.  It never chooses the winner; the effect
critic and task-risk planner do that downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from ..beliefs import DeficitKind
from ..observation import PolicyObservation, VisualAnchor
from .needs import InformationNeed
from .task import TaskKey, TerminalDecision
from .validation import finite_number


class PrimitiveKind(str, Enum):
    DIRECT_ACT = "DIRECT_ACT"
    OPEN_CONTAINER = "OPEN_CONTAINER"
    PULL_DRAWER = "PULL_DRAWER"
    UNCOVER = "UNCOVER"
    CLEAR_OCCLUDER = "CLEAR_OCCLUDER"
    PUSH_ASIDE = "PUSH_ASIDE"
    BRING_CLOSER = "BRING_CLOSER"
    ROTATE_TO_LABEL = "ROTATE_TO_LABEL"
    PICK_AND_INSPECT = "PICK_AND_INSPECT"
    STOP_NOT_FOUND = "STOP_NOT_FOUND"


TERMINAL_KINDS = frozenset({PrimitiveKind.DIRECT_ACT, PrimitiveKind.STOP_NOT_FOUND})


class CoarsePrimitive(str, Enum):
    """The reporting-level action space ``A`` used for baseline comparison.

    :class:`PrimitiveKind` is the execution vocabulary: it distinguishes a
    drawer from a hinged door because the two need different parameters and
    different affordances.  That resolution is unusable when scoring a
    monolithic VLA, which emits joint deltas and no primitive label at all.

    ``A`` is the coarser space every method can be scored in.  Its members are
    separated by *what information the action produces*, not by how the action
    is executed: revealing a hidden region, resolving a label surface, closing
    distance on a small target, executing the task, or abstaining.  Container
    opening therefore folds into :attr:`REMOVE_OCCLUDER` -- a drawer front is an
    occluder that happens to be attached to a joint.

    :attr:`NOT_FOUND` has no realizable decoding from continuous actions.  That
    is a finding rather than a gap: a policy with no abstention channel cannot
    abstain, so its evidence for this member is structurally zero.
    """

    ACT = "ACT"
    NOT_FOUND = "NOT_FOUND"
    ROTATE = "ROTATE"
    MOVE_CLOSER = "MOVE_CLOSER"
    REMOVE_OCCLUDER = "REMOVE_OCCLUDER"


COARSE_ACTION_SPACE: tuple[CoarsePrimitive, ...] = (
    CoarsePrimitive.ACT,
    CoarsePrimitive.NOT_FOUND,
    CoarsePrimitive.ROTATE,
    CoarsePrimitive.MOVE_CLOSER,
    CoarsePrimitive.REMOVE_OCCLUDER,
)


COARSE_BY_PRIMITIVE: dict[PrimitiveKind, CoarsePrimitive] = {
    PrimitiveKind.DIRECT_ACT: CoarsePrimitive.ACT,
    PrimitiveKind.STOP_NOT_FOUND: CoarsePrimitive.NOT_FOUND,
    PrimitiveKind.ROTATE_TO_LABEL: CoarsePrimitive.ROTATE,
    PrimitiveKind.BRING_CLOSER: CoarsePrimitive.MOVE_CLOSER,
    PrimitiveKind.PICK_AND_INSPECT: CoarsePrimitive.MOVE_CLOSER,
    PrimitiveKind.OPEN_CONTAINER: CoarsePrimitive.REMOVE_OCCLUDER,
    PrimitiveKind.PULL_DRAWER: CoarsePrimitive.REMOVE_OCCLUDER,
    PrimitiveKind.UNCOVER: CoarsePrimitive.REMOVE_OCCLUDER,
    PrimitiveKind.CLEAR_OCCLUDER: CoarsePrimitive.REMOVE_OCCLUDER,
    PrimitiveKind.PUSH_ASIDE: CoarsePrimitive.REMOVE_OCCLUDER,
}


PRIMITIVES_BY_COARSE: dict[CoarsePrimitive, tuple[PrimitiveKind, ...]] = {
    coarse: tuple(
        kind for kind, mapped in COARSE_BY_PRIMITIVE.items() if mapped is coarse
    )
    for coarse in COARSE_ACTION_SPACE
}


def to_coarse(kind: PrimitiveKind) -> CoarsePrimitive:
    """Project an execution primitive onto the reporting action space ``A``."""

    if not isinstance(kind, PrimitiveKind):
        raise TypeError("kind must be a PrimitiveKind")
    return COARSE_BY_PRIMITIVE[kind]


@dataclass(frozen=True)
class NoParameters:
    def to_dict(self) -> dict[str, object]:
        return {"type": "NONE"}


@dataclass(frozen=True)
class OpenParameters:
    open_fraction: float = 1.0

    def __post_init__(self) -> None:
        value = finite_number(
            self.open_fraction,
            location="open_fraction",
            minimum=0.0,
            maximum=1.0,
            exclusive_minimum=True,
        )
        if not 0.0 < value <= 1.0:
            raise ValueError("open_fraction must lie in (0, 1]")

    def to_dict(self) -> dict[str, object]:
        return {"type": "OPEN", "open_fraction": self.open_fraction}


@dataclass(frozen=True)
class RotationParameters:
    rotation_budget_degrees: float = 180.0
    max_attempts: int = 1

    def __post_init__(self) -> None:
        value = finite_number(
            self.rotation_budget_degrees,
            location="rotation_budget_degrees",
            minimum=0.0,
            maximum=360.0,
            exclusive_minimum=True,
        )
        if not 0.0 < value <= 360.0:
            raise ValueError("rotation_budget_degrees must lie in (0, 360]")
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or not 1 <= self.max_attempts <= 8
        ):
            raise ValueError("max_attempts must be an integer in [1, 8]")

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "ROTATE",
            "rotation_budget_degrees": self.rotation_budget_degrees,
            "max_attempts": self.max_attempts,
        }


@dataclass(frozen=True)
class InspectionParameters:
    target_distance_m: float = 0.22
    max_attempts: int = 1

    def __post_init__(self) -> None:
        value = finite_number(
            self.target_distance_m,
            location="target_distance_m",
            minimum=0.05,
            maximum=0.6,
        )
        if not 0.05 <= value <= 0.6:
            raise ValueError("target_distance_m must lie in [0.05, 0.6]")
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or not 1 <= self.max_attempts <= 8
        ):
            raise ValueError("max_attempts must be an integer in [1, 8]")

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "INSPECT",
            "target_distance_m": self.target_distance_m,
            "max_attempts": self.max_attempts,
        }


@dataclass(frozen=True)
class DisplacementParameters:
    maximum_displacement_m: float = 0.12
    direction_policy: str = "PUBLIC_FREE_SPACE"

    def __post_init__(self) -> None:
        value = finite_number(
            self.maximum_displacement_m,
            location="maximum_displacement_m",
            minimum=0.01,
            maximum=0.4,
        )
        if not 0.01 <= value <= 0.4:
            raise ValueError("maximum_displacement_m must lie in [0.01, 0.4]")
        if self.direction_policy not in {"PUBLIC_FREE_SPACE", "REGISTERED_DROP_ZONE"}:
            raise ValueError("unsupported displacement direction_policy")

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "DISPLACE",
            "maximum_displacement_m": self.maximum_displacement_m,
            "direction_policy": self.direction_policy,
        }


PrimitiveParameters = (
    NoParameters
    | OpenParameters
    | RotationParameters
    | InspectionParameters
    | DisplacementParameters
)


_PARAMETER_TYPE_BY_KIND: dict[PrimitiveKind, type[PrimitiveParameters]] = {
    PrimitiveKind.DIRECT_ACT: NoParameters,
    PrimitiveKind.STOP_NOT_FOUND: NoParameters,
    PrimitiveKind.OPEN_CONTAINER: OpenParameters,
    PrimitiveKind.PULL_DRAWER: OpenParameters,
    PrimitiveKind.UNCOVER: DisplacementParameters,
    PrimitiveKind.CLEAR_OCCLUDER: DisplacementParameters,
    PrimitiveKind.PUSH_ASIDE: DisplacementParameters,
    PrimitiveKind.BRING_CLOSER: InspectionParameters,
    PrimitiveKind.ROTATE_TO_LABEL: RotationParameters,
    PrimitiveKind.PICK_AND_INSPECT: InspectionParameters,
}


@dataclass(frozen=True)
class PrimitiveCall:
    candidate_id: str
    task_key: TaskKey
    kind: PrimitiveKind
    source_anchor: VisualAnchor | None
    destination_anchor: VisualAnchor | None
    parameters: PrimitiveParameters
    addresses_need_ids: tuple[str, ...]
    proposal_score: float
    proposer_id: str
    stop_condition: str

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or not self.proposer_id.strip():
            raise ValueError("candidate_id and proposer_id must be non-empty")
        proposal_score = finite_number(
            self.proposal_score, location="proposal_score", minimum=0.0
        )
        if not isfinite(proposal_score) or proposal_score < 0.0:
            raise ValueError("proposal_score must be finite and non-negative")
        if type(self.parameters) is not _PARAMETER_TYPE_BY_KIND[self.kind]:
            raise TypeError(f"{self.kind.value} received the wrong typed parameter object")
        if len(set(self.addresses_need_ids)) != len(self.addresses_need_ids) or any(
            not item.strip() for item in self.addresses_need_ids
        ):
            raise ValueError("addresses_need_ids must contain unique non-empty IDs")
        if self.kind in TERMINAL_KINDS:
            if self.addresses_need_ids:
                raise ValueError("terminal calls do not claim to resolve information needs")
        elif self.source_anchor is None or not self.addresses_need_ids:
            raise ValueError("information primitives require grounding and addressed needs")
        if self.kind is PrimitiveKind.DIRECT_ACT:
            if self.source_anchor is None or "task_target" not in self.source_anchor.affordances:
                raise ValueError("DIRECT_ACT requires a public task_target anchor")
        if self.kind is PrimitiveKind.STOP_NOT_FOUND and (
            self.source_anchor is not None or self.destination_anchor is not None
        ):
            raise ValueError("STOP_NOT_FOUND cannot carry a physical anchor")
        if self.source_anchor is not None and self.destination_anchor is not None:
            if self.source_anchor.frame_id != self.destination_anchor.frame_id:
                raise ValueError("source and destination anchors must share a frame")
        if not self.stop_condition.strip():
            raise ValueError("stop_condition must be non-empty")

    @property
    def is_terminal(self) -> bool:
        return self.kind in TERMINAL_KINDS

    @property
    def terminal_decision(self) -> TerminalDecision | None:
        if self.kind is PrimitiveKind.DIRECT_ACT:
            return TerminalDecision.DIRECT_ACT
        if self.kind is PrimitiveKind.STOP_NOT_FOUND:
            return TerminalDecision.NOT_FOUND
        return None

    @property
    def requires_effect_forecast(self) -> bool:
        """Every physical action, including DIRECT_ACT, requires an effect forecast."""

        return self.kind is not PrimitiveKind.STOP_NOT_FOUND

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "task_key": self.task_key.to_dict(),
            "kind": self.kind.value,
            "source_anchor": (
                None if self.source_anchor is None else self.source_anchor.to_dict()
            ),
            "destination_anchor": (
                None if self.destination_anchor is None else self.destination_anchor.to_dict()
            ),
            "parameters": self.parameters.to_dict(),
            "addresses_need_ids": list(self.addresses_need_ids),
            "proposal_score": self.proposal_score,
            "proposer_id": self.proposer_id,
            "stop_condition": self.stop_condition,
            "terminal_decision": (
                None if self.terminal_decision is None else self.terminal_decision.value
            ),
        }


@dataclass(frozen=True)
class CandidateSet:
    task_key: TaskKey
    observation_frame_id: str
    candidates: tuple[PrimitiveCall, ...]

    def __post_init__(self) -> None:
        if not self.observation_frame_id.strip() or not self.candidates:
            raise ValueError("candidate set must not be empty")
        ids = [item.candidate_id for item in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate IDs must be unique")
        if any(item.task_key != self.task_key for item in self.candidates):
            raise ValueError("every candidate must use the same TaskKey")
        stop_count = sum(
            item.kind is PrimitiveKind.STOP_NOT_FOUND for item in self.candidates
        )
        if stop_count != 1:
            raise ValueError("candidate set must include exactly one STOP_NOT_FOUND")

    def by_id(self, candidate_id: str) -> PrimitiveCall:
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise KeyError(candidate_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "task_key": self.task_key.to_dict(),
            "observation_frame_id": self.observation_frame_id,
            "candidates": [item.to_dict() for item in self.candidates],
        }


def _exact_public_anchor(anchor: VisualAnchor, observation: PolicyObservation) -> None:
    if anchor.frame_id != observation.frame_id:
        raise ValueError("primitive uses a stale visual anchor")
    try:
        public = observation.anchor(anchor.token)
    except KeyError as exc:
        raise ValueError("primitive invents a visual anchor") from exc
    if public != anchor:
        raise ValueError("primitive modified a public visual anchor")


def validate_candidate_set(
    candidate_set: CandidateSet,
    observation: PolicyObservation,
    needs: tuple[InformationNeed, ...],
) -> None:
    if candidate_set.observation_frame_id != observation.frame_id:
        raise ValueError("candidate set does not belong to the current observation")
    need_by_id = {item.need_id: item for item in needs}
    if len(need_by_id) != len(needs):
        raise ValueError("information need IDs must be unique")
    if any(item.task_key != candidate_set.task_key for item in needs):
        raise ValueError("information needs and candidate set use different tasks")
    for candidate in candidate_set.candidates:
        if candidate.source_anchor is not None:
            _exact_public_anchor(candidate.source_anchor, observation)
        if candidate.destination_anchor is not None:
            _exact_public_anchor(candidate.destination_anchor, observation)
        if candidate.kind is PrimitiveKind.DIRECT_ACT:
            assert candidate.source_anchor is not None
            if "task_target" not in candidate.source_anchor.affordances:
                raise ValueError("DIRECT_ACT lacks a public task_target grounding")
            continue
        if candidate.kind is PrimitiveKind.STOP_NOT_FOUND:
            continue
        assert candidate.source_anchor is not None
        required = _REQUIRED_AFFORDANCE[candidate.kind]
        if not required.issubset(set(candidate.source_anchor.affordances)):
            raise ValueError(
                f"{candidate.kind.value} lacks required public affordances: "
                f"{sorted(required)}"
            )
        unknown = set(candidate.addresses_need_ids) - set(need_by_id)
        if unknown:
            raise ValueError(f"candidate invents information needs: {sorted(unknown)}")
        for need_id in candidate.addresses_need_ids:
            need = need_by_id[need_id]
            if need.anchor_token != candidate.source_anchor.token:
                raise ValueError("candidate grounding does not match its information need")
            if candidate.kind not in _NEED_TO_FAMILIES[need.deficit_kind]:
                raise ValueError("candidate family is incompatible with its information need")


_NEED_TO_FAMILIES: dict[DeficitKind, tuple[PrimitiveKind, ...]] = {
    DeficitKind.IDENTITY_AMBIGUITY: (
        PrimitiveKind.ROTATE_TO_LABEL,
        PrimitiveKind.PICK_AND_INSPECT,
        PrimitiveKind.BRING_CLOSER,
    ),
    DeficitKind.PRESENCE_UNCERTAINTY: (
        PrimitiveKind.OPEN_CONTAINER,
        PrimitiveKind.PULL_DRAWER,
        PrimitiveKind.UNCOVER,
        PrimitiveKind.CLEAR_OCCLUDER,
    ),
    DeficitKind.OCCLUDED_REGION: (
        PrimitiveKind.OPEN_CONTAINER,
        PrimitiveKind.PULL_DRAWER,
        PrimitiveKind.UNCOVER,
        PrimitiveKind.CLEAR_OCCLUDER,
        PrimitiveKind.PUSH_ASIDE,
    ),
    DeficitKind.UNOBSERVED_SURFACE: (
        PrimitiveKind.ROTATE_TO_LABEL,
        PrimitiveKind.PICK_AND_INSPECT,
    ),
    DeficitKind.LOW_RESOLUTION: (
        PrimitiveKind.BRING_CLOSER,
        PrimitiveKind.PICK_AND_INSPECT,
        PrimitiveKind.ROTATE_TO_LABEL,
    ),
    DeficitKind.SEARCH_COVERAGE_GAP: (
        PrimitiveKind.OPEN_CONTAINER,
        PrimitiveKind.PULL_DRAWER,
        PrimitiveKind.UNCOVER,
        PrimitiveKind.CLEAR_OCCLUDER,
    ),
    DeficitKind.ACCESS_BLOCKED: (
        PrimitiveKind.OPEN_CONTAINER,
        PrimitiveKind.PULL_DRAWER,
        PrimitiveKind.UNCOVER,
        PrimitiveKind.CLEAR_OCCLUDER,
        PrimitiveKind.PUSH_ASIDE,
    ),
}


_REQUIRED_AFFORDANCE: dict[PrimitiveKind, frozenset[str]] = {
    PrimitiveKind.OPEN_CONTAINER: frozenset({"openable"}),
    PrimitiveKind.PULL_DRAWER: frozenset({"drawer"}),
    PrimitiveKind.UNCOVER: frozenset({"cover"}),
    PrimitiveKind.CLEAR_OCCLUDER: frozenset({"movable_occluder"}),
    PrimitiveKind.PUSH_ASIDE: frozenset({"pushable_occluder"}),
    PrimitiveKind.BRING_CLOSER: frozenset({"bring_closer", "graspable"}),
    PrimitiveKind.ROTATE_TO_LABEL: frozenset({"rotatable", "label_surface"}),
    PrimitiveKind.PICK_AND_INSPECT: frozenset({"inspectable", "graspable"}),
}


def _parameters(kind: PrimitiveKind) -> PrimitiveParameters:
    if kind in {PrimitiveKind.DIRECT_ACT, PrimitiveKind.STOP_NOT_FOUND}:
        return NoParameters()
    if kind in {PrimitiveKind.OPEN_CONTAINER, PrimitiveKind.PULL_DRAWER}:
        return OpenParameters()
    if kind is PrimitiveKind.ROTATE_TO_LABEL:
        return RotationParameters()
    if kind in {PrimitiveKind.BRING_CLOSER, PrimitiveKind.PICK_AND_INSPECT}:
        return InspectionParameters()
    return DisplacementParameters()


@dataclass(frozen=True)
class NeedDrivenPrimitiveProposer:
    """Translate information needs into a high-recall grounded candidate set."""

    proposer_id: str = "need_registry:v2"
    max_candidates: int = 32

    def __post_init__(self) -> None:
        if not self.proposer_id.strip() or not 2 <= self.max_candidates <= 256:
            raise ValueError("invalid proposer_id or max_candidates")

    def propose(
        self,
        *,
        task_key: TaskKey,
        observation: PolicyObservation,
        needs: tuple[InformationNeed, ...],
    ) -> CandidateSet:
        direct_anchor = next(
            (
                anchor
                for anchor in observation.anchors
                if "task_target" in set(anchor.affordances)
            ),
            None,
        )
        candidates: list[PrimitiveCall] = [
            PrimitiveCall(
                candidate_id="terminal::not_found",
                task_key=task_key,
                kind=PrimitiveKind.STOP_NOT_FOUND,
                source_anchor=None,
                destination_anchor=None,
                parameters=NoParameters(),
                addresses_need_ids=(),
                proposal_score=0.0,
                proposer_id=self.proposer_id,
                stop_condition="terminal_bounded_search_commit",
            ),
        ]
        if direct_anchor is not None:
            candidates.append(
                PrimitiveCall(
                    candidate_id="terminal::direct",
                    task_key=task_key,
                    kind=PrimitiveKind.DIRECT_ACT,
                    source_anchor=direct_anchor,
                    destination_anchor=None,
                    parameters=NoParameters(),
                    addresses_need_ids=(),
                    proposal_score=0.0,
                    proposer_id=self.proposer_id,
                    stop_condition="terminal_task_execution",
                )
            )
        merged: dict[tuple[PrimitiveKind, str], list[InformationNeed]] = {}
        for need in needs:
            if need.task_key != task_key:
                raise ValueError("information need belongs to another task")
            try:
                anchor = observation.anchor(need.anchor_token)
            except KeyError as exc:
                raise ValueError("information need references an unknown public anchor") from exc
            affordances = set(anchor.affordances)
            for kind in _NEED_TO_FAMILIES[need.deficit_kind]:
                if not _REQUIRED_AFFORDANCE[kind].issubset(affordances):
                    continue
                merged.setdefault((kind, anchor.token), []).append(need)

        ranked_merged = sorted(
            merged.items(),
            key=lambda item: (
                -max(need.priority for need in item[1]),
                item[0][0].value,
                item[0][1],
            ),
        )
        for (kind, anchor_token), addressed in ranked_merged:
            if len(candidates) >= self.max_candidates:
                break
            anchor = observation.anchor(anchor_token)
            needs_sorted = tuple(sorted(addressed, key=lambda item: item.need_id))
            candidates.append(
                PrimitiveCall(
                    candidate_id=f"{kind.value.lower()}::{anchor_token}",
                    task_key=task_key,
                    kind=kind,
                    source_anchor=anchor,
                    destination_anchor=None,
                    parameters=_parameters(kind),
                    addresses_need_ids=tuple(item.need_id for item in needs_sorted),
                    proposal_score=max(item.priority for item in needs_sorted),
                    proposer_id=self.proposer_id,
                    stop_condition="reobserve_public_sensors_after_primitive",
                )
            )
        result = CandidateSet(task_key, observation.frame_id, tuple(candidates))
        validate_candidate_set(result, observation, needs)
        return result
