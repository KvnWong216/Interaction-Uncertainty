"""Belief over where the prompt's target is, and the action that follows from it.

This module replaces two earlier attempts. ``vla_evidence`` read the belief out
of the policy's own action spread, which measures whether the policy's samples
agree rather than whether the scene has been observed -- on a tidy kitchen with
the butter shut in a drawer the samples agree perfectly and the target's
location is still unknown. ``prompt_conditioned`` fixed the conditioning but
still described occlusion with a single ``observable`` flag, which cannot
distinguish the two kinds of hiding this benchmark actually contains.

The structure, and why it is not the standard one
-------------------------------------------------
Interactive perception has treated occlusion as a visibility problem since
Bohg et al. (T-RO 2017), and the volumetric line of work -- Novkovic et al.,
"Object Finding in Cluttered Scenes Using Interactive Perception" (ICRA 2020)
-- represents it as unobserved free space, choosing pushes that shrink it.
That is the right model for clutter, and it is the wrong model for a closed
drawer: no camera pose and no push reveals the inside of a shut container, only
actuating its joint does. Our viewpoint certification measures exactly this
distinction, so the hypothesis space carries it explicitly:

``OBSERVED``            evidence flows; the target would be seen if it were here
``VIEWPOINT_BLOCKED``   hidden now, but some reachable viewpoint would reveal it
``MANIPULATION_ONLY``   no viewpoint reveals it; a joint must move
``ABSENT``              the target is not in the scene at all

Splitting the hidden mass three ways is what lets one belief select among
moving closer, opening a container, and declining -- decisions that a single
scalar uncertainty cannot tell apart.

Formalism
---------
A Dirichlet over hypotheses, read as a subjective-logic opinion (Jøsang,
*Subjective Logic*, 2016; Sensoy et al., "Evidential Deep Learning to Quantify
Classification Uncertainty", NeurIPS 2018)::

    alpha_k = W * a_k  +  omega_k * e_k * [k is OBSERVED]
    S       = sum_k alpha_k
    u       = W / S

``a_k`` is a language-grounded containment prior, of the kind ESC (Zhou et al.,
ICML 2023) and SayPlan (Rana et al., CoRL 2023) obtain from an LLM and
ConceptGraphs (Gu et al., ICRA 2024) attaches to a 3D scene graph. ``omega_k``
is the prompt-to-node relevance, e.g. a CLIP text-image similarity (Radford et
al., ICML 2021). Evidence is admitted only for hypotheses that were actually
observable: a shut drawer accumulates none, so its mass stays at the prior and
``u`` stays high without any hand-added penalty term.

Choosing the action is one-step Bayes risk over the terminal decisions
(Kaelbling, Littman and Cassandra, *Artificial Intelligence*, 1998): commit,
or pay for information. That is what makes ``NOT_FOUND`` a decision rather than
a timeout -- it is chosen when the expected loss of declaring absence falls
below the expected loss of continuing to search.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

__all__ = [
    "EVALUATOR_ONLY_ROLES",
    "InformationAction",
    "LocationHypothesis",
    "Reachability",
    "TargetBelief",
    "build_hypotheses",
    "relevance_weights",
    "select_action",
    "target_belief",
]

EVALUATOR_ONLY_ROLES = frozenset({"task_target"})
"""Benchmark anchor roles the controller must never receive.

``task_target`` resolves to the pose of the object the task is about, which on
every scene worth measuring is the object that starts hidden. Reading it would
make the belief a function of privileged simulator state.
"""


class Reachability(str, Enum):
    """How, if at all, evidence about a hypothesis can be obtained."""

    OBSERVED = "OBSERVED"
    VIEWPOINT_BLOCKED = "VIEWPOINT_BLOCKED"
    MANIPULATION_ONLY = "MANIPULATION_ONLY"
    ABSENT = "ABSENT"


class InformationAction(str, Enum):
    """Coarse primitives, as the benchmark's action space names them."""

    ACT = "ACT"
    NOT_FOUND = "NOT_FOUND"
    MOVE_CLOSER = "MOVE_CLOSER"
    ROTATE = "ROTATE"
    REMOVE_OCCLUDER = "REMOVE_OCCLUDER"


@dataclass(frozen=True)
class LocationHypothesis:
    """One candidate location for the target, and how it could be checked."""

    label: str
    reachability: Reachability
    position: tuple[float, float, float] | None = None
    resolving_action: InformationAction | None = None

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("label must be non-empty")
        if self.reachability is Reachability.ABSENT and self.position is not None:
            raise ValueError("the ABSENT hypothesis cannot have a position")
        if (
            self.reachability
            in (Reachability.VIEWPOINT_BLOCKED, Reachability.MANIPULATION_ONLY)
            and self.resolving_action is None
        ):
            raise ValueError(
                f"{self.label} is hidden but names no resolving action; a "
                "hypothesis the robot cannot act on can never be resolved and "
                "would hold uncertainty open forever"
            )

    @property
    def accepts_evidence(self) -> bool:
        """Whether an observation can bear on this hypothesis.

        ``ABSENT`` accepts evidence even though absence is never seen directly.
        Its evidence source is a search that came back empty -- "the drawer is
        open and the butter is not in it" is an observation, and it is the only
        one that can ever justify declining. Excluding it leaves the belief
        unable to reach ``NOT_FOUND`` by any amount of searching, which turns
        abstention back into a timeout.
        """

        return self.reachability in (Reachability.OBSERVED, Reachability.ABSENT)


def build_hypotheses(
    declared_anchors: Sequence[dict],
    *,
    reachability: dict[str, Reachability],
    resolve: Callable[[str], Sequence[float]],
    include_absent: bool = True,
) -> tuple[LocationHypothesis, ...]:
    """Build the hypothesis space from a benchmark task entry.

    ``resolve`` is only ever called for anchors the controller may see, so an
    evaluator-private resolver can be passed without leaking through: the
    hidden target's position is not merely unused, it is never computed.
    """

    hypotheses: list[LocationHypothesis] = []
    for anchor in declared_anchors:
        role = str(anchor.get("role", ""))
        if role in EVALUATOR_ONLY_ROLES:
            continue
        label = str(anchor["label"])
        kind = reachability.get(label, Reachability.OBSERVED)
        action = {
            Reachability.MANIPULATION_ONLY: InformationAction.REMOVE_OCCLUDER,
            Reachability.VIEWPOINT_BLOCKED: InformationAction.MOVE_CLOSER,
        }.get(kind)
        hypotheses.append(
            LocationHypothesis(
                label=label,
                reachability=kind,
                position=tuple(resolve(str(anchor["ref"]))),
                resolving_action=action,
            )
        )
    if include_absent:
        hypotheses.append(
            LocationHypothesis(label="ABSENT", reachability=Reachability.ABSENT)
        )
    return tuple(hypotheses)


def relevance_weights(
    labels: Sequence[str],
    prompt: str,
    *,
    scorer: Callable[[str, str], float],
    temperature: float = 1.0,
) -> np.ndarray:
    r"""``omega_k = softmax(score(prompt, label_k) / tau)``.

    ``scorer`` is the seam for a CLIP text encoder or an LLM: the uncertainty
    definition does not depend on which model supplies the commonsense, and a
    fixed table stands in for it under test.
    """

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    raw = np.array([float(scorer(prompt, label)) for label in labels], dtype=np.float64)
    if not np.all(np.isfinite(raw)):
        raise ValueError("relevance scores must be finite")
    shifted = (raw - raw.max()) / temperature
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum()


@dataclass(frozen=True)
class TargetBelief:
    """A Dirichlet over target location, read as a subjective-logic opinion."""

    hypotheses: tuple[LocationHypothesis, ...]
    alpha: tuple[float, ...]
    prior_weight: float

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(item.label for item in self.hypotheses)

    @property
    def strength(self) -> float:
        return float(sum(self.alpha))

    @property
    def vacuity(self) -> float:
        """``u = W / S``.

        The numerator is the prior weight, not the hypothesis count. Those
        coincide only under one-pseudo-count-per-category, and writing ``K / S``
        with any other prior lets ``u`` exceed one, which would make a router
        fire on scenes merely for having many candidate locations.
        """

        return self.prior_weight / self.strength

    @property
    def probability(self) -> tuple[float, ...]:
        total = self.strength
        return tuple(float(value / total) for value in self.alpha)

    def mass_where(self, *kinds: Reachability) -> float:
        return float(
            sum(
                probability
                for probability, hypothesis in zip(
                    self.probability, self.hypotheses, strict=True
                )
                if hypothesis.reachability in kinds
            )
        )

    @property
    def manipulation_mass(self) -> float:
        """Belief that the target sits where only manipulation can reveal it."""

        return self.mass_where(Reachability.MANIPULATION_ONLY)

    @property
    def viewpoint_mass(self) -> float:
        return self.mass_where(Reachability.VIEWPOINT_BLOCKED)

    @property
    def absent_mass(self) -> float:
        return self.mass_where(Reachability.ABSENT)

    @property
    def observed_mass(self) -> float:
        return self.mass_where(Reachability.OBSERVED)

    def to_dict(self) -> dict[str, object]:
        return {
            "labels": list(self.labels),
            "alpha": list(self.alpha),
            "probability": list(self.probability),
            "vacuity": self.vacuity,
            "observed_mass": self.observed_mass,
            "viewpoint_mass": self.viewpoint_mass,
            "manipulation_mass": self.manipulation_mass,
            "absent_mass": self.absent_mass,
        }


def target_belief(
    *,
    prompt: str,
    hypotheses: Sequence[LocationHypothesis],
    observation_evidence: Sequence[float],
    containment_prior: dict[str, float],
    scorer: Callable[[str, str], float],
    prior_weight: float = 2.0,
    relevance_temperature: float = 1.0,
    prior_temperature: float = 1.0,
) -> TargetBelief:
    r"""``alpha_k = W a_k + omega_k e_k [k is OBSERVED]``.

    Evidence offered for a hypothesis that was not observable is discarded
    rather than trusted. If the drawer is shut, no amount of model confidence
    about its contents is an observation, and accepting it is exactly how a
    system talks itself out of opening the drawer.
    """

    if len(hypotheses) < 2:
        raise ValueError("at least two hypotheses are required")
    if len(observation_evidence) != len(hypotheses):
        raise ValueError("observation_evidence must align with hypotheses")
    if prior_weight <= 0.0:
        raise ValueError("prior_weight must be positive")
    evidence = np.asarray(observation_evidence, dtype=np.float64)
    if np.any(evidence < 0.0) or not np.all(np.isfinite(evidence)):
        raise ValueError("observation evidence must be finite and non-negative")

    labels = [item.label for item in hypotheses]
    if len(set(labels)) != len(labels):
        raise ValueError("hypothesis labels must be unique")

    omega = relevance_weights(
        labels, prompt, scorer=scorer, temperature=relevance_temperature
    )
    raw_prior = np.array(
        [containment_prior.get(label, 0.0) for label in labels], dtype=np.float64
    )
    shifted = (raw_prior - raw_prior.max()) / prior_temperature
    base = np.exp(shifted)
    base = base / base.sum()

    # Relevance down-weights a *location* by how plausibly the prompt's target
    # belongs there. That question is meaningless for ABSENT: "the target is
    # nowhere" is exactly as pertinent whatever the target is, so scoring it
    # against the prompt would let a semantically odd target suppress the only
    # hypothesis that can justify declining.
    is_absent = np.array(
        [item.reachability is Reachability.ABSENT for item in hypotheses], dtype=bool
    )

    # Evidence of absence is only as strong as the search was exhaustive. A
    # report of "not found" while a drawer is still shut is a statement about
    # where the robot has looked, not about where the target is, and admitting
    # it at full strength is what makes a system declare NOT_FOUND before it
    # has opened anything. Scale it by the share of candidate locations that
    # have actually been resolved; with everything still hidden the factor is
    # zero and no amount of looking at the tabletop can justify declining.
    located = ~is_absent
    searched = np.array([item.accepts_evidence for item in hypotheses], dtype=bool) & located
    located_prior = float(base[located].sum())
    exhaustiveness = (
        float(base[searched].sum()) / located_prior if located_prior > 0.0 else 0.0
    )

    weight = np.where(is_absent, exhaustiveness, omega)
    admits = np.array([item.accepts_evidence for item in hypotheses], dtype=bool)
    alpha = prior_weight * base + np.where(admits, weight * evidence, 0.0)
    return TargetBelief(
        hypotheses=tuple(hypotheses), alpha=tuple(alpha.tolist()), prior_weight=prior_weight
    )


def select_action(
    belief: TargetBelief,
    *,
    cost_of_information: float = 0.1,
    loss_false_commit: float = 1.0,
    loss_false_absent: float = 1.0,
) -> tuple[InformationAction, str | None, dict[str, float]]:
    """One-step Bayes risk over commit, decline, or pay for information.

    Three expected losses are compared directly, so the choice between acting,
    declining and seeking information comes out of the same arithmetic rather
    than a threshold on a scalar:

    * ``ACT`` risks committing while the target is elsewhere;
    * ``NOT_FOUND`` risks declaring absence while the target is merely hidden;
    * an information action pays a fixed cost and resolves the mass sitting on
      the hypotheses that action would reveal.

    Returning the risks alongside the choice keeps the decision auditable: a
    reader can see *why* a primitive won, not merely that it did.
    """

    risk_act = loss_false_commit * (1.0 - belief.observed_mass)
    risk_absent = loss_false_absent * (1.0 - belief.absent_mass)

    by_action: dict[InformationAction, float] = {}
    resolves: dict[InformationAction, str] = {}
    for probability, hypothesis in zip(belief.probability, belief.hypotheses, strict=True):
        action = hypothesis.resolving_action
        if action is None:
            continue
        # Resolving a hypothesis is worth the mass it would settle, so the
        # cheapest useful action wins rather than the most drastic one.
        gain = by_action.get(action, 0.0) + probability
        by_action[action] = gain
        if resolves.get(action) is None or gain > probability:
            resolves.setdefault(action, hypothesis.label)

    risks: dict[str, float] = {
        InformationAction.ACT.value: risk_act,
        InformationAction.NOT_FOUND.value: risk_absent,
    }
    best_information: tuple[float, InformationAction, str] | None = None
    for action, gain in by_action.items():
        risk = loss_false_commit * (1.0 - gain) + cost_of_information
        risks[action.value] = risk
        candidate = (risk, action, resolves[action])
        if best_information is None or candidate[0] < best_information[0]:
            best_information = candidate

    choice = min(risks.items(), key=lambda item: item[1])[0]
    if best_information is not None and choice == best_information[1].value:
        return best_information[1], best_information[2], risks
    return InformationAction(choice), None, risks
