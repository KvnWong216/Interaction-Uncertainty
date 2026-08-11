"""Prompt-conditioned uncertainty over a containment-aware hypothesis space.

The action-induced belief in :mod:`.vla_bridge` answers "how much do the
policy's action samples agree?". That is not the question the task asks. On a
clean kitchen whose butter is inside a shut drawer, the scene is visually
unambiguous and the samples may agree perfectly, yet the *target's location* is
maximally uncertain. Any uncertainty that falls when the tabletop is tidy is
measuring the wrong thing.

What follows conditions the whole quantity on the prompt, in three parts.

1. **Relevance.** A scene node earns evidence weight only in proportion to how
   plausibly it relates to the prompt's target, ``omega_k``. A drawer is a
   candidate location for butter; the wall behind it is not.

2. **Evidence.** Action samples are attributed across nodes and weighted by
   decisiveness *and* relevance, giving a Dirichlet over "where the target is".

3. **Containment.** A closed container yields no observation of its interior.
   The subtle part is what to do about that.

Why containment is prior mass, not an additive penalty
------------------------------------------------------
The natural first formulation adds a term::

    U_final = u_target + lambda * sum_c P(target in c | prompt) * [c is closed]

It has the right instinct and the wrong algebra. ``u_target`` is a vacuity in
``[0, 1]``; adding an unbounded positive term breaks that range, so ``U_final``
stops being comparable across scenes with different container counts, and
``lambda`` has to be retuned whenever the scene changes. Worse, it is applied
*after* the belief is formed, so the containment knowledge never reaches the
planner's posterior -- it only decorates the scalar.

Putting the same knowledge in as **prior mass on hypotheses that cannot yet be
observed** gets the behaviour for free and keeps ``u in [0, 1]``:

* a closed container is a hypothesis that collects no evidence, so its
  Dirichlet mass stays at the prior and total strength ``S`` stays low, which is
  *exactly* high vacuity -- no extra term required;
* opening it converts prior mass into evidence, and vacuity falls because the
  observation happened, not because a hand-tuned constant switched off;
* on a visible-target scene the relevant mass sits on an observable hypothesis
  from the start, so vacuity is low immediately.

The containment prior therefore enters as the Dirichlet base rate, and the
uncertainty stays a single well-formed subjective-logic quantity throughout.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "ContainmentPrior",
    "PromptConditionedBelief",
    "SceneNode",
    "prompt_conditioned_belief",
    "relevance_weights",
]


@dataclass(frozen=True)
class SceneNode:
    """One candidate location for the prompt's target.

    ``observable`` is the load-bearing flag: a shut drawer is a location the
    robot can reason about but cannot see into, and it is the inability to
    gather evidence -- not any explicit penalty -- that keeps uncertainty high.
    """

    label: str
    observable: bool
    position: tuple[float, float, float] | None = None
    is_container: bool = False

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("label must be non-empty")


@dataclass(frozen=True)
class ContainmentPrior:
    """P(target in node | prompt), from commonsense rather than from the sim.

    ``scores`` are unnormalised affinities -- VLM logits, an LLM's ranking, or a
    hand-specified table. They are normalised here, so the caller never has to
    supply a distribution.
    """

    scores: dict[str, float]
    temperature: float = 1.0
    weight: float = 2.0
    """Total prior mass, in evidence units. This is the Dirichlet's ``W``."""

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.weight <= 0.0:
            raise ValueError("prior weight must be positive")
        if any(not np.isfinite(value) for value in self.scores.values()):
            raise ValueError("containment scores must be finite")

    def base_rate(self, labels: Sequence[str]) -> np.ndarray:
        raw = np.array(
            [self.scores.get(label, 0.0) for label in labels], dtype=np.float64
        )
        shifted = (raw - raw.max()) / self.temperature
        exponentiated = np.exp(shifted)
        return exponentiated / exponentiated.sum()


def relevance_weights(
    labels: Sequence[str],
    prompt: str,
    *,
    scorer: Callable[[str, str], float],
    temperature: float = 1.0,
) -> np.ndarray:
    r"""omega_k = softmax(score(prompt, node_k) / tau).

    ``scorer`` is the seam for a VLM: give it ``(prompt, label)`` and return a
    logit. Keeping it a callable means the uncertainty definition does not
    depend on which vision-language model supplies the commonsense, and a fixed
    table can stand in for it in tests.
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
class PromptConditionedBelief:
    """A Dirichlet over target location, plus the readings taken from it."""

    labels: tuple[str, ...]
    alpha: tuple[float, ...]
    relevance: tuple[float, ...]
    evidence: tuple[float, ...]
    prior_mass: tuple[float, ...]
    prior_weight: float

    @property
    def strength(self) -> float:
        return float(sum(self.alpha))

    @property
    def vacuity(self) -> float:
        """u = W / S, the multinomial subjective-logic vacuity.

        It is ``W``, the non-informative prior weight, and not ``K``, the
        number of hypotheses. The two coincide only under the convention that
        every category starts with one pseudo-count, and writing ``K / S`` with
        any other prior lets vacuity exceed 1 -- which is not a conservative
        error, because a router thresholding on it would then fire on scenes
        purely for having many candidate locations.
        """

        return self.prior_weight / self.strength

    @property
    def expected_probability(self) -> tuple[float, ...]:
        total = self.strength
        return tuple(float(value / total) for value in self.alpha)

    @property
    def unobserved_mass(self) -> float:
        """Probability the target sits somewhere no evidence could reach.

        This is the quantity that should drive an interactive action: it is
        high exactly when the belief's own support lies behind a door.
        """

        return float(
            sum(
                probability
                for probability, prior in zip(
                    self.expected_probability, self.prior_mass, strict=True
                )
                if prior > 0.0
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "labels": list(self.labels),
            "alpha": list(self.alpha),
            "vacuity": self.vacuity,
            "expected_probability": list(self.expected_probability),
            "unobserved_mass": self.unobserved_mass,
        }


def prompt_conditioned_belief(
    *,
    prompt: str,
    nodes: Sequence[SceneNode],
    observation_evidence: Sequence[float],
    containment: ContainmentPrior,
    scorer: Callable[[str, str], float],
    relevance_temperature: float = 1.0,
) -> PromptConditionedBelief:
    r"""Fuse relevance, observation and containment into one Dirichlet.

    .. math::

        \alpha_k = \underbrace{W\,a_k}_{\text{containment prior}}
                 + \underbrace{\omega_k\, e_k\, \mathbb{I}[\text{observable}_k]}
                   _{\text{prompt-weighted evidence}}

    ``observation_evidence`` is whatever the perception stack can defend for
    each node -- decisiveness-weighted action attribution, open-vocabulary
    segmentation confidence, or both summed. Evidence offered for an
    unobservable node is discarded rather than trusted: if the drawer is shut,
    no amount of model confidence about its contents is an observation, and
    accepting it is precisely how a system talks itself out of opening the
    drawer.
    """

    if len(nodes) < 2:
        raise ValueError("at least two candidate locations are required")
    if len(observation_evidence) != len(nodes):
        raise ValueError("observation_evidence must align with nodes")
    evidence = np.asarray(observation_evidence, dtype=np.float64)
    if np.any(evidence < 0.0) or not np.all(np.isfinite(evidence)):
        raise ValueError("observation evidence must be finite and non-negative")

    labels = tuple(node.label for node in nodes)
    if len(set(labels)) != len(labels):
        raise ValueError("node labels must be unique")

    omega = relevance_weights(
        labels, prompt, scorer=scorer, temperature=relevance_temperature
    )
    base = containment.base_rate(labels)
    prior = containment.weight * base

    observable = np.array([node.observable for node in nodes], dtype=bool)
    effective = np.where(observable, omega * evidence, 0.0)

    alpha = prior + effective
    return PromptConditionedBelief(
        labels=labels,
        alpha=tuple(alpha.tolist()),
        relevance=tuple(omega.tolist()),
        evidence=tuple(effective.tolist()),
        prior_mass=tuple(np.where(observable, 0.0, prior).tolist()),
        prior_weight=containment.weight,
    )
