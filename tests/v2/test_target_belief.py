"""Target-location belief and the primitive chosen from it."""

from __future__ import annotations

import pytest

from interaction_uncertainty.v2.target_belief import (
    EVALUATOR_ONLY_ROLES,
    InformationAction,
    LocationHypothesis,
    Reachability,
    build_hypotheses,
    select_action,
    target_belief,
)

R = Reachability
A = InformationAction

PRIOR = {"drawer": 3.0, "tabletop": 1.0, "behind_cans": 2.5, "front_row": 1.0, "ABSENT": 0.0}
SCORES = dict(PRIOR, ABSENT=0.5)


def score(_prompt: str, label: str) -> float:
    return SCORES.get(label, 1.0)


def believe(hypotheses, evidence):
    return target_belief(
        prompt="find the butter",
        hypotheses=hypotheses,
        observation_evidence=evidence,
        containment_prior=PRIOR,
        scorer=score,
    )


def drawer_shut() -> LocationHypothesis:
    return LocationHypothesis("drawer", R.MANIPULATION_ONLY, (0.0, 0.0, 1.0), A.REMOVE_OCCLUDER)


def drawer_open() -> LocationHypothesis:
    return LocationHypothesis("drawer", R.OBSERVED, (0.0, 0.0, 1.0))


def tabletop() -> LocationHypothesis:
    return LocationHypothesis("tabletop", R.OBSERVED, (0.0, 1.0, 1.0))


def absent() -> LocationHypothesis:
    return LocationHypothesis("ABSENT", R.ABSENT)


def test_a_clean_tabletop_does_not_resolve_a_shut_drawer() -> None:
    """The failure that motivates the whole module.

    Every pixel of the table is accounted for and the target is still missing,
    so uncertainty must stay high rather than fall with the visual clutter.
    """

    belief = believe([drawer_shut(), tabletop(), absent()], [0.0, 6.0, 0.0])
    assert belief.manipulation_mass > belief.observed_mass
    assert belief.vacuity > 0.5
    action, target, _ = select_action(belief)
    assert action is A.REMOVE_OCCLUDER
    assert target == "drawer"


def test_a_visible_target_is_grasped_rather_than_investigated() -> None:
    belief = believe([drawer_shut(), tabletop(), absent()], [0.0, 60.0, 0.0])
    assert belief.observed_mass > 0.7
    assert select_action(belief)[0] is A.ACT


def test_free_space_occlusion_selects_a_viewpoint_action() -> None:
    """Clutter and a shut container are different problems, and must differ here."""

    belief = believe(
        [
            LocationHypothesis("behind_cans", R.VIEWPOINT_BLOCKED, (0.0, 0.0, 1.0), A.MOVE_CLOSER),
            LocationHypothesis("front_row", R.OBSERVED, (0.0, 1.0, 1.0)),
            absent(),
        ],
        [0.0, 4.0, 0.0],
    )
    assert belief.viewpoint_mass > belief.manipulation_mass
    assert select_action(belief)[0] is A.MOVE_CLOSER


def test_absence_is_not_declared_while_a_container_is_still_shut() -> None:
    """Premature abstention is the failure mode that makes NOT_FOUND worthless."""

    belief = believe([drawer_shut(), tabletop(), absent()], [0.0, 0.0, 3.0])
    action, _, _ = select_action(belief)
    assert action is A.REMOVE_OCCLUDER


def test_absence_is_declared_once_the_search_is_exhaustive() -> None:
    belief = believe([drawer_open(), tabletop(), absent()], [0.0, 0.0, 40.0])
    assert belief.absent_mass > 0.9
    assert select_action(belief)[0] is A.NOT_FOUND


def test_evidence_for_an_unobservable_hypothesis_is_discarded() -> None:
    """Confidence about a closed drawer's contents is not an observation."""

    shut = believe([drawer_shut(), tabletop(), absent()], [50.0, 0.0, 0.0])
    nothing = believe([drawer_shut(), tabletop(), absent()], [0.0, 0.0, 0.0])
    assert shut.alpha == nothing.alpha


def test_vacuity_stays_within_the_unit_interval() -> None:
    for evidence in ([0.0, 0.0, 0.0], [6.0, 6.0, 6.0], [0.0, 100.0, 0.0]):
        belief = believe([drawer_open(), tabletop(), absent()], evidence)
        assert 0.0 < belief.vacuity <= 1.0


def test_evaluator_private_anchors_are_never_resolved() -> None:
    resolved: list[str] = []

    def resolve(ref: str) -> tuple[float, float, float]:
        resolved.append(ref)
        return (0.0, 0.0, 1.0)

    built = build_hypotheses(
        [
            {"label": "drawer", "role": "occluder", "ref": "wooden_cabinet_1"},
            {"label": "butter", "role": "task_target", "ref": "butter_1"},
        ],
        reachability={"drawer": R.MANIPULATION_ONLY},
        resolve=resolve,
    )
    assert "task_target" in EVALUATOR_ONLY_ROLES
    assert "butter_1" not in resolved
    assert [item.label for item in built] == ["drawer", "ABSENT"]


def test_a_hidden_hypothesis_without_a_resolving_action_is_rejected() -> None:
    with pytest.raises(ValueError, match="names no resolving action"):
        LocationHypothesis("drawer", R.MANIPULATION_ONLY, (0.0, 0.0, 1.0))


def test_risks_are_returned_so_a_choice_can_be_audited() -> None:
    belief = believe([drawer_shut(), tabletop(), absent()], [0.0, 6.0, 0.0])
    _, _, risks = select_action(belief)
    assert {"ACT", "NOT_FOUND", "REMOVE_OCCLUDER"} <= set(risks)
    assert all(value == value for value in risks.values())
