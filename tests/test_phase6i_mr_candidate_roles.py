from dataclasses import replace
from pathlib import Path

from rcias_clgri.analysis.phase6i_mr import (
    FrozenArmPrediction,
    decode_forced_candidate,
    select_forced_candidate_roles,
    select_top_eight_audit_candidates,
)
from rcias_clgri.data.loader import load_instance
from rcias_clgri.heuristic.dispatching import solve_dispatching
from rcias_clgri.search.common import candidate_from_actions, decode_candidate


ROOT = Path(__file__).resolve().parents[1]


def arm(
    target_id: str,
    operations: tuple[str, ...],
    score: float,
    *,
    rule: str = "matched_random_1",
    family: str = "MATCHED_RANDOM",
    operator: str = "random",
) -> FrozenArmPrediction:
    return FrozenArmPrediction(
        target_id,
        family,
        operator,
        (rule,),
        operations,
        score,
        0.5,
        0.0,
        0.5,
        0.0,
    )


def test_four_role_selection_is_deterministic_outcome_blind_and_unique():
    candidates = [
        arm("top", ("o1", "o2"), 4.0),
        arm("second", ("o1", "o3"), 3.0),
        arm(
            "fallback",
            ("o2", "o3"),
            1.0,
            rule="operator_related",
            family="ORIGINAL_OPERATOR",
            operator="related",
        ),
        arm("diverse", ("o4", "o5"), 2.0),
        arm("duplicate", ("o1", "o2"), 99.0),
    ]
    first = select_forced_candidate_roles(candidates)
    modified_outcomes = [
        replace(candidate, raw_utility=1000.0 - index)
        for index, candidate in enumerate(candidates)
    ]
    second = select_forced_candidate_roles(modified_outcomes)
    assert [(row.role, row.arm.destroyed_operations) for row in first] == [
        (row.role, row.arm.destroyed_operations) for row in second
    ]
    assert [row.role for row in first] == [
        "FROZEN_NEURAL_TOP1",
        "FROZEN_NEURAL_TOP2",
        "ALNS_RELATED_FALLBACK",
        "DETERMINISTIC_DIVERSE",
    ]
    assert len({row.arm.destroyed_operations for row in first}) == 4
    assert first[2].arm.target_set_id == "fallback"
    assert first[3].arm.target_set_id == "diverse"


def test_duplicate_role_candidate_is_replaced_deterministically():
    candidates = [
        arm(
            "related_top",
            ("o1", "o2"),
            4.0,
            rule="operator_related",
            family="ORIGINAL_OPERATOR",
            operator="related",
        ),
        arm("second", ("o1", "o3"), 3.0),
        arm("replacement", ("o2", "o4"), 2.0, operator="related"),
        arm("diverse", ("o5", "o6"), 1.0),
    ]
    selected = select_forced_candidate_roles(candidates)
    assert selected[2].role == "ALNS_RELATED_FALLBACK"
    assert selected[2].arm.target_set_id == "replacement"
    assert selected[2].replacement is True


def test_all_eight_audit_retains_four_roles_then_fills_by_frozen_score():
    candidates = [
        arm(f"candidate_{index}", (f"o{index}",), float(20 - index))
        for index in range(10)
    ]
    candidates[5] = arm(
        "fallback",
        ("o5",),
        1.0,
        rule="operator_related",
        family="ORIGINAL_OPERATOR",
        operator="related",
    )
    broad = select_forced_candidate_roles(candidates)
    audit = select_top_eight_audit_candidates(candidates, broad)
    broad_targets = {selection.arm.destroyed_operations for selection in broad}
    assert len(audit) == 8
    assert broad_targets.issubset({candidate.destroyed_operations for candidate in audit})
    assert len({candidate.destroyed_operations for candidate in audit}) == 8
    modified_outcomes = [
        replace(candidate, raw_utility=float(index * 100))
        for index, candidate in enumerate(candidates)
    ]
    replay = select_top_eight_audit_candidates(
        modified_outcomes, select_forced_candidate_roles(modified_outcomes)
    )
    assert [candidate.destroyed_operations for candidate in audit] == [
        candidate.destroyed_operations for candidate in replay
    ]


def test_forced_labels_are_deterministic_feasible_and_positive_sign_is_improvement():
    instance = load_instance(ROOT / "instances/tiny/tiny_03.json")
    h1 = solve_dispatching(instance, "H1")
    current = decode_candidate(instance, candidate_from_actions(instance, h1.actions))
    candidate = arm(
        "forced",
        tuple(instance.operations[:2]),
        1.0,
        rule="operator_related",
        family="ORIGINAL_OPERATOR",
        operator="related",
    )
    first = decode_forced_candidate(
        instance,
        current,
        candidate,
        state_id="test_state",
        repair_seed_namespace=684000000,
        candidate_trials=3,
    )
    second = decode_forced_candidate(
        instance,
        current,
        candidate,
        state_id="test_state",
        repair_seed_namespace=684000000,
        candidate_trials=3,
    )
    assert first.candidate.feasible
    assert first.trial_makespans == second.trial_makespans
    utility = (current.makespan - first.candidate.makespan) / current.makespan
    assert (utility > 0) == (first.candidate.makespan < current.makespan)
