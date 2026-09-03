from __future__ import annotations

import random
import math

from rcias_clgri.search.common import Candidate, decode_candidate, random_candidate
from rcias_clgri.search.dabc import (
    DABCConfig,
    DABCIndividual,
    _self_explore,
    hierarchy_first_level,
    solve_dabc,
)
from rcias_clgri.search.dabc_chdg_rules import (
    SOURCE_EXACT,
    audit_intermachine_move,
    audit_intramachine_move,
    source_projection_metrics,
)
from rcias_clgri.search.dabc_chdg import CriticalIslandBlock


class _DecodedStub:
    def __init__(self, makespan):
        self.makespan = makespan


def test_dabc_source_defaults_are_frozen():
    config = DABCConfig()
    assert config.population_size == 120
    assert config.neighborhood_search_intensity == 2
    assert config.max_nonimprovements == 40
    assert config.self_exploration_rate == 0.1
    assert config.first_level_fraction == 0.2
    assert config.restart_self_explorations == 10
    assert config.source_clipping_mode == "shadow"


def test_default_initialization_has_120_feasible_domain_valid_candidates(automotive_instance):
    result = solve_dabc(automotive_instance, 0.0, 7)
    assert result.decoder_evaluations == 120
    assert result.diagnostics["population_size"] == 120
    assert result.diagnostics["first_level_size"] == 24
    assert result.best.feasible


def test_hierarchy_uses_stable_makespan_order_and_ceil_rounding():
    population = [
        DABCIndividual(index, _DecodedStub(value))  # type: ignore[arg-type]
        for index, value in enumerate((9, 7, 7, 8, 6, 10))
    ]
    first = hierarchy_first_level(population, 0.2)
    assert [item.individual_id for item in first] == [4, 1]


def test_self_exploration_sequence_branch_is_insertion(automotive_instance):
    base = random_candidate(automotive_instance, random.Random(5))
    explored, detail = _self_explore(automotive_instance, base, random.Random(1))
    assert detail["branch"] == "sequence"
    assert detail["operator"] == "insertion"
    assert explored.island_assignment == base.island_assignment
    assert explored.w_assignment == base.w_assignment
    assert explored.f_assignment == base.f_assignment
    source = int(detail["source_position"])
    target = int(detail["target_position"])
    expected = list(base.operation_order)
    expected.insert(target, expected.pop(source))
    assert explored.operation_order == tuple(expected)


def test_self_exploration_assignment_changes_at_most_one_gene(automotive_instance):
    base = random_candidate(automotive_instance, random.Random(8))
    explored, detail = _self_explore(automotive_instance, base, random.Random(2))
    assert detail["branch"] == "assignment"
    differences = sum(
        left != right
        for old, new in (
            (base.island_assignment, explored.island_assignment),
            (base.w_assignment, explored.w_assignment),
            (base.f_assignment, explored.f_assignment),
        )
        for left, right in zip(old, new)
    )
    assert differences <= 1


def test_dabc_decoder_count_trace_and_restart_are_exact(automotive_instance):
    config = DABCConfig(
        population_size=8,
        neighborhood_search_intensity=1,
        max_nonimprovements=1,
        restart_self_explorations=3,
        iteration_limit=1,
    )
    result = solve_dabc(automotive_instance, 100.0, 17, config)
    diagnostics = result.diagnostics
    expected = (
        config.population_size
        + 2 * diagnostics["employed_crossover_count"]
        + diagnostics["self_exploration_count"]
        + diagnostics["cns1_generated_candidates"]
        + diagnostics["cns2_generated_candidates"]
    )
    assert result.decoder_evaluations == expected
    assert diagnostics["restart_self_exploration_decodes"] == (
        diagnostics["restart_count"] * config.restart_self_explorations
    )
    trace = [point.current_best_makespan for point in result.convergence_trace]
    assert all(left > right for left, right in zip(trace, trace[1:]))
    assert result.best.feasible


def test_source_projection_f_and_r_recurrences_match_paper(automotive_instance):
    decoded = decode_candidate(
        automotive_instance,
        random_candidate(automotive_instance, random.Random(0)),
    )
    metrics = source_projection_metrics(automotive_instance, decoded)

    def rt(predecessor, successor):
        island = decoded.schedule.operation_schedules[successor].island_id
        source_config = (
            automotive_instance.island_data[island].initial_config
            if predecessor is None
            else decoded.schedule.operation_schedules[predecessor].config_id
        )
        target_config = decoded.schedule.operation_schedules[successor].config_id
        return automotive_instance.reconfiguration_time[(island, source_config, target_config)]

    for operation in automotive_instance.operations:
        mp = metrics.machine_predecessor[operation]
        jp = metrics.job_predecessor[operation]
        machine_head = (
            0.0 if mp is None else metrics.head[mp] + metrics.processing_time[mp]
        ) + rt(mp, operation)
        job_head = 0.0 if jp is None else metrics.head[jp] + metrics.processing_time[jp]
        assert math.isclose(metrics.head[operation], max(machine_head, job_head))

        ms = metrics.machine_successor[operation]
        js = metrics.job_successor[operation]
        machine_tail = (
            0.0
            if ms is None
            else metrics.tail[ms] + metrics.processing_time[ms] + rt(operation, ms)
        )
        job_tail = 0.0 if js is None else metrics.tail[js] + metrics.processing_time[js]
        assert math.isclose(metrics.tail[operation], max(machine_tail, job_tail))


def test_source_theorems_are_exact_shadow_audits(automotive_instance):
    decoded = decode_candidate(
        automotive_instance,
        random_candidate(automotive_instance, random.Random(0)),
    )
    block = CriticalIslandBlock("M2", ("o11", "o12", "o21"), ())
    intra = audit_intramachine_move(
        automotive_instance,
        decoded,
        (block,),
        0,
        "o12",
        2,
        ("o11", "o21", "o12"),
    )
    assert intra.full_dag_reachability_feasible
    assert [rule.theorem for rule in intra.clipping_rules] == [
        "THEOREM_5", "THEOREM_6", "THEOREM_7",
        "THEOREM_8", "THEOREM_9", "THEOREM_10",
    ]
    theorem_9 = next(rule for rule in intra.clipping_rules if rule.theorem == "THEOREM_9")
    assert theorem_9.status == SOURCE_EXACT

    inter = audit_intermachine_move(
        automotive_instance,
        decoded,
        "o12",
        "M1",
        ("o12", "o23"),
    )
    assert inter.full_dag_reachability_feasible
    assert {rule.theorem for rule in inter.feasibility_rules} == {"THEOREM_3", "THEOREM_4"}
    assert all(rule.status == SOURCE_EXACT for rule in inter.feasibility_rules)
    assert not inter.source_clip_predicate


def test_all_six_clipping_theorems_map_to_the_paper_move_cases(automotive_instance):
    decoded = decode_candidate(
        automotive_instance,
        random_candidate(automotive_instance, random.Random(4)),
    )
    operations = tuple(decoded.schedule.island_timelines["M1"])
    assert len(operations) == 4
    block = CriticalIslandBlock("M1", operations, ())
    cases = (
        (operations[1], 2, "THEOREM_5"),
        (operations[2], 1, "THEOREM_6"),
        (operations[0], 2, "THEOREM_7"),
        (operations[2], 0, "THEOREM_8"),
        (operations[1], 3, "THEOREM_9"),
        (operations[3], 1, "THEOREM_10"),
    )
    for moved, target, expected in cases:
        remainder = [operation for operation in operations if operation != moved]
        remainder.insert(target, moved)
        audit = audit_intramachine_move(
            automotive_instance,
            decoded,
            (block,),
            0,
            moved,
            target,
            tuple(remainder),
        )
        applicable = [
            rule.theorem for rule in audit.clipping_rules if rule.status == SOURCE_EXACT
        ]
        assert applicable == [expected]


def test_assignment_exploration_retains_candidate_shape_with_singleton_fleets(
    automotive_instance,
):
    base = random_candidate(automotive_instance, random.Random(3))
    candidate = Candidate(
        base.operation_order,
        base.island_assignment,
        base.w_assignment,
        base.f_assignment,
    )
    for seed in range(20):
        explored, _ = _self_explore(automotive_instance, candidate, random.Random(seed))
        assert len(explored.operation_order) == automotive_instance.num_operations
        assert len(explored.island_assignment) == automotive_instance.num_operations
