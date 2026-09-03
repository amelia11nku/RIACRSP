from __future__ import annotations

import json
import random

import numpy as np

from rcias_clgri.search.common import decode_candidate, random_candidate
from rcias_clgri.search.lghga import (
    LGHGAConfig,
    _topology_mutation,
    generate_knowledge_run,
    solve_lghga,
)
from rcias_clgri.search.lghga_learning import (
    DTRBundle,
    improvement_rate_pct,
    load_dtr_bundle,
    predict_rates,
    save_dtr_bundle,
    select_neighborhood,
    train_dtr_bundle,
)
from rcias_clgri.search.lghga_neighborhoods import NEIGHBORHOODS, propose_neighborhood


class _ConstantModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, values):
        return np.asarray([self.value] * len(values))


def _bundle(selected: str, value: float = 60.0) -> DTRBundle:
    return DTRBundle(
        {
            neighborhood: _ConstantModel(value if neighborhood == selected else 10.0)
            for neighborhood in NEIGHBORHOODS
        },
        {neighborhood: f"hash-{neighborhood}" for neighborhood in NEIGHBORHOODS},
        "knowledge-hash",
    )


def test_lghga_source_defaults_are_frozen():
    config = LGHGAConfig()
    assert config.max_generations == 100
    assert config.population_size == 40
    assert config.crossover_probability == 0.9
    assert config.mutation_probability == 0.4
    assert config.local_search_threshold_pct == 50
    assert config.local_search_population_size == 5
    assert config.local_search_max_iterations == 5
    assert config.neighborhood_size == 20
    assert config.knowledge_generation_runs == 20


def test_improvement_rate_is_percent_and_gate_is_strict():
    better, generated, rate = improvement_rate_pct([10, 10, 10, 10], [9, 9, 10, 11])
    assert (better, generated, rate) == (2, 4, 50.0)
    selected, passed = select_neighborhood(
        {"N1_CTU": 50, "N2_EST": 40, "N3_TOPO": 30, "N4_MMIT": 20}, 50
    )
    assert selected == "N1_CTU"
    assert not passed


def test_n1_and_n2_are_earlier_insertions_using_documented_fields(automotive_instance):
    decoded = decode_candidate(
        automotive_instance, random_candidate(automotive_instance, random.Random(8))
    )
    for neighborhood in ("N1_CTU", "N2_EST"):
        changed = None
        for seed in range(50):
            proposal = propose_neighborhood(
                automotive_instance, decoded, neighborhood, random.Random(seed)
            )
            if proposal.detail.get("changed"):
                changed = proposal
                break
        assert changed is not None
        assert changed.detail["target_position"] < changed.detail["source_position"]
        assert changed.candidate.island_assignment == decoded.candidate.island_assignment
        assert changed.candidate.w_assignment == decoded.candidate.w_assignment
        assert changed.candidate.f_assignment == decoded.candidate.f_assignment
    assert changed.neighborhood_id == "N2_EST"
    assert changed.detail["realized_time_field"] == "OperationSchedule.start_time"


def test_n3_topology_mutation_preserves_dag_and_other_layers(automotive_instance):
    base = random_candidate(automotive_instance, random.Random(11))
    changed = None
    for seed in range(100):
        mutated = _topology_mutation(automotive_instance, base, random.Random(seed))
        if mutated.operation_order != base.operation_order:
            changed = mutated
            break
    assert changed is not None
    assert changed.island_assignment == base.island_assignment
    assert changed.w_assignment == base.w_assignment
    assert changed.f_assignment == base.f_assignment
    changed_products = [
        product
        for product in automotive_instance.products
        if tuple(op for op in changed.operation_order if automotive_instance.product_of[op] == product)
        != tuple(op for op in base.operation_order if automotive_instance.product_of[op] == product)
    ]
    assert len(changed_products) == 1
    product = changed_products[0]
    relative = [
        operation for operation in changed.operation_order
        if automotive_instance.product_of[operation] == product
    ]
    position = {operation: index for index, operation in enumerate(relative)}
    assert all(
        position[source] < position[target]
        for source, target in automotive_instance.product_data[product].precedence
    )


def test_n4_uses_minimum_eligible_island_operation_count(automotive_instance):
    decoded = decode_candidate(
        automotive_instance, random_candidate(automotive_instance, random.Random(13))
    )
    proposal = propose_neighborhood(
        automotive_instance, decoded, "N4_MMIT", random.Random(7)
    )
    operation = proposal.operation_ids[0]
    target = proposal.detail["target_island"]
    counts = {
        island: len(decoded.schedule.island_timelines[island])
        for island in automotive_instance.operation_data[operation].eligible_islands
    }
    assert counts[target] == min(counts.values())
    assert proposal.candidate.w_assignment == decoded.candidate.w_assignment
    assert proposal.candidate.f_assignment == decoded.candidate.f_assignment


def test_online_local_search_sizes_decoder_count_and_population_are_exact(automotive_instance):
    config = LGHGAConfig(
        max_generations=1,
        population_size=8,
        local_search_population_size=2,
        local_search_max_iterations=2,
        neighborhood_size=4,
    )
    result = solve_lghga(
        automotive_instance, 100.0, 19, _bundle("N3_TOPO"), config
    )
    assert result.decoder_evaluations == 8 + 8 + 2 * 4
    assert result.diagnostics["local_decoder_evaluations"] == 8
    assert result.diagnostics["neighborhood_proposal_counts"]["N3_TOPO"] == 8
    assert result.diagnostics["final_population_size"] == 8
    assert result.generations_if_applicable == 1
    assert result.best.feasible
    trace = [point.current_best_makespan for point in result.convergence_trace]
    assert all(left > right for left, right in zip(trace, trace[1:]))


def test_gate_failure_skips_local_search(automotive_instance):
    config = LGHGAConfig(max_generations=1, population_size=6, neighborhood_size=3)
    result = solve_lghga(
        automotive_instance, 100.0, 31, _bundle("N1_CTU", value=50.0), config
    )
    assert result.diagnostics["local_search_gate_passes"] == 0
    assert result.diagnostics["local_decoder_evaluations"] == 0
    assert result.decoder_evaluations == 12


def test_knowledge_mode_evaluates_four_equal_neighborhood_budgets(automotive_instance):
    config = LGHGAConfig(
        max_generations=2,
        population_size=6,
        local_search_population_size=2,
        neighborhood_size=3,
    )
    result = generate_knowledge_run(automotive_instance, 23, config)
    assert len(result.rows) == 2 * 4
    assert all(row["num_generated"] == 3 for row in result.rows)
    assert {row["neighborhood_id"] for row in result.rows} == set(NEIGHBORHOODS)
    assert all(0 <= row["R_pct"] <= 100 for row in result.rows)
    assert result.decoder_evaluations == 6 + 2 * (6 + 4 * 3)
    assert result.best.feasible


def test_four_dtr_models_round_trip_with_hash_verification(tmp_path):
    rows = []
    for neighborhood_index, neighborhood in enumerate(NEIGHBORHOODS):
        for generation in range(1, 5):
            rows.append({
                "neighborhood_id": neighborhood,
                "normalized_generation_index": generation / 4,
                "R_pct": 10 * neighborhood_index + generation,
            })
    bundle = train_dtr_bundle(rows, random_state=41, knowledge_manifest_hash="kb")
    manifest = save_dtr_bundle(bundle, tmp_path)
    loaded = load_dtr_bundle(tmp_path)
    assert set(loaded.models) == set(NEIGHBORHOODS)
    assert set(manifest["model_hashes"]) == set(NEIGHBORHOODS)
    assert predict_rates(bundle, 2, 4) == predict_rates(loaded, 2, 4)
    persisted = json.loads((tmp_path / "model_manifest.json").read_text())
    assert persisted["features"] == ["normalized_generation_index"]
