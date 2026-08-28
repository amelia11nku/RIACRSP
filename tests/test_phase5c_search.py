import random
from pathlib import Path

from rcias_clgri.data.loader import load_instance
from rcias_clgri.search.alns import ALNSConfig, solve_alns
from rcias_clgri.search.common import decode_candidate, random_candidate
from rcias_clgri.search.dcga import DCGAConfig, _crossover, _mutate, _pox, solve_dcga
from rcias_clgri.search.ga import GAConfig, solve_ga


ROOT = Path(__file__).resolve().parents[1]


def _tiny():
    return load_instance(ROOT / "instances/tiny/tiny_01.json")


def test_common_decoder_is_deterministic_and_feasible():
    instance = _tiny()
    candidate = random_candidate(instance, random.Random(7))
    first = decode_candidate(instance, candidate)
    second = decode_candidate(instance, candidate)
    assert first.feasible and second.feasible
    assert first.makespan == second.makespan
    assert first.actions == second.actions


def test_search_smoke():
    instance = _tiny()
    ga = solve_ga(instance, 0.02, 1, GAConfig(population_size=6))
    dcga = solve_dcga(instance, 0.02, 1, DCGAConfig(population_size_each=4))
    alns = solve_alns(instance, 0.02, 1, ALNSConfig(candidate_trials=2))
    assert ga.best.feasible and dcga.best.feasible and alns.best.feasible
    assert dcga.method == "Adapted DCGA"
    assert alns.method == "ALNS-H1"
    assert set(dcga.diagnostics["population_pathway_best"]) == {
        "Decoding1_fixed", "Decoding2_cumulative"
    }


def test_dcga_pox_and_variation_remain_feasible():
    instance = _tiny()
    rng = random.Random(13)
    left = random_candidate(instance, rng)
    right = random_candidate(instance, rng)
    orders = _pox(instance, left.operation_order, right.operation_order, rng)
    assert all(set(order) == set(instance.operations) for order in orders)
    children = _crossover(instance, left, right, rng)
    for child in (*children, _mutate(instance, children[0], rng)):
        assert decode_candidate(instance, child, "fixed").feasible
