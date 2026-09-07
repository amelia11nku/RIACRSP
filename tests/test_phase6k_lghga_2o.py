from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from rcias_clgri.data.loader import load_instance
from rcias_clgri.env.feasibility import check_schedule
from rcias_clgri.search import lghga, lghga_v2, lghga_2o
from rcias_clgri.search.lghga_learning import DTRBundle
from rcias_clgri.search.lghga_neighborhoods import NEIGHBORHOODS


class Regressor:
    def __init__(self, rate):
        self.rate = rate
        self.features = []

    def predict(self, values):
        self.features.append(values[0][0])
        return np.array([self.rate])


def run(monkeypatch, solver, budget, rate):
    tick = 0

    def clock():
        nonlocal tick
        tick += 1
        return tick / 1000

    for module in (lghga, lghga_v2, lghga_2o):
        monkeypatch.setattr(module, "time", SimpleNamespace(perf_counter=clock))
    instance = load_instance("instances/tiny/tiny_01.json")
    models = DTRBundle({name: Regressor(rate) for name in NEIGHBORHOODS}, {}, "smoke")
    config = lghga.LGHGAConfig(population_size=2, neighborhood_size=2,
                              local_search_population_size=2, local_search_max_iterations=1)
    result = solver(instance, budget, 696101, models, config)
    assert check_schedule(instance, result.best.schedule)["feasible"]
    return result, models


@pytest.mark.parametrize("rate", [0.0, 75.0])
def test_same_rng_decoder_and_timer_before_old_cap(monkeypatch, rate):
    canonical, _ = run(monkeypatch, lghga_v2.solve_lghga_v2, 0.09, rate)
    adapted, _ = run(monkeypatch, lghga_2o.solve_lghga_2o, 0.09, rate)
    repeated, _ = run(monkeypatch, lghga_2o.solve_lghga_2o, 0.09, rate)
    assert 0 < canonical.iterations < 100
    assert replace(adapted, method=canonical.method) == canonical
    assert repeated == adapted


def test_only_stopping_differs_past_100_and_dtr_denominator_stays_fixed(monkeypatch):
    canonical, _ = run(monkeypatch, lghga_v2.solve_lghga_v2, 3, 0.0)
    adapted, models = run(monkeypatch, lghga_2o.solve_lghga_2o, 3, 0.0)
    assert canonical.iterations == 100 < adapted.iterations
    assert adapted.runtime >= 3 > canonical.runtime
    assert adapted.diagnostics["generation_records"][:100] == canonical.diagnostics["generation_records"]
    assert all(model.features[99:102] == [1.0, 1.01, 1.02] for model in models.models.values())


def test_budget_notation_and_source_normalization_guard():
    instance = load_instance("instances/tiny/tiny_01.json")
    assert lghga_2o.operation_budget(instance) == 2 * len(instance.operations)
    with pytest.raises(ValueError, match="normalization"):
        lghga_2o.solve_lghga_2o(instance, 1, 0, None, lghga.LGHGAConfig(max_generations=1000))
