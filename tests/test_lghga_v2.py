from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random

import numpy as np

from rcias_clgri.search.common import decode_candidate, random_candidate
from rcias_clgri.search.lghga import LGHGAConfig
from rcias_clgri.search.lghga_learning import DTRBundle
from rcias_clgri.search.lghga_neighborhoods import NEIGHBORHOODS
import rcias_clgri.search.lghga_neighborhoods_v2 as neighborhoods_v2
from rcias_clgri.search.lghga_neighborhoods_v2 import propose_neighborhood
from rcias_clgri.search.lghga_v2 import generate_knowledge_run_v2, solve_lghga_v2
from scripts.run_lghga_knowledge_v2 import _load_configs, _verify_implementation


ROOT = Path(__file__).resolve().parents[1]


class _ConstantModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, values):
        return np.asarray([self.value] * len(values))


def _bundle(selected: str) -> DTRBundle:
    return DTRBundle(
        {
            neighborhood: _ConstantModel(60.0 if neighborhood == selected else 10.0)
            for neighborhood in NEIGHBORHOODS
        },
        {neighborhood: f"v2-{neighborhood}" for neighborhood in NEIGHBORHOODS},
        "v2-knowledge",
    )


def test_v2_protocol_is_versioned_and_keeps_source_threshold():
    algorithm = json.loads(
        (ROOT / "configs/baselines/lghga_v2_riacrsp.json").read_text()
    )
    kb, _, training = _load_configs()
    assert algorithm["method"] == "LG_HGA-RIACRSP-v2-N4M"
    assert algorithm["formal_objective"] == "makespan"
    assert algorithm["local_search_threshold_pct"] == 50.0
    assert algorithm["n4_selection_rule"] == "MINIMAL_PLURAL_TWO_EFFECTIVE_MOVES"
    assert algorithm["knowledge_target"] == "ONE_STEP_EQ11_R_PCT"
    assert kb["dtr_regime"] == "scale_x_CF_level"
    assert len(training["instances"]) == 9
    assert _verify_implementation(kb)


def test_v2_n4_uses_two_effective_minimum_count_moves_when_available(
    automotive_instance, monkeypatch,
):
    monkeypatch.setattr(
        neighborhoods_v2,
        "_critical_context",
        lambda instance, decoded, rng: (None, None, tuple(instance.operations)),
    )
    proposal = None
    decoded = None
    for candidate_seed in range(50):
        decoded = decode_candidate(
            automotive_instance,
            random_candidate(automotive_instance, random.Random(candidate_seed)),
        )
        for proposal_seed in range(50):
            candidate = propose_neighborhood(
                automotive_instance, decoded, "N4_MMIT", random.Random(proposal_seed)
            )
            if candidate.detail["selection_count"] == 2:
                proposal = candidate
                break
        if proposal is not None:
            break
    assert proposal is not None
    assert decoded is not None
    assert proposal.detail["changed"] is True
    assert len(proposal.operation_ids) == 2
    assert proposal.candidate.w_assignment == decoded.candidate.w_assignment
    assert proposal.candidate.f_assignment == decoded.candidate.f_assignment

    load = Counter(
        record.island_id for record in decoded.schedule.operation_schedules.values()
    )
    for change in proposal.detail["changes"]:
        operation = change["operation_id"]
        source = change["source_island"]
        target = change["target_island"]
        eligible = automotive_instance.operation_data[operation].eligible_islands
        assert source != target
        assert load[target] == min(load[island] for island in eligible)
        load[source] -= 1
        load[target] += 1


def test_v2_knowledge_preserves_one_step_r_budget_and_records_noops(
    automotive_instance,
):
    config = LGHGAConfig(
        max_generations=1,
        population_size=6,
        local_search_population_size=2,
        neighborhood_size=3,
    )
    result = generate_knowledge_run_v2(automotive_instance, 23, config)
    assert len(result.rows) == 4
    assert all(row["num_generated"] == 3 for row in result.rows)
    assert all(row["num_changed"] + row["num_unchanged"] == 3 for row in result.rows)
    assert result.decoder_evaluations == 6 + 6 + 4 * 3
    assert result.best.feasible


def test_v2_online_search_uses_frozen_local_budget_and_diagnostics(
    automotive_instance,
):
    config = LGHGAConfig(
        max_generations=1,
        population_size=8,
        local_search_population_size=2,
        local_search_max_iterations=2,
        neighborhood_size=4,
    )
    result = solve_lghga_v2(
        automotive_instance, 100.0, 19, _bundle("N4_MMIT"), config
    )
    assert result.method == "LG_HGA-RIACRSP-v2-N4M"
    assert result.decoder_evaluations == 8 + 8 + 2 * 4
    assert result.diagnostics["local_decoder_evaluations"] == 8
    assert (
        result.diagnostics["changed_proposal_counts"]["N4_MMIT"]
        + result.diagnostics["noop_proposal_counts"]["N4_MMIT"]
        == 8
    )
    assert result.best.feasible
