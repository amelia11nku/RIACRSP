from __future__ import annotations

import math
import random

import pytest

from rcias_clgri.analysis.phase6l_legacy_score import select_score_free_fallback
from rcias_clgri.data.loader import load_instance
from rcias_clgri.ni.phase6p_live_inference import (
    Phase6PBankScores,
    origin_destroy_operators,
)
from rcias_clgri.ni.proposal_bank import build_live_proposal_bank
from rcias_clgri.search.alns import ALNSConfig, REPAIR, _roulette
from rcias_clgri.search.phase6p_adaptive import (
    mapped_destroy_weight,
    portfolio_distribution,
    sample_portfolio_target,
    simulated_annealing_accept,
    solve_phase6p,
    updated_operator_weight,
)


def test_portfolio_uses_mean_origin_weight_and_inverse_rank():
    operator_weights = {
        "random": 2.0,
        "critical": 4.0,
        "overloaded_island": 1.0,
        "high_reconfiguration": 1.0,
        "w_bottleneck": 1.0,
        "f_bottleneck": 1.0,
        "related": 1.0,
    }
    origins = {
        "a": ("random", "critical"),
        "b": ("random",),
        "c": ("related",),
        "d": ("related",),
        "e": ("related",),
        "f": ("related",),
    }
    assert mapped_destroy_weight(origins["a"], operator_weights) == 3.0
    distribution = portfolio_distribution(
        tuple(origins), origins, operator_weights
    )
    raw = [3.0, 1.0, 1 / 3, 1 / 4, 1 / 5, 1 / 6]
    assert [item[1] for item in distribution] == pytest.approx(
        [value / sum(raw) for value in raw]
    )
    assert sample_portfolio_target(distribution, random.Random(17)) == \
        sample_portfolio_target(distribution, random.Random(17))


def test_reward_update_preserves_frozen_floor():
    assert updated_operator_weight(1.0, 0.0, 0.2) == pytest.approx(0.82)
    assert updated_operator_weight(1.0, 1.0, 0.2) == pytest.approx(1.0)
    assert updated_operator_weight(1.0, 5.0, 0.2) == pytest.approx(1.8)


def test_repair_roulette_and_sa_use_the_frozen_semantics():
    weights = {name: 1.0 for name in REPAIR}
    weights["transport_aware"] = 1_000_000.0
    assert _roulette(REPAIR, weights, random.Random(1)) == "transport_aware"
    first = random.Random(8)
    second = random.Random(8)
    assert simulated_annealing_accept(3.0, 2.0, first) == (
        second.random() < math.exp(-3.0 / 2.0)
    )
    rng = random.Random(9)
    before = rng.getstate()
    assert simulated_annealing_accept(0.0, 1.0, rng) is True
    assert rng.getstate() == before


class FakeCritic:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.prepared = 0

    def prepare_instance(self, instance, schedule):
        del instance, schedule
        self.prepared += 1

    def score_bank(
        self,
        instance,
        current,
        *,
        state_id,
        destroy_count,
        proposal_seed_namespace,
        **kwargs,
    ):
        del kwargs
        if self.fail:
            raise RuntimeError("deliberate critic failure")
        generated, _ = build_live_proposal_bank(
            instance,
            current,
            state_id=state_id,
            destroy_count=destroy_count,
            seed_namespace=proposal_seed_namespace,
        )
        fallback = select_score_free_fallback(generated)
        scores = {
            arm.target_set_id: float(index)
            for index, arm in enumerate(generated.arms)
        }
        ranked = tuple(sorted(scores, key=lambda item: (-scores[item], item)))
        return Phase6PBankScores(
            state_id,
            generated,
            fallback,
            scores,
            {},
            ranked,
            ranked[:6],
            origin_destroy_operators(generated),
            {},
            "fake_graph",
            0,
            {"total": 0.0},
        )


def _run(fake, *, iteration_limit=6):
    instance = load_instance("instances/tiny/tiny_01.json")
    events = []
    result = solve_phase6p(
        instance,
        30.0,
        746101,
        fake,
        alns_config=ALNSConfig(iteration_limit=iteration_limit),
        observer=events.append,
    )
    return instance, result, events


def test_solver_is_reproducible_feasible_and_uses_complete_banks():
    first_instance, first, first_events = _run(FakeCritic())
    second_instance, second, second_events = _run(FakeCritic())
    assert first.best.feasible and second.best.feasible
    assert first.best.makespan == second.best.makespan
    assert [event["selected_target_set_id"] for event in first_events] == [
        event["selected_target_set_id"] for event in second_events
    ]
    assert [event["repair_operator"] for event in first_events] == [
        event["repair_operator"] for event in second_events
    ]
    assert [point.current_best_makespan for point in first.convergence_trace] == sorted(
        [point.current_best_makespan for point in first.convergence_trace], reverse=True
    )
    assert first.diagnostics["neural_eligible_iterations"] == 2
    assert first.diagnostics["neural_portfolio_decisions"] == 2
    for event in (first_events[0], first_events[5]):
        assert event["neural_eligible"] is True
        assert event["requested_proposal_count"] == 24
        assert len(event["top_target_ids"]) == 6
        assert event["candidate_trials_completed"] == 8
    assert first_instance.instance_id == second_instance.instance_id


def test_solver_uses_canonical_related_safe_fallback():
    instance, result, events = _run(FakeCritic(fail=True), iteration_limit=1)
    event = events[0]
    assert result.best.feasible
    assert result.diagnostics["safe_fallbacks"] == 1
    assert event["safe_fallback"] is True
    assert event["scoring_error"] == "RuntimeError: deliberate critic failure"
    generated, _ = build_live_proposal_bank(
        instance,
        event["current_before"],
        state_id=event["state_id"],
        destroy_count=max(2, round(instance.num_operations * 0.15)),
        seed_namespace=692000000,
    )
    assert event["selected_target_set_id"] == select_score_free_fallback(
        generated
    ).target_set_id
