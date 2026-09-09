import json
from types import SimpleNamespace

import pytest

from scripts import run_phase6p_development as development


def test_development_tasks_are_complete_fixed_and_method_grouped():
    manifest = json.loads(
        (development.REGISTRY_ROOT / "instance_manifest.json").read_text()
    )
    protocol = {
        "instances": manifest["instances"],
        "development_seeds": [746101, 746102, 746103],
    }
    tasks = development.build_tasks(protocol)
    assert len(tasks) == 270
    assert len({development.task_key(task) for task in tasks}) == 270
    assert [task["method"] for task in tasks[::54]] == list(development.METHODS)
    assert all(task["budget_seconds"] == 2 * task["num_operations"] for task in tasks)
    assert all(task["result_source"] == "NEW_RUN" for task in tasks)


def test_portfolio_observer_aggregates_rank_origins_weights_and_runtime():
    observer = development.PortfolioObserver(10.0)
    weights = {"random": 1.0, "greedy": 1.0}
    observer({
        "accepted": True,
        "new_global_best": False,
        "candidate": SimpleNamespace(makespan=9.0),
        "current_before": SimpleNamespace(makespan=10.0),
        "repair_operator": "greedy",
        "credited_destroy_operators": ("random",),
        "iteration_runtime": 0.2,
        "decoder_runtime": 0.1,
        "repair_excluding_decoder_runtime": 0.05,
        "neural_eligible": True,
        "safe_fallback": False,
        "selected_target_set_id": "b",
        "canonical_fallback_target_set_id": "f",
        "ranked_target_ids": ("a", "b", "c", "d", "e", "f"),
        "selected_origin_rules": ("operator_random",),
        "selected_origin_families": ("ORIGINAL_OPERATOR",),
        "portfolio_distribution": (
            ("a", 0.5, 2.0), ("b", 0.2, 1.0), ("c", 0.1, 1.0),
            ("d", 0.08, 1.0), ("e", 0.07, 1.0), ("f", 0.05, 1.0),
        ),
        "critic_timing_ms": {"total": 4.0},
        "operator_weights_after": weights,
        "elapsed_time": 10.0,
    })
    summary = observer.summary()
    assert summary["neural_decisions"] == 1
    assert summary["sampled_neural_rank_distribution"] == {2: 1}
    assert summary["selected_24_rule_origin_distribution"] == {
        "operator_random": 1
    }
    assert summary["selected_destroy_operator_distribution"] == {"random": 1}
    assert summary["selected_repair_distribution"] == {"greedy": 1}
    assert summary["acceptance_rate"] == 1.0
    assert summary["current_improvement_rate"] == 1.0
    assert summary["portfolio_probability_changed_by_destroy_weights_fraction"] == 1.0
    assert len(summary["adaptive_weight_checkpoints"]) == 11
    assert summary["runtime_component_sums"]["critic_ms_total"] == pytest.approx(4.0)
