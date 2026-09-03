from pathlib import Path
from types import SimpleNamespace

from rcias_clgri.data.loader import load_instance
from rcias_clgri.heuristic.dispatching import solve_dispatching
from rcias_clgri.search.alns import ALNSConfig
from rcias_clgri.search.common import candidate_from_actions, decode_candidate
import rcias_clgri.analysis.phase6i_mr as phase6i_mr
from scripts.run_phase6i_mr_pilot import SnapshotObserver, _rank_and_label


ROOT = Path(__file__).resolve().parents[1]


def test_snapshot_selection_is_unique_and_nearest_to_fixed_progress():
    instance = load_instance(ROOT / "instances/tiny/tiny_03.json")
    h1 = solve_dispatching(instance, "H1")
    current = decode_candidate(instance, candidate_from_actions(instance, h1.actions))
    observer = SnapshotObserver(10.0, [0.1, 0.25, 0.9])
    for iteration, progress in enumerate([0.08, 0.12, 0.24, 0.26, 0.88, 0.92]):
        observer({
            "ni_eligible": True,
            "ni_state_id": f"state_{iteration}",
            "ni_state_feature_summary": {"search_progress": progress},
            "current_before": current,
            "iteration": iteration,
            "elapsed_time": progress * 10.0,
        })
    selected = observer.selected()
    assert [row["search_progress"] for row in selected] == [0.08, 0.24, 0.88]
    assert len({row["state_id"] for row in selected}) == 3


def test_post_decoder_rank_regret_and_sign_labels_are_correct():
    rows = [
        {
            "state_id": "state",
            "candidate_role": "FROZEN_NEURAL_TOP1",
            "target_set_id": "a",
            "raw_score": 4.0,
            "calibrated_utility": 0.2,
            "decoded_immediate_utility": -0.1,
        },
        {
            "state_id": "state",
            "candidate_role": "FROZEN_NEURAL_TOP2",
            "target_set_id": "b",
            "raw_score": 3.0,
            "calibrated_utility": 0.1,
            "decoded_immediate_utility": 0.3,
        },
        {
            "state_id": "state",
            "candidate_role": "ALNS_RELATED_FALLBACK",
            "target_set_id": "c",
            "raw_score": 2.0,
            "calibrated_utility": -0.1,
            "decoded_immediate_utility": 0.1,
        },
        {
            "state_id": "state",
            "candidate_role": "DETERMINISTIC_DIVERSE",
            "target_set_id": "d",
            "raw_score": 1.0,
            "calibrated_utility": -0.2,
            "decoded_immediate_utility": -0.2,
        },
    ]
    _rank_and_label(rows)
    by_id = {row["target_set_id"]: row for row in rows}
    assert by_id["b"]["within_state_true_rank"] == 1
    assert by_id["a"]["within_state_predicted_rank"] == 1
    assert by_id["a"]["regret_to_best"] == 0.4
    assert by_id["a"]["within_state_inversion"] is True
    assert by_id["a"]["sign_error"] is True
    assert by_id["c"]["fallback_decoded_utility"] == 0.1


def test_fixed_horizon_continuation_uses_exact_budget_and_replays_seed(monkeypatch):
    start_candidate = object()
    start = SimpleNamespace(candidate=start_candidate, makespan=100.0)
    instance = SimpleNamespace(num_operations=10)
    monkeypatch.setattr(
        phase6i_mr,
        "_roulette",
        lambda names, weights, rng: names[0],
    )
    monkeypatch.setattr(
        phase6i_mr,
        "_destroy",
        lambda instance, current, destroy, count, rng: {"o1", "o2"},
    )
    monkeypatch.setattr(
        phase6i_mr,
        "_neighbor",
        lambda instance, base, removed, repair, rng: base,
    )
    monkeypatch.setattr(
        phase6i_mr,
        "decode_candidate",
        lambda instance, candidate: SimpleNamespace(
            candidate=candidate,
            makespan=99.0,
        ),
    )
    config = ALNSConfig(candidate_trials=2)
    first = phase6i_mr.continue_frozen_alns_from_candidate(
        instance,
        start,
        state_id="state",
        continuation_seed=685101,
        seed_namespace=685000000,
        iterations=3,
        config=config,
    )
    second = phase6i_mr.continue_frozen_alns_from_candidate(
        instance,
        start,
        state_id="state",
        continuation_seed=685101,
        seed_namespace=685000000,
        iterations=3,
        config=config,
    )
    assert first.derived_seed == second.derived_seed
    assert first.iterations == 3
    assert first.decoder_evaluations == 6
    assert first.best_makespan == 99.0
    assert first.continuation_value == 0.01
    assert first.operator_selections == {"random": 3, "greedy": 3}
