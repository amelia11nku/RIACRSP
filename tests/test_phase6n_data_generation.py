import json
from pathlib import Path

from scripts import run_phase6n_data_generation as generation


ROOT = Path(__file__).resolve().parents[1]


def test_phase6n_data_tasks_are_equal_across_all_r12_fit_instances():
    config = json.loads(
        (ROOT / "configs/phase6n_candidate_conditioned_csg_v1.json").read_text()
    )
    plan = json.loads(
        (
            ROOT
            / "outputs/phase6n_candidate_conditioned_csg_v1/preregistration/data_generation_plan.json"
        ).read_text()
    )
    tasks = generation.build_tasks(config, plan)
    assert len(tasks) == 72
    assert len({row["instance_id"] for row in tasks}) == 18
    assert {row["trajectory_seed"] for row in tasks} == {721201, 721202, 721203, 721204}
    per_instance = {}
    for row in tasks:
        per_instance.setdefault(row["instance_id"], []).append(row)
    assert {len(rows) for rows in per_instance.values()} == {4}
    assert {row["scale"] for row in tasks} == {"S", "M", "L"}
    assert {row["CF_level"] for row in tasks} == {"CF1", "CF2", "CF3"}


def test_phase6n_alns_snapshot_selection_is_deterministic_and_stratified():
    observer = generation.ALNSSnapshotObserver(
        "instance", 721201, 100.0, [0.05, 0.18, 0.32]
    )

    class Decoded:
        makespan = 10.0
        feasible = True
        candidate = generation.Candidate(("o1",), ("i1",), ("w1",), ("f1",))

    for iteration, elapsed in enumerate((4.0, 6.0, 17.0, 19.0, 31.0, 33.0)):
        observer(
            {
                "iteration": iteration,
                "elapsed_time": elapsed,
                "decoder_evaluations": iteration + 1,
                "current_before": Decoded(),
            }
        )
    first = observer.selected()
    second = observer.selected()
    assert first == second
    assert [row["iteration"] for row in first] == [0, 2, 4]
    assert [row["target_progress"] for row in first] == [0.05, 0.18, 0.32]
    assert len({row["state_id"] for row in first}) == 3


def test_phase6n_generation_path_has_zero_historical_score_dependency():
    source = Path(generation.__file__).read_text()
    assert "score_frozen_candidate_bank" not in source
    assert "FrozenLiveInference" not in source
    assert "load_policy" not in source
    plan = json.loads(generation.DATA_PLAN.read_text())
    assert plan["source_sampler"] == "frozen H1-seeded ALNS with no neural intervention"
    assert plan["historical_score_calls"] == 0


def test_phase6n_progress_accounting_uses_all_states_without_filtering():
    statuses = [
        {"elapsed_seconds": 4.0, "unique_candidates": 24},
        {"elapsed_seconds": 6.0, "unique_candidates": 23},
    ]
    value = generation.progress_payload(
        implementation={"implementation_commit": "x", "worker_script_sha256": "y"},
        process_started=0.0,
        source_complete=1,
        completed=statuses,
        active=None,
        status="RUNNING",
    )
    assert value["states_complete"] == 2
    assert value["states_expected"] == 576
    assert value["unique_candidates_complete"] == 47
    assert value["measured_state_seconds"] == 10.0
    assert value["measured_seconds_per_state"] == 5.0
    assert value["historical_score_calls"] == 0
