import json

import numpy as np

from scripts.finalize_phase6i_mr import bootstrap_interval
from scripts.run_phase6i_mr_r11_validation import (
    CONFIG_PATH,
    R11Observer,
    TARGET_FRACTIONS,
    build_tasks,
)


def test_r11_protocol_expands_to_288_matched_runs_without_forbidden_methods():
    tasks = build_tasks(json.loads(CONFIG_PATH.read_text()))
    counts = {
        method: sum(task["method"] == method for task in tasks)
        for method in {task["method"] for task in tasks}
    }
    assert len(tasks) == 288
    assert counts == {
        "H1": 18,
        "ALNS": 90,
        "PHASE6H_CSGNI": 90,
        "PHASE6I_MR_CSGNI": 90,
    }
    assert not ({"GA", "DCGA", "GUROBI"} & set(counts))


def test_r11_forced_snapshots_are_outcome_blind_fixed_progress_targets():
    observer = R11Observer({"instance_id": "test"})
    observer.events = [
        {
            "state_id": f"state-{index}",
            "iteration": index,
            "search_progress": index / 100,
            "elapsed_wall_time": float(index),
            "decoder_evaluations": index + 1,
            "current": object(),
        }
        for index in range(100)
    ]
    snapshots = observer.snapshots()
    assert [row["target_progress"] for row in snapshots] == list(TARGET_FRACTIONS)
    assert len({row["state_id"] for row in snapshots}) == 10
    assert [row["search_progress"] for row in snapshots] == [
        0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95
    ]


def test_grouped_bootstrap_is_deterministic_and_instance_level():
    values = np.asarray([-0.01, 0.02, 0.03, 0.04])
    first = bootstrap_interval(values)
    second = bootstrap_interval(values)
    assert first == second
    assert first[0] <= values.mean() <= first[1]
