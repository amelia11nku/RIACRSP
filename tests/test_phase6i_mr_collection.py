import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from rcias_clgri.search.common import Candidate
from scripts.run_phase6i_mr_collection import CollectionObserver, build_tasks


ROOT = Path(__file__).resolve().parents[1]


def test_collection_observer_selects_30_unique_stage_balanced_states():
    fractions = [(index + 0.5) / 30 for index in range(30)]
    observer = CollectionObserver(fractions)
    candidate = Candidate(("o1",), ("M1",), ("W1",), ("F1",))
    current = SimpleNamespace(candidate=candidate, makespan=100.0)
    for iteration in range(100):
        progress = (iteration + 0.25) / 100
        observer({
            "ni_eligible": True,
            "ni_state_id": f"state_{iteration}",
            "ni_state_feature_summary": {"search_progress": progress},
            "current_before": current,
            "iteration": iteration,
            "elapsed_time": progress * 10,
            "decoder_evaluations": iteration * 8,
            "ni_timing_ms": {"csg_build": 1.0},
            "repair_runtime": 0.002,
        })
    selected = observer.selected()
    assert len(selected) == 30
    assert len({row["state_id"] for row in selected}) == 30
    assert [row["search_stage"] for row in selected].count("EARLY") == 10
    assert [row["search_stage"] for row in selected].count("MIDDLE") == 10
    assert [row["search_stage"] for row in selected].count("LATE") == 10
    assert observer.component_ms["csg_build"] == 100.0
    assert observer.component_ms["search_repair_decoder"] == 200.0


def test_r09_collection_task_and_audit_allocation_is_exact():
    config = json.loads(
        (ROOT / "configs/phase6i_mr_live_utility_revision.json").read_text()
    )
    manifest = pd.read_csv(ROOT / config["instance_suite"]["manifest"])
    tasks = build_tasks(config, manifest, "R09")
    assert len(tasks) == 54
    assert len({task["instance_id"] for task in tasks}) == 18
    top = [task for task in tasks if task["top_eight_audit_target"] is not None]
    full = [task for task in tasks if task["full_bank_audit_target"] is not None]
    assert len(top) == 18
    assert len(full) == 9
    assert {task["seed"] for task in top} == {681201}
    assert {task["seed"] for task in full} == {681201}
    assert pd.Series([task["top_eight_audit_target"] for task in top]).value_counts().to_dict() == {
        0.15: 6,
        0.50: 6,
        0.85: 6,
    }
    assert {task["cell_replicate"] for task in full} == {"C02"}
