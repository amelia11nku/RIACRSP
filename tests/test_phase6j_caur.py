from __future__ import annotations

import json
from pathlib import Path

import pytest

from rcias_clgri.analysis.phase6j_caur import (
    CANDIDATE_INPUT_COLUMNS,
    FULL_BANK_SCOPE,
    LABEL_COLUMNS,
    REDUCED_AUDIT_SCOPE,
    aggregate_paired_advantages,
    build_candidate_source_features,
    choose_caur_action,
    continue_frozen_alns_at_horizons,
    critical_and_bottleneck_operations,
    evaluate_paired_continuation_advantage,
    fallback_relative_advantage,
    grouped_oof_fold,
    select_shortest_adequate_horizon,
    validate_full_bank_feature_rows,
    validate_grouped_label_records,
)
from rcias_clgri.analysis.phase6i_mr import continue_frozen_alns_from_candidate
from rcias_clgri.data.loader import load_instance
from rcias_clgri.data.phase6j_access import (
    R13_SPLIT,
    Phase6JAccessError,
    begin_one_time_split_access,
    complete_one_time_split_access,
    load_phase6j_instance,
    sha256_file,
)
from rcias_clgri.heuristic.dispatching import solve_dispatching
from rcias_clgri.search.alns import ALNSConfig
from rcias_clgri.search.common import candidate_from_actions, decode_candidate
from rcias_clgri.search.phase6c import generate_revised_target_arms
from scripts.run_phase6j_caur_pilot import SnapshotObserver, build_tasks


ROOT = Path(__file__).resolve().parents[1]


def test_phase6j_config_freezes_scope_splits_and_disjoint_namespaces():
    config = json.loads((ROOT / "configs/phase6j_caur.json").read_text())
    assert config["scope"]["version"] == "CSG-NI v1 model revision"
    assert config["instance_suite"]["splits"] == {
        "R12": "CAUR_FIT", "R13": "CAUR_SELECT", "R14": "CAUR_HOLDOUT"
    }
    assert config["candidate_bank"]["proposal_rules"] == 24
    assert config["candidate_bank"]["candidate_trials_per_target"] == 8
    values = []
    for key, value in config["rng"].items():
        values.extend(value if isinstance(value, list) else [value])
    assert len(values) == len(set(values))
    assert min(values) > 688001
    assert "Phase6I-MR R11 outcome payloads" in config["forbidden_data"][0]


def test_phase6j_historical_evidence_manifest_has_six_immutable_roles():
    record = json.loads(
        (ROOT / "configs/phase6j_caur_phase6i_mr_evidence_manifest.json").read_text()
    )
    assert record["status"] == "IMMUTABLE_HISTORICAL_MOTIVATION_ONLY"
    assert {row["role"] for row in record["files"]} == {
        "final_decision", "selected_artifact", "r11_result_manifest",
        "completion_integrity_audit", "ranking_calibration_report",
        "anytime_runtime_report",
    }
    assert all(len(row["sha256"]) == 64 for row in record["files"])


def _write_unlock(tmp_path: Path, split: str):
    artifact = tmp_path / f"{split.lower()}_artifact.json"
    artifact.write_text('{"frozen":true}\n')
    freeze = tmp_path / f"{split.lower()}_freeze.json"
    number = "r13" if split == R13_SPLIT else "r14"
    status = "FROZEN_BEFORE_R13" if split == R13_SPLIT else "FROZEN_BEFORE_R14"
    freeze.write_text(json.dumps({
        "schema": f"phase6j-caur-{number}-freeze-v1",
        "status": status,
        "split": split,
        "content_accessed": False,
        "artifact_sha256": sha256_file(artifact),
    }))
    return artifact, freeze


def test_r13_one_time_access_allows_same_run_resume_and_then_closes(tmp_path):
    directory = tmp_path / "r13_caur_select"
    directory.mkdir()
    instance_path = directory / "tiny_R13.json"
    instance_path.write_bytes((ROOT / "instances/tiny/tiny_03.json").read_bytes())
    artifact, freeze = _write_unlock(tmp_path, R13_SPLIT)
    ledger = tmp_path / "r13_access_ledger.json"
    first = begin_one_time_split_access(
        R13_SPLIT,
        freeze_record_path=freeze,
        artifact_path=artifact,
        ledger_path=ledger,
        run_id="frozen-pass-1",
    )
    resumed = begin_one_time_split_access(
        R13_SPLIT,
        freeze_record_path=freeze,
        artifact_path=artifact,
        ledger_path=ledger,
        run_id="frozen-pass-1",
    )
    assert first == resumed
    assert load_phase6j_instance(
        instance_path,
        freeze_record_path=freeze,
        artifact_path=artifact,
        ledger_path=ledger,
        run_id="frozen-pass-1",
    ).instance_id == "tiny_03"
    complete_one_time_split_access(ledger, run_id="frozen-pass-1")
    with pytest.raises(Phase6JAccessError, match="already complete"):
        begin_one_time_split_access(
            R13_SPLIT,
            freeze_record_path=freeze,
            artifact_path=artifact,
            ledger_path=ledger,
            run_id="frozen-pass-1",
        )


def test_r13_r14_and_legacy_r11_are_locked_before_authorization(tmp_path):
    for folder, suffix in (("r13_caur_select", "R13"), ("r14_caur_holdout", "R14")):
        path = tmp_path / folder / f"tiny_{suffix}.json"
        with pytest.raises(Phase6JAccessError, match="locked"):
            load_phase6j_instance(path)
    with pytest.raises(Phase6JAccessError, match="never read"):
        load_phase6j_instance(tmp_path / "outputs/phase6i_mr/r11_validation/final_decision.json")


def test_r12_loader_accepts_only_registered_phase6j_paths(tmp_path):
    directory = tmp_path / "r12_caur_fit"
    directory.mkdir()
    path = directory / "tiny_R12.json"
    path.write_bytes((ROOT / "instances/tiny/tiny_03.json").read_bytes())
    assert load_phase6j_instance(path).instance_id == "tiny_03"
    with pytest.raises(Phase6JAccessError, match="not a registered"):
        load_phase6j_instance(ROOT / "instances/tiny/tiny_03.json")


def test_true_full_bank_features_are_complete_deterministic_and_outcome_blind():
    instance = load_instance(ROOT / "instances/tiny/tiny_03.json")
    h1 = solve_dispatching(instance, "H1")
    current = decode_candidate(instance, candidate_from_actions(instance, h1.actions))
    critical, bottleneck, proxy = critical_and_bottleneck_operations(instance, current)
    assert critical
    assert bottleneck
    assert proxy
    generated = generate_revised_target_arms(
        instance, current, "phase6j-full-bank", 2, 692000000
    )
    scores = {arm.target_set_id: float(index) for index, arm in enumerate(generated.arms)}
    fallback = generated.arms[0].target_set_id
    kwargs = {
        "state_id": "phase6j-full-bank",
        "operation_count": instance.num_operations,
        "fallback_target_set_id": fallback,
        "frozen_scores": scores,
        "critical_operations": critical,
        "bottleneck_operations": bottleneck,
    }
    first = build_candidate_source_features(generated, **kwargs)
    second = build_candidate_source_features(generated, **kwargs)
    assert first == second
    validate_full_bank_feature_rows(generated, first)
    assert generated.requested_arm_count == 24
    assert len(first) == generated.unique_arm_count
    assert all(row["label_scope"] == FULL_BANK_SCOPE for row in first)
    assert REDUCED_AUDIT_SCOPE == "REDUCED_TOP8_AUDIT_ONLY"
    assert set(CANDIDATE_INPUT_COLUMNS).isdisjoint(LABEL_COLUMNS)
    assert sum(bool(row["is_fallback"]) for row in first) == 1

    labels = [{
        **row,
        "full_bank_unique_count": len(first),
        "continuation_advantage_mean": 0.0,
        "beats_fallback": False,
        "immediate_utility": 0.0,
        "continuation_best_makespan": current.makespan,
        "fallback_continuation_best_makespan": current.makespan,
    } for row in first]
    validate_grouped_label_records(labels)
    with pytest.raises(ValueError, match="invalid"):
        validate_grouped_label_records(labels[:-1])


def test_fallback_relative_continuation_sign_and_crn_pairing_are_deterministic():
    assert fallback_relative_advantage(100.0, 98.0, 200.0) == pytest.approx(0.01)
    assert fallback_relative_advantage(98.0, 100.0, 200.0) == pytest.approx(-0.01)
    instance = load_instance(ROOT / "instances/tiny/tiny_03.json")
    current = decode_candidate(
        instance, candidate_from_actions(instance, solve_dispatching(instance, "H1").actions)
    )
    kwargs = {
        "state_id": "phase6j-crn",
        "target_set_id": "same",
        "fallback_target_set_id": "same",
        "incumbent_makespan": current.makespan,
        "continuation_seed": 695101,
        "seed_namespace": 694000000,
        "horizon": 4,
        "config": ALNSConfig(candidate_trials=2),
    }
    first = evaluate_paired_continuation_advantage(instance, current, current, **kwargs)
    second = evaluate_paired_continuation_advantage(instance, current, current, **kwargs)
    assert first.advantage == second.advantage == 0.0
    assert first.candidate.derived_seed == first.fallback.derived_seed
    aggregate = aggregate_paired_advantages([first])
    assert aggregate["continuation_advantage_mean"] == 0.0


def test_multi_horizon_prefix_matches_frozen_continuation_at_h12():
    instance = load_instance(ROOT / "instances/tiny/tiny_03.json")
    current = decode_candidate(
        instance, candidate_from_actions(instance, solve_dispatching(instance, "H1").actions)
    )
    common = {
        "state_id": "phase6j-prefix",
        "continuation_seed": 695102,
        "seed_namespace": 694000000,
        "config": ALNSConfig(candidate_trials=2),
    }
    checkpoints = continue_frozen_alns_at_horizons(
        instance, current, horizons=(4, 8, 12), **common
    )
    frozen = continue_frozen_alns_from_candidate(
        instance, current, iterations=12, **common
    )
    observed = checkpoints[12].result
    assert set(checkpoints) == {4, 8, 12}
    assert observed.candidate == frozen.candidate
    assert observed.best_makespan == frozen.best_makespan
    assert observed.continuation_value == frozen.continuation_value
    assert observed.derived_seed == frozen.derived_seed
    assert observed.decoder_evaluations == frozen.decoder_evaluations
    assert observed.accepted_moves == frozen.accepted_moves
    assert observed.improving_moves == frozen.improving_moves
    assert observed.operator_selections == frozen.operator_selections
    assert checkpoints[12].decoder_seconds > 0
    assert checkpoints[12].neighbor_seconds > 0


def test_horizon_selection_uses_shortest_passing_agreement_rule_only():
    scale_ok = {"S": 0.1, "M": 0.2, "L": 0.3}
    metrics = {
        4: {"median_within_state_spearman": 0.69, "mean_ndcg_at_1": 0.9,
            "top1_agreement": 0.8, "mean_spearman_by_scale": scale_ok},
        8: {"median_within_state_spearman": 0.71, "mean_ndcg_at_1": 0.81,
            "top1_agreement": 0.6, "mean_spearman_by_scale": scale_ok},
        12: {},
    }
    assert select_shortest_adequate_horizon(metrics) == 8
    metrics[4]["median_within_state_spearman"] = 0.70
    assert select_shortest_adequate_horizon(metrics) == 4
    metrics[4]["mean_spearman_by_scale"] = {**scale_ok, "L": -0.01}
    metrics[8]["mean_spearman_by_scale"] = {**scale_ok, "M": -0.01}
    assert select_shortest_adequate_horizon(metrics) == 12


def test_grouped_oof_folds_cover_each_structural_cell_once():
    assignments = {
        (scale, cf): grouped_oof_fold(scale, cf)
        for scale in ("S", "M", "L") for cf in ("CF1", "CF2", "CF3")
    }
    assert set(assignments.values()) == {0, 1, 2}
    assert list(assignments.values()).count(0) == 3
    assert list(assignments.values()).count(1) == 3
    assert list(assignments.values()).count(2) == 3


def test_selection_aware_lcb_gate_is_deterministic_and_falls_back_on_harm():
    rows = [
        {"target_set_id": "fallback", "continuation_advantage_mean": 0.0,
         "continuation_advantage_std": 0.0, "beats_fallback_probability": 0.5,
         "supported": True, "immediate_utility_prediction": 0.0},
        {"target_set_id": "neural", "continuation_advantage_mean": 0.02,
         "continuation_advantage_std": 0.005, "beats_fallback_probability": 0.8,
         "supported": True, "immediate_utility_prediction": 0.0},
    ]
    decision = choose_caur_action(
        rows, fallback_target_set_id="fallback", p_min=0.65,
        lcb_lambda=1.0, delta_min=0.0025, immediate_harm_floor=-0.005,
    )
    assert decision.intervened and decision.selected_target_set_id == "neural"
    rows[1]["immediate_utility_prediction"] = -0.01
    abstained = choose_caur_action(
        rows, fallback_target_set_id="fallback", p_min=0.65,
        lcb_lambda=1.0, delta_min=0.0025, immediate_harm_floor=-0.005,
    )
    assert not abstained.intervened
    assert abstained.reason == "IMMEDIATE_HARM"


def test_phase6j_learning_and_stage_sources_do_not_name_r11_payload_paths():
    sources = [
        ROOT / "rcias_clgri/analysis/phase6j_caur.py",
        ROOT / "scripts/run_phase6j_caur_stage.py",
        ROOT / "scripts/generate_phase6j_caur_instances.py",
        ROOT / "scripts/run_phase6j_caur_pilot.py",
    ]
    forbidden = ("outputs/phase6i_mr/r11_validation", "r11_live_rev_holdout")
    for path in sources:
        text = path.read_text(encoding="utf-8").lower()
        assert all(value not in text for value in forbidden)


def test_r12_pilot_task_and_snapshot_selection_are_frozen_and_deterministic():
    config = json.loads((ROOT / "configs/phase6j_caur.json").read_text())
    tasks = build_tasks(config)
    assert len(tasks) == 9
    assert [task["scale"] for task in tasks[:3]] == ["S", "S", "S"]
    assert {task["trajectory_seed"] for task in tasks} == {691101}
    assert all(task["instance_id"].endswith("R12_C02") for task in tasks)
    observer = SnapshotObserver([0.15, 0.5, 0.85])
    observer.events = [{
        "state_id": f"state-{index}",
        "iteration": index,
        "search_progress": index / 10,
        "source_elapsed_wall_time": float(index),
        "source_decoder_evaluations": index,
        "current_makespan": 100.0,
        "current_candidate": {},
    } for index in range(10)]
    selected = observer.selected()
    assert [row["state_id"] for row in selected] == ["state-1", "state-5", "state-8"]
    assert [row["target_progress"] for row in selected] == [0.15, 0.5, 0.85]
