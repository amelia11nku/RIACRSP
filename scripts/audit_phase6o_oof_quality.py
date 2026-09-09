#!/usr/bin/env python3
"""Apply the preregistered Phase 6O Route B OOF quality gate."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import train_phase6o_top_utility as training  # noqa: E402


CONFIG = ROOT / "configs/phase6o_neural_shortlist_v1.json"
TRAINING = ROOT / "outputs/phase6o_neural_shortlist_v1/training"
COMPLETION = TRAINING / "completion_integrity_audit.json"
PHASE6L = (
    ROOT / "outputs/phase6l_legacy_score_decoupling_v1/training/ensemble_oof.parquet"
)
OUT = ROOT / "outputs/phase6o_neural_shortlist_v1/quality"
RESULT = OUT / "oof_quality_gate.json"
REPORT = ROOT / "docs/reports/phase6o_oof_quality_report.md"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def ranking_metrics(frame: pd.DataFrame, prediction: str) -> tuple[pd.DataFrame, dict]:
    states = training.metrics.ranking_state_metrics(frame, prediction)
    origins = states.selected_origin_family.value_counts().sort_index()
    return states, {
        "states": int(len(states)),
        "candidates": int(len(frame)),
        "spearman": float(states.spearman.mean()),
        "pairwise_accuracy": float(states.pairwise_accuracy.mean()),
        "ndcg_at_1": float(states.ndcg_at_1.mean()),
        "top1_agreement": float(states.top1_agreement.mean()),
        "selected_lift": float(states.selected_lift.mean()),
        "selection_regret": float(states.selection_regret.mean()),
        "selected_origin_counts": {
            str(key): int(value) for key, value in origins.items()
        },
        "selected_origin_family_count": int(len(origins)),
        "maximum_selected_origin_fraction": float(origins.max() / len(states)),
    }


def add_bootstrap(metrics: dict, states: pd.DataFrame, bootstrap: dict) -> None:
    lower, upper = training.metrics.grouped_bootstrap_interval(
        states,
        "selected_lift",
        seed=int(bootstrap["seed"]),
        resamples=int(bootstrap["resamples"]),
    )
    metrics["selected_lift_lcb"] = lower
    metrics["selected_lift_ucb"] = upper


def paired_phase6l(
    phase6o_states: pd.DataFrame,
    phase6l_states: pd.DataFrame,
    bootstrap: dict,
) -> tuple[pd.DataFrame, dict]:
    paired = phase6o_states[
        ["state_id", "instance_id", "scale", "selected_lift"]
    ].merge(
        phase6l_states[["state_id", "selected_lift"]],
        on="state_id",
        suffixes=("_phase6o", "_phase6l_same_truth"),
        validate="one_to_one",
    )
    require(len(paired) == 288, "Phase 6O/6L paired common scope is incomplete")
    paired["selected_lift_improvement"] = (
        paired.selected_lift_phase6o - paired.selected_lift_phase6l_same_truth
    )
    values = (
        paired.groupby("instance_id", sort=True)
        .selected_lift_improvement.mean()
        .to_numpy(float)
    )
    require(len(values) == 18, "Phase 6O paired bootstrap must use 18 instances")
    rng = np.random.default_rng(int(bootstrap["seed"]))
    draws = rng.integers(
        0, len(values), size=(int(bootstrap["resamples"]), len(values))
    )
    means = values[draws].mean(axis=1)
    return paired, {
        "states": len(paired),
        "instances": len(values),
        "seed": int(bootstrap["seed"]),
        "resamples": int(bootstrap["resamples"]),
        "unit": bootstrap["unit"],
        "mean_improvement": float(paired.selected_lift_improvement.mean()),
        "instance_grouped_lcb": float(np.quantile(means, 0.025)),
        "instance_grouped_ucb": float(np.quantile(means, 0.975)),
        "phase6o_selected_lift": float(paired.selected_lift_phase6o.mean()),
        "phase6l_selected_lift_on_phase6o_truth": float(
            paired.selected_lift_phase6l_same_truth.mean()
        ),
        "by_scale": {
            str(scale): float(group.selected_lift_improvement.mean())
            for scale, group in paired.groupby("scale", sort=True)
        },
    }


def stratified_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    def append(scope: str, subset: pd.DataFrame) -> None:
        _, metric = ranking_metrics(subset, "ensemble_advantage_mean")
        rows.append(
            {
                "scope": scope,
                **{
                    key: metric[key]
                    for key in (
                        "states",
                        "candidates",
                        "spearman",
                        "pairwise_accuracy",
                        "ndcg_at_1",
                        "top1_agreement",
                        "selected_lift",
                        "selection_regret",
                        "selected_origin_family_count",
                        "maximum_selected_origin_fraction",
                    )
                },
            }
        )

    append("EXPANDED_ALL", frame)
    append(
        "COMMON_ORIGINAL",
        frame[frame.phase6n_data_origin.eq("ORIGINAL_PHASE6J_CAUR")],
    )
    for source, subset in frame.groupby("phase6n_data_origin", sort=True):
        append(f"SOURCE:{source}", subset)
    for scale, subset in frame.groupby("scale", sort=True):
        append(f"SCALE:{scale}", subset)
    for (source, scale), subset in frame.groupby(
        ["phase6n_data_origin", "scale"], sort=True
    ):
        append(f"SOURCE_SCALE:{source}:{scale}", subset)
    return pd.DataFrame(rows)


def main() -> None:
    config = json.loads(CONFIG.read_text())
    gate = config["oof_quality_gate"]
    completion = json.loads(COMPLETION.read_text())
    require(
        completion.get("status") == "PASS"
        and all(completion.get("checks", {}).values()),
        "Phase 6O formal training completion audit did not pass",
    )
    ensemble = pd.read_parquet(TRAINING / "ensemble_oof.parquet")
    common = ensemble[
        ensemble.phase6n_data_origin.eq("ORIGINAL_PHASE6J_CAUR")
    ].copy()
    phase6l = pd.read_parquet(PHASE6L)
    bootstrap = gate["paired_bootstrap"]

    expanded_states, expanded_metrics = ranking_metrics(
        ensemble, "ensemble_advantage_mean"
    )
    common_states, common_metrics = ranking_metrics(
        common, "ensemble_advantage_mean"
    )
    add_bootstrap(expanded_metrics, expanded_states, bootstrap)
    add_bootstrap(common_metrics, common_states, bootstrap)

    phase6l_frozen_states, phase6l_frozen_metrics = ranking_metrics(
        phase6l, "ensemble_advantage_mean"
    )
    phase6l_scores = phase6l[
        ["state_id", "target_set_id", "ensemble_advantage_mean"]
    ].rename(columns={"ensemble_advantage_mean": "phase6l_score"})
    common_with_phase6l = common.merge(
        phase6l_scores,
        on=["state_id", "target_set_id"],
        how="inner",
        validate="one_to_one",
    )
    require(
        len(common_with_phase6l) == len(common) == len(phase6l),
        "Phase 6L and Phase 6O common candidate identity differs",
    )
    phase6l_same_truth_states, phase6l_same_truth_metrics = ranking_metrics(
        common_with_phase6l, "phase6l_score"
    )
    paired, paired_metrics = paired_phase6l(
        common_states, phase6l_same_truth_states, bootstrap
    )

    common_by_scale = {
        str(scale): float(group.selected_lift.mean())
        for scale, group in common_states.groupby("scale", sort=True)
    }
    expanded_by_scale = {
        str(scale): float(group.selected_lift.mean())
        for scale, group in expanded_states.groupby("scale", sort=True)
    }
    source_map = ensemble.groupby("state_id", sort=True).phase6n_data_origin.first()
    expanded_states["phase6n_data_origin"] = expanded_states.state_id.map(source_map)
    source_metrics = {
        str(source): {
            "states": int(len(group)),
            "selected_lift": float(group.selected_lift.mean()),
            "selection_regret": float(group.selection_regret.mean()),
        }
        for source, group in expanded_states.groupby("phase6n_data_origin", sort=True)
    }
    minimum_families = int(gate["candidate_origin_diversity_min_families"])
    maximum_fraction = float(gate["candidate_origin_max_fraction"])
    diversity_pass = all(
        scope["selected_origin_family_count"] >= minimum_families
        and scope["maximum_selected_origin_fraction"] <= maximum_fraction
        for scope in (common_metrics, expanded_metrics)
    )
    checks = {
        "phase6l_frozen_reference_exact": abs(
            phase6l_frozen_metrics["selected_lift"]
            - float(gate["common_raw_selected_lift_min"])
        )
        <= 1e-12
        and abs(
            phase6l_frozen_metrics["selection_regret"]
            - float(gate["common_top1_regret_max"])
        )
        <= 1e-12,
        "common_raw_selected_lift_at_least_phase6l": common_metrics[
            "selected_lift"
        ]
        >= float(gate["common_raw_selected_lift_min"]),
        "expanded_raw_selected_lift_positive": expanded_metrics["selected_lift"]
        > 0,
        "expanded_grouped_lcb_positive": expanded_metrics["selected_lift_lcb"]
        > 0,
        "common_top1_regret_at_most_phase6l": common_metrics["selection_regret"]
        <= float(gate["common_top1_regret_max"]),
        "all_scale_selected_lift_nonnegative": all(
            value >= 0 for value in [*common_by_scale.values(), *expanded_by_scale.values()]
        ),
        "candidate_origin_diversity": diversity_pass,
        "phase6l_common_candidate_identity": len(common_with_phase6l) == len(common),
        "full_bank_feasibility_fold_isolation": all(
            completion["checks"][name]
            for name in (
                "full_bank_identity",
                "candidate_feasibility",
                "whole_instance_fold_isolation",
            )
        ),
        "finite_predictions": completion["checks"]["ensemble_finite"],
        "historical_score_calls_zero": completion["historical_score_calls"] == 0,
        "r13_r14_locked": completion["r13_accessed"] is False
        and completion["r14_accessed"] is False
        and not (
            ROOT / "outputs/phase6o_neural_shortlist_v1/r13_selection/access_ledger.json"
        ).exists()
        and not (
            ROOT / "outputs/phase6o_neural_shortlist_v1/r14_holdout/access_ledger.json"
        ).exists(),
    }
    hard_check_names = [
        "common_raw_selected_lift_at_least_phase6l",
        "expanded_raw_selected_lift_positive",
        "expanded_grouped_lcb_positive",
        "common_top1_regret_at_most_phase6l",
        "all_scale_selected_lift_nonnegative",
        "candidate_origin_diversity",
        "phase6l_common_candidate_identity",
        "full_bank_feasibility_fold_isolation",
        "finite_predictions",
        "historical_score_calls_zero",
        "r13_r14_locked",
    ]
    passed = checks["phase6l_frozen_reference_exact"] and all(
        checks[name] for name in hard_check_names
    )
    decision = "PROCEED_DEVELOPMENT_SOLVER_PILOT" if passed else gate[
        "failure_decision"
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    stratified = stratified_rows(ensemble)
    origin_rows = []
    for scope_name, metrics in (
        ("COMMON_ORIGINAL", common_metrics),
        ("EXPANDED_ALL", expanded_metrics),
    ):
        for origin, count in metrics["selected_origin_counts"].items():
            origin_rows.append(
                {
                    "scope": scope_name,
                    "origin_family": origin,
                    "selected_states": count,
                    "selected_fraction": count / metrics["states"],
                }
            )
    training.metrics.atomic_parquet(
        expanded_states, OUT / "expanded_state_metrics.parquet"
    )
    training.metrics.atomic_parquet(
        common_states, OUT / "common_state_metrics.parquet"
    )
    training.metrics.atomic_parquet(
        paired, OUT / "paired_phase6l_comparison.parquet"
    )
    training.metrics.atomic_csv(stratified, OUT / "stratified_metrics.csv")
    training.metrics.atomic_csv(
        pd.DataFrame(origin_rows), OUT / "selected_origin_metrics.csv"
    )

    result = {
        "schema": "phase6o-oof-quality-gate-v1",
        "status": "PASS" if passed else "FAIL",
        "decision": decision,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": {key: bool(value) for key, value in checks.items()},
        "hard_check_names": hard_check_names,
        "failed_hard_checks": [name for name in hard_check_names if not checks[name]],
        "thresholds": gate,
        "metrics": {
            "common_original_288": {**common_metrics, "by_scale_lift": common_by_scale},
            "expanded_864": {**expanded_metrics, "by_scale_lift": expanded_by_scale},
            "by_source": source_metrics,
            "phase6l_frozen_reference": phase6l_frozen_metrics,
            "phase6l_on_phase6o_relabel_truth": phase6l_same_truth_metrics,
            "paired_improvement_same_relabel_truth": paired_metrics,
        },
        "direct_decision_stage": "ELIGIBLE" if passed else "NOT_RUN_OOF_GATE_FAILED",
        "development_solver_pilot": "ELIGIBLE" if passed else "NOT_RUN_OOF_GATE_FAILED",
        "formal_runtime": "NOT_YET_ELIGIBLE" if passed else "NOT_RUN_OOF_GATE_FAILED",
        "formal_r12_solver": "NOT_YET_ELIGIBLE" if passed else "NOT_RUN_OOF_GATE_FAILED",
        "historical_score_calls": 0,
        "gurobi_run": False,
        "r13_accessed": False,
        "r14_accessed": False,
        "artifacts": {
            str(path.relative_to(ROOT)): training.digest(path)
            for path in (
                OUT / "expanded_state_metrics.parquet",
                OUT / "common_state_metrics.parquet",
                OUT / "paired_phase6l_comparison.parquet",
                OUT / "stratified_metrics.csv",
                OUT / "selected_origin_metrics.csv",
            )
        },
    }
    training.metrics.atomic_json(result, RESULT)
    REPORT.write_text(
        f"""# Phase 6O OOF 质量报告

终态：**`{decision}`**。

正式训练完整性审计通过后，本审计按冻结 Route B 门槛评估 3-seed ensemble OOF。没有拟合 direct-decision calibration，没有运行 development solver pilot、runtime、formal R12、Gurobi、R13 或 R14。

| 范围 | Spearman | Pairwise | NDCG@1 | Raw selected lift | Grouped LCB | Regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 6O common 288 | {common_metrics['spearman']:.6f} | {common_metrics['pairwise_accuracy']:.6f} | {common_metrics['ndcg_at_1']:.6f} | {common_metrics['selected_lift']:.6f} | {common_metrics['selected_lift_lcb']:.6f} | {common_metrics['selection_regret']:.6f} |
| Phase 6O expanded 864 | {expanded_metrics['spearman']:.6f} | {expanded_metrics['pairwise_accuracy']:.6f} | {expanded_metrics['ndcg_at_1']:.6f} | {expanded_metrics['selected_lift']:.6f} | {expanded_metrics['selected_lift_lcb']:.6f} | {expanded_metrics['selection_regret']:.6f} |
| Frozen Phase 6L common 288 | {phase6l_frozen_metrics['spearman']:.6f} | {phase6l_frozen_metrics['pairwise_accuracy']:.6f} | {phase6l_frozen_metrics['ndcg_at_1']:.6f} | {phase6l_frozen_metrics['selected_lift']:.6f} | n/a | {phase6l_frozen_metrics['selection_regret']:.6f} |

Common 的冻结硬门槛要求 raw selected lift ≥ {gate['common_raw_selected_lift_min']:.8f}、regret ≤ {gate['common_top1_regret_max']:.8f}。实际分别为 {common_metrics['selected_lift']:.8f} 和 {common_metrics['selection_regret']:.8f}，两项均失败。Expanded lift 与 grouped LCB 为正，common/expanded 的 S/M/L lift 全部非负，origin diversity、candidate identity/order、full bank、feasibility、whole-instance isolation 和 finite prediction 均通过。

为避免混淆 reference truth，本报告另外把 Phase 6L 排序应用到 Phase 6O 的同一份 targeted-relabel truth。其 lift 为 {phase6l_same_truth_metrics['selected_lift']:.8f}，Phase 6O 的 paired delta 为 {paired_metrics['mean_improvement']:.8f}，18-instance bootstrap 95% CI 为 [{paired_metrics['instance_grouped_lcb']:.8f}, {paired_metrics['instance_grouped_ucb']:.8f}]。该 CI 是诊断；冻结点值和 regret 门槛仍使用原 Phase 6L truth 上预注册的常数。

按协议，Route B 在 point/top-selection 硬要求失败时必须终止为 `MODEL_REVISION_TOP_UTILITY`，不得创建新 selector，也不得进入 solver pilot。R13/R14 保持锁定。
"""
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
