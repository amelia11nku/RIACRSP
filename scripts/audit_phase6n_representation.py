#!/usr/bin/env python3
"""Apply the preregistered Phase 6N N5 raw representation gate."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import train_phase6n_candidate_conditioned as training  # noqa: E402


CONFIG = ROOT / "configs/phase6n_candidate_conditioned_csg_v1.json"
TRAINING = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/training"
N4_AUDIT = TRAINING / "completion_integrity_audit.json"
PHASE6L = ROOT / "outputs/phase6l_legacy_score_decoupling_v1/training/ensemble_oof.parquet"
PHASE6L_SUMMARY = ROOT / "outputs/phase6l_legacy_score_decoupling_v1/training/oof_summary.json"
OUT = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/quality"
RESULT = OUT / "raw_representation_gate.json"
REPORT = ROOT / "docs/reports/phase6n_representation_quality_report.md"


def ranking_metrics(frame: pd.DataFrame, prediction: str, *, seed: int, resamples: int):
    states = training.metrics.ranking_state_metrics(frame, prediction)
    lower, upper = training.metrics.grouped_bootstrap_interval(
        states,
        "selected_lift",
        seed=seed,
        resamples=resamples,
    )
    scale = {
        name: {
            "states": int(len(group)),
            "spearman": float(group.spearman.mean()),
            "pairwise_accuracy": float(group.pairwise_accuracy.mean()),
            "ndcg_at_1": float(group.ndcg_at_1.mean()),
            "top1_agreement": float(group.top1_agreement.mean()),
            "selected_lift": float(group.selected_lift.mean()),
            "selection_regret": float(group.selection_regret.mean()),
        }
        for name, group in states.groupby("scale", sort=True)
    }
    origin_counts = states.selected_origin_family.value_counts().sort_index()
    return states, {
        "states": int(len(states)),
        "candidates": int(len(frame)),
        "spearman": float(states.spearman.mean()),
        "pairwise_accuracy": float(states.pairwise_accuracy.mean()),
        "ndcg_at_1": float(states.ndcg_at_1.mean()),
        "top1_agreement": float(states.top1_agreement.mean()),
        "selected_lift": float(states.selected_lift.mean()),
        "selected_lift_lcb": lower,
        "selected_lift_ucb": upper,
        "selection_regret": float(states.selection_regret.mean()),
        "by_scale": scale,
        "selected_origin_counts": {
            str(key): int(value) for key, value in origin_counts.items()
        },
        "selected_origin_family_count": int(len(origin_counts)),
        "maximum_selected_origin_fraction": float(origin_counts.max() / len(states)),
    }


def paired_bootstrap(
    phase6n_states: pd.DataFrame,
    phase6l_states: pd.DataFrame,
    *,
    seed: int,
    resamples: int,
) -> tuple[pd.DataFrame, dict]:
    columns = ["state_id", "instance_id", "scale", "selected_lift"]
    paired = phase6n_states[columns].merge(
        phase6l_states[["state_id", "selected_lift"]],
        on="state_id",
        suffixes=("_phase6n", "_phase6l"),
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != 288:
        raise RuntimeError("Phase 6N/6L common-state pairing is incomplete")
    paired["selected_lift_improvement"] = (
        paired.selected_lift_phase6n - paired.selected_lift_phase6l
    )
    instance = paired.groupby("instance_id").selected_lift_improvement.mean()
    values = instance.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[draws].mean(axis=1)
    return paired, {
        "states": len(paired),
        "instances": len(instance),
        "mean_improvement": float(paired.selected_lift_improvement.mean()),
        "instance_grouped_lcb": float(np.quantile(means, 0.025)),
        "instance_grouped_ucb": float(np.quantile(means, 0.975)),
        "by_scale": {
            str(scale): float(group.selected_lift_improvement.mean())
            for scale, group in paired.groupby("scale", sort=True)
        },
        "phase6n_selected_lift": float(paired.selected_lift_phase6n.mean()),
        "phase6l_selected_lift": float(paired.selected_lift_phase6l.mean()),
    }


def main() -> None:
    config = json.loads(CONFIG.read_text())
    gate = config["raw_representation_gate"]
    n4 = json.loads(N4_AUDIT.read_text())
    if n4.get("status") != "PASS" or not all(n4.get("checks", {}).values()):
        raise RuntimeError("Phase 6N N4 completion audit did not pass")
    phase6n = pd.read_parquet(TRAINING / "ensemble_oof.parquet")
    common = phase6n[
        phase6n.phase6n_data_origin.eq("ORIGINAL_PHASE6J_CAUR")
    ].copy()
    phase6l = pd.read_parquet(PHASE6L)
    bootstrap = gate["bootstrap"]
    seed = int(bootstrap["seed"])
    resamples = int(bootstrap["resamples"])
    expanded_states, expanded_metrics = ranking_metrics(
        phase6n, "ensemble_advantage_mean", seed=seed, resamples=resamples
    )
    common_states, common_metrics = ranking_metrics(
        common, "ensemble_advantage_mean", seed=seed, resamples=resamples
    )
    phase6l_states, phase6l_metrics = ranking_metrics(
        phase6l, "ensemble_advantage_mean", seed=seed, resamples=resamples
    )
    paired, paired_metrics = paired_bootstrap(
        common_states,
        phase6l_states,
        seed=seed,
        resamples=resamples,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    training.metrics.atomic_parquet(expanded_states, OUT / "expanded_state_metrics.parquet")
    training.metrics.atomic_parquet(common_states, OUT / "common_state_metrics.parquet")
    training.metrics.atomic_parquet(paired, OUT / "paired_phase6l_comparison.parquet")

    reference = gate["phase6l_reference"]
    phase6l_frozen_summary = json.loads(PHASE6L_SUMMARY.read_text())["metrics"]
    phase6l_reference_matches = all((
        abs(phase6l_metrics["spearman"] - reference["overall_spearman"]) <= 1e-12,
        abs(phase6l_metrics["pairwise_accuracy"] - reference["pairwise_accuracy"]) <= 1e-12,
        abs(phase6l_metrics["ndcg_at_1"] - reference["ndcg_at_1"]) <= 1e-12,
        abs(phase6l_metrics["selected_lift"] - reference["raw_selected_lift"]) <= 1e-12,
        abs(
            phase6l_frozen_summary["selected_lift_lcb"]
            - reference["raw_selected_lift_lcb"]
        ) <= 1e-12,
    ))
    diagnostic_improvements = {
        "spearman": common_metrics["spearman"] > phase6l_metrics["spearman"],
        "pairwise_accuracy": common_metrics["pairwise_accuracy"]
        > phase6l_metrics["pairwise_accuracy"],
        "ndcg_at_1": common_metrics["ndcg_at_1"] > phase6l_metrics["ndcg_at_1"],
    }
    source = pd.read_parquet(training.SOURCE)
    grouped = source.groupby("state_id", sort=True)
    unique_counts = grouped.size()
    advertised_counts = grouped.full_bank_unique_count.first()
    full_bank_integrity = all((
        source.requested_bank_count.eq(24).all(),
        unique_counts.between(21, 24).all(),
        unique_counts.eq(advertised_counts).all(),
        grouped.target_set_id.nunique().eq(unique_counts).all(),
        grouped.is_fallback.sum().eq(1).all(),
    ))
    no_scale_spearman_reversal = all(
        scope["spearman"] > 0
        and all(value["spearman"] >= 0 for value in scope["by_scale"].values())
        for scope in (common_metrics, expanded_metrics)
    )
    common_lift_positive = (
        common_metrics["selected_lift"] > 0
        and all(value["selected_lift"] > 0 for value in common_metrics["by_scale"].values())
    )
    diversity_pass = all(
        scope["selected_origin_family_count"] >= 3
        and scope["maximum_selected_origin_fraction"] <= 0.75
        for scope in (common_metrics, expanded_metrics)
    )
    checks = {
        "phase6l_reference_exact": phase6l_reference_matches,
        "common_overall_and_scale_lift_positive": common_lift_positive,
        "common_and_expanded_spearman_no_sign_reversal": no_scale_spearman_reversal,
        "common_and_expanded_raw_lift_lcb_positive": common_metrics[
            "selected_lift_lcb"
        ] > 0 and expanded_metrics["selected_lift_lcb"] > 0,
        "paired_phase6l_lift_improvement_lcb_positive": paired_metrics[
            "instance_grouped_lcb"
        ] > 0,
        "common_spearman_improvement_at_least_0_02": common_metrics["spearman"]
        - phase6l_metrics["spearman"] >= 0.02,
        "at_least_two_diagnostics_improve": sum(diagnostic_improvements.values()) >= 2,
        "candidate_origin_diversity": diversity_pass,
        "full_bank_identity": full_bank_integrity
        and n4["checks"]["ensemble_candidate_identity"],
        "all_candidates_feasible": source.candidate_feasible.astype(bool).all(),
        "finite_predictions": n4["checks"]["ensemble_finite"],
        "whole_instance_isolation": n4["checks"]["whole_instance_isolation"],
        "historical_score_calls_zero": n4["historical_score_calls"] == 0,
        "r13_r14_locked": n4["r13_accessed"] is False
        and n4["r14_accessed"] is False
        and not (ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json").exists()
        and not (ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json").exists(),
    }
    hard_check_names = [
        "common_overall_and_scale_lift_positive",
        "common_and_expanded_spearman_no_sign_reversal",
        "common_and_expanded_raw_lift_lcb_positive",
        "paired_phase6l_lift_improvement_lcb_positive",
        "common_spearman_improvement_at_least_0_02",
        "at_least_two_diagnostics_improve",
        "candidate_origin_diversity",
        "full_bank_identity",
        "all_candidates_feasible",
        "finite_predictions",
        "whole_instance_isolation",
        "historical_score_calls_zero",
        "r13_r14_locked",
    ]
    passed = checks["phase6l_reference_exact"] and all(
        checks[name] for name in hard_check_names
    )
    preferred = gate["preferred_diagnostics"]
    preferred_checks = {
        "common_spearman": common_metrics["spearman"]
        >= preferred["overall_spearman_min"],
        "common_pairwise_accuracy": common_metrics["pairwise_accuracy"]
        >= preferred["pairwise_accuracy_min"],
        "common_ndcg_at_1": common_metrics["ndcg_at_1"]
        >= preferred["ndcg_at_1_min"],
        "expanded_spearman": expanded_metrics["spearman"]
        >= preferred["overall_spearman_min"],
        "expanded_pairwise_accuracy": expanded_metrics["pairwise_accuracy"]
        >= preferred["pairwise_accuracy_min"],
        "expanded_ndcg_at_1": expanded_metrics["ndcg_at_1"]
        >= preferred["ndcg_at_1_min"],
    }
    decision = "PASS_TO_N6" if passed else gate["failure_decision"]
    result = {
        "schema": "phase6n-raw-representation-gate-v1",
        "status": "PASS" if passed else "FAIL",
        "decision": decision,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": {key: bool(value) for key, value in checks.items()},
        "hard_check_names": hard_check_names,
        "failed_hard_checks": [name for name in hard_check_names if not checks[name]],
        "preferred_checks": {key: bool(value) for key, value in preferred_checks.items()},
        "diagnostic_improvements_over_phase6l": {
            key: bool(value) for key, value in diagnostic_improvements.items()
        },
        "metrics": {
            "common_original_288": common_metrics,
            "expanded_864": expanded_metrics,
            "phase6l_common_reference": phase6l_metrics,
            "phase6l_frozen_reference_lcb": float(
                phase6l_frozen_summary["selected_lift_lcb"]
            ),
            "paired_improvement": paired_metrics,
            "common_spearman_improvement": common_metrics["spearman"]
            - phase6l_metrics["spearman"],
        },
        "calibration_stage": "ELIGIBLE" if passed else "NOT_RUN_STOPPED_ON_N5_REPRESENTATION",
        "decision_stage": "ELIGIBLE" if passed else "NOT_RUN_STOPPED_ON_N5_REPRESENTATION",
        "runtime_stage": "ELIGIBLE" if passed else "NOT_RUN_STOPPED_ON_N5_REPRESENTATION",
        "solver_stage": "ELIGIBLE" if passed else "NOT_RUN_STOPPED_ON_N5_REPRESENTATION",
        "historical_score_calls": 0,
        "r13_accessed": False,
        "r14_accessed": False,
        "artifacts": {
            str((OUT / "expanded_state_metrics.parquet").relative_to(ROOT)): training.digest(
                OUT / "expanded_state_metrics.parquet"
            ),
            str((OUT / "common_state_metrics.parquet").relative_to(ROOT)): training.digest(
                OUT / "common_state_metrics.parquet"
            ),
            str((OUT / "paired_phase6l_comparison.parquet").relative_to(ROOT)): training.digest(
                OUT / "paired_phase6l_comparison.parquet"
            ),
        },
    }
    training.metrics.atomic_json(result, RESULT)
    report = f"""# Phase 6N N5 原始表示质量报告

终态：**`{decision}`**。

N4 的 9-run outer OOF 完整性审计通过后，按预注册规则在 common original 288 states 上与冻结 Phase 6L 做 paired comparison，并独立报告 expanded 864-state OOF。没有执行 empirical calibration、direct-decision gate、ablation、runtime 或 solver qualification。

| 范围 | Spearman | Pairwise | NDCG@1 | Raw selected lift | Grouped LCB | Regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 6N common 288 | {common_metrics['spearman']:.6f} | {common_metrics['pairwise_accuracy']:.6f} | {common_metrics['ndcg_at_1']:.6f} | {common_metrics['selected_lift']:.6f} | {common_metrics['selected_lift_lcb']:.6f} | {common_metrics['selection_regret']:.6f} |
| Phase 6N expanded 864 | {expanded_metrics['spearman']:.6f} | {expanded_metrics['pairwise_accuracy']:.6f} | {expanded_metrics['ndcg_at_1']:.6f} | {expanded_metrics['selected_lift']:.6f} | {expanded_metrics['selected_lift_lcb']:.6f} | {expanded_metrics['selection_regret']:.6f} |
| Phase 6L common 288 | {phase6l_metrics['spearman']:.6f} | {phase6l_metrics['pairwise_accuracy']:.6f} | {phase6l_metrics['ndcg_at_1']:.6f} | {phase6l_metrics['selected_lift']:.6f} | {phase6l_metrics['selected_lift_lcb']:.6f} | {phase6l_metrics['selection_regret']:.6f} |

表内 Phase 6L LCB 按 N5 的 5,000 次、seed 727001 口径重算；配置中冻结的 Phase 6L 原始 LCB 0.002866 使用其 Phase 6L 2,000 次、seed 707001 口径。二者均为正，点指标与冻结 reference 精确一致，这一 bootstrap 口径差异不参与 paired improvement 判定。

Phase 6N 在 common evidence 上的 Spearman 提升 {result['metrics']['common_spearman_improvement']:.6f}，Spearman 与 pairwise 两项严格优于 Phase 6L；expanded Spearman 达到 preferred 0.25。候选来源没有坍缩，full-bank、feasibility、finite predictions、whole-instance isolation、zero historical scorer 和 R13/R14 lock 均通过。

阻断项是预注册的 paired utility improvement：Phase 6N common raw selected lift 为 {paired_metrics['phase6n_selected_lift']:.6f}，Phase 6L 为 {paired_metrics['phase6l_selected_lift']:.6f}，均值差为 {paired_metrics['mean_improvement']:.6f}；18-instance paired bootstrap 95% 区间为 [{paired_metrics['instance_grouped_lcb']:.6f}, {paired_metrics['instance_grouped_ucb']:.6f}]。LCB 未严格大于 0，因此 `paired_phase6l_lift_improvement_lcb_positive=FALSE`。

这表明显式 candidate-conditioned CSG representation 改善了相关性，但没有在同一 288-state evidence 上证明优于 Phase 6L 的实际 raw selection utility。按冻结 gate，N5 必须判为 `MODEL_REVISION_REPRESENTATION`，并停止 calibration、decision、runtime、solver 及 R13/R14。
"""
    REPORT.write_text(report)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
