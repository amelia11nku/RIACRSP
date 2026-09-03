#!/usr/bin/env python3
"""Evaluate the immutable Phase 6I-MR R11 promotion gates."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6h import (  # noqa: E402
    first_common_target_hit,
    normalized_gap_auc,
    validate_incumbent_trace,
)
from rcias_clgri.ni.calibration import calibration_metrics  # noqa: E402
from rcias_clgri.ni.phase6i_policy import (  # noqa: E402
    select_immediate_actions,
    state_ranking_metrics,
    summarize_ranking_metrics,
)


OUT = ROOT / "outputs/phase6i_mr/r11_validation"
RUNS = OUT / "runs"
ARTIFACT_PATH = ROOT / "outputs/phase6i_mr/frozen/selected_artifact.json"
FREEZE_RECORD = ROOT / "outputs/phase6i_mr/frozen/artifact_freeze.json"
CONFIG_PATH = ROOT / "configs/phase6i_mr_live_utility_revision.json"
METHODS = ("ALNS", "PHASE6H_CSGNI", "PHASE6I_MR_CSGNI")
TARGET_GAPS = (0.05, 0.02, 0.01, 0.005)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def bootstrap_interval(values: np.ndarray, seed: int = 688001) -> tuple[float, float]:
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("bootstrap requires a finite non-empty instance vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(10000, len(values)))
    samples = values[indices].mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(low), float(high)


def load_runs() -> tuple[pd.DataFrame, dict[tuple[str, str, int], dict]]:
    progress = load_json(OUT / "progress.json")
    if progress.get("status") != "COMPLETE" or progress.get("completed_runs") != 288:
        raise RuntimeError("R11 validation is not complete")
    summary = pd.read_csv(OUT / "validation_run_summary.csv")
    if len(summary) != 288 or not summary.feasible.astype(bool).all():
        raise RuntimeError("R11 run-summary integrity failed")
    payloads = {}
    for path in sorted(RUNS.glob("*/*/*.json")):
        payload = load_json(path)
        validate_incumbent_trace(
            payload["incumbent_trace"], final_best=payload["final_best"]
        )
        if payload["seed"] is not None:
            payloads[(payload["method"], payload["instance_id"], int(payload["seed"]))] = payload
    if len(payloads) != 270:
        raise RuntimeError("R11 iterative payload count must be 270")
    return summary, payloads


def ranking_and_calibration(forced: pd.DataFrame, artifact: dict, config: dict) -> tuple[dict, pd.DataFrame]:
    if len(forced) != 3600 or forced.state_id.nunique() != 900:
        raise RuntimeError("R11 forced diagnostics must contain 900 states and 3600 actions")
    revised_states = state_ranking_metrics(forced)
    u0_states = state_ranking_metrics(forced, prediction_column="ensemble_raw_score")
    revised = summarize_ranking_metrics(revised_states)
    u0 = summarize_ranking_metrics(u0_states)
    selected = select_immediate_actions(
        forced,
        probability_threshold=artifact["thresholds"]["probability"],
        utility_threshold=artifact["thresholds"]["utility"],
    )
    probability = calibration_metrics(
        forced.calibrated_probability.to_numpy(dtype=float),
        forced.positive_label.astype(int).to_numpy(),
    )
    references = config["statistics"]["probability_reference"]
    material_pair_gain = (
        revised["overall_per_instance_pairwise_accuracy"]
        - u0["overall_per_instance_pairwise_accuracy"]
    )
    material_ndcg_gain = (
        revised["overall_per_instance_ndcg_at_1"]
        - u0["overall_per_instance_ndcg_at_1"]
    )
    scale_rows = []
    for scale in ("S", "M", "L"):
        part = selected[selected.scale.eq(scale)]
        interventions = part[part.intervened]
        abstained = part[~part.intervened]
        instance_abstention = (
            abstained.groupby("instance_id").best_forced_lift_over_fallback.mean()
        )
        upper = (
            bootstrap_interval(instance_abstention.to_numpy())[1]
            if len(instance_abstention) else math.nan
        )
        scale_rows.append({
            "scale": scale,
            "state_count": len(part),
            "selected_action_count": len(interventions),
            "coverage": float(part.intervened.mean()),
            "unsupported_intervention_fraction": (
                float((~interventions.selected_supported).mean())
                if len(interventions) else 0.0
            ),
            "predicted_abstained_count": len(abstained),
            "mean_best_forced_utility_abstained": float(
                abstained.best_forced_utility.mean()
            ) if len(abstained) else math.nan,
            "mean_best_forced_lift_abstained": float(
                abstained.best_forced_lift_over_fallback.mean()
            ) if len(abstained) else math.nan,
            "abstained_best_lift_bootstrap_upper": upper,
            "selected_realized_utility": float(
                interventions.selected_realized_utility.mean()
            ) if len(interventions) else math.nan,
            "selected_lift_over_fallback": float(
                interventions.selected_lift_over_fallback.mean()
            ) if len(interventions) else math.nan,
            "selected_sign_error_rate": float(
                interventions.selected_sign_error.mean()
            ) if len(interventions) else math.nan,
        })
    scales = pd.DataFrame(scale_rows)
    report = {
        "schema": "phase6i-mr-r11-ranking-calibration-v1.2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "forced_actions": len(forced),
        "forced_states": forced.state_id.nunique(),
        "revised_ranking": revised,
        "u0_ranking": u0,
        "revised_minus_u0_pairwise_accuracy": material_pair_gain,
        "revised_minus_u0_ndcg_at_1": material_ndcg_gain,
        "selected_interventions": int(selected.intervened.sum()),
        "selected_immediate_utility": float(
            selected.loc[selected.intervened, "selected_realized_utility"].mean()
        ) if selected.intervened.any() else math.nan,
        "selected_lift_over_fallback": float(
            selected.loc[selected.intervened, "selected_lift_over_fallback"].mean()
        ) if selected.intervened.any() else math.nan,
        "probability_metrics": probability,
        "probability_reference": references,
        "scale_support": scale_rows,
    }
    return report, scales


def solver_and_anytime(
    summary: pd.DataFrame,
    payloads: dict[tuple[str, str, int], dict],
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    iterative = summary[summary.method.isin(METHODS)].copy()
    instance = iterative.groupby(
        ["method", "instance_id", "scale", "CF_level"], as_index=False
    ).agg(
        mean_final=("final_best", "mean"),
        mean_runtime=("total_runtime", "mean"),
        mean_decoder_evals=("total_decoder_evals", "mean"),
        median_time_to_best=("time_to_best", "median"),
        median_evals_to_best=("evals_to_best", "median"),
    )
    pivot = instance.pivot(index="instance_id", columns="method", values="mean_final")
    improvement = (
        (pivot.ALNS - pivot.PHASE6I_MR_CSGNI) / pivot.ALNS
    ).sort_index()
    bootstrap = bootstrap_interval(improvement.to_numpy())
    nonzero = improvement.to_numpy()[improvement.to_numpy() != 0]
    if len(nonzero):
        test = wilcoxon(nonzero, alternative="greater", method="exact")
        statistic, pvalue = float(test.statistic), float(test.pvalue)
    else:
        statistic, pvalue = 0.0, 1.0

    run_rows = []
    for instance_id in sorted(iterative.instance_id.unique()):
        for seed in sorted(iterative[iterative.instance_id.eq(instance_id)].seed.dropna().astype(int).unique()):
            keys = [(method, instance_id, seed) for method in METHODS]
            reference = min(float(payloads[key]["final_best"]) for key in keys)
            for key in keys:
                payload = payloads[key]
                budget = float(payload["time_limit_seconds"])
                row = {
                    "method": key[0], "instance_id": instance_id, "seed": seed,
                    "scale": payload["scale"], "CF_level": payload["CF_level"],
                    "reference_makespan": reference,
                    "normalized_gap_auc": normalized_gap_auc(
                        payload["incumbent_trace"], budget=budget,
                        reference_makespan=reference,
                    ),
                }
                for gap in TARGET_GAPS:
                    hit = first_common_target_hit(
                        payload["incumbent_trace"],
                        target_makespan=reference * (1.0 + gap),
                    )
                    label = str(gap).replace(".", "p")
                    row[f"target_{label}_reached"] = hit["reached"]
                    row[f"target_{label}_time"] = hit["elapsed_wall_time"]
                    row[f"target_{label}_evals"] = hit["decoder_evaluations"]
                run_rows.append(row)
    anytime = pd.DataFrame(run_rows)
    anytime_instance = anytime.groupby(["method", "instance_id"], as_index=False).agg(
        mean_auc=("normalized_gap_auc", "mean")
    )
    auc_pivot = anytime_instance.pivot(index="instance_id", columns="method", values="mean_auc")
    auc_improvement = float(
        ((auc_pivot.ALNS - auc_pivot.PHASE6I_MR_CSGNI)
         / auc_pivot.ALNS.replace(0, np.nan)).mean()
    )
    method_summary = instance.groupby("method", as_index=False).agg(
        mean_of_instance_final=("mean_final", "mean"),
        mean_runtime=("mean_runtime", "mean"),
        mean_decoder_evals=("mean_decoder_evals", "mean"),
        median_time_to_best=("median_time_to_best", "median"),
        median_evals_to_best=("median_evals_to_best", "median"),
    )
    evals = method_summary.set_index("method").mean_decoder_evals
    decoder_reduction = float(
        1.0 - evals.PHASE6I_MR_CSGNI / evals.ALNS
    )
    relative_worse = -improvement
    instance_meta = instance.drop_duplicates("instance_id").set_index("instance_id")
    subgroup = pd.DataFrame({
        "instance_id": relative_worse.index,
        "relative_worse_than_alns": relative_worse.values,
        "scale": [instance_meta.loc[item, "scale"] for item in relative_worse.index],
        "CF_level": [instance_meta.loc[item, "CF_level"] for item in relative_worse.index],
    })
    scale_relative = subgroup.groupby("scale").relative_worse_than_alns.mean().to_dict()
    cf_relative = subgroup.groupby("CF_level").relative_worse_than_alns.mean().to_dict()
    report = {
        "schema": "phase6i-mr-r11-anytime-runtime-v1.2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_summary": method_summary.to_dict("records"),
        "aggregate_revised_relative_to_alns": float(relative_worse.mean()),
        "aggregate_revised_improvement_vs_alns": float(improvement.mean()),
        "instance_improvements": improvement.to_dict(),
        "paired_instance_bootstrap_95_percent": {"low": bootstrap[0], "high": bootstrap[1]},
        "wilcoxon": {
            "alternative": "revised improvement greater than zero",
            "paired_unit": "18 instance means over five matched seeds",
            "zero_handling": "discard exact-zero paired differences before exact test",
            "nonzero_pairs": len(nonzero), "statistic": statistic, "p_value": pvalue,
        },
        "scale_relative_worse_than_alns": scale_relative,
        "cf_relative_worse_than_alns": cf_relative,
        "maximum_instance_relative_worse_than_alns": float(relative_worse.max()),
        "decoder_evaluation_reduction_vs_alns": decoder_reduction,
        "normalized_gap_auc_improvement_vs_alns": auc_improvement,
        "runtime_components": summary.groupby("method")[[
            "candidate_proposal_seconds", "csg_seconds", "neural_seconds",
            "calibration_gate_seconds", "repair_seconds", "decoder_seconds",
        ]].mean().reset_index().to_dict("records"),
        "target_gaps": list(TARGET_GAPS),
        "runtime_claim_scope": "DECLARED_PLATFORM_ONLY_NO_HARDWARE_INDEPENDENT_SPEED_CLAIM",
    }
    return report, instance, anytime


def main() -> None:
    artifact = load_json(ARTIFACT_PATH)
    freeze = load_json(FREEZE_RECORD)
    config = load_json(CONFIG_PATH)
    if not all([
        artifact.get("status") == "FROZEN_BEFORE_R11",
        artifact.get("r11_accessed") is False,
        freeze.get("artifact_sha256") == digest(ARTIFACT_PATH),
        artifact.get("code_hashes", {}).get("scripts/finalize_phase6i_mr.py")
        == digest(Path(__file__)),
    ]):
        raise RuntimeError("selected artifact/finalizer hash validation failed")
    summary, payloads = load_runs()
    forced = pd.read_parquet(OUT / "forced_diagnostics.parquet")
    ranking, scales = ranking_and_calibration(forced, artifact, config)
    runtime, instance, anytime = solver_and_anytime(summary, payloads)
    atomic_json(ranking, OUT / "r11_ranking_calibration.json")
    atomic_json(runtime, OUT / "r11_anytime_runtime.json")
    atomic_csv(scales, OUT / "scale_support_gate.csv")
    atomic_csv(instance, OUT / "instance_method_summary.csv")
    atomic_csv(anytime, OUT / "anytime_target_metrics.csv")

    gates = artifact["frozen_gate_constants"]
    scale_quality = runtime["scale_relative_worse_than_alns"]
    direct_or_exception = []
    for row in ranking["scale_support"]:
        direct = (
            row["selected_action_count"] >= 200
            and row["unsupported_intervention_fraction"] <= 0.10
        )
        exception = (
            row["predicted_abstained_count"] >= 120
            and (
                row["mean_best_forced_utility_abstained"] <= 0
                or row["mean_best_forced_lift_abstained"] <= 0
            )
            and row["abstained_best_lift_bootstrap_upper"] <= 0.0025
            and scale_quality[row["scale"]] <= 0.01
        )
        direct_or_exception.append(direct or exception)
    positive_coverages = [
        row["coverage"] for row in ranking["scale_support"] if row["coverage"] > 0
    ]
    ratio = (
        max(positive_coverages) / min(positive_coverages)
        if len(positive_coverages) >= 2 else 1.0
    )
    chosen = ranking["selected_interventions"]
    checks = {
        "correctness_integrity": bool(
            summary.feasible.astype(bool).all()
            and len(summary) == 288
            and freeze.get("r11_content_accessed") is False
        ),
        "overall_utility_spearman_positive": ranking["revised_ranking"][
            "overall_per_instance_spearman"
        ] > 0,
        "each_scale_spearman_nonnegative": all(
            ranking["revised_ranking"][f"scale_{scale}_per_instance_spearman"] >= 0
            for scale in ("S", "M", "L")
        ),
        "pairwise_and_ndcg_materially_better_than_u0": (
            ranking["revised_minus_u0_pairwise_accuracy"]
            >= gates["u0_material_pairwise_accuracy_gain"]
            and ranking["revised_minus_u0_ndcg_at_1"]
            >= gates["u0_material_ndcg_at_1_gain"]
        ),
        "selected_utility_positive_and_above_fallback": bool(
            chosen > 0
            and ranking["selected_immediate_utility"] > 0
            and ranking["selected_lift_over_fallback"] > 0
        ),
        "support_aware_coverage": all(direct_or_exception),
        "coverage_balance": bool(ratio <= 8.0 or all(direct_or_exception)),
        "probability_reliability": bool(
            ranking["probability_metrics"]["expected_calibration_error"] <= 0.10
            and ranking["probability_metrics"]["brier_score"]
            <= 1.10 * config["statistics"]["probability_reference"]["brier"]
            and ranking["probability_metrics"]["negative_log_likelihood"]
            <= 1.10 * config["statistics"]["probability_reference"]["nll"]
        ),
        "solver_noninferiority": runtime["aggregate_revised_relative_to_alns"] <= 0.01,
        "no_catastrophic_collapse": bool(
            max(runtime["scale_relative_worse_than_alns"].values()) <= 0.02
            and max(runtime["cf_relative_worse_than_alns"].values()) <= 0.02
            and runtime["maximum_instance_relative_worse_than_alns"] <= 0.05
        ),
        "search_efficiency_advantage": bool(
            runtime["aggregate_revised_relative_to_alns"] <= 0.01
            and (
                runtime["decoder_evaluation_reduction_vs_alns"] >= 0.25
                or runtime["normalized_gap_auc_improvement_vs_alns"] >= 0.10
            )
        ),
        "runtime_reporting_complete": bool(
            summary[[
                "total_runtime", "total_decoder_evals", "time_to_best",
                "evals_to_best", "candidate_proposal_seconds", "csg_seconds",
                "neural_seconds", "calibration_gate_seconds", "repair_seconds",
                "decoder_seconds",
            ]].notna().all().all()
        ),
    }
    correctness = checks["correctness_integrity"]
    stability = checks["solver_noninferiority"] and checks["no_catastrophic_collapse"]
    if not correctness or not stability:
        decision = "HOLD"
    elif all(checks.values()):
        decision = "PROCEED_FREEZE_V1"
    else:
        decision = "MODEL_REVISION"
    gate_table = pd.DataFrame([
        {"gate": name, "passed": passed} for name, passed in checks.items()
    ])
    atomic_csv(gate_table, OUT / "promotion_gate_table.csv")
    final = {
        "schema": "phase6i-mr-final-decision-v1.2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "checks": checks,
        "selected_artifact_sha256": digest(ARTIFACT_PATH),
        "r11_accessed_once": True,
        "no_r11_retuning": True,
        "preferred_evidence": {
            "utility_spearman_at_least_0_20": ranking["revised_ranking"][
                "overall_per_instance_spearman"
            ] >= 0.20,
            "solver_improvement_at_least_0_005": runtime[
                "aggregate_revised_improvement_vs_alns"
            ] >= 0.005,
            "bootstrap_lower_bound_positive": runtime[
                "paired_instance_bootstrap_95_percent"
            ]["low"] > 0,
            "one_sided_wilcoxon_p_below_0_05": runtime["wilcoxon"]["p_value"] < 0.05,
        },
        "evidence": {
            "ranking_calibration": "outputs/phase6i_mr/r11_validation/r11_ranking_calibration.json",
            "anytime_runtime": "outputs/phase6i_mr/r11_validation/r11_anytime_runtime.json",
            "gate_table": "outputs/phase6i_mr/r11_validation/promotion_gate_table.csv",
        },
    }
    atomic_json(final, OUT / "final_decision.json")
    print(f"PHASE6I_MR_FINAL_DECISION={decision}")


if __name__ == "__main__":
    main()
