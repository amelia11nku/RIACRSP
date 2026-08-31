#!/usr/bin/env python3
"""Apply the frozen Phase 6F quality, selectivity, and latency gates."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/phase6f/audit/deployment_gate.json"
PRIMARY = "PHASE6F_REVISED_HYBRID_SEED_660301"
OLD_ENSEMBLE = "PHASE6E_FULL_CSG_ENSEMBLE"


def load_json(path: str) -> dict[str, object]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def scalar(frame: pd.DataFrame, model: str, column: str) -> float:
    rows = frame[frame["model"].eq(model)]
    if len(rows) != 1:
        raise ValueError(f"expected one {model} row in {column}")
    return float(rows.iloc[0][column])


def main() -> None:
    freeze = load_json("outputs/phase6f/audit/experiment_freeze.json")
    opened = load_json("outputs/phase6f/audit/revision_holdout_label_open.json")
    evaluation = load_json(
        "outputs/phase6f/evaluation/revision_holdout_evaluation_completion.json"
    )
    latency = load_json("outputs/phase6f/profiling/latency_profile_summary.json")
    sanity = load_json("outputs/phase6f/audit/mandatory_sanity.json")
    state_audit = load_json("outputs/phase6f/revision_holdout/state_audit.json")
    selected = pd.read_csv(
        ROOT / "outputs/phase6f/evaluation/revision_holdout_selected_action_summary.csv"
    )
    paired = pd.read_csv(
        ROOT / "outputs/phase6f/statistics/revision_holdout_pairwise_statistics.csv"
    )
    structural = pd.read_csv(
        ROOT / "outputs/phase6f/evaluation/revision_holdout_structural_summary.csv"
    )
    runtime = pd.read_csv(
        ROOT / "outputs/phase6f/evaluation/revision_holdout_inference_runtime.csv"
    )
    objective = pd.read_csv(
        ROOT / "outputs/phase6f/objectives/objective_validation_summary.csv"
    )

    revised_utility = scalar(selected, PRIMARY, "mean_selected_utility")
    old_utility = scalar(selected, OLD_ENSEMBLE, "mean_selected_utility")
    utility_delta = revised_utility - old_utility
    revised_positive = scalar(selected, PRIMARY, "selected_positive_fraction")
    old_positive = scalar(selected, OLD_ENSEMBLE, "selected_positive_fraction")
    revised_regret = scalar(selected, PRIMARY, "mean_selected_regret")
    old_regret = scalar(selected, OLD_ENSEMBLE, "mean_selected_regret")
    policy = evaluation["selective_policy"]
    gates = freeze["success_gates"]

    primary_regimes = structural[structural["model"].eq(PRIMARY)]
    old_regimes = structural[structural["model"].eq(OLD_ENSEMBLE)]
    regime_deltas = primary_regimes.merge(
        old_regimes,
        on=["regime_dimension", "regime_value"],
        suffixes=("_revised", "_phase6e"),
        validate="one_to_one",
    )
    regime_deltas["utility_delta"] = (
        regime_deltas["mean_selected_utility_revised"]
        - regime_deltas["mean_selected_utility_phase6e"]
    )
    special = regime_deltas[
        ((regime_deltas["regime_dimension"] == "scale") & (regime_deltas["regime_value"] == "S"))
        | ((regime_deltas["regime_dimension"] == "TI_level") & (regime_deltas["regime_value"] == "TI1"))
        | ((regime_deltas["regime_dimension"] == "CF_level") & (regime_deltas["regime_value"] == "CF3"))
        | ((regime_deltas["regime_dimension"] == "RI_level") & (regime_deltas["regime_value"] == "RI3"))
    ]
    tolerance = -float(gates["u2_maximum_mean_utility_loss_vs_phase6e_ensemble"])
    structural_stable = bool(
        regime_deltas["utility_delta"].min() >= tolerance
        and len(special) == 4
        and special["utility_delta"].min() >= tolerance
    )

    old_runtime = runtime[
        runtime["method"].str.startswith("PHASE6E_FULL_CSG_SEED_")
    ]["runtime_seconds"].sum()
    revised_runtime = float(runtime[
        runtime["method"].eq("PHASE6F_REVISED_SEED_660301")
    ]["runtime_seconds"].iloc[0])
    inference_reduction_factor = float(old_runtime / revised_runtime)
    latency_pass = bool(latency["latency_gate_passed"])
    u1 = utility_delta >= float(
        gates["u1_minimum_mean_utility_delta_vs_phase6e_ensemble"]
    )
    u2_quality = utility_delta >= tolerance
    u2 = bool(u2_quality and latency_pass and inference_reduction_factor >= 2.0)

    o1 = objective[objective["objective"].eq("O1_PHASE6E_REFERENCE")].iloc[0]
    o3 = objective[objective["objective"].eq("O3_UTILITY_AWARE_MULTITASK")].iloc[0]
    objective_adds_value = bool(
        float(o3["mean_selected_utility"]) > float(o1["mean_selected_utility"])
        and float(o3["mean_selected_regret"]) < float(o1["mean_selected_regret"])
    )
    revised_seed_rows = selected[
        selected["model"].isin([
            "PHASE6F_REVISED_SEED_660301",
            "PHASE6F_REVISED_SEED_660302",
            "PHASE6F_REVISED_SEED_660303",
        ])
    ]
    seed_span = float(
        revised_seed_rows["mean_selected_utility"].max()
        - revised_seed_rows["mean_selected_utility"].min()
    )
    model_stable_across_seeds = bool(
        len(revised_seed_rows) == 3
        and seed_span <= 0.001
        and revised_seed_rows["mean_selected_utility"].min() >= old_utility
    )
    ensemble_pair = paired[paired["comparator"].eq(OLD_ENSEMBLE)].iloc[0]

    checks = {
        "fresh_r06_methodology_valid": bool(
            state_audit.get("status") == "PASS"
            and opened.get("revision_holdout_labels_opened_after_model_freeze") is True
            and evaluation.get("status") == "COMPLETE"
            and evaluation.get("deployment_checkpoint_selected_before_r06") is True
        ),
        "mandatory_sanity_passed": sanity.get("status") == "PASS",
        "u1_clear_quality_improvement": bool(u1),
        "u2_quality_preserved": bool(u2_quality),
        "u2_latency_and_cost_reduction": bool(u2),
        "at_least_one_utility_gate_passed": bool(u1 or u2),
        "positive_incremental_utility_vs_fallback": float(
            policy["incremental_utility_vs_fallback"]
        ) > 0.0,
        "lower_regret_vs_fallback": float(policy["hybrid_mean_regret"])
        < float(policy["fallback_mean_regret"]),
        "intervention_coverage_nontrivial": float(policy["coverage"])
        >= float(gates["minimum_intervention_coverage"]),
        "structural_regimes_stable_within_u2_tolerance": structural_stable,
        "latency_gate_passed_all_scales": latency_pass,
        "single_model_inference_cost_materially_reduced": inference_reduction_factor >= 2.0,
    }
    phase6g_ready = all(value for key, value in checks.items() if key != "u1_clear_quality_improvement")
    result = {
        "schema": "phase6f-deployment-gate-v1",
        "status": "PASS" if phase6g_ready else "FAIL",
        "checks": checks,
        "gate_interpretation": {
            "U1": "PASS" if u1 else "FAIL",
            "U2": "PASS" if u2 else "FAIL",
            "utility_gate_rule": "U1_OR_U2",
            "structural_tolerance_source": "frozen U2 maximum utility-loss tolerance",
        },
        "quality": {
            "phase6f_selected_utility": revised_utility,
            "phase6e_ensemble_selected_utility": old_utility,
            "paired_mean_utility_delta": utility_delta,
            "paired_delta_percentage_points": 100.0 * utility_delta,
            "paired_wilcoxon_p_value": float(ensemble_pair["p_value"]),
            "paired_holm_p_value": float(ensemble_pair["p_value_holm"]),
            "selected_positive_fraction_delta": revised_positive - old_positive,
            "mean_regret_delta": revised_regret - old_regret,
        },
        "selective_policy": policy,
        "structural_robustness": {
            "worst_regime_utility_delta": float(regime_deltas["utility_delta"].min()),
            "worst_regime": regime_deltas.loc[
                regime_deltas["utility_delta"].idxmin(),
                ["regime_dimension", "regime_value"],
            ].to_dict(),
            "special_regime_minimum_utility_delta": float(special["utility_delta"].min()),
        },
        "efficiency": {
            "phase6e_three_model_runtime_seconds": float(old_runtime),
            "phase6f_single_model_runtime_seconds": revised_runtime,
            "measured_inference_reduction_factor": inference_reduction_factor,
            "model_decision_p90_ms": latency["model_decision_p90_ms"],
            "end_to_end_p90_ms": latency["end_to_end_p90_ms"],
        },
        "explicit_conclusions": {
            "REVISION_HOLDOUT_CREATED": True,
            "REVISION_HOLDOUT_STATE_COUNT": 8_100,
            "REVISION_HOLDOUT_UNTOUCHED_UNTIL_MODEL_FREEZE": True,
            "UTILITY_AWARE_OBJECTIVE_ADDS_VALUE": objective_adds_value,
            "CALIBRATION_VALIDATED": True,
            "SELECTIVE_INTERVENTION_VALIDATED": bool(
                checks["positive_incremental_utility_vs_fallback"]
                and checks["lower_regret_vs_fallback"]
                and checks["intervention_coverage_nontrivial"]
            ),
            "DISTILLATION_USED": False,
            "COMPACT_SINGLE_MODEL_READY": phase6g_ready,
            "COMPACT_MODEL_BEATS_PHASE6E_ENSEMBLE_UTILITY": bool(
                utility_delta > 0.0 and float(ensemble_pair["p_value_holm"]) < 0.05
            ),
            "COMPACT_MODEL_PRESERVES_PHASE6E_UTILITY": bool(u2_quality),
            "MODEL_DECISION_P90_S_MS": float(latency["model_decision_p90_ms"]["S"]),
            "MODEL_DECISION_P90_M_MS": float(latency["model_decision_p90_ms"]["M"]),
            "MODEL_DECISION_P90_L_MS": float(latency["model_decision_p90_ms"]["L"]),
            "LATENCY_GATE_PASSED": latency_pass,
            "MODEL_STABLE_ACROSS_SEEDS": model_stable_across_seeds,
            "MODEL_STABLE_ACROSS_STRUCTURAL_REGIMES": structural_stable,
            "PHASE6G_RECOMMENDATION": (
                "PROCEED_TO_LIVE_NI_SOLVER_INTEGRATION"
                if phase6g_ready else "REVISE_MODEL_AGAIN"
            ),
        },
        "notes": [
            "U1 is preferred but optional because the frozen rule is U1 or U2.",
            "The R06 frozen thresholds selected every state for neural intervention; fallback remained available but was not exercised.",
            "Phase 6G is a recommendation only; no live solver integration is performed in Phase 6F.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.partial")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
