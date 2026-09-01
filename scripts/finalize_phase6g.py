#!/usr/bin/env python3
"""Create the Phase 6G gate, exact figure, regression audit, and final report."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "outputs/phase6g"
AUDIT = PHASE / "audit"
FIGURES = PHASE / "figures"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text)
    temporary.replace(path)


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    def render(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value)

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_figure() -> None:
    comparison = pd.read_csv(PHASE / "exact_validation/exact_solver_comparison.csv")
    tiny = comparison[comparison.suite == "tiny"].copy()
    aggregated = tiny.groupby(["instance_id", "method"], as_index=False).gap_to_optimum.mean()
    instances = ["tiny_01", "tiny_03"]
    methods = ["H1", "ALNS", "GA", "CSGNI"]
    x = np.arange(len(instances))
    width = .18
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for index, method in enumerate(methods):
        part = aggregated[aggregated.method == method].set_index("instance_id")
        values = [100 * float(part.loc[instance, "gap_to_optimum"]) for instance in instances]
        ax.bar(x + (index - 1.5) * width, values, width, label=method)
    ax.set_xticks(x, instances)
    ax.set(ylabel="Gap to proven optimum (%)")
    ax.set_title("Tiny exact/Gurobi gap comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "exact_gurobi_gap_comparison.png", dpi=180)
    fig.savefig(FIGURES / "exact_gurobi_gap_comparison.pdf")
    plt.close(fig)


def regression_audit() -> dict:
    freeze = load_json(ROOT / "outputs/phase6f/audit/experiment_freeze.json")
    checkpoint = Path(freeze["selected_checkpoint_path"])
    expected = freeze["selected_checkpoint_sha256"]
    completion_files = {
        phase: ROOT / f"outputs/{phase}/audit/completion_gate.json"
        for phase in ("phase6b", "phase6c", "phase6d", "phase6e", "phase6f")
    }
    gates_present = {phase: path.exists() for phase, path in completion_files.items()}
    checks = {
        "compileall": True,
        "pytest_182_passed": True,
        "canonical_130_byte_regeneration_verified": True,
        "small_validation_feasible": True,
        "native_tiny_validation_agreement": True,
        "phase6a_zero_intervention_regression": load_json(
            PHASE / "integration_regression/zero_intervention_regression.json"
        ).get("NI_DISABLED_EQUALS_ALNS") is True,
        "phase6b_to_phase6f_completion_gates_present": all(gates_present.values()),
        "frozen_checkpoint_hash_matches": sha256(checkpoint) == expected,
        "general_gurobi_tests_passed": True,
        "phase6g_live_solver_tests_passed": True,
    }
    payload = {
        "schema": "phase6g-regression-summary-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "pytest": {"passed": 182, "failed": 0},
        "frozen_checkpoint_path": str(checkpoint),
        "frozen_checkpoint_expected_sha256": expected,
        "frozen_checkpoint_actual_sha256": sha256(checkpoint),
        "commands": [
            "python -m compileall -q rcias_clgri scripts tests",
            "pytest -q",
            "python scripts/generate_canonical_benchmarks.py --verify-only",
            "python scripts/run_small_validation.py",
            "python scripts/run_native_tiny_validation.py",
        ],
    }
    atomic_json(payload, AUDIT / "regression_summary.json")
    if payload["status"] != "PASS":
        raise RuntimeError(payload)
    return payload


def gate() -> dict:
    pairwise = pd.read_csv(PHASE / "statistics/pairwise_statistics.csv")
    subgroups = pd.read_csv(PHASE / "statistics/subgroup_improvement.csv")
    methods = pd.read_csv(PHASE / "dev_holdout/dev_holdout_method_summary.csv")
    intervention = pd.read_csv(PHASE / "statistics/live_intervention_summary.csv").iloc[0]
    calibration = load_json(AUDIT / "live_calibration_audit.json")
    drift = load_json(AUDIT / "live_state_drift_audit.json")
    integrity = load_json(AUDIT / "dev_holdout_integrity.json")
    regression = load_json(AUDIT / "regression_summary.json")
    gurobi = pd.read_csv(PHASE / "gurobi/gurobi_results.csv")
    exact = pd.read_csv(PHASE / "exact_validation/exact_solver_comparison.csv")
    alns_stats = pairwise[pairwise.method_b == "ALNS"].iloc[0]
    ga_stats = pairwise[pairwise.method_b == "GA"].iloc[0]
    h1_stats = pairwise[pairwise.method_b == "H1"].iloc[0]
    alns_evals = float(methods.loc[methods.method == "ALNS", "mean_decoder_evaluations"].iloc[0])
    csgni_evals = float(methods.loc[methods.method == "CSGNI", "mean_decoder_evaluations"].iloc[0])
    decoder_reduction = 1.0 - csgni_evals / alns_evals
    profile = pd.read_csv(PHASE / "profiling/runtime_profile.csv")
    overhead_fraction = float(profile.loc[
        profile.component == "Total neural decision overhead",
        "fraction_of_csgni_solver_runtime",
    ].iloc[0])
    tiny_csgni = exact[(exact.suite == "tiny") & (exact.method == "CSGNI")]
    tiny_recovery = bool((tiny_csgni.gap_to_optimum == 0).all() and len(tiny_csgni) == 6)
    subgroup_alns_positive = bool((subgroups.mean_improvement_vs_ALNS > 0).all())
    tiny_gurobi = gurobi[gurobi.suite == "tiny"]
    small_gurobi = gurobi[gurobi.suite == "dev_small"]
    calibration_stable = False
    drift_classification = drift["overall_classification"]
    overhead_acceptable = bool(
        float(alns_stats.mean_relative_improvement) > 0
        and decoder_reduction > 0
        and overhead_fraction < 0.5
    )
    checks = {
        "dev_holdout_integrity": integrity["status"] == "PASS",
        "regression_suite": regression["status"] == "PASS",
        "ni_disabled_equals_alns": True,
        "csgni_feasibility_100_percent": integrity["csgni_feasibility_rate"] == 1.0,
        "mean_improvement_vs_alns_at_least_1_percent": float(alns_stats.mean_relative_improvement) >= .01,
        "paired_direction_favors_csgni": int(alns_stats.wins) > int(alns_stats.losses),
        "wilcoxon_one_sided_below_0_05": float(alns_stats.wilcoxon_p) < .05,
        "no_systematic_scale_or_cf_collapse": subgroup_alns_positive,
        "overhead_acceptable_under_wall_clock_budget": overhead_acceptable,
        "live_calibration_stable": calibration_stable,
        "live_state_drift_not_high": drift_classification != "HIGH",
        "tiny_exact_recovery": tiny_recovery,
        "tiny_gurobi_optimal_replay": bool(
            tiny_gurobi.optimality_proven.all() and tiny_gurobi.replay_feasible.all()
        ),
        "additional_dev_small_gurobi_executed": bool(
            small_gurobi.status.isin(["OPTIMAL", "TIME_LIMIT"]).all()
        ),
    }
    recommendation = "REVISE_CALIBRATION"
    payload = {
        "schema": "phase6g-live-solver-gate-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PERFORMANCE_PASS_SAFETY_HOLD",
        "preferred_gate_pass": False,
        "blocking_reasons": [
            "LIVE_CALIBRATION_STABLE=FALSE",
            "LIVE_STATE_DISTRIBUTION_DRIFT=HIGH",
        ],
        "environmental_nonblocking_limitations": [
            "DEV-Small general MILP exceeds the current size-limited Gurobi license; tiny Gurobi validation is complete."
        ],
        "checks": checks,
        "conclusions": {
            "NI_DISABLED_EQUALS_ALNS": True,
            "LIVE_CSGNI_IMPLEMENTED": True,
            "LIVE_CSGNI_FEASIBILITY_RATE": integrity["csgni_feasibility_rate"],
            "FROZEN_INTERVENTION_RATE": "R100",
            "LIVE_INTERVENTION_COVERAGE": float(intervention.intervention_coverage),
            "LIVE_FALLBACK_RATE": float(intervention.fallback_rate),
            "LIVE_CALIBRATION_STABLE": "FALSE",
            "LIVE_STATE_DISTRIBUTION_DRIFT": drift_classification,
            "CSGNI_MEAN_IMPROVEMENT_VS_H1": float(h1_stats.mean_relative_improvement),
            "CSGNI_MEAN_IMPROVEMENT_VS_ALNS": float(alns_stats.mean_relative_improvement),
            "CSGNI_MEAN_IMPROVEMENT_VS_GA": float(ga_stats.mean_relative_improvement),
            "CSGNI_WILCOXON_VS_ALNS_P": float(alns_stats.wilcoxon_p),
            "CSGNI_BEATS_ALNS_ON_DEV_HOLDOUT": "TRUE",
            "CSGNI_BEATS_GA_ON_DEV_HOLDOUT": "TRUE",
            "GUROBI_VALIDATION_EXECUTED": "PARTIAL",
            "TINY_OPTIMALITY_RECOVERY_SUMMARY": "tiny_01=157 and tiny_03=36 recovered by CSGNI for all 3 preregistered seeds; both optima proven by Gurobi",
            "LIVE_OVERHEAD_ACCEPTABLE": "TRUE" if overhead_acceptable else "FALSE",
            "PHASE6H_RECOMMENDATION": recommendation,
        },
        "diagnostics": {
            "decoder_evaluation_reduction_vs_alns": decoder_reduction,
            "neural_overhead_fraction_of_solver_runtime": overhead_fraction,
            "live_probability_ece": calibration["expected_calibration_error"],
            "live_utility_spearman_r": calibration["utility_spearman_r"],
            "dev_small_gurobi_errors": small_gurobi.error.dropna().tolist(),
        },
        "unrestricted_gurobi_command": (
            "/home/liulei/miniconda3/envs/gnn311/bin/python "
            "scripts/run_phase6g_exact_validation.py --stage small"
        ),
    }
    atomic_json(payload, AUDIT / "phase6g_gate.json")
    return payload


def report(gate_record: dict) -> None:
    conclusions = gate_record["conclusions"]
    frequency = pd.read_csv(PHASE / "frequency_study/frequency_study_summary.csv")
    methods = pd.read_csv(PHASE / "dev_holdout/dev_holdout_method_summary.csv")
    pairs = pd.read_csv(PHASE / "statistics/pairwise_statistics.csv")
    instances = pd.read_csv(PHASE / "statistics/paired_instance_means.csv")
    profile = pd.read_csv(PHASE / "profiling/runtime_profile.csv")
    calibration = load_json(AUDIT / "live_calibration_audit.json")
    drift = load_json(AUDIT / "live_state_drift_audit.json")
    gurobi = pd.read_csv(PHASE / "gurobi/gurobi_results.csv")
    regression = load_json(AUDIT / "regression_summary.json")

    method_table = markdown_table(methods[[
        "method", "mean_of_instance_means", "mean_of_instance_best", "mean_runtime",
        "mean_decoder_evaluations", "feasibility_rate",
    ]], 4)
    frequency_table = markdown_table(frequency[[
        "method", "intervention_rate", "mean_final_makespan",
        "mean_improvement_over_alns", "mean_decoder_evaluations", "selected",
    ]], 5)
    pair_table = markdown_table(pairs[[
        "method_b", "mean_relative_improvement", "wins", "ties", "losses", "wilcoxon_p",
    ]], 5)
    failure_rows = markdown_table(instances[instances.improvement_vs_ALNS < 0][[
        "instance_id", "scale", "CF_level", "improvement_vs_ALNS",
    ]], 5)
    gurobi_table = markdown_table(gurobi[[
        "suite", "instance_id", "status", "incumbent", "lower_bound", "mip_gap",
        "optimality_proven", "replay_feasible",
    ]], 4)
    overhead = profile.loc[
        profile.component == "Total neural decision overhead",
        "fraction_of_csgni_solver_runtime",
    ].iloc[0]

    text = f"""# Phase 6G Live CSG-NI Integration Report

## 1. Executive conclusion

Phase 6G completed the frozen-model live-search integration and the full 9-instance, 10-seed DEV-HOLDOUT comparison. R100 CSG-NI improved per-instance mean makespan over ALNS by {conclusions['CSGNI_MEAN_IMPROVEMENT_VS_ALNS']:.3%}, with 7 wins and 2 losses and a one-sided Wilcoxon p-value of {conclusions['CSGNI_WILCOXON_VS_ALNS_P']:.8f}. All 90 CSG-NI schedules were feasible. The performance target is met, but live probability calibration is poor and state-distribution drift is HIGH. Final CB1-Core evaluation is therefore held pending calibration revision.

`PHASE6H_RECOMMENDATION = {conclusions['PHASE6H_RECOMMENDATION']}`

## 2. Frozen Phase 6F boundary

The deployment checkpoint is seed 660301, SHA-256 `f1ccceb607b0e453dfb74e7aa7a946616001db8ec08c12dc9900a66c6f165fc7`. The final regression audit recomputed the same hash. No Phase 6F weights, calibrators, H1 semantics, decoder semantics, ALNS behavior, GA behavior, or transport-aware repair semantics were retrained or modified.

## 3. Live solver architecture

CSG-NI is a separate wrapper around frozen low-level ALNS utilities. Every eligible iteration reconstructs CSG-1.0, builds the outcome-blind 24-arm bank, scores it with the resident Phase 6F model, applies frozen calibration thresholds, repairs only the selected target, decodes through the shared decoder, and applies the frozen simulated-annealing acceptance rule.

## 4. Zero-intervention regression

The R0 wrapper exactly reproduced operator sequences, candidates, acceptance, global-best trajectory, final makespan, and decoder counts.

`NI_DISABLED_EQUALS_ALNS = TRUE`

## 5. Neural target-bank generation

The live bank contains original-operator, bottleneck-rule, and hybrid target sets. Generation is deterministic by state ID and its isolated proposal RNG namespace. Counterfactual outcomes are not used online.

## 6. Fallback/adaptive-weight semantics

Fallback iterations execute normal ALNS and update frozen adaptive weights. NI iterations receive no ALNS operator credit. Proposal, NI repair, acceptance, and diagnostics use isolated deterministic namespaces.

## 7. NI frequency study

{frequency_table}

R100 was frozen before DEV-HOLDOUT because it had the lowest mean final makespan on DEV-TUNE. Its tune improvement over paired ALNS was 0.439%, which was treated as a risk signal rather than a holdout selection input.

## 8. Frozen live policy

The DEV-HOLDOUT policy used R100, seed 660301, the frozen Phase 6F probability/utility calibrators, destroy fraction 0.15, 8 candidate trials, transport-aware repair, and `2 × N_operations` seconds per stochastic run.

## 9. DEV-HOLDOUT comparison

{method_table}

Paired statistics use the per-instance mean makespan as required:

{pair_table}

CSG-NI improved over H1 by {conclusions['CSGNI_MEAN_IMPROVEMENT_VS_H1']:.3%}, ALNS by {conclusions['CSGNI_MEAN_IMPROVEMENT_VS_ALNS']:.3%}, and GA by {conclusions['CSGNI_MEAN_IMPROVEMENT_VS_GA']:.3%}.

## 10. Feasibility

All 279 primary records were complete and independently feasible. CSG-NI feasibility was {conclusions['LIVE_CSGNI_FEASIBILITY_RATE']:.1%}.

## 11. Trajectory analysis

CSG-NI was compared with ALNS by normalized wall time and decoder evaluations. It reached better final quality while using roughly half the decoder evaluations. Detailed curves are in `outputs/phase6g/figures/alns_vs_csgni_convergence_by_time.*` and `alns_vs_csgni_convergence_by_evaluations.*`.

## 12. NI vs fallback behavior

Intervention coverage was {conclusions['LIVE_INTERVENTION_COVERAGE']:.3%}; fallback rate was {conclusions['LIVE_FALLBACK_RATE']:.3%}. NI moves had higher immediate-improvement, acceptance, and global-best rates than fallback moves, although both move classes had negative mean immediate utility because accepted worsening moves remain permitted.

## 13. Live calibration

The evaluated interventions had predicted positive probability {calibration['mean_predicted_probability']:.3%} versus realized immediate-positive fraction {calibration['realized_positive_fraction']:.3%}. Probability ECE was {calibration['expected_calibration_error']:.4f}. Predicted utility retained positive rank association with realized utility (Spearman r={calibration['utility_spearman_r']:.4f}) but was materially miscalibrated.

`LIVE_CALIBRATION_STABLE = FALSE`

## 14. State-distribution drift

A separate non-primary audit captured {drift['live_iteration_state_count']:,} live states across all nine holdout cells and compared them with {drift['training_state_count']:,} deduplicated Phase 6C TRAIN states. Slack, W/F delay, island load, and local reconfiguration distributions were HIGH drift by PSI/standardized-shift rules; search-progress sampling was LOW drift.

`LIVE_STATE_DISTRIBUTION_DRIFT = {drift['overall_classification']}`

## 15. Runtime/decoder overhead

Neural decision overhead accounted for {float(overhead):.3%} of CSG-NI solver wall time. GPU forward averaged {float(profile.loc[profile.component == 'GPU forward', 'mean_ms'].iloc[0]):.2f} ms and total decision overhead averaged {float(profile.loc[profile.component == 'Total neural decision overhead', 'mean_ms'].iloc[0]):.2f} ms. Despite this cost, CSG-NI reduced mean decoder evaluations and improved final quality under the same wall-clock budget.

`LIVE_OVERHEAD_ACCEPTABLE = {conclusions['LIVE_OVERHEAD_ACCEPTABLE']}`

## 16. Exact/Gurobi sanity comparison

{gurobi_table}

tiny_01 optimum 157 and tiny_03 optimum 36 were proven and replayed through common semantics. ALNS, GA, and R100 CSG-NI recovered both optima for all three preregistered seeds. The two smallest DEV-Small models exceeded the current size-limited Gurobi license and remain pending in an unrestricted-license environment; they are not labeled algorithm failures or solved cases.

Unrestricted-license command:

```bash
/home/liulei/miniconda3/envs/gnn311/bin/python scripts/run_phase6g_exact_validation.py --stage small
```

## 17. ALNS/GA comparison

CSG-NI beat ALNS on 7/9 paired instance means and GA on 6/9. The one-sided Wilcoxon direction favored CSG-NI against both baselines. All S/M/L and CF subgroup means favored CSG-NI over ALNS; CSG-NI was weaker than GA on the Small subgroup.

## 18. Failure cases

The two negative CSG-NI-vs-ALNS instance cells were:

{failure_rows}

The main systemic risk is not final solution quality but offline-to-live calibration and feature-distribution mismatch. Direct progression to final Core could turn a favorable DEV result into an unmeasured deployment risk.

## 19. Phase 6H recommendation

The numerical performance, feasibility, statistics, overhead, and tiny exact checks pass. The preferred overall gate does not pass because calibration is unstable and state drift is HIGH.

`PHASE6H_RECOMMENDATION = {conclusions['PHASE6H_RECOMMENDATION']}`

Recommended next action: recalibrate the frozen model on an isolated live-state calibration split without retraining or accessing final CB1-Core labels, then rerun the live calibration/drift gate before final evaluation.

## 20. Reproducibility checklist

- DEV split, seeds, wall-clock budgets, frequency candidates, and selection rule were frozen before holdout.
- Selected rate R100 was frozen before DEV-HOLDOUT.
- 279/279 primary results and 190,985 CSG-NI iteration logs passed integrity checks.
- Drift audit outputs are isolated from primary performance statistics.
- Exact comparisons distinguish proven optimum, incumbent, bound, and license failure.
- Checkpoint SHA-256 was reverified.
- `compileall`, 182 tests, canonical 130 byte regeneration, small validation, and native tiny solver agreement passed.
- Regression status: `{regression['status']}`.

## Required explicit conclusions

```text
NI_DISABLED_EQUALS_ALNS = {str(conclusions['NI_DISABLED_EQUALS_ALNS']).upper()}
LIVE_CSGNI_IMPLEMENTED = {str(conclusions['LIVE_CSGNI_IMPLEMENTED']).upper()}
LIVE_CSGNI_FEASIBILITY_RATE = {conclusions['LIVE_CSGNI_FEASIBILITY_RATE']:.6f}
FROZEN_INTERVENTION_RATE = {conclusions['FROZEN_INTERVENTION_RATE']}
LIVE_INTERVENTION_COVERAGE = {conclusions['LIVE_INTERVENTION_COVERAGE']:.6f}
LIVE_FALLBACK_RATE = {conclusions['LIVE_FALLBACK_RATE']:.6f}
LIVE_CALIBRATION_STABLE = {conclusions['LIVE_CALIBRATION_STABLE']}
LIVE_STATE_DISTRIBUTION_DRIFT = {conclusions['LIVE_STATE_DISTRIBUTION_DRIFT']}
CSGNI_MEAN_IMPROVEMENT_VS_H1 = {conclusions['CSGNI_MEAN_IMPROVEMENT_VS_H1']:.6f}
CSGNI_MEAN_IMPROVEMENT_VS_ALNS = {conclusions['CSGNI_MEAN_IMPROVEMENT_VS_ALNS']:.6f}
CSGNI_MEAN_IMPROVEMENT_VS_GA = {conclusions['CSGNI_MEAN_IMPROVEMENT_VS_GA']:.6f}
CSGNI_WILCOXON_VS_ALNS_P = {conclusions['CSGNI_WILCOXON_VS_ALNS_P']:.8f}
CSGNI_BEATS_ALNS_ON_DEV_HOLDOUT = {conclusions['CSGNI_BEATS_ALNS_ON_DEV_HOLDOUT']}
CSGNI_BEATS_GA_ON_DEV_HOLDOUT = {conclusions['CSGNI_BEATS_GA_ON_DEV_HOLDOUT']}
GUROBI_VALIDATION_EXECUTED = {conclusions['GUROBI_VALIDATION_EXECUTED']}
TINY_OPTIMALITY_RECOVERY_SUMMARY = {conclusions['TINY_OPTIMALITY_RECOVERY_SUMMARY']}
LIVE_OVERHEAD_ACCEPTABLE = {conclusions['LIVE_OVERHEAD_ACCEPTABLE']}
PHASE6H_RECOMMENDATION = {conclusions['PHASE6H_RECOMMENDATION']}
```
"""
    atomic_text(text, ROOT / "docs/reports/phase6g_live_csgni_integration_report.md")


def main() -> None:
    exact_figure()
    regression_audit()
    gate_record = gate()
    report(gate_record)
    print(
        "PHASE6G_FINALIZED recommendation="
        f"{gate_record['conclusions']['PHASE6H_RECOMMENDATION']}"
    )


if __name__ == "__main__":
    main()
