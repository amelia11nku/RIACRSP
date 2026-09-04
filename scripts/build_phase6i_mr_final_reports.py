#!/usr/bin/env python3
"""Build the terminal Phase 6I-MR report and project handoff from frozen R11 evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
R11 = ROOT / "outputs/phase6i_mr/r11_validation"
FINAL = ROOT / "outputs/phase6i_mr/final"
REPORT = ROOT / "docs/reports/phase6i_mr_live_utility_revision_report.md"
HANDOFF = ROOT / "docs/reports/phase6i_mr_project_handoff.md"
TARGETS = (("0p05", "5%"), ("0p02", "2%"), ("0p01", "1%"), ("0p005", "0.5%"))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def outcome(values: pd.Series) -> dict[str, int]:
    return {
        "wins": int((values > 0).sum()),
        "ties": int((values == 0).sum()),
        "losses": int((values < 0).sum()),
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def main() -> int:
    decision = load_json(R11 / "final_decision.json")
    ranking = load_json(R11 / "r11_ranking_calibration.json")
    runtime = load_json(R11 / "r11_anytime_runtime.json")
    integrity = load_json(R11 / "completion_integrity_audit.json")
    compatibility = load_json(R11 / "summary_compatibility_audit.json")
    artifact = load_json(ROOT / "outputs/phase6i_mr/frozen/selected_artifact.json")
    freeze = load_json(ROOT / "outputs/phase6i_mr/frozen/artifact_freeze.json")
    split_audit = load_json(
        ROOT / "instances/controlled/RCIAS-CB1-TRAIN-LIVE-R09R11/manifests/phase6i_integrity_audit.json"
    )
    selection = load_json(ROOT / "outputs/phase6i_mr/r10_selection/selection_decision.json")
    translation = load_json(ROOT / "outputs/phase6i_mr/r10_translation/translation_decision.json")
    continuation = load_json(ROOT / "outputs/phase6i_mr/continuation/continuation_branch_decision.json")
    full_bank = load_json(ROOT / "outputs/phase6i_mr/pre_r10/r09_full_bank_audit.json")
    if not (
        decision.get("decision") == "MODEL_REVISION"
        and integrity.get("status") == "PASS"
        and compatibility.get("status") == "PASS_REPAIRED_DERIVED_SUMMARY"
        and split_audit.get("status") == "PASS"
    ):
        raise RuntimeError("terminal Phase 6I-MR evidence gates are not ready")

    instances = pd.read_csv(R11 / "instance_method_summary.csv")
    pivot = instances.pivot(index="instance_id", columns="method", values="mean_final")
    pairwise = pd.DataFrame({
        "phase6i_improvement_vs_alns": (pivot.ALNS - pivot.PHASE6I_MR_CSGNI) / pivot.ALNS,
        "phase6h_improvement_vs_alns": (pivot.ALNS - pivot.PHASE6H_CSGNI) / pivot.ALNS,
        "phase6i_improvement_vs_phase6h": (
            pivot.PHASE6H_CSGNI - pivot.PHASE6I_MR_CSGNI
        ) / pivot.PHASE6H_CSGNI,
    }).reset_index()
    metadata = instances.drop_duplicates("instance_id")[["instance_id", "scale", "CF_level"]]
    pairwise = pairwise.merge(metadata, on="instance_id", how="left", validate="one_to_one")
    atomic_csv(FINAL / "r11_pairwise_quality.csv", pairwise)

    anytime = pd.read_csv(R11 / "anytime_target_metrics.csv")
    target_rows = []
    for method, group in anytime.groupby("method", sort=True):
        for suffix, label in TARGETS:
            reached = group[f"target_{suffix}_reached"].astype(bool)
            times = group.loc[reached, f"target_{suffix}_time"].dropna()
            evaluations = group.loc[reached, f"target_{suffix}_evals"].dropna()
            target_rows.append({
                "method": method,
                "target_gap": label,
                "runs": len(group),
                "hits": int(reached.sum()),
                "hit_rate": float(reached.mean()),
                "median_conditional_time_seconds": float(times.median()) if len(times) else None,
                "median_conditional_decoder_evaluations": float(evaluations.median()) if len(evaluations) else None,
                "right_censored_runs": int((~reached).sum()),
            })
    targets = pd.DataFrame(target_rows)
    atomic_csv(FINAL / "r11_target_summary.csv", targets)

    method_summary = pd.DataFrame(runtime["method_summary"])[[
        "method",
        "mean_of_instance_final",
        "mean_runtime",
        "mean_decoder_evals",
        "median_time_to_best",
        "median_evals_to_best",
    ]]
    atomic_csv(FINAL / "r11_method_summary.csv", method_summary)
    revised = ranking["revised_ranking"]
    scale_support = {row["scale"]: row for row in ranking["scale_support"]}
    phase6i_vs_alns = outcome(pairwise.phase6i_improvement_vs_alns)
    phase6i_vs_phase6h = outcome(pairwise.phase6i_improvement_vs_phase6h)
    status = {
        "phase_decision": "MODEL_REVISION",
        "v1_frozen": False,
        "feasibility_rate": 1.0,
        "r11_instance_count": 18,
        "r11_iterative_seeds_per_method": 5,
        "r11_runs": 288,
        "utility_ranking": {
            "spearman_overall": revised["overall_per_instance_spearman"],
            "spearman_by_scale": {
                scale: revised[f"scale_{scale}_per_instance_spearman"] for scale in ("S", "M", "L")
            },
            "kendall_overall": revised["overall_per_instance_kendall"],
            "ndcg_at_1": revised["overall_per_instance_ndcg_at_1"],
            "ndcg_at_2": revised["overall_per_instance_ndcg_at_2"],
            "pairwise_accuracy": revised["overall_per_instance_pairwise_accuracy"],
            "top1_agreement": revised["overall_per_instance_top1_agreement"],
        },
        "intervention_reliability": {
            "selected_actions": ranking["selected_interventions"],
            "selected_immediate_utility": ranking["selected_immediate_utility"],
            "selected_lift_over_fallback": ranking["selected_lift_over_fallback"],
            "coverage_by_scale": {scale: scale_support[scale]["coverage"] for scale in ("S", "M", "L")},
            "selected_actions_by_scale": {
                scale: scale_support[scale]["selected_action_count"] for scale in ("S", "M", "L")
            },
            "fallback_coverage_by_scale": {
                scale: 1.0 - scale_support[scale]["coverage"] for scale in ("S", "M", "L")
            },
        },
        "probability": ranking["probability_metrics"],
        "solver_quality": {
            "improvement_vs_alns": runtime["aggregate_revised_improvement_vs_alns"],
            "wins_ties_losses_vs_alns": phase6i_vs_alns,
            "wins_ties_losses_vs_phase6h": phase6i_vs_phase6h,
            "bootstrap_95_percent": runtime["paired_instance_bootstrap_95_percent"],
            "wilcoxon": runtime["wilcoxon"],
            "relative_worse_than_alns_by_scale": runtime["scale_relative_worse_than_alns"],
        },
        "search_efficiency": {
            "decoder_reduction_vs_alns": runtime["decoder_evaluation_reduction_vs_alns"],
            "normalized_gap_auc_improvement_vs_alns": runtime["normalized_gap_auc_improvement_vs_alns"],
            "method_summary": runtime["method_summary"],
            "target_summary": target_rows,
            "runtime_components": runtime["runtime_components"],
        },
        "chosen_model": {
            "family": artifact["model_family"],
            "candidate_id": artifact["candidate_id"],
            "ensemble_rule": artifact["ensemble_rule"],
            "training_seeds": [item["training_seed"] for item in artifact["model_artifacts"]],
            "r10_mean_training_seed_spearman": selection["selected"]["mean_seed_spearman"],
            "r10_worst_training_seed_spearman": selection["selected"]["worst_seed_spearman"],
            "r10_training_seed_sd": selection["selected"]["seed_standard_deviation"],
        },
        "artifact_hashes": {
            "selected_artifact": digest(ROOT / "outputs/phase6i_mr/frozen/selected_artifact.json"),
            "artifact_freeze": digest(ROOT / "outputs/phase6i_mr/frozen/artifact_freeze.json"),
            "r11_result_manifest": digest(R11 / "r11_result_hash_manifest.csv"),
            "r11_forced_diagnostics": digest(R11 / "forced_diagnostics.parquet"),
            "final_decision": digest(R11 / "final_decision.json"),
        },
        "holdout_access_integrity": {
            "split_audit": split_audit["status"],
            "completion_audit": integrity["status"],
            "r11_accessed_once": decision["r11_accessed_once"],
            "no_r11_retuning": decision["no_r11_retuning"],
        },
        "gurobi_executed": False,
    }
    atomic_text(FINAL / "final_status.json", json.dumps(status, indent=2, sort_keys=True, allow_nan=False) + "\n")

    failed = [name for name, passed in decision["checks"].items() if not passed]
    target_markdown = targets.copy()
    target_markdown["hit_rate"] = target_markdown.hit_rate.map(lambda value: f"{100 * value:.1f}%")
    target_markdown["median_conditional_time_seconds"] = target_markdown.median_conditional_time_seconds.map(
        lambda value: "--" if pd.isna(value) else f"{value:.1f}"
    )
    target_markdown["median_conditional_decoder_evaluations"] = target_markdown.median_conditional_decoder_evaluations.map(
        lambda value: "--" if pd.isna(value) else f"{value:.0f}"
    )
    target_table = markdown_table(target_markdown)
    method_markdown = method_summary.copy()
    for column in method_markdown.columns:
        if column != "method":
            method_markdown[column] = method_markdown[column].map(lambda value: f"{float(value):.3f}")
    method_table = markdown_table(method_markdown)
    status_block = json.dumps(status, indent=2, sort_keys=True, allow_nan=False)
    report = f"""```json
{status_block}
```

# Phase 6I-MR live utility revision report

## 1. Final decision

The immutable R11 gate returns **`MODEL_REVISION`**. Correctness and solver
non-inferiority remain intact, but the selected-action utility and support-aware
coverage gates fail. The selected artifact is therefore not frozen as CSG-NI
v1, and Phase 6I-MR does not authorize Core/Sensitivity/Legacy replacement or
the v2 architecture backlog.

Failed mandatory checks: `{', '.join(failed)}`.

## 2. Data and leakage boundary

R09, R10 and R11 contain 18 disjoint instances each, with two instances in
every Scale × CF cell. The split audit reports unique hashes, no historical
ID/hash/seed overlap, R11 content withheld before artifact freeze, and no R10
refit. R11 was opened once after freezing artifact `{status['artifact_hashes']['selected_artifact']}`.
The final matrix contains 18 H1 runs and 90 runs for each iterative method.
All 288 action sequences replay to their recorded makespans, all schedules are
feasible, all traces are monotone, and all individual live/forced-log hashes
match.

## 3. Why Phase 6H failed

The R09 pilot localized the dominant problems to gate-selection bias (40/54,
74.1%), within-state inversion (37/54, 68.5%), candidate-source bias (23/54,
42.6%), and cross-state miscalibration/sign error (22/54, 40.7%). Inversions
were present at every scale and increased from 61.1% early to 72.2% in middle
and late search. This is not explained by low-support extrapolation alone.

The continuation diagnostic also found a target mismatch: immediate utility
and 12-iteration continuation value had median within-state Spearman
{continuation['median_within_state_spearman']:.3f} and top-1 agreement
{100 * continuation['top1_agreement']:.2f}%, activating `TARGET_MISMATCH`.

## 4. Candidate truncation and representation

The audited bank averaged {full_bank['mean_bank_size']:.2f} unique targets.
The true full-bank best was absent from broad-4 in
{100 * full_bank['true_best_absent_from_broad_four_rate']:.1f}% of states and
from top-8 in {100 * full_bank['true_best_absent_from_top_eight_rate']:.1f}%.
Thus broad forced labeling has material truncation bias, but the frozen protocol
did not permit candidate-bank redesign. The frozen-embedding probe achieved
only 0.549 pairwise accuracy; U3 was activated under the preregistered underfit
rule, but R10 ultimately selected U2 Mixed Old/New.

## 5. Selected model and training-seed stability

R10 selected `{artifact['candidate_id']}`: a three-seed arithmetic ensemble
with training seeds `{[item['training_seed'] for item in artifact['model_artifacts']]}`.
Its seed-level mean/worst Spearman was
{selection['selected']['mean_seed_spearman']:.3f}/
{selection['selected']['worst_seed_spearman']:.3f}, SD
{selection['selected']['seed_standard_deviation']:.4f}. The one-time R10 solver
translation remained within the preregistered non-inferiority envelope
({pct(translation['aggregate_revised_relative_to_alns'])} worse than ALNS), but
did not show directional improvement.

## 6. R11 ranking and calibration

R11 utility Spearman is {revised['overall_per_instance_spearman']:.3f} overall
(S {revised['scale_S_per_instance_spearman']:.3f}, M
{revised['scale_M_per_instance_spearman']:.3f}, L
{revised['scale_L_per_instance_spearman']:.3f}); Kendall is
{revised['overall_per_instance_kendall']:.3f}, NDCG@1/2 is
{revised['overall_per_instance_ndcg_at_1']:.3f}/
{revised['overall_per_instance_ndcg_at_2']:.3f}. Pairwise accuracy and NDCG@1
improve over U0 by {ranking['revised_minus_u0_pairwise_accuracy']:.3f} and
{ranking['revised_minus_u0_ndcg_at_1']:.3f}, respectively. Ranking therefore
improves modestly and stays non-negative by scale, but remains below the
preferred 0.20 Spearman target.

Probability reliability passes: ECE
{ranking['probability_metrics']['expected_calibration_error']:.3f}, Brier
{ranking['probability_metrics']['brier_score']:.3f}, and NLL
{ranking['probability_metrics']['negative_log_likelihood']:.3f}.

## 7. Cross-state intervention failure

Only {ranking['selected_interventions']}/900 forced diagnostic states pass the
frozen gate: coverage is S {100 * scale_support['S']['coverage']:.2f}%, M
{100 * scale_support['M']['coverage']:.2f}%, and L
{100 * scale_support['L']['coverage']:.2f}%. Their realized immediate utility is
{pct(ranking['selected_immediate_utility'])}, with lift over fallback
{pct(ranking['selected_lift_over_fallback'])}. Both are negative. Moreover,
predicted-abstained states still contain mean best-forced lift of 3.02–3.64%,
and their grouped-bootstrap upper bounds are about 4.1–4.5%, so low coverage
cannot be justified as safe abstention. This is the decisive Phase 6I-MR
failure: weak cross-state gating, not feasibility.

## 8. Final solver quality

Phase 6I-MR is {pct(-runtime['aggregate_revised_improvement_vs_alns'])} worse
than ALNS on the mean of 18 paired instance means, while remaining inside the
+1% non-inferiority margin. It records
{phase6i_vs_alns['wins']}/{phase6i_vs_alns['ties']}/{phase6i_vs_alns['losses']}
wins/ties/losses. The paired bootstrap interval for improvement is
[{pct(runtime['paired_instance_bootstrap_95_percent']['low'])},
{pct(runtime['paired_instance_bootstrap_95_percent']['high'])}], and the exact
one-sided Wilcoxon statistic is {runtime['wilcoxon']['statistic']:.0f}
(`p={runtime['wilcoxon']['p_value']:.6f}`). Small instances are
{pct(runtime['scale_relative_worse_than_alns']['S'])} worse than ALNS on average;
there is no catastrophic subgroup collapse under the frozen numerical rule.

Relative to the Phase 6H reference, Phase 6I-MR records
{phase6i_vs_phase6h['wins']}/{phase6i_vs_phase6h['ties']}/{phase6i_vs_phase6h['losses']}
wins/ties/losses across the 18 instance means. Ranking improvements therefore
did not translate into better final search behavior.

## 9. Search efficiency and runtime fairness

Decoder evaluations fall by {100 * runtime['decoder_evaluation_reduction_vs_alns']:.2f}%
versus ALNS, satisfying the disjunctive efficiency gate without a final-quality
collapse. However, normalized-gap AUC is
{100 * abs(runtime['normalized_gap_auc_improvement_vs_alns']):.2f}% worse, and
median time to final best is 125.55 s versus 108.58 s for ALNS and 101.95 s for
Phase 6H. Raw wall-clock results apply only to the recorded RTX 4060 Ti / i7-14700
single-worker platform; checkpoint loading is reported separately.

{method_table}

## 10. Time-to-target evidence

Hit times and decoder counts are conditional on reaching the target; misses are
right-censored and retained in the hit-rate column.

{target_table}

## 11. Evidence boundary and next phase

No Gurobi work was run. R11 may not be revisited for tuning, threshold changes,
model selection, or rescue analysis. The failed artifact remains a frozen
experimental candidate, not CSG-NI v1. Any further model revision requires a
new preregistered development/selection/holdout boundary. Until a future fresh
holdout yields `PROCEED_FREEZE_V1`, the v2 mechanisms in the Phase 6I-MR backlog
remain design-only and the manuscript's Phase 6H CSG-NI results remain the
operative Core45 evidence.

## 12. Reproducibility artifacts

- Completion audit: `outputs/phase6i_mr/r11_validation/completion_integrity_audit.json`
- Summary compatibility audit: `outputs/phase6i_mr/r11_validation/summary_compatibility_audit.json`
- Gate decision: `outputs/phase6i_mr/r11_validation/final_decision.json`
- Ranking/calibration: `outputs/phase6i_mr/r11_validation/r11_ranking_calibration.json`
- Anytime/runtime: `outputs/phase6i_mr/r11_validation/r11_anytime_runtime.json`
- Pairwise quality: `outputs/phase6i_mr/final/r11_pairwise_quality.csv`
- Target summary: `outputs/phase6i_mr/final/r11_target_summary.csv`
- Machine-readable status: `outputs/phase6i_mr/final/final_status.json`
"""
    atomic_text(REPORT, report)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    handoff = f"""# Phase 6I-MR project handoff

## Current gate

- Decision: **`MODEL_REVISION`**
- CSG-NI v1 frozen: **false**
- R11: 288/288 complete; completion/replay audit PASS
- Frozen candidate: `{artifact['candidate_id']}` three-seed ensemble
- Source commit at handoff: `{commit}`
- Gurobi executed: false

## Why the phase stops here

The solver remains feasible and within the frozen +1% ALNS non-inferiority
margin, and decoder evaluations are 51.59% lower. Nevertheless, only 9/900
diagnostic states intervene, selected realized utility and fallback-relative
lift are negative, and forced abstention evidence shows useful actions were
missed. These failures trigger `MODEL_REVISION`; R11 retuning is forbidden.

## Authoritative evidence

- `docs/reports/phase6i_mr_live_utility_revision_report.md`
- `outputs/phase6i_mr/final/final_status.json`
- `outputs/phase6i_mr/r11_validation/final_decision.json`
- `outputs/phase6i_mr/r11_validation/promotion_gate_table.csv`
- `outputs/phase6i_mr/r11_validation/completion_integrity_audit.json`
- `outputs/phase6i_mr/r11_validation/r11_result_hash_manifest.csv`
- `outputs/phase6i_mr/r11_validation/r11_ranking_calibration.json`
- `outputs/phase6i_mr/r11_validation/r11_anytime_runtime.json`

## Frozen boundaries

- Do not alter or rerun the existing R11 results.
- Do not retune thresholds, normalization, calibration, losses, or model choice
  using R11.
- Do not label the current artifact CSG-NI v1.
- Do not replace the manuscript Phase6H Core45 column with Phase6I-MR.
- Do not open Core/Sensitivity/Legacy for this failed promotion attempt.
- Do not begin the Phase6I-MR v2 backlog; it requires a prior
  `PROCEED_FREEZE_V1` decision.

## Corrected post-processing boundary

The frozen runner omitted `decoder_seconds` only from its derived
`validation_run_summary.csv`; all 288 immutable run JSONs contained the value.
`repair_phase6i_mr_r11_summary.py` restored that projection, recorded before
and after hashes, and did not alter scientific result payloads. The independent
completion audit then replayed every stored action sequence and validated every
trace.

## Next authorized work

The only scientifically valid continuation is to design a new, explicitly
versioned utility/gating revision with fresh selection and holdout splits. It
should address the immediate-versus-continuation target mismatch, candidate
label truncation, and support-aware cross-state abstention before any new long
experiment. That design must be frozen before accessing new selection data.
"""
    atomic_text(HANDOFF, handoff)
    print(json.dumps({
        "status": "PASS_REPORTS_BUILT",
        "decision": decision["decision"],
        "report": str(REPORT.relative_to(ROOT)),
        "handoff": str(HANDOFF.relative_to(ROOT)),
        "final_status": str((FINAL / "final_status.json").relative_to(ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
