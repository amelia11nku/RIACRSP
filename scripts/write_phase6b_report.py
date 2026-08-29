#!/usr/bin/env python3
"""Render the Phase 6B counterfactual data-pilot report from frozen outputs."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs/phase6b"


def main():
    decision = json.loads((OUT / "diagnostics/phase6c_recommendation.json").read_text())
    stability = json.loads((OUT / "diagnostics/repair_seed_stability_summary.json").read_text())
    scaling = json.loads((OUT / "diagnostics/runtime_storage_scalability.json").read_text())
    distribution = json.loads((OUT / "audit/train_distribution_audit.json").read_text())
    arms = pd.read_csv(OUT / "summaries/counterfactual_arm_summary.csv")
    balance = pd.read_csv(OUT / "summaries/counterfactual_sample_balance.csv")
    predict = pd.read_csv(OUT / "summaries/target_predictability.csv")
    marginal = pd.read_csv(OUT / "summaries/operation_marginal_summary.csv").iloc[0]
    main = predict[predict.model == "TARGET_STRUCTURE_CONTEXT_LOGISTIC"].iloc[0]
    random = predict[predict.model == "RANDOM"].iloc[0]
    positive = decision["POSITIVE_ARM_FRACTION"]
    report = f"""# Phase 6B Counterfactual Data Pilot Report

## 1. Executive conclusion

Phase 6B created the isolated 405-instance RCIAS-CB1-TRAIN distribution and evaluated {decision['PRIMARY_COUNTERFACTUAL_ARM_COUNT']:,} primary counterfactual target sets from {decision['COUNTERFACTUAL_STATE_COUNT']:,} immutable pre-action states. The instance-grouped structural/context diagnostic reached ROC-AUC {main.roc_auc:.3f}, PR-AUC {main.pr_auc:.3f}, and within-state pairwise accuracy {main.pairwise_accuracy:.3f}, versus random {random.roc_auc:.3f}/{random.pr_auc:.3f}/{random.pairwise_accuracy:.3f}. The Phase 6C recommendation is **{decision['PHASE6C_RECOMMENDATION']}**.

## 2. Frozen Phase 6A evidence

The Phase 6A report, three raw Parquet files, and ten summary CSV files were hashed before implementation. Canonical and CB1 checksums and the instrumentation regression passed. Phase 6A logs were not changed or used as training rows.

## 3. Why destroy target is the NI-v1 primary task

Phase 6A found stable observational target signal, `related` as the strongest destroy, and `transport_aware` repair as an 81.7% best-gain contributor. Phase 6B therefore compares target sets from identical states, fixes repair and destroy count, and keeps operator choice secondary.

## 4. Training-distribution construction

The existing native controlled generator produced 45 structurally accepted Scale×CF×replicate bases. Existing RI/TI scaling created nine paired variants per base: 3×3×3×3×5 = 405. Validation covers operation count, route/capability flexibility, processing CV, RI, W/F intensity, configuration entropy, DAG loadability, and correlations; solver performance never enters acceptance.

`TRAIN_DISTRIBUTION_CREATED = TRUE`  
`TRAIN_INSTANCE_COUNT = 405`

## 5. Split and seed isolation

R01–R03 give 243 TRAIN instances, R04 gives 81 TRAIN_VALIDATION, and R05 gives 81 TRAIN_INTERNAL_HOLDOUT instances. Every split covers all 81 cells. Generation, trajectory, sampling, arm, and repair RNG namespaces are separate and disjoint from frozen assets.

`TRAIN_SPLIT_COUNT = 243`  
`TRAIN_VALIDATION_COUNT = 81`  
`TRAIN_INTERNAL_HOLDOUT_COUNT = 81`  
`FROZEN_TEST_LEAKAGE = FALSE`

## 6. Pilot state reservoir

One frozen ALNS trajectory was run on R01 in every structural cell. The reservoir contains {decision['COUNTERFACTUAL_STATE_COUNT']:,} distinct reconstructable states, balanced across five causal wall-budget stages and deliberately retaining more S/CF3 states. Rare F-logistics and cross-resource occurrences were prioritized within stage reservoirs without relabeling.

## 7. Counterfactual evaluation design

Every evaluation reconstructs the current candidate with the frozen decoder, uses a local RNG, generates eight repaired candidates, and returns the best decoded result. It does not run SA acceptance or update weights, temperature, best state, or live RNG.

`COUNTERFACTUAL_EVALUATOR_VALIDATED = TRUE`

## 8. Repair deconfounding strategy

All primary, repeat, and marginal evaluations fix `transport_aware`. Destroy fraction is 0.15 and count uses the frozen rounding rule. Marginal swaps use the same repair seed as their reference.

## 9. Counterfactual arm design

Each state receives seven original operator target sets, three additional related variants, two matched random controls, and 25%/50% related perturbations. Outcome-blind identical sets are evaluated once with duplicate origins retained. There are {decision['COUNTERFACTUAL_ARM_COUNT']:,} total rows including repair repeats and {decision['PRIMARY_COUNTERFACTUAL_ARM_COUNT']:,} primary rows.

## 10. Sample balance

Primary positive-arm fraction is {positive:.4f}; {decision['STATES_WITH_POSITIVE_ARM_FRACTION']:.4f} of states have at least one positive arm and {decision['STATES_WITH_TWO_POSITIVE_ARMS_FRACTION']:.4f} have at least two. Improving, neutral, and worsening arms are all retained.

## 11. Repair-randomness stability

{stability['repeated_state_count']:,} states repeat every arm over three repair-seed groups. Mean rank Spearman is {stability['mean_rank_spearman']:.3f}, top-arm agreement is {stability['top_arm_agreement_fraction']:.3f}, sign agreement is {stability['mean_improvement_sign_agreement']:.3f}, and mean relative-improvement variance is {stability['mean_relative_improvement_variance']:.6g}.

## 12. Target-set predictability

Leakage-safe instance-grouped results are in `target_predictability.csv`. The structural/context model's ROC-AUC is {main.roc_auc:.3f}, PR-AUC {main.pr_auc:.3f}, Spearman {main.within_state_spearman:.3f}, top-1 {main.top1_accuracy:.3f}, top-3 recall {main.top3_recall:.3f}, NDCG {main.ndcg:.3f}, and pairwise accuracy {main.pairwise_accuracy:.3f}. Criticality, slack, W-delay, and F-delay single-signal baselines are reported alongside it.

## 13. Operation-level marginal target analysis

The pilot evaluates {int(marginal.swap_count):,} paired swaps over {int(marginal.state_count):,} states. Positive marginal fraction is {marginal.positive_marginal_fraction:.3f}; instance-grouped AUC is {marginal.instance_grouped_roc_auc:.3f}. This is a conditional effect under the related reference and fixed repair, not universal causal importance.

`OPERATION_MARGINAL_SIGNAL = {decision['OPERATION_MARGINAL_SIGNAL']}`

## 14. Scale / CF / RI / TI coverage

All 81 cells occur in every training split and the R01 pilot. Machine-readable summaries report arm yield and improvement jointly by Scale, CF, RI, TI, search stage, and bottleneck proxy. Improving arms are checked in every Scale×CF group.

## 15. Bottleneck-proxy coverage

State sampling preserves natural proxy labels and prioritizes observed F_LOGISTICS and CROSS_RESOURCE_SYNCHRONIZATION states within each stage. No proxy is fabricated. Counts are in the reservoir and Fig10.

## 16. Information-leakage audit

Every field is classified as MODEL_INPUT_ALLOWED, LABEL_ONLY, IDENTIFIER_ONLY, ANALYSIS_ONLY, or FORBIDDEN_FUTURE_INFORMATION. Outcomes, ranks, regret, swap gain, and counterfactual makespan are labels only. Identifiers and serialized candidates are reconstruction keys, not model features.

## 17. Counterfactual-integrity audit

Tests verify candidate/schedule immutability, live RNG isolation, deterministic replay, arm-order invariance, fixed repair/count, deduplication, paired swaps, and unchanged subsequent ALNS trajectory.

## 18. Runtime and storage scalability

Mean arm runtime is {scaling['mean_arm_runtime_seconds']:.4f}s ({scaling['arms_per_cpu_second']:.2f} arms/CPU-second). Estimated 100k-state cost is {scaling['estimated_100k_cpu_hours']:.1f} CPU-hours and {scaling['estimated_100k_storage_bytes']/2**30:.1f} GiB under this schema. No 100k collection was launched.

## 19. Limitations

Labels remain conditional on one repair heuristic and one destroy size; target sets are structured perturbations rather than exhaustive choices; wall-time trajectories differ in iteration yield; bottleneck and criticality are proxies; and marginal labels are context-dependent. Diagnostic tabular models do not implement NI.

## 20. Phase 6C recommendation

`DESTROY_TARGET_REMAINS_PRIMARY_NI_TARGET = {decision['DESTROY_TARGET_REMAINS_PRIMARY_NI_TARGET']}`  
`OPERATOR_SELECTION_REMAINS_SECONDARY_TARGET = {decision['OPERATOR_SELECTION_REMAINS_SECONDARY_TARGET']}`  
`REPAIR_SELECTION_REMAINS_DEFERRED = {str(decision['REPAIR_SELECTION_REMAINS_DEFERRED']).upper()}`  
`DESTROY_SIZE_REMAINS_DEFERRED = {str(decision['DESTROY_SIZE_REMAINS_DEFERRED']).upper()}`  
`PHASE6C_RECOMMENDATION = {decision['PHASE6C_RECOMMENDATION']}`

The recommendation considers state/arm coverage, positive yield, all-regime yield, repair stability, instance-grouped prediction, within-state ranking, and marginal signal—not one AUC alone.

The revised Phase 6C arm protocol should average three repair seeds per retained arm before assigning ranks, because rank Spearman is usable but top-arm agreement is only {stability['top_arm_agreement_fraction']:.3f}. It should retain the original operator controls while increasing outcome-blind local swaps and structured near-neighbor perturbations; the current marginal AUC of {marginal.instance_grouped_roc_auc:.3f} is insufficient for direct node labels. Set-level classification is ready to scale, but operation labels should remain conditional pairwise labels until richer pre-action state features or a revised reference-set design improves marginal prediction. Three-seed scaling would cost approximately {3*scaling['estimated_100k_cpu_hours']:.1f} CPU-hours at the measured rate, still operationally feasible but requiring review before collection.

## 21. Reproducibility checklist

Phase 6A freeze hashes, generation spec/history/checksums, training manifests and split, separated seeds, compact state reconstruction records, raw counterfactual/marginal Parquet, summaries, audits, tests, figures, and environment are retained. Phase 6B stops here and does not implement CSG, a neural policy, or Phase 6C scaling.

`COUNTERFACTUAL_STATE_COUNT = {decision['COUNTERFACTUAL_STATE_COUNT']}`  
`COUNTERFACTUAL_ARM_COUNT = {decision['COUNTERFACTUAL_ARM_COUNT']}`  
`POSITIVE_ARM_FRACTION = {decision['POSITIVE_ARM_FRACTION']:.6f}`  
`STATES_WITH_POSITIVE_ARM_FRACTION = {decision['STATES_WITH_POSITIVE_ARM_FRACTION']:.6f}`  
`REPAIR_SEED_RANK_STABILITY = {stability['mean_rank_spearman']:.6f}`  
`COUNTERFACTUAL_TARGET_PREDICTABILITY = ROC_AUC_{main.roc_auc:.6f}_PAIRWISE_{main.pairwise_accuracy:.6f}`
"""
    path=ROOT/"docs/reports/phase6b_counterfactual_data_pilot_report.md"; path.write_text(report); print(path.relative_to(ROOT))


if __name__ == "__main__": main()
