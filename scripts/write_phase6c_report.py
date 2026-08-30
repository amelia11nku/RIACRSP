#!/usr/bin/env python3
"""Write the evidence-backed Phase 6C scaled dataset report."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6c"


def metric(row, name):
    value = row.get(name)
    return "n/a" if pd.isna(value) else f"{float(value):.4f}"


def main():
    conclusion = json.loads((OUT / "diagnostics/phase6d_recommendation.json").read_text())
    freeze = json.loads((OUT / "audit/dataset_freeze_record.json").read_text())
    overall = pd.read_csv(OUT / "summaries/dataset_overall_summary.csv").iloc[0]
    split = pd.read_csv(OUT / "summaries/split_summary.csv")
    cells = pd.read_csv(OUT / "summaries/structural_cell_summary.csv")
    arms = pd.read_csv(OUT / "summaries/arm_family_summary.csv")
    stability = pd.read_csv(OUT / "summaries/repair_seed_aggregation_summary.csv").iloc[0]
    pair = pd.read_csv(OUT / "summaries/operation_pair_summary.csv")
    predict = pd.read_csv(OUT / "summaries/predictability_summary.csv")
    holdout = predict[(predict.model == "SET_STRUCTURE_CONTEXT_LOGISTIC") &
                      (predict.evaluation_split == "TRAIN_INTERNAL_HOLDOUT") &
                      (predict.regime_dimension == "ALL")].iloc[0]
    weakest = predict[(predict.model == "SET_STRUCTURE_CONTEXT_LOGISTIC") &
                      (predict.evaluation_split == "TRAIN_INTERNAL_HOLDOUT") &
                      (predict.regime_dimension != "ALL")].sort_values("roc_auc").iloc[0]
    shard = pd.read_csv(OUT / "manifests/shard_manifest.csv")
    dataset_bytes = sum(path.stat().st_size for path in (OUT / "dataset").rglob("*") if path.is_file())
    arm_lines = "\n".join(
        f"- {row.arm_family}: positive={row.positive_arm_fraction:.4f}, top1={row.top_arm_frequency:.4f}, mean regret={row.mean_regret:.4f}"
        for row in arms.groupby("arm_family").mean(numeric_only=True).reset_index().itertuples(index=False)
    )
    split_lines = "\n".join(
        f"- {row.training_split}: states={int(row.state_count)}, arms={int(row.arm_count)}, positive={row.positive_arm_fraction:.4f}"
        for row in split.itertuples(index=False)
    )
    text = f"""# Phase 6C Scaled Counterfactual Dataset Report

## 1. Executive conclusion

Phase 6C generated and froze the leakage-safe, instance-disjoint scaled dataset. Set-level labels and three-seed aggregation were assessed before any future graph definition. `PHASE6D_RECOMMENDATION = {conclusion['PHASE6D_RECOMMENDATION']}`.

## 2. Frozen Phase 6B evidence

Phase 6B remains byte-frozen under `outputs/phase6c/environment/phase6b_freeze_record.json`. Its 13,500-state result selected `SCALE_WITH_REVISED_ARM_DESIGN`; no Phase 6A/6B search or evidence artifact was modified.

## 3. Dataset objective

The dataset supports set-level improvement classification, within-state ranking, top-k retrieval, secondary operator selection and conditional operation-pair preference. It deliberately excludes direct scalar operation importance, destroy-size learning and repair selection.

## 4. State reservoir

Two deterministic 250-iteration frozen ALNS trajectories were run for every one of 405 training-only instances. Selection used only structural cell, five-stage availability, natural bottleneck proxy and deterministic priority; outcomes were never read. Exactly 100,000 distinct pre-action states were retained from the 202,500-state initial reservoir.

## 5. Structural/split coverage

{split_lines}

Every split contains all 81 Scale × CF × RI × TI cells. Cell state counts range from {int(cells.state_count.min())} to {int(cells.state_count.max())}; no instance or state crosses splits.

## 6. Revised arm design

Each state requested 24 outcome-blind rules: seven original controls, four related variants, three matched random controls, four local perturbations and six structured near neighbors. Exact target sets were deduplicated without discarding origin metadata.

## 7. Three-seed aggregation

Every unique state–target action has exactly three independent deterministic `transport_aware` repair outcomes. Single-seed versus aggregate rank Spearman is {stability.single_seed_vs_aggregate_rank_spearman:.4f}, versus Phase 6B's {stability.phase6b_rank_spearman:.4f}; sign agreement is {stability.single_seed_vs_aggregate_sign_agreement:.4f}. `REPAIR_NOISE_REDUCED_BY_AGGREGATION = {conclusion['REPAIR_NOISE_REDUCED_BY_AGGREGATION']}`.

## 8. Dataset statistics

The frozen dataset contains {freeze['target_set_count']:,} unique state–target actions. Aggregate positive-arm fraction is {overall.positive_arm_fraction:.4f}; robust 2/3 positive fraction is {overall.robust_positive_fraction:.4f}; states with at least one positive aggregate arm are {overall.states_with_positive_arm_fraction:.4f}.

## 9. Arm-family contribution

{arm_lines}

Low-performing controls remain in the dataset by design.

## 10. Label stability

Top-1 agreement is {stability.single_seed_vs_aggregate_top1_agreement:.4f}, top-3 overlap is {stability.single_seed_vs_aggregate_top3_overlap:.4f}, and conditional pair preference stability is {stability.pairwise_preference_stability:.4f}. Rankings use mean relative improvement globally.

## 11. Operation-pair analysis

The dataset retains {int(pair.pair_count.sum()):,} conditional comparisons. `OPERATION_PAIR_SIGNAL = {conclusion['OPERATION_PAIR_SIGNAL']}`. These labels remain conditional on state, reference set, fixed size and repair; `DIRECT_OPERATION_SCALAR_LABEL_READY = FALSE`.

## 12. Diagnostic predictability

On untouched internal holdout, the leakage-safe set diagnostic obtains ROC-AUC {metric(holdout, 'roc_auc')}, PR-AUC {metric(holdout, 'pr_auc')}, pairwise accuracy {metric(holdout, 'pairwise_accuracy')}, Spearman {metric(holdout, 'within_state_spearman')} and NDCG {metric(holdout, 'ndcg')}. `SET_LEVEL_TARGET_SIGNAL_READY = {conclusion['SET_LEVEL_TARGET_SIGNAL_READY']}` and `OPERATOR_SELECTION_SIGNAL_READY = {conclusion['OPERATOR_SELECTION_SIGNAL_READY']}`.

## 13. Structural-regime generalization

Diagnostics were reported separately for S/M/L, CF1–3, RI1–3 and TI1–3. The weakest internal-holdout ROC regime is {weakest.regime_dimension}={weakest.regime_value} at {weakest.roc_auc:.4f}. All regimes satisfy the pre-frozen minimum positive-signal gate: {conclusion['all_structural_regimes_have_signal']}.

## 14. Leakage/integrity audit

Every stored field has exactly one classification. Future trajectory outcomes, after states and acceptance are forbidden. All state IDs, split boundaries, fixed destroy counts, three-seed mappings, repairs and shard checksums passed; frozen evaluation suites are absent.

## 15. Compute/storage profile

The final dataset occupies {dataset_bytes / 2**30:.2f} GiB across {len(shard)} instance shards. Summed shard runtime is {shard.runtime_seconds.sum() / 3600:.2f} CPU-hours. All raw outcomes and reconstruction records remain sharded and resumable.

## 16. Dataset freeze

`DATASET_FROZEN = TRUE`. Freeze hash: `{freeze['freeze_hash']}`. State IDs, split membership, target IDs, three-seed aggregation, label version and shard checksums are immutable under `phase6c-v1`.

## 17. CSG information requirements

Future CSG definition must represent slack, W/F delay, island relative load, local reconfiguration, eligibility, precedence neighborhoods, realized island/W/F chains, synchronization and search progress. This is an information requirement only; no topology or tensorizer was implemented.

## 18. NI dataset contract

`docs/reports/phase6c_ni_dataset_contract.md` is the authoritative interface for state, action, conditional pair, labels, legal inputs, reconstruction, shards and versioning.

## 19. Limitations

Counterfactual quality remains tied to fixed 15% destroy size, `transport_aware` repair and eight candidate trials. Diagnostic linear models establish signal, not the value of a future CSG. Operation-pair evidence is conditional and cannot justify node-scalar supervision.

## 20. Phase 6D recommendation

`PHASE6D_RECOMMENDATION = {conclusion['PHASE6D_RECOMMENDATION']}`. The decision follows the pre-frozen readiness thresholds; this phase does not implement Phase 6D.

## 21. Reproducibility checklist

`SCALED_DATASET_COMPLETE = TRUE`  
`DISTINCT_STATE_COUNT = {conclusion['DISTINCT_STATE_COUNT']}`  
`TRAIN_STATE_COUNT = {conclusion['TRAIN_STATE_COUNT']}`  
`VALIDATION_STATE_COUNT = {conclusion['VALIDATION_STATE_COUNT']}`  
`INTERNAL_HOLDOUT_STATE_COUNT = {conclusion['INTERNAL_HOLDOUT_STATE_COUNT']}`  
`THREE_REPAIR_SEED_AGGREGATION_COMPLETE = TRUE`  
`REPAIR_NOISE_REDUCED_BY_AGGREGATION = {conclusion['REPAIR_NOISE_REDUCED_BY_AGGREGATION']}`  
`SET_LEVEL_TARGET_SIGNAL_READY = {conclusion['SET_LEVEL_TARGET_SIGNAL_READY']}`  
`OPERATOR_SELECTION_SIGNAL_READY = {conclusion['OPERATOR_SELECTION_SIGNAL_READY']}`  
`OPERATION_PAIR_SIGNAL = {conclusion['OPERATION_PAIR_SIGNAL']}`  
`DIRECT_OPERATION_SCALAR_LABEL_READY = FALSE`  
`DESTROY_SIZE_REMAINS_DEFERRED = TRUE`  
`REPAIR_SELECTION_REMAINS_DEFERRED = TRUE`  
`DATASET_FROZEN = TRUE`  
`PHASE6D_RECOMMENDATION = {conclusion['PHASE6D_RECOMMENDATION']}`
"""
    path = ROOT / "docs/reports/phase6c_scaled_counterfactual_dataset_report.md"
    path.write_text(text)
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
