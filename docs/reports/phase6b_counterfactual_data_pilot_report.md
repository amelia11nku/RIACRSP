# Phase 6B Counterfactual Data Pilot Report

## 1. Executive conclusion

Phase 6B created the isolated 405-instance RCIAS-CB1-TRAIN distribution and evaluated 185,643 primary counterfactual target sets from 13,500 immutable pre-action states. The instance-grouped structural/context diagnostic reached ROC-AUC 0.900, PR-AUC 0.428, and within-state pairwise accuracy 0.572, versus random 0.500/0.063/0.500. The Phase 6C recommendation is **SCALE_WITH_REVISED_ARM_DESIGN**.

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

One frozen ALNS trajectory was run on R01 in every structural cell. The reservoir contains 13,500 distinct reconstructable states, balanced across five causal wall-budget stages and deliberately retaining more S/CF3 states. Rare F-logistics and cross-resource occurrences were prioritized within stage reservoirs without relabeling.

## 7. Counterfactual evaluation design

Every evaluation reconstructs the current candidate with the frozen decoder, uses a local RNG, generates eight repaired candidates, and returns the best decoded result. It does not run SA acceptance or update weights, temperature, best state, or live RNG.

`COUNTERFACTUAL_EVALUATOR_VALIDATED = TRUE`

## 8. Repair deconfounding strategy

All primary, repeat, and marginal evaluations fix `transport_aware`. Destroy fraction is 0.15 and count uses the frozen rounding rule. Marginal swaps use the same repair seed as their reference.

## 9. Counterfactual arm design

Each state receives seven original operator target sets, three additional related variants, two matched random controls, and 25%/50% related perturbations. Outcome-blind identical sets are evaluated once with duplicate origins retained. There are 224,021 total rows including repair repeats and 185,643 primary rows.

## 10. Sample balance

Primary positive-arm fraction is 0.0631; 0.2396 of states have at least one positive arm and 0.1404 have at least two. Improving, neutral, and worsening arms are all retained.

## 11. Repair-randomness stability

1,397 states repeat every arm over three repair-seed groups. Mean rank Spearman is 0.644, top-arm agreement is 0.174, sign agreement is 0.932, and mean relative-improvement variance is 0.000608082.

## 12. Target-set predictability

Leakage-safe instance-grouped results are in `target_predictability.csv`. The structural/context model's ROC-AUC is 0.900, PR-AUC 0.428, Spearman 0.199, top-1 0.141, top-3 recall 0.317, NDCG 0.878, and pairwise accuracy 0.572. Criticality, slack, W-delay, and F-delay single-signal baselines are reported alongside it.

## 13. Operation-level marginal target analysis

The pilot evaluates 7,290 paired swaps over 1,215 states. Positive marginal fraction is 0.461; instance-grouped AUC is 0.544. This is a conditional effect under the related reference and fixed repair, not universal causal importance.

`OPERATION_MARGINAL_SIGNAL = FALSE`

## 14. Scale / CF / RI / TI coverage

All 81 cells occur in every training split and the R01 pilot. Machine-readable summaries report arm yield and improvement jointly by Scale, CF, RI, TI, search stage, and bottleneck proxy. Improving arms are checked in every Scale×CF group.

## 15. Bottleneck-proxy coverage

State sampling preserves natural proxy labels and prioritizes observed F_LOGISTICS and CROSS_RESOURCE_SYNCHRONIZATION states within each stage. No proxy is fabricated. Counts are in the reservoir and Fig10.

## 16. Information-leakage audit

Every field is classified as MODEL_INPUT_ALLOWED, LABEL_ONLY, IDENTIFIER_ONLY, ANALYSIS_ONLY, or FORBIDDEN_FUTURE_INFORMATION. Outcomes, ranks, regret, swap gain, and counterfactual makespan are labels only. Identifiers and serialized candidates are reconstruction keys, not model features.

## 17. Counterfactual-integrity audit

Tests verify candidate/schedule immutability, live RNG isolation, deterministic replay, arm-order invariance, fixed repair/count, deduplication, paired swaps, and unchanged subsequent ALNS trajectory.

## 18. Runtime and storage scalability

Mean arm runtime is 0.1277s (7.83 arms/CPU-second). Estimated 100k-state cost is 50.8 CPU-hours and 0.6 GiB under this schema. No 100k collection was launched.

## 19. Limitations

Labels remain conditional on one repair heuristic and one destroy size; target sets are structured perturbations rather than exhaustive choices; wall-time trajectories differ in iteration yield; bottleneck and criticality are proxies; and marginal labels are context-dependent. Diagnostic tabular models do not implement NI.

## 20. Phase 6C recommendation

`DESTROY_TARGET_REMAINS_PRIMARY_NI_TARGET = TRUE`  
`OPERATOR_SELECTION_REMAINS_SECONDARY_TARGET = TRUE`  
`REPAIR_SELECTION_REMAINS_DEFERRED = TRUE`  
`DESTROY_SIZE_REMAINS_DEFERRED = TRUE`  
`PHASE6C_RECOMMENDATION = SCALE_WITH_REVISED_ARM_DESIGN`

The recommendation considers state/arm coverage, positive yield, all-regime yield, repair stability, instance-grouped prediction, within-state ranking, and marginal signal—not one AUC alone.

The revised Phase 6C arm protocol should average three repair seeds per retained arm before assigning ranks, because rank Spearman is usable but top-arm agreement is only 0.174. It should retain the original operator controls while increasing outcome-blind local swaps and structured near-neighbor perturbations; the current marginal AUC of 0.544 is insufficient for direct node labels. Set-level classification is ready to scale, but operation labels should remain conditional pairwise labels until richer pre-action state features or a revised reference-set design improves marginal prediction. Three-seed scaling would cost approximately 152.5 CPU-hours at the measured rate, still operationally feasible but requiring review before collection.

## 21. Reproducibility checklist

Phase 6A freeze hashes, generation spec/history/checksums, training manifests and split, separated seeds, compact state reconstruction records, raw counterfactual/marginal Parquet, summaries, audits, tests, figures, and environment are retained. Phase 6B stops here and does not implement CSG, a neural policy, or Phase 6C scaling.

`COUNTERFACTUAL_STATE_COUNT = 13500`  
`COUNTERFACTUAL_ARM_COUNT = 224021`  
`POSITIVE_ARM_FRACTION = 0.063089`  
`STATES_WITH_POSITIVE_ARM_FRACTION = 0.239556`  
`REPAIR_SEED_RANK_STABILITY = 0.643633`  
`COUNTERFACTUAL_TARGET_PREDICTABILITY = ROC_AUC_0.900488_PAIRWISE_0.572486`
