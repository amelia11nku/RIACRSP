# Phase 6A ALNS Search Diagnosis Report

## 1. Executive conclusion

Phase 6A collected 180 frozen-budget ALNS trajectories over 18 CB1-DEV instances and 10 seeds, yielding 359,573 transitions and 5,904,329 destroyed-operation rows. The instance-grouped diagnostic logistic AUC for target outcomes is 0.603. On the prespecified interpretation rule, the recommended NI-v1 primary target is **DESTROY_TARGET**.

## 2. Frozen Phase 6A boundary

No CSG, neural model, training set, H1 change, decoder/checker change, benchmark change, or GA/DCGA change was made. The only search-file change is an optional post-decision observer plus an unset fixed-iteration regression hook. Formal runs retain the Phase 5C ALNS parameters and wall-clock budget.

## 3. ALNS implementation audit

The exact initializer, seven destroy operators, five repair operators, destroy-size rule, adaptive weighting, simulated-annealing acceptance, cooling, seed handling, decoder, and checker are documented in `phase6a_alns_implementation_audit.md`.

## 4. Instrumentation design and regression validation

The observer is invoked after all stochastic decisions and weight updates and makes no RNG calls. Across fixed 40-iteration S/M/L cases, best makespan, operator sequence, acceptance sequence, convergence trajectory, and decoder count were identical. Mean measured overhead was 1.16%.

`INSTRUMENTATION_CHANGES_SEARCH_BEHAVIOR = FALSE`

## 5. Diagnostic dataset

The balanced 3-scale × 3-CF DEV design used seeds 610001–610010 and the frozen `2 * operation_count` second budget. All runs were feasible: True. Compact Parquet occupies 133.2 MiB, averaging 1998 transitions per run. No extra TRAIN instances were required.

## 6. Destroy-operator behavior

The best destroy operator by global-best rate is `related`. Full rates, deltas, costs, and best-reduction contribution appear in `operator_summary.csv`; frequency is not used as an importance proxy.

## 7. Repair-operator behavior

The best repair operator by global-best rate is `transport_aware`. Current repair heuristics do not explicitly enumerate insertion positions, so candidate-insertion counts are unavailable; naturally executed trial decodes and runtime are logged.

## 8. Destroy/repair interaction

The best pair is `related/transport_aware`. Pairwise selection, acceptance, improvement, new-best, delta, runtime, decoder count, and best-reduction contribution are in `operator_pair_summary.csv` and Fig03.

## 9. Destroy-size effects

The frozen fraction is 0.15, while integer counts vary only with instance operation count. The dataset therefore supports cost/outcome characterization but not causal comparison of alternative sizes. `DESTROY_SIZE_IS_LEARNING_WORTHY = UNCERTAIN`.

## 10. Destroy-target structural analysis

Positive and non-positive target distributions, standardized mean differences, Mann–Whitney diagnostics, and counts are in `target_feature_summary.csv`. Only pre-action fields enter the grouped 5-fold diagnostic model. `DESTROY_TARGETS_APPEAR_PREDICTABLE = TRUE` (AUC 0.603). This is data-readiness evidence, not a trained NI model.

## 11. Critical-chain overlap analysis

Every move records processing zero-gap overlap, island-terminal resource overlap, W/F chain position, and reconfiguration-local structure. The proxy definition explicitly avoids claiming causal criticality and does not construct a CSG.

## 12. Bottleneck-conditioned behavior

The deterministic terminal-binding proxy and conditional operator statistics are in `bottleneck_operator_summary.csv` and Fig11. The least represented observed proxy categories are ['F_LOGISTICS', 'CROSS_RESOURCE_SYNCHRONIZATION']; these require attention in a future training distribution.

## 13. Search-stage behavior

Runs are divided into five normalized stages. `search_stage_summary.csv` and Figs12/14 show stage-conditional success and new-best probability, supporting use of a causal budget-progress feature if later models demonstrate stable stage dependence.

## 14. Scale and CF behavior

All major statistics are stratified by scale and CF. Mean operator improvement is lowest for `S` scale and `CF3`; these regimes should be oversampled if their positive state yield remains low after counterfactual labeling.

## 15. Positive/neutral/negative sample balance

Strong positive is relative improvement ≥1%; weak positive is between 0 and 1%; neutral has zero delta; worsening is split by acceptance. Observed positive fraction is 0.036. Exact counts and definitions are in `sample_balance.csv` and Fig13.

## 16. Counterfactual-data readiness

Candidate and schedule states can be cloned, repair is reproducible with captured RNG state, decoder evaluation is isolated, and calls can be counted per arm. Future generation must use dedicated arm RNG states and must never update live weights. `COUNTERFACTUAL_GENERATION_FEASIBLE = TRUE`.

## 17. NI target predictability

`OPERATOR_SELECTION_IS_LEARNING_WORTHY = TRUE`  
`DESTROY_SIZE_IS_LEARNING_WORTHY = UNCERTAIN`  
`DESTROY_TARGETS_APPEAR_PREDICTABLE = TRUE`  
`REPAIR_SELECTION_IS_LEARNING_WORTHY = FALSE`

## 18. Information-leakage audit

The schema report and `information_leakage_audit.csv` classify every field. Candidate outcome, acceptance, after-state, new-best, and future-window fields are labels only. Identifiers and timing diagnostics are not model inputs.

## 19. Recommended NI-v1 learning target

`RECOMMENDED_NI_V1_PRIMARY_TARGET = DESTROY_TARGET`  
`RECOMMENDED_NI_V1_SECONDARY_TARGETS = ['OPERATOR_SELECTION']`  
`DEFERRED_NI_TARGETS = ['DESTROY_SIZE', 'REPAIR_SELECTION']`

## 20. Recommended Phase 6B training-distribution requirements

If authorized, oversample `S` and `CF3` initially, then rebalance using achieved label yield rather than raw instance count. Cover all controlled RI/TI levels, especially the underrepresented bottleneck proxies ['F_LOGISTICS', 'CROSS_RESOURCE_SYNCHRONIZATION']. Preserve rejected worsening actions and target a usable positive/non-positive ratio near 1:3 through counterfactual choices rather than discarding negatives. A reasonable first collection gate is at least 100,000 distinct pre-action states, with multiple frozen counterfactual arms per state and instance-grouped splits. Do not generate this distribution in Phase 6A.

## 21. Known limitations

Wall-clock experiments under parallel load do not equal fixed iteration counts. Destroy fraction was not randomized, target labels are move-level and confounded by repair/operator choice, critical/bottleneck measures are proxies, and observational trajectories cannot identify the best unchosen action. These motivate counterfactual labels before final model selection.

## 22. Reproducibility checklist

Environment, commit, dependency versions, checksums, tests, seeds, configs, regression evidence, raw Parquet, summaries, and plot inputs are saved. All 14 figures are reproducible from the raw logs and analysis script.

`BEST_DESTROY_OPERATOR_OVERALL = related`  
`BEST_REPAIR_OPERATOR_OVERALL = transport_aware`  
`BEST_OPERATOR_PAIR_OVERALL = related/transport_aware`  
`PHASE6B_RECOMMENDED = TRUE`
