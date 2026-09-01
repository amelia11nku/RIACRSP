# Phase 6G Live CSG-NI Integration Report

## 1. Executive conclusion

Phase 6G completed the frozen-model live-search integration and the full 9-instance, 10-seed DEV-HOLDOUT comparison. R100 CSG-NI improved per-instance mean makespan over ALNS by 1.133%, with 7 wins and 2 losses and a one-sided Wilcoxon p-value of 0.01367188. All 90 CSG-NI schedules were feasible. The performance target is met, but live probability calibration is poor and state-distribution drift is HIGH. Final CB1-Core evaluation is therefore held pending calibration revision.

`PHASE6H_RECOMMENDATION = REVISE_CALIBRATION`

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

| method | intervention_rate | mean_final_makespan | mean_improvement_over_alns | mean_decoder_evaluations | selected |
| --- | --- | --- | --- | --- | --- |
| R100 | 100 | 2963.66667 | 0.00439 | 15514.62963 | True |
| R20 | 20 | 2978.44444 | 0.00302 | 26291.88889 | False |
| R50 | 50 | 2978.96296 | 0.00815 | 20638.33333 | False |

R100 was frozen before DEV-HOLDOUT because it had the lowest mean final makespan on DEV-TUNE. Its tune improvement over paired ALNS was 0.439%, which was treated as a risk signal rather than a holdout selection input.

## 8. Frozen live policy

The DEV-HOLDOUT policy used R100, seed 660301, the frozen Phase 6F probability/utility calibrators, destroy fraction 0.15, 8 candidate trials, transport-aware repair, and `2 × N_operations` seconds per stochastic run.

## 9. DEV-HOLDOUT comparison

| method | mean_of_instance_means | mean_of_instance_best | mean_runtime | mean_decoder_evaluations | feasibility_rate |
| --- | --- | --- | --- | --- | --- |
| H1 | 3406.2222 | 3406.2222 | 0.2060 | 1.0000 | 1.0000 |
| ALNS | 3063.7556 | 2962.7778 | 247.1162 | 34621.6000 | 1.0000 |
| GA | 3211.3000 | 3031.8889 | 247.1162 | 33856.0222 | 1.0000 |
| CSGNI | 3027.6778 | 2930.5556 | 247.1280 | 16972.1222 | 1.0000 |

Paired statistics use the per-instance mean makespan as required:

| method_b | mean_relative_improvement | wins | ties | losses | wilcoxon_p |
| --- | --- | --- | --- | --- | --- |
| H1 | 0.10610 | 9 | 0 | 0 | 0.00195 |
| ALNS | 0.01133 | 7 | 0 | 2 | 0.01367 |
| GA | 0.03335 | 6 | 0 | 3 | 0.04883 |

CSG-NI improved over H1 by 10.610%, ALNS by 1.133%, and GA by 3.335%.

## 10. Feasibility

All 279 primary records were complete and independently feasible. CSG-NI feasibility was 100.0%.

## 11. Trajectory analysis

CSG-NI was compared with ALNS by normalized wall time and decoder evaluations. It reached better final quality while using roughly half the decoder evaluations. Detailed curves are in `outputs/phase6g/figures/alns_vs_csgni_convergence_by_time.*` and `alns_vs_csgni_convergence_by_evaluations.*`.

## 12. NI vs fallback behavior

Intervention coverage was 29.512%; fallback rate was 70.488%. NI moves had higher immediate-improvement, acceptance, and global-best rates than fallback moves, although both move classes had negative mean immediate utility because accepted worsening moves remain permitted.

## 13. Live calibration

The evaluated interventions had predicted positive probability 39.061% versus realized immediate-positive fraction 5.482%. Probability ECE was 0.3358. Predicted utility retained positive rank association with realized utility (Spearman r=0.3214) but was materially miscalibrated.

`LIVE_CALIBRATION_STABLE = FALSE`

## 14. State-distribution drift

A separate non-primary audit captured 20,836 live states across all nine holdout cells and compared them with 60,000 deduplicated Phase 6C TRAIN states. Slack, W/F delay, island load, and local reconfiguration distributions were HIGH drift by PSI/standardized-shift rules; search-progress sampling was LOW drift.

`LIVE_STATE_DISTRIBUTION_DRIFT = HIGH`

## 15. Runtime/decoder overhead

Neural decision overhead accounted for 40.822% of CSG-NI solver wall time. GPU forward averaged 22.53 ms and total decision overhead averaged 47.54 ms. Despite this cost, CSG-NI reduced mean decoder evaluations and improved final quality under the same wall-clock budget.

`LIVE_OVERHEAD_ACCEPTABLE = TRUE`

## 16. Exact/Gurobi sanity comparison

| suite | instance_id | status | incumbent | lower_bound | mip_gap | optimality_proven | replay_feasible |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dev_small | CB1_DEV_S_CF2_R02 | ENVIRONMENT_ERROR |  |  |  | False | False |
| dev_small | CB1_DEV_S_CF3_R02 | ENVIRONMENT_ERROR |  |  |  | False | False |
| tiny | tiny_01 | OPTIMAL | 157.0000 | 157.0000 | 0.0000 | True | True |
| tiny | tiny_03 | OPTIMAL | 36.0000 | 36.0000 | 0.0000 | True | True |

tiny_01 optimum 157 and tiny_03 optimum 36 were proven and replayed through common semantics. ALNS, GA, and R100 CSG-NI recovered both optima for all three preregistered seeds. The two smallest DEV-Small models exceeded the current size-limited Gurobi license and remain pending in an unrestricted-license environment; they are not labeled algorithm failures or solved cases.

Unrestricted-license command:

```bash
/home/liulei/miniconda3/envs/gnn311/bin/python scripts/run_phase6g_exact_validation.py --stage small
```

## 17. ALNS/GA comparison

CSG-NI beat ALNS on 7/9 paired instance means and GA on 6/9. The one-sided Wilcoxon direction favored CSG-NI against both baselines. All S/M/L and CF subgroup means favored CSG-NI over ALNS; CSG-NI was weaker than GA on the Small subgroup.

## 18. Failure cases

The two negative CSG-NI-vs-ALNS instance cells were:

| instance_id | scale | CF_level | improvement_vs_ALNS |
| --- | --- | --- | --- |
| CB1_DEV_L_CF3_R02 | L | CF3 | -0.00045 |
| CB1_DEV_S_CF3_R02 | S | CF3 | -0.00652 |

The main systemic risk is not final solution quality but offline-to-live calibration and feature-distribution mismatch. Direct progression to final Core could turn a favorable DEV result into an unmeasured deployment risk.

## 19. Phase 6H recommendation

The numerical performance, feasibility, statistics, overhead, and tiny exact checks pass. The preferred overall gate does not pass because calibration is unstable and state drift is HIGH.

`PHASE6H_RECOMMENDATION = REVISE_CALIBRATION`

Recommended next action: recalibrate the frozen model on an isolated live-state calibration split without retraining or accessing final CB1-Core labels, then rerun the live calibration/drift gate before final evaluation.

## 20. Reproducibility checklist

- DEV split, seeds, wall-clock budgets, frequency candidates, and selection rule were frozen before holdout.
- Selected rate R100 was frozen before DEV-HOLDOUT.
- 279/279 primary results and 190,985 CSG-NI iteration logs passed integrity checks.
- Drift audit outputs are isolated from primary performance statistics.
- Exact comparisons distinguish proven optimum, incumbent, bound, and license failure.
- Checkpoint SHA-256 was reverified.
- `compileall`, 182 tests, canonical 130 byte regeneration, small validation, and native tiny solver agreement passed.
- Regression status: `PASS`.

## Required explicit conclusions

```text
NI_DISABLED_EQUALS_ALNS = TRUE
LIVE_CSGNI_IMPLEMENTED = TRUE
LIVE_CSGNI_FEASIBILITY_RATE = 1.000000
FROZEN_INTERVENTION_RATE = R100
LIVE_INTERVENTION_COVERAGE = 0.295117
LIVE_FALLBACK_RATE = 0.704883
LIVE_CALIBRATION_STABLE = FALSE
LIVE_STATE_DISTRIBUTION_DRIFT = HIGH
CSGNI_MEAN_IMPROVEMENT_VS_H1 = 0.106102
CSGNI_MEAN_IMPROVEMENT_VS_ALNS = 0.011328
CSGNI_MEAN_IMPROVEMENT_VS_GA = 0.033352
CSGNI_WILCOXON_VS_ALNS_P = 0.01367188
CSGNI_BEATS_ALNS_ON_DEV_HOLDOUT = TRUE
CSGNI_BEATS_GA_ON_DEV_HOLDOUT = TRUE
GUROBI_VALIDATION_EXECUTED = PARTIAL
TINY_OPTIMALITY_RECOVERY_SUMMARY = tiny_01=157 and tiny_03=36 recovered by CSGNI for all 3 preregistered seeds; both optima proven by Gurobi
LIVE_OVERHEAD_ACCEPTABLE = TRUE
PHASE6H_RECOMMENDATION = REVISE_CALIBRATION
```
