# NGAS A1.7A-R final report

## Terminal classification

`NGAS_A1_7AR_PASS_CONTAMINATION_RECOVERED`

A1.7A-R reconstructs and repairs the experimental evidence boundary. It does not
retrain C1, alter the production solver, run Gurobi, or access R13, R14, or
RCIAS-CB1-CORE45 for performance evaluation.

## Required conclusions

1. **Exact final-C1 instances.** The final frozen C1 used these 18 R12 instances:

- `CB1_CAUR_L_CF1_RI2_TI2_R12`
- `CB1_CAUR_L_CF1_RI2_TI2_R12_C02`
- `CB1_CAUR_L_CF2_RI2_TI2_R12`
- `CB1_CAUR_L_CF2_RI2_TI2_R12_C02`
- `CB1_CAUR_L_CF3_RI2_TI2_R12`
- `CB1_CAUR_L_CF3_RI2_TI2_R12_C02`
- `CB1_CAUR_M_CF1_RI2_TI2_R12`
- `CB1_CAUR_M_CF1_RI2_TI2_R12_C02`
- `CB1_CAUR_M_CF2_RI2_TI2_R12`
- `CB1_CAUR_M_CF2_RI2_TI2_R12_C02`
- `CB1_CAUR_M_CF3_RI2_TI2_R12`
- `CB1_CAUR_M_CF3_RI2_TI2_R12_C02`
- `CB1_CAUR_S_CF1_RI2_TI2_R12`
- `CB1_CAUR_S_CF1_RI2_TI2_R12_C02`
- `CB1_CAUR_S_CF2_RI2_TI2_R12`
- `CB1_CAUR_S_CF2_RI2_TI2_R12_C02`
- `CB1_CAUR_S_CF3_RI2_TI2_R12`
- `CB1_CAUR_S_CF3_RI2_TI2_R12_C02`

2. **Training volume.** The cache contains **72 states** and **6,465 joint-action
   labels**.
3. **Continuation label scope.** `no_continuation_rollout=true`.
4. **Four sources per instance.** `H1`, `NATIVE16_746101`,
   `NATIVE16_746102`, and `NATIVE16_746103`.
5. **Instance overlap.** Final C1 and A1.6R overlap on **18/18 instances (100%)**.
6. **Seed overlap.** All three Native16/A1.6R seeds overlap:
   `746101`, `746102`, and `746103` (**3/3; 100%**).
7. **State overlap.** The 18 replayable A1.6R snapshots are all
   same-instance/same-seed but non-identical states. Exact matches: **0**;
   same schedule with incomplete metadata: **0**; non-identical provenance
   matches: **18**.
8. **Remaining A1.6R claims.** A1.6R remains valid for production integration,
   budget/concurrency accounting, feasibility, reproducibility, runtime
   qualification, and R12 development-benchmark performance.
9. **Removed A1.6R claims.** It does not establish independent-test performance,
   unseen-instance generalization, leakage-free held-out evaluation, or an
   unbiased final comparison against learning-free baselines.
10. **Comparator exposure.** R12-specific training/adaptation is confirmed for
    `PHASE6N_TOP1` and `NGAS_C1`. No R12-specific tuning evidence was found for
    GA, DCGA, DABC, LG_HGA_2O, ALNS, or PHASE6H.
11. **Permanent R12 role.** `DEVELOPMENT_EXPOSED`.
12. **Final-evaluation locks.** R13 remains `LOCKED_FINAL_EVAL`; R14 remains
    `LOCKED_GENERALIZATION_EVAL`. Their performance data remain untouched.
13. **New C1-v2 training pool.** The governed pool contains 81 instances:

- `CB1_TRAIN_L_CF1_RI1_TI1_R03`
- `CB1_TRAIN_L_CF1_RI1_TI2_R01`
- `CB1_TRAIN_L_CF1_RI1_TI3_R02`
- `CB1_TRAIN_L_CF1_RI2_TI1_R01`
- `CB1_TRAIN_L_CF1_RI2_TI2_R02`
- `CB1_TRAIN_L_CF1_RI2_TI3_R03`
- `CB1_TRAIN_L_CF1_RI3_TI1_R02`
- `CB1_TRAIN_L_CF1_RI3_TI2_R03`
- `CB1_TRAIN_L_CF1_RI3_TI3_R01`
- `CB1_TRAIN_L_CF2_RI1_TI1_R01`
- `CB1_TRAIN_L_CF2_RI1_TI2_R02`
- `CB1_TRAIN_L_CF2_RI1_TI3_R03`
- `CB1_TRAIN_L_CF2_RI2_TI1_R02`
- `CB1_TRAIN_L_CF2_RI2_TI2_R03`
- `CB1_TRAIN_L_CF2_RI2_TI3_R01`
- `CB1_TRAIN_L_CF2_RI3_TI1_R03`
- `CB1_TRAIN_L_CF2_RI3_TI2_R01`
- `CB1_TRAIN_L_CF2_RI3_TI3_R02`
- `CB1_TRAIN_L_CF3_RI1_TI1_R02`
- `CB1_TRAIN_L_CF3_RI1_TI2_R03`
- `CB1_TRAIN_L_CF3_RI1_TI3_R01`
- `CB1_TRAIN_L_CF3_RI2_TI1_R03`
- `CB1_TRAIN_L_CF3_RI2_TI2_R01`
- `CB1_TRAIN_L_CF3_RI2_TI3_R02`
- `CB1_TRAIN_L_CF3_RI3_TI1_R01`
- `CB1_TRAIN_L_CF3_RI3_TI2_R02`
- `CB1_TRAIN_L_CF3_RI3_TI3_R03`
- `CB1_TRAIN_M_CF1_RI1_TI1_R02`
- `CB1_TRAIN_M_CF1_RI1_TI2_R03`
- `CB1_TRAIN_M_CF1_RI1_TI3_R01`
- `CB1_TRAIN_M_CF1_RI2_TI1_R03`
- `CB1_TRAIN_M_CF1_RI2_TI2_R01`
- `CB1_TRAIN_M_CF1_RI2_TI3_R02`
- `CB1_TRAIN_M_CF1_RI3_TI1_R01`
- `CB1_TRAIN_M_CF1_RI3_TI2_R02`
- `CB1_TRAIN_M_CF1_RI3_TI3_R03`
- `CB1_TRAIN_M_CF2_RI1_TI1_R03`
- `CB1_TRAIN_M_CF2_RI1_TI2_R01`
- `CB1_TRAIN_M_CF2_RI1_TI3_R02`
- `CB1_TRAIN_M_CF2_RI2_TI1_R01`
- `CB1_TRAIN_M_CF2_RI2_TI2_R02`
- `CB1_TRAIN_M_CF2_RI2_TI3_R03`
- `CB1_TRAIN_M_CF2_RI3_TI1_R02`
- `CB1_TRAIN_M_CF2_RI3_TI2_R03`
- `CB1_TRAIN_M_CF2_RI3_TI3_R01`
- `CB1_TRAIN_M_CF3_RI1_TI1_R01`
- `CB1_TRAIN_M_CF3_RI1_TI2_R02`
- `CB1_TRAIN_M_CF3_RI1_TI3_R03`
- `CB1_TRAIN_M_CF3_RI2_TI1_R02`
- `CB1_TRAIN_M_CF3_RI2_TI2_R03`
- `CB1_TRAIN_M_CF3_RI2_TI3_R01`
- `CB1_TRAIN_M_CF3_RI3_TI1_R03`
- `CB1_TRAIN_M_CF3_RI3_TI2_R01`
- `CB1_TRAIN_M_CF3_RI3_TI3_R02`
- `CB1_TRAIN_S_CF1_RI1_TI1_R01`
- `CB1_TRAIN_S_CF1_RI1_TI2_R02`
- `CB1_TRAIN_S_CF1_RI1_TI3_R03`
- `CB1_TRAIN_S_CF1_RI2_TI1_R02`
- `CB1_TRAIN_S_CF1_RI2_TI2_R03`
- `CB1_TRAIN_S_CF1_RI2_TI3_R01`
- `CB1_TRAIN_S_CF1_RI3_TI1_R03`
- `CB1_TRAIN_S_CF1_RI3_TI2_R01`
- `CB1_TRAIN_S_CF1_RI3_TI3_R02`
- `CB1_TRAIN_S_CF2_RI1_TI1_R02`
- `CB1_TRAIN_S_CF2_RI1_TI2_R03`
- `CB1_TRAIN_S_CF2_RI1_TI3_R01`
- `CB1_TRAIN_S_CF2_RI2_TI1_R03`
- `CB1_TRAIN_S_CF2_RI2_TI2_R01`
- `CB1_TRAIN_S_CF2_RI2_TI3_R02`
- `CB1_TRAIN_S_CF2_RI3_TI1_R01`
- `CB1_TRAIN_S_CF2_RI3_TI2_R02`
- `CB1_TRAIN_S_CF2_RI3_TI3_R03`
- `CB1_TRAIN_S_CF3_RI1_TI1_R03`
- `CB1_TRAIN_S_CF3_RI1_TI2_R01`
- `CB1_TRAIN_S_CF3_RI1_TI3_R02`
- `CB1_TRAIN_S_CF3_RI2_TI1_R01`
- `CB1_TRAIN_S_CF3_RI2_TI2_R02`
- `CB1_TRAIN_S_CF3_RI2_TI3_R03`
- `CB1_TRAIN_S_CF3_RI3_TI1_R02`
- `CB1_TRAIN_S_CF3_RI3_TI2_R03`
- `CB1_TRAIN_S_CF3_RI3_TI3_R01`

14. **New validation pool.** The governed pool contains 27 instances:

- `CB1_TRAIN_L_CF1_RI1_TI3_R04`
- `CB1_TRAIN_L_CF1_RI2_TI1_R04`
- `CB1_TRAIN_L_CF1_RI3_TI2_R04`
- `CB1_TRAIN_L_CF2_RI1_TI1_R04`
- `CB1_TRAIN_L_CF2_RI2_TI2_R04`
- `CB1_TRAIN_L_CF2_RI3_TI3_R04`
- `CB1_TRAIN_L_CF3_RI1_TI2_R04`
- `CB1_TRAIN_L_CF3_RI2_TI3_R04`
- `CB1_TRAIN_L_CF3_RI3_TI1_R04`
- `CB1_TRAIN_M_CF1_RI1_TI2_R04`
- `CB1_TRAIN_M_CF1_RI2_TI3_R04`
- `CB1_TRAIN_M_CF1_RI3_TI1_R04`
- `CB1_TRAIN_M_CF2_RI1_TI3_R04`
- `CB1_TRAIN_M_CF2_RI2_TI1_R04`
- `CB1_TRAIN_M_CF2_RI3_TI2_R04`
- `CB1_TRAIN_M_CF3_RI1_TI1_R04`
- `CB1_TRAIN_M_CF3_RI2_TI2_R04`
- `CB1_TRAIN_M_CF3_RI3_TI3_R04`
- `CB1_TRAIN_S_CF1_RI1_TI1_R04`
- `CB1_TRAIN_S_CF1_RI2_TI2_R04`
- `CB1_TRAIN_S_CF1_RI3_TI3_R04`
- `CB1_TRAIN_S_CF2_RI1_TI2_R04`
- `CB1_TRAIN_S_CF2_RI2_TI3_R04`
- `CB1_TRAIN_S_CF2_RI3_TI1_R04`
- `CB1_TRAIN_S_CF3_RI1_TI3_R04`
- `CB1_TRAIN_S_CF3_RI2_TI1_R04`
- `CB1_TRAIN_S_CF3_RI3_TI2_R04`

15. **Split disjointness.** Training and validation IDs and content hashes are
    mutually disjoint and are also disjoint from R12, R13, R14, and CORE45.
16. **Scale/stage ranking.** U0 is too sparse to establish monotonic degradation.
    Clean L has 0/6 informative states; clean M and S have mean rho 0.0529 and
    0.0349. R12 S/M/L mean rho is 0.0812/0.0819/0.1436 on only 1/2/2 informative
    states. R12 middle and late audited states are all U0-constant. This is a
    support and trajectory-coverage problem, not evidence of reliable scaling.
17. **U0 versus U1.** Conditional on informative states, mean action-rank rho is
    **0.9330** on clean non-R12 and **0.7664** on R12, but only 4/18 and 5/18
    states are informative. The apparent all-state Top-1 agreement is inflated by
    all-zero banks.
18. **Cost normalization.** U1 and U2 rankings are nearly identical: mean rho is
    **0.999999** (clean) and **0.999991** (R12), with the same Top-1 in all 36
    audited states. The current measured cost normalization does not materially
    change the action order.
19. **Portfolio interaction.** Deterministic portfolio adjustment changes neural
    Top-1 in 1/18 states per origin. The one clean change loses 13 U0 units; the
    R12 change is neutral. Archived sampled production actions differ from neural
    Top-1 in every state, so that comparison combines frozen exploration with
    portfolio weighting. There is no evidence of a systematic correction benefit.
20. **Trials 1...8.** Mean raw best-makespan loss versus eight trials is
    187.8078, 101.4400, and 43.1263 at caps 1, 2, and 4. The probability that a
    later trial changes the best is 0.8807, 0.7472, and 0.4973. Positive-gain loss
    is much smaller (0.0564, 0.0442, 0.0353), while the no-new-best rate reaches
    0.8869 at trial 8.
21. **Downstream decision.** Evidence justifies separate development pilots for
    **both A1.7B and A1.7C**. A1.7B should test adaptive trial racing against the
    raw-quality/positive-gain tradeoff. A1.7C should train a trajectory-aware C1-v2
    because current local ranking is sparse and weak. Neither phase starts here.

## Clean trajectory and staleness evidence

The frozen revision-3 collection completed 108/108 runs and 540/540 states: 405
TRAIN and 135 VALIDATION states, balanced across S/M/L and five search-stage
targets. The full-bank audit evaluated 12,775 actions from 36 frozen states.

The staleness audit completed 36 states, 180 offsets, and 63,875 action-offset
evaluations. At offset 5, semantic bank Jaccard drops to 0.1413 (clean) and
0.1325 (R12), even where compact features and typed edges are unchanged. Common
semantic actions retain rank rho around 0.98-1.00 and mean percentile-rank drift
around 0.03 through offset 19. The drop is therefore state-ID-conditioned
candidate-bank churn, not evidence of RT-HGT numerical instability. Fresh Top-1
does not consistently improve U0, so the frozen refresh interval is unchanged.

## Evidence locations

- Exposure reconstruction: `reports/ngas_a17ar_r12_exposure_audit.json`
- Dataset registry: `configs/dataset_role_registry.json`
- Exposure ledger: `artifacts/dataset_exposure_ledger.jsonl`
- Trajectory protocol: `artifacts/ngas_a17ar/trajectory_protocol_manifest.json`
- Completion audits: `outputs/ngas_a1/trajectory_utility_a17ar_v1/audit`
- Diagnostic reports: `reports/ngas_a17ar_critic_ranking_audit.md` through
  `reports/ngas_a17ar_prior_staleness.md`
- Figures and source data: `reports/figures/ngas_a17ar`
- Final decision: `outputs/ngas_a1/trajectory_utility_a17ar_v1/final_decision.json`
- Result manifest: `outputs/ngas_a1/trajectory_utility_a17ar_v1/result_manifest.json`
