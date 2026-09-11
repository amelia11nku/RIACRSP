# NGAS A1.6R Integrity-Revalidated Baseline and Search-Efficiency Diagnosis

**A1.7A-R evidence-class correction:** all 18 R12 instances used here were also
used to train the final frozen C1 critic. This report therefore establishes an
R12 development-exposed comparison; it supplies no fresh-instance evidence for
generalization. The A1.6R integrity decision remains unchanged.

## Failure repaired and terminal decision

- Terminal decision: **`NGAS_A1_6R_PASS_REVALIDATED`**
- A1.6 failure: a detached worker and a duplicate systemd worker overlapped the first formal run, so A1.6 remains `NGAS_A1_6_INVALID_COMPARISON` and is not merged here.
- Repair: atomic filesystem ownership was acquired before CUDA initialization and queue/raw inspection; a second worker fails immediately. The formal session was PID 322359 on `ustb` with 434 recorded heartbeats.
- Execution-integrity audit: **PASS**; session count 1; all 54 run-start and run-complete identities occurred exactly once.
- Formal scope: 54/54 NGAS runs plus four immutable 54-run comparators
- Feasibility and replay: 54/54 NGAS; all reused comparator payloads remain feasible
- Budget audit: median overshoot 0.004184 s; maximum 0.058358 s; atomic starts after deadline = 0
- Regression: 505 tests passed
- Behavior invariance: frozen GPU check passed before formal execution; identical H1, action, repair/candidate, acceptance, and final-solution trajectories were observed under a fixed iteration boundary.
- R13/R14 remain locked. A1.6R does not authorize either holdout.
- Gurobi: not run

## R12 development-exposed quality

NGAS mean RPD is 1.4872% and median RPD is
1.4503%. Against PHASE6N_TOP1, its mean instance-level
gain is 3.4141% with W/T/L
17/0/1. Against LG_HGA_2O, the
corresponding result is 1.7200% and
12/0/6.

Average ranks are NGAS_A1_6R=1.556, PHASE6N_TOP1=3.833, PHASE6H=3.722, ALNS=3.222, LG_HGA_2O=2.667. The Friedman result is
`chi2(4)=24.933333`,
`p=5.18866e-05`. Holm-corrected comparisons are: PHASE6N_TOP1: p_holm=0.000152588, effect=0.965; PHASE6H: p_holm=0.00257874, effect=0.813; ALNS: p_holm=0.000434875, effect=0.918; LG_HGA_2O: p_holm=0.154045, effect=0.392.

The invalid A1.6 diagnostic gain versus PHASE6N_TOP1 was 3.4298%. It is shown only
as a consistency reference and was never pooled with A1.6R.

## Search-efficiency diagnosis

1. **PHASE6N_TOP1:** NGAS paired gain is 3.4141% with W/T/L 17/0/1.
2. **When quality appears:** Paired gains at normalized checkpoints are 10%=3.618%, 25%=3.593%, 50%=3.430%, 75%=3.283%, 100%=3.375%; these causal checkpoints use only incumbents already present at each deadline.
3. **Scale:** Paired NGAS gains versus PHASE6N_TOP1 are S=5.211%, M=3.499%, L=1.532%. This shows whether improvement strengthens or collapses with scale without extrapolating beyond the 18 instances.
4. **Best-of-k:** trials 5-8 contributed 0.396348 mean summed marginal makespan units across their per-trial summaries. Full per-prefix probability, time, and gain/ms are in `diagnostics/best_of_k.json`.
5. **Prior staleness:** age bins 1, 2-5, 6-10, 11-15, and 16-20 are reported in `diagnostics/prior_staleness.json`. Structural changes were sampled at the existing refresh boundary only, so this evidence is descriptive.
6. **Critic and portfolio:** portfolio changed the critic top-1 at rate 0.0357; mean top-5 overlap is 4.852. Promotion/demotion outcomes are retained separately.
7. **Passive critic evidence:** selected-action score calibration and correlations are associative because unselected actions were not evaluated online.
8. **Offline regret:** 18 frozen states enumerated the full unique production bank. Mean critic top-1 realized rank is 153.778, mean regret is 0.555556, and mean recall@5 of best actions is 0.0133.
9. **Runtime:** mean neural-refresh budget share is 1.874%; mean guided iterations per call is 19.949. Across 11753 refreshes, p99 is 26.735 ms and maximum is 198.481 ms.
10. **Capacity and ownership:** all 18 current instances passed derived node/edge/event/target/action bounds with no resize or truncation. `ProductionRefreshRuntime` rejects cross-thread/process ownership.

## Residual risks and next-stage recommendation

The passive correlations do not identify counterfactual causality. The offline
audit uses 18 stratified states and one matched stochastic realization per full-bank
action. Capacity bounds are established for the frozen R12 instance model and do
not authorize larger R13/R14 cases.

Active diagnostic classification: **H5_UNRESOLVED**. Recommended work is recorded in
`diagnostics/next_stage_recommendation.json`. No optimization was implemented and
no subsequent stage starts automatically.

## Evidence

- Protocol: `outputs/ngas_a1/solver_comparison_a16r_v1/preregistration/protocol.json`
- Completion audit: `outputs/ngas_a1/solver_comparison_a16r_v1/audit/completion_audit.json`
- Decision: `outputs/ngas_a1/solver_comparison_a16r_v1/final_decision.json`
- Raw manifest: `outputs/ngas_a1/solver_comparison_a16r_v1/raw_manifest.json`
- Derived and manuscript-ready tables: `outputs/ngas_a1/solver_comparison_a16r_v1/derived/`
- Machine-readable diagnostics: `outputs/ngas_a1/solver_comparison_a16r_v1/diagnostics/`
- Capacity audit: `outputs/ngas_a1/solver_comparison_a16r_v1/preregistration/capacity_audit.json`

BKS v001 remains unchanged. No comparator was rerun, no post-outcome tuning was
performed, and the prior A1.6 raw data remains separate diagnostic-only evidence.
