# RIACRSP Phase 6H Project Handoff

## 1. Authoritative status

```text
PHASE6H_STATUS = MODEL_REVISION
PHASE6H_CALIBRATION_STABLE = FALSE
PHASE6H_LIVE_STATE_DRIFT = HIGH
PHASE6H_DRIFT_MITIGATION_ACCEPTED = FALSE
PHASE6H_FEASIBILITY_RATE = 1.000000
PHASE6H_CSGNI_VS_ALNS_MEAN_IMPROVEMENT = 0.0110848867911569
PHASE6H_CSGNI_VS_GA_MEAN_IMPROVEMENT = 0.0414438261275338
PHASE6H_CSGNI_VS_DCGA_MEAN_IMPROVEMENT = 0.1171084478359543
PHASE6H_DECODER_REDUCTION_VS_ALNS = 0.4189220585477993
PHASE6H_TIME_TO_BEST_MEDIAN_CSGNI = 68.87314046698157
PHASE6H_TIME_TO_BEST_MEDIAN_ALNS = 94.71570290898671
PHASE6H_ANYTIME_AUC_IMPROVEMENT_VS_ALNS = 0.2111008299383802
PHASE6H_TARGET_1PCT_HIT_RATE_CSGNI = 0.1111111111111111
PHASE6H_TARGET_1PCT_HIT_RATE_ALNS = 0.0888888888888889
PHASE6H_TARGET_1PCT_TIME_MEDIAN_CSGNI = 1.2299583260319196
PHASE6H_TARGET_1PCT_TIME_MEDIAN_ALNS = 91.6495627395052
PHASE6H_CSGNI_V1_FROZEN = FALSE
PHASE6H_V2_ARCHITECTURE_CHANGED = FALSE
PHASE6H_GUROBI_VALIDATION_EXECUTED = PARTIAL
PHASE6H_GUROBI_PROVEN_OPTIMA_COUNT = 2
PHASE6H_CORE_ACCESSED = FALSE
PHASE6I_RECOMMENDATION = MODEL_REVISION
```

The primary report is
`docs/reports/phase6h_live_calibration_report.md`. The authoritative
machine-readable gate is
`outputs/phase6h_validation/audit/phase6h_gate.json`.

## 2. What completed

- Generated and audited a disjoint 18-instance `CB1-CAL` suite.
- Collected 62,624 post-decoder CAL-FIT labels in 27/27 complete runs.
- Compared Phase6G, sigmoid identity, Platt, isotonic, and beta probability
  mappings using grouped CV.
- Selected Platt probability calibration and retained the Phase6G utility
  mapping after the replacement failed the CAL-FIT rank constraint.
- Ran a 72-result CAL-FIT policy study and froze
  `CALIBRATED_PROBABILITY_UTILITY` before opening CAL-HOLDOUT.
- Completed the 234-result CAL-HOLDOUT comparison with raw monotone incumbent
  traces, normalized-budget checkpoints, pooled-BKS target hits, and decoder
  counts.
- Generated calibration, reliability, drift, intervention/fallback, subgroup,
  runtime, pairwise, anytime, and fairness audits.
- Preserved existing Phase6G exact evidence. A redundant Phase6H tiny batch had
  already returned 6/6 optimum recoveries before the user-approved reuse rule
  was applied; do not rerun it.
- Added 6 Phase6H tests over the Phase6G baseline, reaching 188 passing tests.

## 3. Why the decision is MODEL_REVISION

The result is not a correctness hold. All schedules are feasible, all result
and trace counts are complete, the candidate label path is outcome-blind until
decoder evaluation, and Core was not accessed.

It is also not a solver-performance failure. Phase6H improves mean makespan by
1.108% versus ALNS, reduces decoder evaluations by 41.89%, and improves mean
normalized gap AUC by 21.11%. The paired 95% interval is
`[-0.0388%, 2.2510%]`, so the preferred strictly positive interval is not met,
but the minimum aggregate performance gate passes.

The blocking result is reliability. Probability ECE improves from 0.3531 to
0.0300, but utility Spearman changes from +0.3439 to -0.1342, below the frozen
minimum of +0.3239. Drift remains HIGH and intervention coverage ranges from
0.69% on Small to 35.70% on Large. Simple global post-hoc gating therefore did
not stabilize every live decision signal.

## 4. Frozen evidence versus promoted solver

The pre-holdout policy artifact remains immutable evidence:

```text
policy = outputs/phase6h_calibration/frozen/phase6h_policy.json
policy_sha256 = 4d2da03e13a036569bebf3897135e5da139e292e3260a2f452754d5c9ae3d239
checkpoint_sha256 = f1ccceb607b0e453dfb74e7aa7a946616001db8ec08c12dc9900a66c6f165fc7
```

“Frozen before holdout” does not mean “promoted to CSG-NI v1.” Because the
utility gate failed:

- do not label the artifact as stable CSG-NI v1;
- do not use it to unlock final evaluation;
- do not tune it on CAL-HOLDOUT and report the same results again;
- preserve the raw evidence and hashes unchanged.

## 5. Boundaries for the next phase

The next phase should be a separately preregistered training-side live-state
model revision. It may use new TRAIN-distribution live trajectories and a new
fit/validation split. It must keep the following locked until a revised model
passes fresh reliability and solver gates:

- CB1-Core;
- CB1-Sensitivity;
- Legacy-130;
- Phase6G DEV-HOLDOUT;
- Phase6H CAL-HOLDOUT;
- online RL/PPO;
- claims that the current policy is CSG-NI v1.

The immediate technical priority is to repair utility learning under live
state distribution shift, not to lower the ECE further. A clean next protocol
should collect forced, outcome-labelled top actions on new training-side live
states, fit probability and utility heads without reusing CAL-HOLDOUT, then
evaluate once on a new isolated holdout.

## 6. CSG-NI v2 design backlog — documentation only

No v2 change was implemented in Phase 6H. The following backlog is evidence
based and must remain separate from the current solver:

1. **Search-to-policy self-improvement.** Promising after a controlled model
   revision because selected-action utility ranking is negative and mean
   immediate utility remains negative. Use only training-side CSG-NI
   trajectories with a new holdout boundary.
2. **Search-state curriculum.** High priority. HIGH feature drift and the
   0.69%/8.00%/35.70% Small/Medium/Large coverage split show that raw instance
   size is not an adequate description of live-state difficulty.
3. **Diversity-aware multi-candidate improvement.** Medium priority. The
   deterministic selected action preserves final performance but does not beat
   the Phase6G reference on average; complementary candidate directions may
   reduce ranking saturation. This needs a separate ablation.
4. **Adaptive computation allocation.** High priority. Neural decisions consume
   a mean 46.30% per-run runtime fraction while decoder evaluations fall by
   41.89%. State-dependent computation could retain decoder efficiency with
   lower inference overhead.
5. **Construction-improvement interaction.** Low current priority. H1 remains a
   fast, feasible initializer and the improvement search already supplies the
   main quality gain. Revisit only after utility reliability is stable.

## 7. Exact validation reuse

Phase6G already proved `tiny_01 = 157` and `tiny_03 = 36`; Phase6H does not
change schedule-feasibility semantics. Per user instruction, this evidence is
sufficient and exact/tiny must not be rerun merely because calibration code
changed.

The already-finished redundant Phase6H batch is retained, not deleted:

- Phase6H CSG-NI: 6/6 feasible optimum recoveries;
- Gurobi: two proven tiny optima;
- two CAL-FIT Small cases: size-limited license, no incumbent/optimum claim.

## 8. Runtime fairness caveat

ALNS, GA, Phase6G CSG-NI, and Phase6H CSG-NI respected the nominal budget to
small timing tolerance. Frozen DCGA executes coarse generations and overshot by
3.94% on average, 12.18% maximum. This gives DCGA additional time and is
conservative against Phase6H, but must be disclosed in future tables.

## 9. Reproduction map

Tracked protocol and code:

- `configs/phase6h_live_calibration.json`;
- `configs/phase6h_command_manifest.json`;
- `configs/phase6h_exact_validation.json`;
- `scripts/analyze_phase6h_validation.py`;
- `scripts/finalize_phase6h.py`;
- `rcias_clgri/analysis/phase6h.py`;
- `tests/test_phase6h_calibration.py`.

Authoritative ignored evidence:

- `outputs/phase6h_calibration/`;
- `outputs/phase6h_validation/`;
- `outputs/phase6h_exact_validation/`.

Key summaries:

- `outputs/phase6h_validation/statistics/method_summary.csv`;
- `outputs/phase6h_validation/statistics/pairwise_statistics.csv`;
- `outputs/phase6h_validation/statistics/holdout_calibration_summary.csv`;
- `outputs/phase6h_validation/statistics/intervention_diagnostics.csv`;
- `outputs/phase6h_validation/statistics/csgni_runtime_efficiency_summary.csv`;
- `outputs/phase6h_validation/anytime/method_anytime_summary.csv`;
- `outputs/phase6h_validation/anytime/target_hit_summary.csv`;
- `outputs/phase6h_validation/audit/phase6h_gate.json`;
- `outputs/phase6h_validation/audit/artifact_manifest.json`.

Hashes:

```text
START_COMMIT = 87ef6837c13dcad034ecd5062b76eeb3ef836872
PHASE6H_ANALYSIS_SOURCE_COMMIT = 339814f29f387bdf9af454bc2d4e409759638ce9
CONFIG_SHA256 = e6ac2ab2aeaef7d799e58a6f7b2a5bcbbcc6342e8c8e7a5d2e5330eaea99fe22
MANIFEST_SHA256 = 698f3811294d50c24c4bc4ffe20d6d99338da3c15a8a1cc3d78ee793e8787783
CALIBRATION_ARTIFACT_SHA256 = 834d0824c84c784114b5e42bb5d5fc43385d0de4760ae81a16c0616117069888
POLICY_SHA256 = 4d2da03e13a036569bebf3897135e5da139e292e3260a2f452754d5c9ae3d239
CHECKPOINT_SHA256 = f1ccceb607b0e453dfb74e7aa7a946616001db8ec08c12dc9900a66c6f165fc7
```

## 10. Process and workspace state

CAL-FIT collection, policy gate, CAL-HOLDOUT validation, analysis, and the
already-finished redundant exact batch have all returned. No Phase6H process
needs monitoring or resumption.

Before beginning a model-revision phase, inspect `git status`, read this handoff
and the final report completely, and verify the gate/artifact hashes. Do not
delete ignored output evidence simply because it is not tracked by Git.
