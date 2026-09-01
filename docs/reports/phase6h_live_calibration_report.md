# RIACRSP Phase 6H Live Calibration Revision Report

## 1. Final gate

Phase 6H is complete. The final decision is `MODEL_REVISION`.

The calibrated policy passed correctness, probability-calibration, aggregate
solver-performance, decoder-efficiency, and anytime-efficiency checks. It did
not pass the preregistered utility-rank stability check under the fresh
CAL-HOLDOUT live-state distribution. Therefore the policy frozen before
CAL-HOLDOUT is retained as experimental evidence but is **not** promoted to
the production baseline `CSG-NI v1`. CB1-Core, CB1-Sensitivity, and Legacy-130
remain locked.

```text
PHASE6H_STATUS = MODEL_REVISION
PHASE6H_CALIBRATION_STABLE = FALSE
PHASE6H_PROBABILITY_CALIBRATION_STABLE = TRUE
PHASE6H_UTILITY_CALIBRATION_STABLE = FALSE
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

The 1% target times above are medians conditional on a successful hit. The hit
rates are reported alongside them because only 5/45 Phase6H runs and 4/45 ALNS
runs reached that target.

## 2. Scope and frozen boundary

Phase 6H started from Phase 6G commit
`87ef6837c13dcad034ecd5062b76eeb3ef836872`. It introduced only:

- probability/utility calibration artifacts around the frozen Phase 6F model;
- an abstaining live intervention gate;
- an isolated `CB1-CAL` fit/holdout split;
- raw-score, intervention, timing, and anytime-search instrumentation;
- analysis and reproducibility tooling.

It did not retrain the neural checkpoint, change CSG-1.0, change H1, change the
decoder or feasibility checker, alter candidate-bank semantics, alter the
transport-aware repair, or alter simulated-annealing acceptance. No v2
architecture was implemented.

The Phase 6F checkpoint remained:

```text
f1ccceb607b0e453dfb74e7aa7a946616001db8ec08c12dc9900a66c6f165fc7
```

## 3. Calibration split and integrity

`CB1-CAL` contains 18 new canonical instances with no historical ID/hash
overlap:

- CAL-FIT: R07, 9 instances;
- CAL-HOLDOUT: R08, 9 instances;
- cells: Small/Medium/Large × CF1/CF2/CF3, fixed RI2/TI2.

CAL-FIT collection completed 27/27 runs and 62,624 labelled live states. Every
label was created only after common decoder evaluation. CAL-HOLDOUT was not
opened until the policy artifact and concurrency protocol were frozen.

Integrity evidence:

- `instances/controlled/RCIAS-CB1-CAL/manifests/calibration_instance_audit.json`;
- `outputs/phase6h_calibration/collection/collection_integrity.json`;
- `outputs/phase6h_calibration/gate_study/gate_study_integrity.json`;
- `outputs/phase6h_validation/audit/analysis_integrity.json`.

All report `PASS`.

## 4. CAL-FIT calibration and policy freeze

Grouped three-fold CV was performed by `instance_id`. The probability results
on all out-of-fold samples were:

| Method | ECE | Brier | NLL | Mean predicted | Realized positive |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase6G mapping | 0.316596 | 0.144171 | 0.465355 | 0.353387 | 0.036791 |
| Platt | 0.006828 | 0.035124 | 0.151664 | 0.040068 | 0.036791 |
| Isotonic | 0.015928 | 0.037192 | 0.157590 | 0.046722 | 0.036791 |
| Beta | 0.013860 | 0.037257 | 0.157944 | 0.043329 | 0.036791 |

Platt calibration was selected. Its fitted coefficient is
`0.8973868827907767` and intercept is `-3.6617671779884624`.

The new isotonic utility candidate improved MAE but reduced OOF Spearman from
`0.294207` to `0.200118`, beyond the allowed degradation. The Phase 6G utility
calibrator was therefore retained rather than selecting the lower-MAE mapping.

The 72-run CAL-FIT solver gate selected
`CALIBRATED_PROBABILITY_UTILITY` by the frozen tie-breaker:

| Policy | Mean final makespan | Decoder evaluations | Coverage | Selected |
| --- | ---: | ---: | ---: | --- |
| Calibrated probability | 3131.389 | 18869.389 | 0.367492 | No |
| Calibrated probability + utility | 3131.389 | 18795.167 | 0.401946 | Yes |
| Probability + utility + support | 3147.500 | 18926.167 | 0.145332 | No |
| Phase6G reference | 3167.222 | 19036.556 | 0.364462 | No |

The selected thresholds were probability `0.06` and predicted utility `-0.05`;
the support guard was disabled. The frozen deployment policy SHA-256 is:

```text
4d2da03e13a036569bebf3897135e5da139e292e3260a2f452754d5c9ae3d239
```

## 5. Primary CAL-HOLDOUT comparison

The unbiased comparison contains 234 results: H1 once per instance and five
matched seeds per instance for each iterative solver. All 234 returned and all
final schedules passed the independent feasibility checker.

| Method | Mean of instance means | Mean decoder evals | Mean runtime (s) | Improvement vs ALNS |
| --- | ---: | ---: | ---: | ---: |
| H1 | 3293.000 | 1.0 | 0.186 | — |
| ALNS | 3086.333 | 32711.7 | 244.227 | 0.000% |
| GA | 3264.489 | 32212.5 | 244.227 | -3.746% |
| DCGA | 3540.156 | 11381.2 | 255.946 | -12.558% |
| Phase6G CSG-NI | 3036.867 | 16187.8 | 244.239 | +1.441% |
| Phase6H CSG-NI | 3042.467 | 19008.1 | 244.238 | +1.108% |

Primary paired statistics for Phase6H versus ALNS:

- mean relative improvement: `1.1085%`;
- instance wins/ties/losses: `6/0/3`;
- instance bootstrap 95% interval: `[-0.0388%, 2.2510%]`;
- one-sided Wilcoxon p-value: `0.064453125`.

Thus the minimum aggregate non-worse gate passes by point estimate and the
effect exceeds the preferred 0.5% threshold, but the preferred positive paired
confidence interval is not achieved. Phase6H is 0.3425% worse than the
uncalibrated Phase6G reference on this holdout; that comparison is descriptive
and does not change the pre-frozen selection.

All scale and CF subgroup means are non-negative versus ALNS. The worst
individual instance is Medium/CF2 at `-1.4097%`; no mean subgroup collapse was
observed.

## 6. CAL-HOLDOUT reliability and intervention behavior

| Metric | Phase6G reference | Phase6H calibrated |
| --- | ---: | ---: |
| Eligible states | 91,076 | 106,946 |
| Intervention coverage | 0.326705 | 0.141604 |
| Fallback coverage | 0.673295 | 0.858396 |
| ECE | 0.353092 | 0.029953 |
| Brier | 0.184216 | 0.076154 |
| NLL | 0.554492 | 0.288516 |
| Mean predicted positive | 0.407659 | 0.102282 |
| Realized positive | 0.054579 | 0.080032 |
| Utility Spearman | +0.343922 | -0.134199 |
| Mean realized immediate utility | -0.064017 | -0.088742 |

Probability calibration is a clear pass: ECE is below the preferred 0.05
threshold, Brier and NLL both improve, and the predicted-positive error falls
from 0.3531 to 0.0223.

Utility reliability is a fail. The preregistered minimum acceptable Spearman
was `0.323922` (Phase6G minus 0.02), while Phase6H achieved `-0.134199`.
Although final solver quality held, the retained utility signal reversed its
holdout ranking on the actions actually selected by the calibrated gate. This
is not safe enough for v1 promotion.

The abstention pattern also varies sharply by scale:

| Scale | Intervention coverage | Positive rate | Mean immediate utility |
| --- | ---: | ---: | ---: |
| Small | 0.006898 | 0.215139 | -0.042475 |
| Medium | 0.080025 | 0.178151 | -0.038954 |
| Large | 0.356997 | 0.052693 | -0.102144 |

This is evidence that global post-hoc calibration is not uniformly stable
across live-state regimes, even though fallback prevents final-quality
collapse.

## 7. Anytime efficiency

The reference is a clearly labelled pooled BKS built only after all compared
runs completed. It was not used for tuning.

| Method | Mean normalized gap AUC | Median AUC | Median time to final best (s) | Median evals to final best |
| --- | ---: | ---: | ---: | ---: |
| ALNS | 0.078318 | 0.070086 | 94.716 | 15,297 |
| GA | 0.132748 | 0.118167 | 224.009 | 23,790 |
| DCGA | 0.238126 | 0.224971 | 203.266 | 8,130 |
| Phase6G CSG-NI | 0.064189 | 0.050035 | 102.889 | 8,057 |
| Phase6H CSG-NI | 0.061785 | 0.049553 | 68.873 | 6,489 |

Phase6H improves mean normalized gap AUC by `21.11%` versus ALNS. Target-hit
rates for Phase6H versus ALNS are:

| Shared gap target | Phase6H | ALNS |
| --- | ---: | ---: |
| 5% | 0.6667 | 0.5111 |
| 2% | 0.2667 | 0.1111 |
| 1% | 0.1111 | 0.0889 |
| 0.5% | 0.0667 | 0.0444 |

There is no anytime collapse hidden by the final objective. Right-censored
non-hits are retained as non-hits rather than assigned a false hit time.

## 8. Runtime and fairness audit

Phase6H uses 19,008 mean decoder evaluations versus ALNS's 32,712, a 41.89%
reduction. The calibrated solver averages 42.87 ms per eligible neural decision.
Mean per-run neural decision overhead is 46.30% of runtime when the fraction is
computed per run and then averaged. The calibration gate itself is small; CSG
construction and neural inference dominate the overhead.

The frozen validation protocol used four concurrent stochastic workers, one
thread per worker, and one shared GPU worker. Phase6G and Phase6H CSG-NI ran
sequentially in that GPU worker.

The frozen DCGA implementation checks its budget between coarse generations.
It exceeded the nominal wall-clock budget by 3.94% on average and 12.18% at
maximum. This favors DCGA rather than Phase6H, so the large Phase6H advantage
over DCGA is conservative, but DCGA wall-clock figures must not be described as
exactly matched.

## 9. Drift

Overall live-state drift remains `HIGH` in five of six tracked features;
`search_progress` remains `LOW`. Particularly large PSI values remain in local
reconfiguration and F-delay features. The calibration gate was not allowed to
manipulate this diagnostic.

Probability reliability and final performance held under the drift, but the
utility-ranking reversal and scale-dependent abstention mean that drift
mitigation is not accepted as stable.

## 10. Exact/tiny evidence

Phase6G had already proven `tiny_01 = 157` and `tiny_03 = 36` with Gurobi and
verified schedules through the common replay/checker. Phase6H did not change
schedule construction semantics, so that evidence is sufficient under the
user-approved reuse rule.

The Phase6H batch had already completed before it could be stopped. It is kept
only as redundant confirmation: all six Phase6H CSG-NI tiny runs recovered the
same proven optima and were feasible. The two smallest CAL-FIT Small MILPs were
rejected by the size-limited license. Their errors and solver logs are
preserved; no restriction was bypassed and no incumbent was called optimal.

```text
GUROBI_TINY_PROVEN_OPTIMA = 2
PHASE6H_TINY_RECOVERY = 6/6
CAL_FIT_SMALL_LICENSE_LIMITED = 2/2
```

## 11. Decision

| Gate | Result | Reason |
| --- | --- | --- |
| Correctness | PASS | 100% feasible, traces valid, splits/integrity pass |
| Probability calibration | PASS | ECE/Brier/NLL and rate alignment improve |
| Utility calibration | FAIL | Spearman +0.344 to -0.134 |
| Solver minimum | PASS | +1.108% vs ALNS, fewer decodes, better AUC |
| Preferred solver confidence | NOT MET | bootstrap lower bound slightly negative |
| Drift mitigation | FAIL | HIGH drift plus utility/coverage instability |
| CSG-NI v1 promotion | NO | all required gates did not pass |

The next phase must be a separate `MODEL_REVISION` phase using training-side
live-state data. It must create a new fit/validation boundary and must not tune
on CAL-HOLDOUT, DEV-HOLDOUT, Core, Sensitivity, or Legacy-130.

## 12. Reproducibility

Key hashes:

```text
START_COMMIT = 87ef6837c13dcad034ecd5062b76eeb3ef836872
PHASE6H_ANALYSIS_SOURCE_COMMIT = 339814f29f387bdf9af454bc2d4e409759638ce9
PHASE6H_CONFIG_SHA256 = e6ac2ab2aeaef7d799e58a6f7b2a5bcbbcc6342e8c8e7a5d2e5330eaea99fe22
CALIBRATION_MANIFEST_SHA256 = 698f3811294d50c24c4bc4ffe20d6d99338da3c15a8a1cc3d78ee793e8787783
CALIBRATION_ARTIFACT_SHA256 = 834d0824c84c784114b5e42bb5d5fc43385d0de4760ae81a16c0616117069888
FROZEN_POLICY_SHA256 = 4d2da03e13a036569bebf3897135e5da139e292e3260a2f452754d5c9ae3d239
CHECKPOINT_SHA256 = f1ccceb607b0e453dfb74e7aa7a946616001db8ec08c12dc9900a66c6f165fc7
```

Seeds:

- CAL-FIT collection: 671201–671203;
- CAL-HOLDOUT: 671301–671305;
- CAL-FIT policy gate: 671401–671402;
- tiny redundant confirmation: 671501–671503;
- bootstrap namespace: 671901;
- frozen proposal/repair/acceptance/diagnostic namespaces: 670102–670105.

The command manifest is `configs/phase6h_command_manifest.json`. The complete
key-file and raw-tree hashes are in
`outputs/phase6h_validation/audit/artifact_manifest.json`. The machine-readable
decision is `outputs/phase6h_validation/audit/phase6h_gate.json`.

Environment: Python 3.11.15, PyTorch 2.11.0+cu128, CUDA 12.8, NVIDIA RTX 4060
Ti. The final implementation suite reached 188 passing tests before report-only
finalization.
