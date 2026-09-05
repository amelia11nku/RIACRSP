# Phase 6J: R12 completion and J1 deployment preparation

## Decision boundary

All 27 OOF runs are complete: J1/J2 18, conditional J3 9. The separate
readiness audit reproduces the prior regular-family audit and all J3 labels,
folds, transforms, checkpoints, ensemble predictions, calibration and gates.
No historical protocol or training artifact is changed.

| Family | Spearman | Pairwise | Selected lift / lower CI | Winner ECE | Retained gates |
| --- | ---: | ---: | ---: | ---: | ---: |
| J1 | 0.175995 | 0.562392 | 0.007156 / 0.003222 | 0.047758 | 6 |
| J2 | 0.171750 | 0.561150 | 0.006556 / 0.002312 | 0.009216 | 0 |
| J3 | 0.160839 | 0.556186 | 0.005635 / 0.000878 | 0.114795 | 0 |

These are R12 OOF continuation-target measurements, **not solver makespan
improvements**. J3 fails ECE and retained-gate requirements; J2 has no retained
gate. J1 is the only data-eligible family, not yet an R13-selected artifact.
Preferred ranking targets remain unmet. The J1 gate has 94 interventions
(S/M/L: 25/35/34), gated mean lift 0.004976 and lower CI 0.001926.

Origin shares and intervention lift by scale, CF and search stage are saved
under `outputs/phase6j_caur/r12_acceptance/`. The origin audit checks observed
single-origin degeneration by scale and reports all shares; it does not
introduce an entropy threshold after looking at results. No observed scale
degenerates to a single origin in J1's raw winners or interventions.

## Frozen full-R12 fit rule

Before any final-fit optimizer step, freeze a new independent addendum at
`outputs/phase6j_caur/frozen/r12_j1_deployment_protocol.json`.

- Family: J1 only; unchanged Phase 6F encoder and original J1 heads/losses.
- Seeds 696101, 696102, 696103; all 288 R12 state lists in each fit.
- Fixed epochs: per-seed median of three nested-OOF selected epoch counts.
  The observed fold counts are [6,5,3], [25,16,3], [3,15,4], yielding 5,16,4.
  No holdout, early stopping, seed selection or new hyperparameter sweep.
- Fit candidate feature vocabularies and robust normalization on full R12.
- Reinitialize each seed from the same frozen base, using the unchanged
  optimizer, loss weights, batch size and deterministic shuffle with salt 303.
- Reuse the existing deployment Platt calibrator fitted to OOF selected
  winners and the existing selected gate. Do not recalibrate or select gates
  from final-fit in-sample predictions.
- Save only three small head checkpoints plus hashes of the shared base.
  Reload and validate identities, exact trainable keys, finite parameters and
  selected epoch counts before profiling.

## Inference equivalence and latency

J1's three encoders have identical, frozen parameters. The deployment wrapper
checks this equality, evaluates the common encoder once, and evaluates all
three independent heads. This is exact computation reuse, not a new model or
seed reduction. Require bit-exact three-head predictions and identical final
gate decisions against three separately evaluated models on every R12 state.
Per-seed caps are unchanged; also report the physical shared-ensemble size.

CUDA profile: all 288 cached full banks, batch size one, ten warmup decisions,
three timed repetitions per state, synchronized CUDA, four CPU threads.
Report median/p90/p99 overall and by scale. Neural p90 must be <=30 ms.
Cached total includes batching, transfer, three-head inference, CPU output,
the frozen calibration and the gate. Its p90 is compared with 100 ms only as
a diagnostic: live CSG construction, proposals and frozen-score feature
generation are excluded and **must be measured before R13 eligibility**.
Training tensors may carry labels, but inference never consumes outcome
fields; regression tests poison outcome tensors to verify this boundary.

## Commands and completion

```bash
/home/liulei/miniconda3/envs/gnn311/bin/python scripts/audit_phase6j_caur_readiness.py
/home/liulei/miniconda3/envs/gnn311/bin/python scripts/prepare_phase6j_caur_deployment.py --mode freeze
/home/liulei/miniconda3/envs/gnn311/bin/python scripts/prepare_phase6j_caur_deployment.py --mode launch
```

The launch command requires host CUDA and uses the validated `gnn311` Python.
The worker is detached, single-lock protected and resumable by complete,
hash-validated seed records; it will not overwrite corrupt or orphaned
checkpoints. Existing completed latency evidence is not remeasured to obtain
a more favorable result. The parent epoch logger names fixed fits
`OUTER_FINAL_FIT`; this addendum's run records identify them as full-R12 fits.

Outputs: `outputs/phase6j_caur/deployment/j1_full_r12/`, with three seed
checkpoints/records, progress, worker status, launch record, raw cached latency
samples and report. A successful worker ends at
`CACHED_PROFILE_COMPLETE_LIVE_TOTAL_PENDING`, not `R13_READY`.

Next after completion: audit final-fit hashes and parity, implement/verify
outcome-blind live feature construction and complete live total latency, then
freeze the R13 candidate bundle and one-time selection execution protocol.
R13 and R14 stay locked throughout this preparation.
