# Phase 6J CAUR Project Handoff

## Current state

- Final decision: `MODEL_REVISION`, stop boundary `BEFORE_R13`; integrity PASS.
- Selected model: none.
- CSG-NI v1 frozen: no.
- Phase 6H replacement authorized: none; Phase 6H remains the reference.
- R12 pilot and formal fit split accessed: yes, within the frozen protocol.
- R12 collection: `PASS`, 288/288 states, 36/36 source trajectories,
  6,809 candidate groups and 13,618 paired-seed label rows.
- R12 tensor cache: 18 shards, 288 states, 6,809 candidates; hashes verified.
- J1/J2 training: `COMPLETE_J1_J2`, 18/18 OOF folds, worker exit code zero.
- The regular-family completion audit reproduces 40,854 OOF rows,
  normalization, metrics, calibration and selected gates, and verifies all
  checkpoint and tensor-shard hashes.
- J1/J2 pairwise accuracies: 0.562392 / 0.561150. Their maximum is below the
  preregistered 0.60 activation threshold, so J3 is required.
- J1 has six retained gates; J2/J3 have none. J1 alone passes the audited R12
  data gates but fails the frozen neural-latency cap. No family is R13 eligible.
- Frozen continuation horizon: H=4. It is the shortest horizon satisfying the
  preregistered agreement rule against H=12 and was selected without solver
  outcome evidence.
- R13 content accessed: no; locked.
- R14 content accessed: no; locked.
- Conditional J3 adds one rank-eight state/candidate/fallback/origin
  interaction: 3,168 new parameters, 5,336,159 total, 2,597,855 trainable.
- J3 uses the unchanged R12 objective, features, seeds, nested OOF folds,
  calibration and gate grid. Its independent protocol was frozen before the
  first J3 optimizer step; the real-data GPU smoke passed.
- J3 completed 9/9 OOF folds with worker exit code zero. The new completion
  audit verifies 20,427 predictions, all checkpoint identities and hashes,
  exact outer-fit normalization, nested folds and reproducible summaries.
- J3 Spearman/pairwise/ECE: 0.160839 / 0.556186 / 0.114795. Its ECE exceeds
  0.10 and no gate survives; J3 is not eligible for R13. No retuning is allowed
  to rescue these results outside a new explicit revision boundary.
- J1 ECE is 0.047758, with 94 gated interventions (S/M/L: 25/35/34), mean
  gated continuation lift 0.004976 and lower CI 0.001926. These are not
  final-solver makespan gains. Preferred ranking targets remain unmet.
- Origin shares by scale and intervention lift by scale/CF/stage are under
  `outputs/phase6j_caur/r12_acceptance/`. No single-origin degeneration was
  observed in J1's scale-level raw winners or interventions.
- J1 full-R12 fitting is complete, 3/3 seeds, with fixed 5/16/4 epochs and
  validated checkpoint hashes. Deployment worker exit code is zero.
- Shared-encoder output and decision parity: exact on 288/288 states, maximum
  absolute output difference zero. All 864 timing records are complete.
- Neural p90: 32.716758 ms > frozen 30 ms cap (9.0559% over); S/M/L p90:
  32.237169/32.868555/32.761459 ms. No favorable latency rerun was performed.
- Cached-total p90: 38.031468 ms, excluding live graph/proposal/source-feature
  construction. Full online total latency is unmeasured, not PASS.
- The independent CPU audit reproduces the OOF evidence, validates complete
  final-fit artifacts and recomputes timing quantiles/gates from the raw CSV.
- Final regression: `300 passed in 13.60s` (baseline 287 plus thirteen
  terminal-audit checks for completeness, corrupt samples, gate boundaries
  and immutable results).

## Frozen files

- `configs/phase6j_caur.json`
- `configs/phase6j_caur_command_manifest.json`
- `configs/phase6j_caur_phase6i_mr_evidence_manifest.json`
- `docs/reports/phase6j_caur_preregistered_protocol.md`
- `instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14/manifests/phase6j_instance_manifest.csv`
- `outputs/phase6j_caur/frozen/r12_training_protocol.json`
  (`a73d9c914a0803305a1246e746970ed14302d7e86f87d8f337a8aa7c642786b3`)
- `outputs/phase6j_caur/frozen/r12_j3_training_protocol.json`
  (`bf596e506b5fbe832e25c9ed8e5edd49db2da14d94cad0c28da29222ef656ac7`)
- `outputs/phase6j_caur/frozen/r12_j1_deployment_protocol.json`
  (`6691d6655b249ba1a1c2a92c30044aa252e2eec3b5d8f40588b798b3ef49c882`)

The authoritative hashes are recorded at the top of the preregistered
protocol. R11 evidence remains immutable and may not be used to adjust Phase
6J.

## Current gate and authorized next action

This Phase 6J run is closed with `MODEL_REVISION` before R13. Do not relaunch
training or profiling, implement the live adapter as if the gate passed, or
unlock R13/R14. No artifact may replace Phase 6H; CSG-NI v1 is not frozen.

Authoritative final decision and status:

- `outputs/phase6j_caur/final/final_decision.json`
- `outputs/phase6j_caur/final/final_status.json`
- `outputs/phase6j_caur/deployment/j1_full_r12/completion_integrity_audit.json`

Reproduce the terminal audit (CPU, no retraining or timing rerun):

```bash
/home/liulei/miniconda3/envs/gnn311/bin/python scripts/finalize_phase6j_caur_r12.py
```

Original deployment evidence stays at
`outputs/phase6j_caur/deployment/j1_full_r12/`: three seed checkpoints/records,
`cached_latency_samples.csv`, `cached_latency_report.json`, log, progress and
worker status. Implementation commit: `91a0347`. Worker PID 91577 completed
at 2026-09-05 11:08:04 CST and was confirmed no longer present during this
audit. The old running snapshot and ETA are superseded by completion evidence.

The only proposed next revision is separately preregistered, runtime-only,
exact-equivalent J1 implementation work: profile the real bottleneck before
changing execution, keep model/normalization/calibrator/gate/candidate bank
and latency caps fixed, and preserve this failed result. This proposal has
not been launched and requires a new explicit continuation decision.
Do not infer solver improvement from R12 continuation lift or substitute
Core/Sensitivity/Legacy results for the unopened promotion gates.

Failure diagnosis and proposal:
`docs/reports/phase6j_caur_r12_final_report.md`.
Historical implementation details remain in the unchanged J3 and J1
deployment protocol addenda. All six registered historical Phase 6I-MR
evidence files retain their original hashes and sizes.
