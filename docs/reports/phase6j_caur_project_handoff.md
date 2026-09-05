# Phase 6J CAUR Project Handoff

## Current state

- Substage decision: `R12_DATA_AUDIT_COMPLETE_J1_DEPLOYMENT_PENDING`; no Phase 6J scientific
  final decision has been made.
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
  data gates. Deployment fitting, parity, latency and the bundle remain pending.
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
- Full regression with deployment preparation: 287 tests passed, including
  eleven new parity, outcome-isolation, gate and resumability checks.

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

The authoritative hashes are recorded at the top of the preregistered
protocol. R11 evidence remains immutable and may not be used to adjust Phase
6J.

## Current gate and authorized next action

The authorized stage is J1 three-seed full-R12 fitting and deployment
validation. Host-level CUDA verification passes on the RTX 4060 Ti in
`gnn311`; the restricted sandbox does not expose the GPU. All OOF training
is complete; do not launch J3 again.

R12 data readiness and the separate J3 audit are under
`outputs/phase6j_caur/r12_acceptance/`. The immutable J1/J2 audit and all
original J1/J2/J3 outputs remain in their original locations.

The independent deployment addendum fixes full-fit epochs to the median of
each seed's three nested-OOF epoch counts: 5,16,4. It retains the original
OOF-selected-winner calibrator and gate. All three base encoders are identical;
shared-encoder inference must match three independent evaluations exactly.
Freeze before fitting and launch/resume only missing hash-validated seeds:

```bash
/home/liulei/miniconda3/envs/gnn311/bin/python scripts/prepare_phase6j_caur_deployment.py --mode freeze
/home/liulei/miniconda3/envs/gnn311/bin/python scripts/prepare_phase6j_caur_deployment.py --mode launch
```

Deployment output root: `outputs/phase6j_caur/deployment/j1_full_r12/`.
Inspect `launch_record.json`, `progress.json`, `worker_status.json`, three
seed checkpoint/record pairs and `cached_latency_report.json`. The worker
holds a file lock and the launcher records the PID before its startup probe.
Corrupt or orphaned checkpoints require inspection, not silent replacement.

Cached latency does not include live CSG construction, proposals or frozen
score feature generation. A completed cached profile is not R13 clearance.
After fitting, audit hashes and parity, implement and verify the live adapter,
measure complete live decision latency, then freeze the R13 bundle and
one-time selection execution protocol. R13/R14 remain locked.

Implementation details and bounded commands:
`docs/reports/phase6j_caur_j1_deployment_addendum.md`.
Historical J3 implementation: `1d54200`, with unchanged J3 protocol addendum.
