# Phase 6J CAUR Project Handoff

## Current state

- Substage decision: `J1_J2_COMPLETE_J3_REQUIRED`; no Phase 6J scientific
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
- J1 has six retained gates; J2 has none. Family eligibility is not final:
  latency, origin behavior, J3 and the deployable bundle are still pending.
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
- The first formal J3 fold (seed 696101, held fold 0) completed in 66.51 s,
  using 26 inner epochs and 14 outer refit epochs. Checkpoint reload reproduced
  all 2,270 predictions from 96 held states exactly (maximum absolute error 0).
- Full regression after the J3 implementation: `276 passed in 13.02s`.

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

The authorized stage is J3 grouped OOF training, three seeds times three
outer folds. Host-level CUDA verification passes on the RTX 4060 Ti in
`gnn311`; the restricted sandbox does not expose the GPU.

J3 runtime evidence is under
`outputs/phase6j_caur/training/j3_relational/`: `progress.json` updates after
every epoch, `worker_status.json` records completion/failure, and
`launch_record.json` identifies the PID, log and measured ETA. Consult these
files for live counts; do not infer completion from an old ETA.

Resume only missing, hash-validated folds with:

```bash
/home/liulei/miniconda3/envs/gnn311/bin/python scripts/launch_phase6j_caur_j3.py
```

The worker holds an operating-system file lock throughout training and
summary generation. The launcher checks actual Python process arguments and
the lock, and records the PID before its startup probe. This addresses the
previous regular-family launcher's orphan/duplicate-worker failure.

After J3 completes, verify nine run records and all checkpoint/prediction
hashes, then audit J1/J2/J3 ranking, selected lift, support, origin behavior,
three-seed stability and latency. Freeze eligible deployable families before
R13 access. No essential gate may be bypassed. R13/R14 remain locked.

Implementation details and exact bounded commands:
`docs/reports/phase6j_caur_j3_protocol_addendum.md`.
J3 implementation commit: `1d54200`.
