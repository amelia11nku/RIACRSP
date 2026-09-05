# Phase 6J conditional J3 implementation

This addendum implements the J3 family already authorized by the original
Phase 6J configuration. R13/R14 content remains locked. It does not replace
the original R12 training freeze or alter any J1/J2 checkpoint or prediction.

## Activation evidence

The regular-family worker finished 18/18 OOF runs with exit code zero.
`outputs/phase6j_caur/training/completion_integrity_audit.json` reproduces
all 40,854 prediction rows, per-fold features and labels, normalization,
ensemble metrics, calibration and selected gates. It verifies checkpoint
hashes and all 18 tensor shard hashes.

| Family | Spearman | Pairwise accuracy | Selected lift | Selected-lift LCB | ECE | Retained gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| J1 | 0.175995 | 0.562392 | 0.00715589 | 0.00322185 | 0.0477584 | 6 |
| J2 | 0.171750 | 0.561150 | 0.00655598 | 0.00231181 | 0.00921595 | 0 |

The pairwise maximum is below 0.60, so the preregistered J3 activation is
required. These are R12 continuation-label metrics, not solver makespan gains.
Gate retention alone does not establish R13 eligibility or latency compliance.

## Single fixed architecture

J3 keeps the J2 encoder, trainability boundary, origin embeddings, candidate
features and three heads. A rank-eight interaction adds a residual to each
candidate embedding:

`residual = W_out[tanh(W_action(a-f)) * tanh(W_state(s) + W_origin(o))]`

Here `s`, `a`, `f` are 128-dimensional state, candidate and fallback
embeddings; `o` is the same 12-dimensional concatenation of the three origin
embeddings. Candidate and fallback share the action projection. All four
linear maps are bias-free, making the fallback residual exactly zero.

The layer adds 3,168 parameters. J3 has 5,336,159 total and 2,597,855
trainable parameters, below the unchanged 5.35M/2.60M limits. Rank eight is
fixed before J3 training and is not selected from an architecture sweep.

## Training and evidence

- Seeds 696101, 696102, 696103; three whole-cell outer folds: nine runs.
- Same nested inner epoch selection and reinitialized outer fixed-epoch fit.
- Same R12 labels, features, train-fold normalization, objective, gap scales,
  batch size, AdamW settings, gradient clipping, patience and epoch cap.
- Learning rate 1e-4, as preregistered for J3.
- Same selected-winner calibration, uncertainty formula and gate grid.
- One worker holds an operating-system file lock throughout training and
  aggregation. Progress updates after every epoch and completed fold.
- Completed folds resume only with matching protocol/checkpoint/prediction
  hashes; corrupt completed records are rejected.

The independent J3 freeze is
`outputs/phase6j_caur/frozen/r12_j3_training_protocol.json`. It hashes the
parent freeze, regular-family completion audit, activation decision and
J3 implementation plus reused neural dependencies. The original frozen
trainer is imported for data, loss, prediction and evaluation helpers; its
source remains unchanged.

Outputs live in `outputs/phase6j_caur/training/j3_relational/`.

```bash
/home/liulei/miniconda3/envs/gnn311/bin/python scripts/audit_phase6j_caur_training.py
/home/liulei/miniconda3/envs/gnn311/bin/python scripts/run_phase6j_caur_j3.py --mode freeze
/home/liulei/miniconda3/envs/gnn311/bin/python scripts/run_phase6j_caur_j3.py --mode smoke --device cuda
/home/liulei/miniconda3/envs/gnn311/bin/python scripts/run_phase6j_caur_j3.py --mode train --device cuda --max-new-runs 1
/home/liulei/miniconda3/envs/gnn311/bin/python scripts/launch_phase6j_caur_j3.py
```

After all nine runs finish, audit J3 and evaluate family eligibility,
candidate-origin behavior and three-model latency. A cached latency result
cannot alone establish the live total-decision cap. Every essential R12 gate
and a deployable family bundle must be complete before the R13 freeze.
