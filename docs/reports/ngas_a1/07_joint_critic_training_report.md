# A1.3 joint critic: implementation and pre-training boundary

Status: **GPU_PROTOCOL_IMPLEMENTED_AND_SMOKE_PASSED; FORMAL_GPU_TRAINING_NOT_STARTED**.
This delivery starts from `098a58e9d42b1dcee84ca04eb288258c70bc0ccb`.
The independently verified V2 T8_R9 label pilot passed all six gates, and all six
source states exceeded 10% informative pairs. Expanded collection and its independent
raw-contract audit passed: 72 states, 18 instances, 6,465 actions and 58,185 paired
replicates. No learned quality, production readiness or latency claim is made yet.

## Model and input contract

`Q(G,k,D,R)` uses one 64-dimensional, two-layer CSG message-passing encoder with
typed nodes and relations. The complete realized CSG includes OP, island, config,
W/F AGV, W/F activity and reconfiguration nodes, 20 original relations and explicit
reverse directions. Selected normalized/current-state node features are signed-log
transformed with a fixed rule; there is no fitted normalization crossing folds.
Historical completion-tail slack/proxies are excluded from the OP input and replaced
by true makespan-DAG slack, operation zero-slack, projected critical membership and
normalized critical-sync score. Temporal margins and binding indicators accompany
edges. IDs and source/fold labels remain lookup-only and are not value-model inputs.

One state forward is shared across the entire joint-action batch. Target OP pooling
combines all-origin rule/family/operator multi-hot provenance and actual cardinality.
Separate size and repair embeddings enter the fusion head with size×target,
target×repair and size×repair interactions. Heads predict joint continuation advantage
and beats-fallback logit; this is not a flat action classifier or target-only scorer.

The loss combines pairwise ranking, listwise relative value, robust advantage
regression and a beats-fallback auxiliary BCE (weights 1/.5/1/.2). Pairwise hard
preferences require a paired-CRN mean gap exceeding max(.001, twice paired SE).
Noise/tie-only states generate no ranking term. Continuous advantage regression and
frequency targets remain available. Advantage scaling/softmax temperature is .01.

## Verification and training boundary

Tests cover batched versus single-action scoring, finite outputs, deterministic
save/load, distinct repair inputs, all-origin permutation invariance, finite gradients,
noise-aware tie handling, cell-group fold isolation and full-rule sampling coverage.
These tests use tiny schedules and synthetic loss targets; they are implementation
checks and are not learned-model results. The current full project regression passes
475 tests. A deterministic 2.74 MB training cache preserves every raw paired advantage,
CSG/action feature and semantic action record; it applies no fitted preprocessing.

The frozen training plan has three seeds and three grouped OOF folds, 60 fixed epochs,
AdamW (lr .0003, decay .0001), gradient clipping 1, and equal instance contribution.
OOF labels may not choose epochs/checkpoints. Ranking, regret, lift, top-5 opportunity,
NDCG and calibration diagnostics must be reported; exact-top1 recall is not the gate.
At least two training seeds must satisfy the preregistered OOF requirements.
Only then fit the single production model using the fixed first seed and all
development instances. The runner verifies frozen code/input hashes, resumes only
complete hash-valid seed/fold units, publishes per-state OOF diagnostics, and performs
the final fit only after writing a passing OOF gate. Formal optimizer steps start only
after the training-specific protocol is committed.

The prior CPU protocol SHA-256 `18dc256f9fc28a880028bf05e65a8f73f764d7f7d680dd0661af6973c0436fd6`
was started and then explicitly retired as incomplete before any OOF quality result
was inspected. Its 29 model/prediction/run artifacts were deleted; protocol, logs,
input cache, last-progress snapshot and a deletion-hash ledger remain. No CPU model
or prediction may be reused by the GPU execution.

The outcome-blind CUDA smoke passed on an RTX 4060 Ti with PyTorch 2.11.0+cu128.
For the largest representative input (722 nodes, 8,148 directed edges, 90 actions),
peak reserved memory was 90 MiB (1.15% of 8,182,628,352 bytes), leaving 7.53 GiB.
Five measured FP32 backward steps averaged 24.63 ms/state; twenty inference repeats
averaged 1.17 ms/state. Losses, gradients, parameters and outputs stayed finite and
repeated inference was bitwise deterministic. Labels were synthetic and independent
of development outcomes. The replacement protocol requires CUDA:0, FP32, TF32 off,
deterministic algorithms, CUBLAS workspace `:4096:8`, no CPU fallback and a complete
restart from 0/9 OOF models.

The replacement GPU protocol is frozen at implementation commit
`656716d76a7efc7119ec41019003c610fe4c5417` with SHA-256
`a65f2ed10a725a002a818c0b75f8058588c56fe8850d98ed079f9610915ef77d`.
The freeze independently reran all 475 project tests and recorded zero formal GPU
optimizer steps at the boundary.

R12 remains DEVELOPMENT after repeated architectural use. R13/R14 stay locked;
no Gurobi or comparator reruns are performed. V1 and V2 raw evidence remain immutable.
