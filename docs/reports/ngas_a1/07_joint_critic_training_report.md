# A1.3 joint critic: implementation and pre-training boundary

Status: **TRAINING_RUNNER_IMPLEMENTED_AND_UNIT_TESTED; NOT_TRAINED**.
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

The formal training boundary is frozen at implementation commit `10c3c7e80722a4ab3fd545e5ee7bdb34c06d6f0b`.
Training protocol SHA-256: `18dc256f9fc28a880028bf05e65a8f73f764d7f7d680dd0661af6973c0436fd6`;
the freeze reran and passed all 475 project tests. The validated project environment
is Python 3.11.15 with PyTorch 2.11.0; CUDA runtime is unavailable, so the formal
worker will use deterministic CPU execution.

R12 remains DEVELOPMENT after repeated architectural use. R13/R14 stay locked;
no Gurobi or comparator reruns are performed. V1 and V2 raw evidence remain immutable.
