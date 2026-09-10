# A1.3 joint critic: implementation and pre-training boundary

Status: **IMPLEMENTED_AND_UNIT_TESTED; NOT_TRAINED**.
This delivery starts from `098a58e9d42b1dcee84ca04eb288258c70bc0ccb`.
The independently verified V2 T8_R9 label pilot passed all six gates, and all six
source states exceeded 10% informative pairs. Expanded data collection is now the
next gate; no learned quality, production readiness or latency qualification is claimed.

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

## Verification and subsequent training

Tests cover batched versus single-action scoring, finite outputs, deterministic
save/load, distinct repair inputs, all-origin permutation invariance, finite gradients,
noise-aware tie handling, cell-group fold isolation and full-rule sampling coverage.
These tests use tiny schedules and synthetic loss targets; they are implementation
checks and are not learned-model results. Full project regression runs before freeze.

The frozen training plan has three seeds and three grouped OOF folds, 60 fixed epochs,
AdamW (lr .0003, decay .0001), gradient clipping 1, and equal instance contribution.
OOF labels may not choose epochs/checkpoints. Ranking, regret, lift, top-5 opportunity,
NDCG and calibration diagnostics must be reported; exact-top1 recall is not the gate.
At least two training seeds must satisfy the preregistered OOF requirements.
Only then fit the single production model using the fixed first seed and all
development instances. The training runner/OOF evaluation and model fitting are the
next implementation step after complete expanded data passes its integrity gate.
No automatic training is attached to the long collection job.

R12 remains DEVELOPMENT after repeated architectural use. R13/R14 stay locked;
no Gurobi or comparator reruns are performed. V1 and V2 raw evidence remain immutable.
