# Phase 6I-MR Preregistered Protocol

```text
phase = Phase 6I-MR
protocol_revision = 1.2
status = REVISION_1_2_PREREGISTERED_BEFORE_FORMAL_R09_COLLECTION_OR_R10_R11_CONTENT_ACCESS
starting_commit = 7c448000be69084511af181f52cad0a8db71479a
config_sha256 = 535ab64b10be26593da4716488dd0189e22fb6d22c413b6b2569251171e64683
command_manifest_sha256 = e449154578de42c47c043b0a6d9b457b176a3659a8b102bd4baa50a11900a147
r10_content_accessed = false
r11_content_accessed = false
phase6h_cal_holdout_reused = false
gurobi_or_tiny_planned = false
```

## 1. Scope and success criterion

Phase 6I-MR repairs the training-side live utility model and runs one fresh,
leakage-safe promotion gate. It does not extend the solver to neural repair,
multiple trajectories, reinforcement learning, curriculum learning, adaptive
computation, or any other CSG-NI v2 mechanism.

Success requires a single frozen Phase 6I-MR artifact to pass every R11 gate in
Section 12. A competitive final makespan cannot compensate for an unreliable
utility gate. The only final decisions are `PROCEED_FREEZE_V1`,
`MODEL_REVISION`, and `HOLD`.

The machine-readable sources of truth are
`configs/phase6i_mr_live_utility_revision.json` and
`configs/phase6i_mr_command_manifest.json`. This report explains the fixed
choices; it does not create additional alternatives after R10 or R11 access.

Revision 1.2 supersedes the initial one-instance-per-cell draft before any R10
or R11 content was opened. The Revision 1.1 pilot process ended without a
completion marker after 4/9 runs. Its files remain under
`outputs/phase6i_mr/pilot/` as cost/schema diagnostics only; they cannot select
a branch, model, threshold, or gate. Formal Revision 1.2 pilot evidence uses a
new output root and fresh C02 cell replicas.

## 2. Starting-state audit

- The starting worktree was clean and `main` contained Phase 6H commit
  `7c448000be69084511af181f52cad0a8db71479a` in its ancestry.
- The Phase 6F checkpoint was verified directly as
  `f1ccceb607b0e453dfb74e7aa7a946616001db8ec08c12dc9900a66c6f165fc7`.
- The frozen Phase 6H policy was verified directly as
  `4d2da03e13a036569bebf3897135e5da139e292e3260a2f452754d5c9ae3d239`.
- Existing ignored Phase 6H evidence was preserved. No Phase 6H job was still
  running.
- The pre-edit test command covering Phase 6F, Phase 6G, Phase 6H, decoder, and
  feasibility checks passed: `19 passed in 1.23s`.
- At the Revision 1.2 boundary, the expanded regression set including common
  search/trace and the newly added Phase 6I tests passed: `31 passed in 1.38s`.
- R08 CAL-HOLDOUT, Phase 6G DEV-HOLDOUT, CB1-Core, CB1-Sensitivity,
  Legacy-130, and final/paper splits are locked. Phase 6H aggregates may be
  cited, but their cases and labels may not guide this revision.
- Existing exact evidence (`tiny_01=157`, `tiny_03=36`, and 6/6 redundant
  optimum recovery) is reused. Phase 6I-MR will not invoke Gurobi or tiny exact
  validation because utility-only changes do not alter schedule feasibility
  semantics.

## 3. Frozen solver semantics and candidate-bank clarification

H1, CSG-1.0, the decoder, the independent feasibility checker,
transport-aware repair, simulated-annealing acceptance, destroy fraction
0.15, the `2N` production budget, RNG separation, ALNS settings, and raw
monotone incumbent tracing remain frozen.

The execution brief describes an unchanged eight-candidate bank, while the
frozen repository implementation constructs 24 deterministic proposal rules
and then deduplicates identical target sets. In the implementation,
`candidate_trials=8` controls transport-aware repair/decoder trials per target;
it is not the number of proposal arms. Phase 6I-MR therefore preserves the
actual frozen 24-rule/deduplicated proposal-bank semantics and the eight repair
trials. Reinterpreting the parameter as an eight-arm bank would change the
Phase 6H solver and invalidate the comparison.

Revision 1.2's requested all-eight audit is therefore implemented as a fixed
audit layer: retain the four broad roles and append highest frozen-score unseen
target sets until eight unique targets are present. In addition, a smaller
true-full-bank subset evaluates every unique target from the frozen 24-rule
generator. Results separately report 4→8 and 4/8→actual-full truncation. This is
strictly diagnostic and cannot change proposal generation.

If a schedule-construction correctness defect is found, experimental work
stops. The defect must first be reproduced by a minimal failing test, fixed
surgically, and recorded as a new evidence boundary.

## 4. Fresh split suite and access lock

The suite root is
`instances/controlled/RCIAS-CB1-TRAIN-LIVE-R09R11/`. Each split contains the
nine `S/M/L × CF1/CF2/CF3` cells with fixed `RI2/TI2`, with two independent
cell replicas (`C01`, `C02`) per cell: 18 instances per split and 54 total.
The minimum-compliant size is frozen instead of the preferred 27 because the
single-GPU, five-seed R11 design is already compute-intensive. It cannot be
reduced after R10 access.

| Replicate | Role | Permitted use |
| --- | --- | --- |
| R09 | `LIVE_REV_FIT` | pilot, diagnostics, fitting, normalization, calibration, thresholds, and one hard-state round |
| R10 | `LIVE_REV_SELECT` | exactly one frozen candidate/threshold selection pass; no refit |
| R11 | `LIVE_REV_HOLDOUT` | exactly one final promotion gate after artifact freeze |

Instance generation uses base namespace `680000`. Content hashes must be
pairwise disjoint across R09/R10/R11 and disjoint from prior controlled suites.
Before the selected artifact is frozen, R11 access is limited to manifest IDs
and content hashes. A guarded loader must reject R11 JSON access without a
valid frozen-artifact hash. The access ledger and guard are tested before any
R11 evaluation.

The new namespaces are disjoint from the Phase 6H `671xxx` and Phase 6G
`670xxx` experiment seeds:

- pilot trajectory: `681101`;
- R09 full trajectories: `681201..681203`;
- R10 full trajectories: `681301..681303`;
- R11 final gate: `681401..681405`;
- R10 solver-translation sanity: `681501`, `681502`;
- state/role/repair/continuation namespaces: `682000000`, `683000000`,
  `684000000`, `685000000`;
- continuation CRN seeds: `685101`, `685102`;
- model seeds: `686101..686103`;
- hard-state aggregation namespace: `687000000`;
- grouped bootstrap seed: `688001`.

## 5. Required R09 pilot before full work

The formal Revision 1.2 first stage is a nine-instance R09 C02 pilot, one
instance per structural cell, using the frozen Phase 6H trajectory policy and
a diagnostic budget of `0.25N` seconds. Six
states per trajectory are selected nearest normalized progress
`0.10, 0.25, 0.45, 0.60, 0.75, 0.90`, giving two states each in early, middle,
and late search. The expected pilot size is 54 states and at most 216 forced
candidate rows.

Snapshots are captured during the source trajectory, but forced candidates
are chosen and decoded only after that trajectory ends, using isolated
diagnostic RNG. Consequently, a forced outcome cannot alter trajectory
acceptance, timing, proposals, or future states.

At every sampled state, four unique target sets are selected without outcome
information:

1. frozen-score top 1;
2. frozen-score top 2;
3. the first unseen ALNS-related candidate under the fixed origin priority;
4. the remaining candidate maximizing minimum Jaccard distance from the first
   three.

Score ties use `target_set_id`; the diverse role then uses score and ID as tie
breaks. A duplicate is replaced by the first unseen target under the same
fixed ordering. If fewer than four unique target sets exist, the shortfall is
recorded rather than synthesizing a new arm.

Immediate utility is
`(current_makespan - decoded_makespan) / current_makespan`, so positive values
are improvements. Labels, ranks, regret, feasibility, fallback identity,
graph/scale context, support features, and component timings are written only
after common frozen decoding. Compact top-error cases are stored without
dumping complete graph tensors into logs.

The failure analysis reports grouped counts, rates, regret/utility loss, and
instance-bootstrap intervals for:

- `WITHIN_STATE_INVERSION`;
- `CROSS_STATE_MISCALIBRATION`;
- `SIGN_ERROR`;
- `SCALE_DRIFT`;
- `SEARCH_STAGE_DRIFT`;
- `CANDIDATE_SOURCE_BIAS`;
- `ONE_STEP_VALUE_MISMATCH`;
- `REPRESENTATION_LIMIT`;
- `LOW_SUPPORT_EXTRAPOLATION`;
- `GATE_SELECTION_BIAS`.

Full R09/R10 collection and broad training cannot start until this pilot,
failure table, scale audit, and continuation branch decision are complete.

The pilot also verifies the audit implementation. In formal R09 and R10, the
top-eight subset uses one predetermined state per instance (18 per split), with
six states each at progress 0.15/0.50/0.85. The four roles are always retained
and score-ranked unseen targets fill positions five through eight. A separate
true-full-bank subset uses one C02 state in each of nine cells; CF1/CF2/CF3 map
to progress 0.15/0.50/0.85. It evaluates every deduplicated arm. These labels
are diagnostic only and cannot redesign the bank or add model families.

## 6. Scale audit

Every numeric model/gate input is classified as `STATIC_ABSOLUTE`,
`DYNAMIC_ABSOLUTE`, `RATIO`, `STANDARDIZED`, or `CATEGORICAL`, with its growth
drivers recorded. Robust centering, clipping, and scale constants are fitted
on R09 only.

U2 context contains log/robust-standardized operation, node, and edge counts;
edge/node, DAG depth/operation, DAG width/operation, eligibility density, and
resource-load CV; makespan and critical-path ratios to H1; W/F/reconfiguration
delay ratios; the existing five structural support features; and normalized
search progress. Raw and normalized values remain separately auditable.

The current local-search representation always holds a complete schedule, so
its literal remaining-operation count is zero. It is recorded for diagnostic
completeness but excluded from U2 rather than treated as informative. No
candidate-set normalization is allowed to erase cross-state scale information.

## 7. One-step versus continuation diagnostic

The diagnostic uses 27 R09 pilot states: the snapshot nearest progress 0.15,
0.50, and 0.85 in each of the nine cells. The four fixed candidate roles are
evaluated with two common seeds and exactly 12 frozen ALNS continuation
iterations. Each candidate starts from its decoded schedule; candidate roles
within a state share the same continuation random streams.

Continuation value is
`(candidate_start_makespan - best_makespan_after_12_iterations) /
candidate_start_makespan`. The branch is immutable:

- `IMMEDIATE_TARGET_VALID` only if median within-state Spearman is at least
  0.30 and top-1 agreement is at least 50%;
- otherwise `TARGET_MISMATCH`.

`TARGET_MISMATCH` activates U2-H. U2-H retains the immediate head and adds a
separate continuation-value head; it never relabels one target as the other.

## 8. Full collection and training candidates

R09 and R10 each use three trajectories per instance and 30 sampled states per
trajectory, balanced as ten early, ten middle, and ten late states. Across 18
instances this gives an expected 1,620 states and at most 6,480 broad forced
actions per split. R10 outcomes
are generated and evaluated once only after every model artifact, calibration,
threshold grid, and lexicographic rule has been frozen.

U1 and U2 each have explicit `R09_ONLY` and `MIXED_OLD_NEW` data variants. Mixed
minibatches contain 25% earlier Phase 6F TRAIN rows and 75% R09 live rows,
irrespective of raw dataset cardinality. Hard-state refits use 20% old, 60% R09
base, and 20% one-round hard rows. Sampling balances instances, scales, CF
groups, search stage, and candidate role. Mixed is ineligible as negative
transfer if its three-seed mean Spearman is more than 0.02 below R09-only, its
worst seed is more than 0.05 below, or it creates a scale sign reversal. All
normalization, clipping, temperatures, and class weights are R09/training-side
only.

The fixed candidates are:

- `U0`: unchanged Phase 6F reference;
- `U1_BASE` and `U1_ONE_ROUND_HARD_AGG`: frozen state/action encoders and score
  head, retrained immediate-utility head;
- `U2_BASE` and `U2_ONE_ROUND_HARD_AGG`: the same frozen encoder/score plus a
  compact 32-hidden-unit regime-conditioned utility head;
- U2-H base/aggregation variants only under `TARGET_MISMATCH`;
- `U3_ONE_ROUND_HARD_AGG` only if the best grouped-OOF U1/U2 R09 overall
  per-instance Spearman is below 0.20 or any scale mean is negative.

U3 may train only the last state relation block, final action projection, and
utility head. Its caps are 5.35M total and 2.60M trainable parameters, p90
neural-decision latency 30 ms, and p90 total decision latency 100 ms. The score
head remains frozen and is protected by fixed-reference score stability.

All trained candidates use seeds `686101..686103`. The objective combines
pairwise gap-scaled clipped margin loss (weight 1.0), ListNet loss (0.5), Huber
regression (0.5), and positive-sign consistency (0.1). Gap weights are clipped
to `[0.25,4]`, margins to `[0.05,1]`, and cross-state utility is divided by the
R09 p99 absolute utility then clipped to `[-1,1]`. Huber delta is 0.1. The
ListNet temperature is the R09 median positive within-state utility standard
deviation with a 0.001 floor. Per-state standardization is confined to ranking
losses.

Every activated family/data variant is trained independently with all three
seeds under identical data and budgets. The deployable family artifact is not
a lucky best seed: it shares the frozen encoder and averages the three
seed-specific normalized utility-head outputs. R10 eligibility requires
positive mean overall Spearman, non-negative worst-seed overall Spearman,
non-negative mean Spearman at every scale, at least two of three non-negative
seeds per scale, and no scale/seed Spearman below -0.10. Selection prioritizes
worst seed, then mean, then lower seed variance.

Contrastive loss is not a default. It activates only for U3 if a grouped-OOF
frozen-embedding probe has pairwise accuracy below 0.55 while the raw
normalized-context probe reaches at least 0.60. The weight is 0.05 and a
no-contrastive U3 ablation is mandatory. It cannot be activated after R10.

## 9. One leakage-safe hard-state round

Exactly one R09-only grouped out-of-fold aggregation round is allowed. The
fixed cell folds are:

- fold 0: `S_CF1`, `M_CF2`, `L_CF3`;
- fold 1: `S_CF2`, `M_CF3`, `L_CF1`;
- fold 2: `S_CF3`, `M_CF1`, `L_CF2`.

The deterministic priority union contains top-decile regret, sign errors,
neural/fallback disagreement, absolute scale residual z-score at least 2, and
low-support states. It is capped at 20 states per instance and 360 total, with
priority count, regret, and state ID as tie breaks. Its four-candidate labels
use the same post-trajectory rules. No R10 or R11 row may enter this round, and
there is no second mining pass.

## 10. Live policy, calibration, and one-time R10 selection

For U1/U2 the score head is bitwise frozen, so the Phase 6H Platt probability
mapping is reused. Any candidate with an adapted score representation would
require a new R09-only mapping. Probability thresholds are
`0.04, 0.06, 0.08, 0.10`; immediate-utility thresholds are
`-0.01, 0, 0.0025, 0.005, 0.01`; continuation thresholds, if U2-H activates,
are `0, 0.0025, 0.005, 0.01`.

The fixed immediate policy first filters by probability and R09-defined support,
then selects maximum revised utility, breaking ties by frozen raw score and
target-set ID, and finally applies the utility threshold. Failure at any stage
uses the frozen ALNS-related fallback. U2-H instead ranks eligible arms by its
separate continuation head and additionally requires predicted immediate
utility to be non-negative.

R09 threshold candidates must have at least 100 selected actions per scale for
direct reliability estimation and at most 10% unsupported interventions. There
is no mandatory intervention-percentage range: lower coverage is admissible
when forced evaluation supports abstention and the scale has no solver-quality
collapse. R10 applies the following lexicographic rule exactly once:

1. no schema, replay, or feasibility failure;
2. positive aggregate per-instance utility Spearman;
3. no negative scale mean Spearman;
4. no unresolved scale sign-error or cross-state calibration collapse;
5. highest selected immediate utility relative to fallback, or the separately
   preregistered U2-H continuation value;
6. lowest selected regret, then pair inversion rate;
7. highest NDCG@1, then NDCG@2;
8. best training-seed stability by worst seed, mean, then variance;
9. adequate support and non-pathological intervention coverage;
10. lowest p90 neural-decision latency;
11. smallest model, then lexicographically smallest model ID.

R10 does not refit weights, calibration, normalization, or thresholds. No new
candidate family may be added after R10 inspection.

After offline selection is immutable, R10 receives one solver-translation
sanity check on the nine C02 instances. It uses H1 plus ALNS and revised CSG-NI
with matched seeds `681501,681502` and a `1N` budget. It passes only with 100%
feasibility, aggregate revised-vs-ALNS makespan no worse than +1%, no scale mean
worse than +3%, no instance worse than +7.5%, and complete intervention/fallback
replay integrity. Failure is terminal `MODEL_REVISION`: no R10 retuning and no
R11 access.

## 11. Artifact freeze and R11 protocol

The selected artifact must contain weights/configuration, tensor-schema hash,
training/manifest hashes, calibration and thresholds, support/context
definition, code/config hashes, deterministic replay, and the split-access
ledger. Only its immutable recorded hash unlocks R11 JSON.

R11 uses five matched seeds for `H1`, frozen ALNS, Phase 6H CSG-NI
(descriptive continuity only), and Phase 6I-MR CSG-NI. Iterative methods receive
the same `2N` wall-clock budget, one CPU thread, and one controlled sequential
GPU worker for CSG methods. GA, DCGA, Gurobi, Core, Sensitivity, and Legacy-130
are not run.

The primary paired unit is the instance mean over five seeds. Bootstrap
resampling groups by instance with 10,000 replicates. The exact one-sided
Wilcoxon test uses 18 paired instance means and reports statistic,
direction, zero handling, and p-value. Timing includes proposal, CSG, neural,
calibration/gate, repair, decoder, and total components; only one-time checkpoint
loading is consistently separated.

For coverage diagnostics, ten fixed-progress states per completed R11 run are
evaluated post-trajectory with the same outcome-blind four roles. They cannot
alter the production trajectory.

## 12. Frozen R11 gates and decision logic

`PROCEED_FREEZE_V1` requires all of the following:

1. 100% accepted-schedule feasibility, complete monotone traces, and integrity;
2. no leakage and no pre-freeze R11 content access;
3. overall per-instance utility Spearman strictly positive; 0.20 or higher is
   the frozen preferred target;
4. mean utility Spearman non-negative at every scale;
5. selected immediate utility positive and above the frozen fallback; if U2-H
   activates, continuation Spearman is also at least 0.20 overall, non-negative
   at every scale, and selected continuation value is positive and above its
   fallback;
6. direct reliability uses at least 200 selected actions per scale and no more
   than 10% unsupported interventions. If selected count is below 200, the
   scale needs at least 120 forced-evaluated predicted-abstained actions, mean
   best forced utility non-positive or no lift over fallback, grouped-bootstrap
   upper bound at most 0.0025, and solver non-inferiority to ALNS;
7. cross-scale coverage ratio should not exceed 8. Above 8 passes only with the
   same abstention evidence and no scale-quality collapse; no universal 5–60%
   intervention range is mandatory;
8. probability ECE at most 0.10 and Brier/NLL no more than 10% relatively worse
   than the Phase 6H references 0.076154/0.288516;
9. aggregate mean makespan non-inferior to ALNS within a 1% relative margin;
   at least 0.5% improvement is preferred;
10. no catastrophic collapse, defined as any scale or CF-group mean more than
    2% worse than ALNS or any instance mean more than 5% worse;
11. at least one strong search-efficiency advantage without final-quality loss;
    preferred evidence is decoder evaluations at least 25% below ALNS and/or
    normalized-gap AUC at least 10% better;
12. wall-clock, time/evaluations to targets and final best, AUC, decoder count,
    and proposal/CSG/neural/gate/repair/decoder timings are all reported under
    the declared CPU/GPU protocol. No hardware-independent speed claim is made.

A strictly positive instance-bootstrap lower bound and one-sided exact
Wilcoxon `p<0.05` are preferred evidence, not substitutes for any gate.

- `PROCEED_FREEZE_V1`: every required gate passes; label the artifact
  `CSG-NI v1`.
- `MODEL_REVISION`: correctness and solver non-inferiority hold, but at least
  one utility, probability, drift, coverage, or efficiency gate fails.
- `HOLD`: correctness, leakage integrity, feasibility, or aggregate solver
  stability fails materially.

No gate may be weakened after R11 is unlocked.

## 13. Execution checkpoints

Work proceeds in this order and stops at each long-job boundary for evidence
verification:

1. extend manifests to 18 instances per split, verify cross-suite disjointness,
   and retest the R11 lock;
2. run the formal nine-cell C02 R09 pilot, failure/scale/full-bank diagnostics,
   and continuation branch;
3. freeze the diagnostic branch and full-collection protocol;
4. collect 18-instance R09 and R10 in resumable detached jobs;
5. train R09-only/mixed candidates across three seeds and perform one R09-only
   hard-state round;
6. evaluate R10 once, select by seed-robust lexicographic rules, and run the
   one-time solver-translation sanity check;
7. only on translation pass, freeze one artifact and hash, then unlock/run R11;
8. finalize grouped gates, full tests, report, and handoff.

Every detached launch records its command, configuration, PID/status file, log,
output root, completeness count, throughput, ETA, and resume check. After a
healthy launch is established, interactive polling stops. This repository will
not be pushed by Codex.
