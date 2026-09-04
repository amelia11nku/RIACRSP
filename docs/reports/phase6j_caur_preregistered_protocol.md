# Phase 6J CAUR Preregistered Protocol

```text
phase = Phase 6J
family = phase6j_caur
protocol_revision = 1.0
status = PREREGISTERED_BEFORE_R12_PILOT_OR_R13_R14_CONTENT_ACCESS
starting_commit = a6ef53df2d18fbf31320b49f5a5859ab0bddb62a
phase6i_mr_handoff_commit_in_ancestry = 6a8ac523adc2186c7cb359c2902c207d90184b5b
config_sha256 = 399f03e33fad4567b8ffade5d632b2c6f32da4a9f3a51c3fea0ceb87e4da03d5
command_manifest_sha256 = 46a0ce301275e89b6a4e5754a47ff9f46d3abf16e1e4cb8fe5c25884ad0aa551
instance_manifest_sha256 = dc06cc3b49e0724bf3bc54255ba95f6a8b29ca07d383f96bcd01f0f6d8f17619
r12_pilot_accessed = false
r13_content_accessed = false
r14_content_accessed = false
```

## 1. Scientific boundary

Phase 6I-MR ended with the immutable decision `MODEL_REVISION`. Its selected
artifact is not CSG-NI v1 and cannot replace the Phase 6H manuscript or
production reference. Phase 6J is a v1 utility/ranking and gate revision. It
does not introduce PPO, actor-critic training, learned repair, multiple starts,
curriculum learning, adaptive computation, or another decoder.

The frozen problem, H1 initializer, deterministic decoder, independent
feasibility checker, transport-aware repair, simulated-annealing acceptance,
destroy fraction, `2N` production budget, RNG separation, ALNS baseline,
Phase 6H reference, monotone incumbent trace, and schedule-feasibility
semantics remain unchanged.

The six Phase 6I-MR evidence files are recorded in
`configs/phase6j_caur_phase6i_mr_evidence_manifest.json`. The setup audit may
verify their bytes. Phase 6J feature, fit, calibration, threshold, selection,
and holdout code may read only that manifest and must never open R11 payloads.
R11 values are historical motivation, not training or selection data.

## 2. Starting-state audit

- Initial branch: `main`; worktree: clean.
- Initial HEAD: `a6ef53df2d18fbf31320b49f5a5859ab0bddb62a`.
- Phase 6I-MR handoff commit is in the current ancestry.
- Python 3.11.15 and PyTorch 2.11.0+cu128 were recorded.
- Host-level verification found an NVIDIA GeForce RTX 4060 Ti with 8188 MiB,
  driver 575.64.03, and a successful `gnn311` CUDA tensor operation. The
  restricted command sandbox cannot see the driver, so GPU jobs require the
  verified host execution boundary.
- CPU: Intel Core i7-14700, 28 logical CPUs visible.
- Filesystem usage was 94%, with approximately 57 GiB available.
- Pre-edit regression: `250 passed in 12.88s`.
- All six historical evidence hashes and sizes passed the setup audit.

## 3. Fresh split and access contract

The suite root is
`instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14/`. R12, R13, and R14 each
contain 18 instances: the nine `S/M/L × CF1/CF2/CF3` cells with `RI2/TI2` and
two independent cell replicas. The generation namespace begins at `690000`;
experimental, continuation, model, and bootstrap namespaces occupy new
`691xxx`–`697xxx` or corresponding nine-digit ranges and do not reuse the
Phase 6I-MR `680xxx`–`688xxx` boundary.

The generated suite passed all ID, content-hash, generation-seed, split-hash,
cell-count, and historical-disjointness checks. R14 was not parsed for
structural metrics; before artifact freeze its access scope is manifest ID,
hash, and size only.

- R12 `CAUR_FIT`: diagnostics, pilot, fitting, normalization, grouped OOF,
  selection-aware calibration, support fitting, and threshold design.
- R13 `CAUR_SELECT`: one resumable selection pass after all eligible family
  artifacts, normalization, calibrators, threshold grid, latency caps, and
  lexicographic logic are frozen. No refit.
- R14 `CAUR_HOLDOUT`: one resumable promotion pass after one deployable
  artifact and all dependencies are hashed.

R13 and R14 use artifact-hash locks plus a single access ledger. The same run
ID may resume an active pass; another run cannot open it, and a completed
ledger cannot be reopened.

## 4. Candidate and label contract

The production bank remains the 24 deterministic proposal rules followed by
target-set deduplication. `candidate_trials=8` means eight transport-aware
repair/decoder trials per target. It never means eight proposal candidates.
Training uses every unique target in each sampled state. `REDUCED_TOP8_AUDIT_ONLY`
is an explicitly diagnostic scope and cannot satisfy full-bank completeness.

For state `s`, candidate `a`, frozen ALNS-related fallback `f`, continuation
seed `r`, and horizon `H`:

`A_H(s,a,r) = (V_H(s,f,r) - V_H(s,a,r)) / C_incumbent(s)`.

Candidate and fallback use the same derived continuation RNG stream. The
primary label is the arithmetic mean of `A_H` over seeds `695101` and `695102`.
Positive is better. Immediate utility remains an auxiliary regression and
bounded-harm signal only.

Outcome-blind candidate inputs are frozen as:

- primary origin rule, destroy operator, and origin family;
- origin rule/family counts and target cardinality/fraction;
- overlap and Jaccard features against the fallback;
- overlap with critical and bottleneck operations;
- Jaccard against the best frozen-score arm;
- normalized frozen-score rank and candidate-diversity rank;
- fallback indicator.

Continuation outcomes, immediate outcomes, and decoded candidate makespan are
label-only and cannot enter these features.

## 5. R12 horizon pilot

The pilot uses the nine R12 C02 instances, one per structural cell, a frozen
Phase 6H source trajectory with `0.25N` diagnostic budget, and snapshots at
progress 0.15, 0.50, and 0.85. It therefore contains exactly 27 state groups.
Every deduplicated candidate is decoded and evaluated under both CRN seeds.

One frozen continuation run records prefixes at H=4, H=8, and H=12. This
prefix implementation must match the existing frozen H=12 continuation result
for candidate, makespan, value, derived seed, evaluation count, accepted and
improving moves, and operator counts.

Choose the shortest of H=4 then H=8 satisfying all conditions against H=12:

- median within-state Spearman at least 0.70;
- mean NDCG@1 at least 0.80;
- top-1 agreement at least 60%;
- non-negative mean Spearman in S, M, and L.

Otherwise choose H=12. Solver performance cannot select the horizon.

Planned pilot cost is 27 states, approximately 629.1 and at most 648 unique
candidate rows, two CRN seeds, eight repair trials per target, and prefix
continuation through H=12. Expected total decoder evaluations are 125,820;
the exact hard maximum is 129,600. The historical local timing estimate is
20–35 minutes end-to-end. It must be replaced by an observed throughput/ETA
after the first completed R12 state.

## 6. R12 collection, models, and OOF calibration

If the pilot freezes a horizon, full R12 uses 18 instances, two source
trajectories per instance, and eight fixed-progress states per trajectory:
288 full-bank state lists. If the pre-collection cost cap is breached, only
the config-defined fallback may activate: the first trajectory remains full
bank and the second uses deterministic stratified sampling with recorded
inclusion probabilities.

Grouped OOF folds hold out whole instances and cells:

- fold 0: S_CF1, M_CF2, L_CF3;
- fold 1: S_CF2, M_CF3, L_CF1;
- fold 2: S_CF3, M_CF1, L_CF2.

The fixed families are J0 Phase 6H reference, J1 with frozen encoder and new
context/three heads, J2 with only the last state block and final action
projection additionally trainable, and conditional J3 relational interaction.
J3 activates only if both regular families remain below 0.60 pairwise accuracy
or any scale has negative mean continuation Spearman. Every activated family
uses seeds 696101–696103 and is selected as a family ensemble, never a lucky
single seed.

J1 is capped at 5.35M total and 0.50M trainable parameters. J2/J3 are capped
at 5.35M total and 2.60M trainable parameters. Training uses AdamW with
family learning rates 3e-4/1e-4/1e-4, weight decay 1e-4, eight whole-state
groups per batch, gradient norm 1.0, deterministic Torch algorithms, at most
120 epochs, and patience 12 on the R12 grouped-OOF selected-lift lower bound.

Loss weights are pairwise logistic 1.0, ListNet 0.75, continuation Huber 0.5,
beats-fallback BCE 0.25, and immediate-utility Huber 0.10. R12-only median/IQR
normalization is clipped to robust z in [-8,8]. Within-state standardization is
limited to ranking losses; the deployed advantage head keeps cross-state
magnitude.

Calibration rows are only the argmax-selected winners from held-out groups.
Candidate methods are Platt and, with at least 200 selected rows, isotonic.
Selection uses ECE, then Brier score, then Platt simplicity.

The frozen gate chooses maximum ensemble mean advantage with target ID tie
break and intervenes only when calibrated probability, LCB, support, and
immediate-harm conditions all pass. The grid contains 18 combinations of
`p_min={0.55,0.65,0.75}`, `lambda={0.5,1.0}`, and
`delta_min={0,0.0025,0.005}`. The immediate-harm floor is -0.005. There is no
intervention-rate target. A scale requires at least 20 direct interventions,
or at least 40 forced-abstention outcomes with a non-positive 95% upper bound
on their lift. Gate selection first enforces positive overall lift and lower
bound plus non-negative scale lift, then maximizes lift lower bound, minimizes
regret and ECE, maximizes supported count, and uses the frozen lexical
threshold tie break.

The full R12 collection cost fallback activates only before collection when
projected decoder evaluations exceed 1.5M or the pilot-throughput ETA exceeds
8 hours. It cannot be activated after outcome inspection.

R13 remains locked unless every essential R12 gate in the config passes,
including positive selected lift and grouped-bootstrap lower bound, no
negative scale Spearman, ECE at most 0.10, complete replay/feasibility and
full-bank labels, and no origin collapse.

## 7. R13 and R14 frozen decisions

R13 labels 144 full-bank states: one trajectory and eight fixed-progress
states on each of 18 instances. Its immutable selection is lexicographic in
integrity, ranking sign, selected lift, scale coverage, selected-lift lower
bound, regret, NDCG, seed stability, calibration, latency, model size, and
model ID. No R13 refit is permitted.

The R13 `1N` translation uses nine C02 cell representatives and matched seeds
691501/691502 for H1, ALNS, Phase 6H, and Phase 6J. It requires 100%
feasibility/replay, at most +0.5% aggregate regression versus ALNS, and at most
+0.25% versus Phase 6H. Failure returns `MODEL_REVISION` without R14 access.

R14 contains 288 runs: H1 once per instance and five matched seeds for ALNS,
Phase 6H, and Phase 6J at `2N`. No Gurobi, GA, DCGA, Core45, Sensitivity, or
Legacy-130 run enters this gate. Full-bank ranking diagnostics use five fixed
states from only the first matched Phase 6J run per instance.

Promotion requires every mandatory gate in `configs/phase6j_caur.json`,
including 100% feasibility, complete replay/leakage integrity, positive overall
and non-negative per-scale continuation ranking, positive selected lift,
acceptable selected-winner ECE, no unsupported coverage collapse, at most
+0.25% aggregate degradation versus ALNS and Phase 6H, no catastrophic group
collapse, and complete wall-clock/decoder accounting. At least one efficiency
advantage must hold without final-quality loss.

Only `PROCEED_FREEZE_V1`, `MODEL_REVISION`, or `HOLD` may be emitted. Gates
cannot be weakened after R14 access.

## 8. Runtime, outputs, and stopping

Long jobs must be persistent, resumable, single-worker launches with a live
PID, advancing log, output growth, measured throughput, and ETA. Interactive
polling stops after a healthy launch. Completion requires a completion marker,
expected counts, parseable outputs, logs without exceptions, and integrity
hashes; elapsed time alone is insufficient.

The exact command shapes and completeness rules are frozen in
`configs/phase6j_caur_command_manifest.json`. Stages after bounded setup remain
implementation-gated until their code and stage-specific tests pass. A command
being listed does not authorize bypassing its preceding gate.

Stop before R13 on an essential R12 failure, before R14 on an R13 or translation
failure, and immediately with `HOLD` on leakage, integrity, feasibility, or
solver-stability failure. Do not push or rewrite Git history.
