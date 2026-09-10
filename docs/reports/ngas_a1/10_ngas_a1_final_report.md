# NGAS-A1 stage delivery and continuation boundary

Starting commit: `156b017b1651e803f5a6bd4894e2713cfec1cc9d`.
This is an incremental A1.0–A1.2 delivery, not a completed NGAS solver or a
claim of improved algorithm quality. The current machine-readable decision is
`outputs/ngas_a1/final_decision.json`; the pilot's terminal gate is
`outputs/ngas_a1/label_pilot/gate.json` when available.

## Implemented and verified

- Independent `rcias_ngas` package, eight stable RNG namespaces and deterministic
  repair iteration order; fixed-work decode trajectories reproduce across processes.
- True completed-work budget checkpoints separate from best-event and last-best
  telemetry. The historical derived anytime work-count field is explicitly deprecated
  in the new registry; its original file remains intact.
- 270 imported immutable raw results, an 18-instance BKS v001, and derived RPD
  carrying the BKS version/hash. Historical RPD is reproduced from the same raw scope.
- All-chain CSG critical synchronization analysis using the shared realized event DAG,
  with earliest/latest times, slack, active edge margins, and non-operation projection.
- New full 24-rule-per-size bank, exact-set deduplication, symmetric all-origin
  features, and conservative 8%/15%/22% sizes. All 18 canonical instances have distinct
  cardinalities. The critical rule and its noncritical padding have explicit provenance.
- Joint `(size,target,repair)` IDs, canonical inherited repair execution, balanced
  sampling and paired-CRN continuation labels. Label action identity is checked.
- Frozen, resumable pilot with per-state snapshots, action manifests, self-hashed
  raw labels, source/config/instance checks, exclusive worker lock and a final gate.

Full regression and bank qualification evidence is in
`outputs/ngas_a1/audit/infrastructure_bank_gate.json` and `regression.txt`.
Detailed A1.0/A1.1 audits are reports 00–05. No existing historical source file or
raw result was edited; protected file hashes are checked again before and after pilot.

## Bounded label pilot

The preregistered three-instance subset is S/CF1, M/CF2, L/CF3 (C01), with equal
H1 and four-step native source states. Canonical CRN seeds are 746101–746103.
Each state samples up to 75 joint actions: three sizes, five target rules covering
five families, and every repair. Deduplicated collisions reduce actual count.
Three paired replicates, two repair trials per move and two continuation moves
bound labels at 16,200 decoder calls, plus source construction and bank analysis.

The fallback is versioned uniform native search with independent component streams,
annealing temperature 5% of source makespan and cooling .995 per step. Candidate
and fallback share per-step neighbor/acceptance keys. The initial move follows the
same acceptance rule in both branches. Both best-so-far values include the source
state. Primary advantage is `(fallback_best - action_best) / source_makespan`.
Raw labels preserve repair identity, immediate improvement, mean, variance, beats-
fallback frequency, feasibility, and the complete short trajectory.

Noise-aware pair ordering requires an absolute mean paired advantage difference
above max(.001, twice its paired standard error). Preregistered gates require full
feasibility/coverage, at least three informative states, positive opportunities, and
bounded cost. Diagnostics test repair effects within the same size/target, target
rank reversals across repairs, and size/repair residual interactions.

This design does not estimate scale/CF subgroup quality; scale and CF are confounded.
It does not estimate eventual eight-trial online utility. No trained neural model
exists at this boundary, so critical-sync priority and matched random controls are
used explicitly. Expanded labels and training need a new pre-outcome protocol after
a passing pilot; no large label campaign or training is started automatically.

## Not yet reached

A1.3 joint critic training, A1.4 persistent-prior online integration and its ablations,
A1.5 latency qualification, and A1.6 the three-seed 2|O| solver comparison remain
gated. Reports 07–09 will be created when those stages are actually executed.
There are no neural-latency or neural-quality measurements to report yet.
The intended later architecture is a jointly value-consistent CSG critic over size,
target and repair, with a cached persistent prior and an independent online portfolio.

R13/R14 stay locked. Gurobi was not run. Historical Phase 6H/6N/6P, ALNS and LG_HGA_2O
remain comparators. No push is performed. New source and pre-outcome protocol are
committed locally before the pilot is launched.

## Resume and decision

Read `outputs/ngas_a1/label_pilot/progress.json` and `launch_record.json` first.
Check the recorded PID before resuming. If interrupted, run:

```bash
/home/liulei/miniconda3/envs/gnn311/bin/python -u scripts/run_ngas_label_pilot.py
```

Complete hash-validated labels are skipped. Do not delete or regenerate successful
rows, change frozen code/config, or select favorable reruns. A changed design needs
a separately named protocol/output boundary.

On `NGAS_A1_REVISE_LABELS`, stop before training and diagnose the recorded failed
checks. On `NGAS_A1_LABEL_PILOT_PASS`, preregister expanded R12 DEVELOPMENT labeling
with broader scale/CF/source coverage and instance-level train/validation separation,
then implement the joint critic and multiple training seeds. This pilot decision
alone does not authorize claims of `NGAS_A1_PASS_DEVELOPMENT`.
