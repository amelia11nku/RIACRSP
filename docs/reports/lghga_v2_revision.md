# LG_HGA-RIACRSP v2 protocol amendment

## Decision

The original frozen implementation remains archived as `LG_HGA-RIACRSP` v1. It is not overwritten. The v2 method is named `LG_HGA-RIACRSP-v2-N4M` and uses separate source files, configuration, knowledge artifacts, models, and Core outputs.

This amendment was frozen before any LG_HGA Core result existed. At the amendment point, the stopped three-seed matrix contained 120 feasible DABC results and zero LG_HGA results. No Core objective was used to choose the v2 mechanics.

## Changes from v1

1. N4 uses the minimum plural interpretation of the source wording “several process stages”: it attempts two effective critical-operation moves. If only one effective move exists, it uses one; if none exists, it records a no-op.
2. Each move still targets an eligible island with the minimum current operation count. Counts are updated after the first move. Processing-time load is not substituted for the paper's explicit operation-count rule.
3. N4 excludes the current island from its target set, so a reported effective move changes the island chromosome. Knowledge and online diagnostics record changed and no-op proposal counts.
4. The knowledge target remains the one-step Eq. (11)-style percentage of newly generated individuals whose makespan strictly improves on the source individual. The single-objective RIACRSP adaptation remains `U=Cmax`.
5. DTRs remain generation-only decision-tree regressors, but are frozen separately for each pre-existing `scale × CF` structural regime. This avoids averaging heterogeneous trajectories into a single gate while introducing no Core outcome features.

## Intentionally unchanged

- `Tls=50` and strict `predicted_R > Tls`;
- `MAXGEN=100`, `NIND=40`, `Pc=0.9`, and `Pm=0.4`;
- `lsize=5`, `MaxIterNum=5`, and `nsize=20`;
- common decoder, makespan objective, `2N` wall-clock Core budget, formal instances, and seeds;
- v1 N1--N3 definitions.

The source threshold levels `{35, 40, 45, 50}` may later be reported as a separately named sensitivity analysis. They must not replace the T50 primary result after Core outcomes are observed.

## Initial smoke evidence

The preregistered one-instance, one-seed, one-generation smoke completed 160 decoder evaluations and returned a feasible schedule. N4 generated 20 proposals, of which 16 changed the chromosome, and its immediate improvement rate was 70%. On the same v1 knowledge key, N4's improvement rate was 40%. This is an implementation diagnostic, not a Core performance comparison and not evidence for selecting a threshold.

## Execution gate

The complete v2 knowledge matrix remains 9 training-only instances × 20 seeds = 180 runs. Training must refuse incomplete data, mixed implementation hashes, infeasible knowledge runs, or any training/Core ID or content-hash overlap. It creates nine `scale × CF` bundles, each containing four DTRs. Core execution remains blocked until that freeze exists.
