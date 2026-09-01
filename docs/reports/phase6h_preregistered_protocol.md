# Phase 6H Preregistered Live-Calibration Protocol

This protocol was frozen before collecting any Phase 6H live outcome labels. The
starting repository commit is `87ef6837c13dcad034ecd5062b76eeb3ef836872`.

## Scope and frozen boundary

Phase 6H may change only post-hoc calibration and an auditable intervention
gate. The Phase 6F checkpoint, H1, the decoder/checker, ALNS/GA/DCGA settings,
CSG-1.0, proposal construction, transport-aware repair, simulated-annealing
acceptance, and the Phase6G RNG namespace/isolation rules remain frozen.
CB1-Core, CB1-Sensitivity,
Legacy-130, DEV-HOLDOUT threshold tuning, model retraining, online RL, and CSG-NI
v2 architecture work are prohibited.

## Independent data boundary

`RCIAS-CB1-CAL` contains two newly generated training-distribution replicates.
R07 is `CAL-FIT`; R08 is `CAL-HOLDOUT`. Each contains exactly one instance in
every Scale x CF cell (3 x 3), with RI2/TI2 fixed to the unscaled core setting.
Generation IDs, hashes, and seed namespaces must be disjoint from every prior
training, development, and final-evaluation suite.

R07 may be used for calibration fitting, threshold selection, and the solver
gate study. R08 may be opened once, only after the final calibrator and gate
artifact has been serialized and hashed. It is never used to fit or select a
mapping or threshold.

## Live label collection

CAL-FIT collection runs the frozen top-1 action at every R100-eligible state so
that each selected action receives an outcome label. Proposal creation remains
outcome-blind; realized utility is computed only after common-decoder evaluation.
This collection policy is diagnostic and is not a deployment candidate.

## Selection rules

Probability calibration compares Phase6G, sigmoid identity, Platt, isotonic,
and beta calibration with instance-grouped three-fold cross-validation. Mean
NLL is primary, followed by Brier score, ECE, and method simplicity. Utility
calibration compares the Phase6G mapping with isotonic regression using grouped
CV MAE/RMSE, while forbidding more than 0.02 Spearman degradation.

Four preregistered solver gates are evaluated on CAL-FIT only: Phase6G
reference, calibrated probability, calibrated probability plus utility, and the
same gate with a deterministic empirical-support guard. Offline threshold grids
and coverage constraints are fixed in
`configs/phase6h_live_calibration.json`. Among feasible policies, the primary
solver selection criterion is the lowest mean of per-instance mean final
makespan; decoder evaluations, decision overhead, and name order are frozen
tie-breakers.

## Unbiased gate and anytime analysis

After the selected artifact is frozen, CAL-HOLDOUT compares H1, ALNS, GA, the
existing Adapted DCGA, Phase6G CSG-NI, and Phase6H CSG-NI under `2N` seconds.
Final quality is primary. Every iterative run retains its full incumbent trace,
first time/evaluation of final best, normalized-budget checkpoints, common-target
hitting times with explicit censoring, and normalized-gap AUC.

As in Phase 6G, the frozen model is loaded once and remains resident in each GPU
worker. One-time checkpoint loading is reported separately and is not charged to
every run; all per-decision CSG, transfer, inference, scoring, calibration, and
guard overhead remains inside the common `2N` wall-clock budget.

Acceptance requires 100% Phase6H CSG-NI feasibility, ECE <= 0.10, improved
Brier and NLL versus the Phase6G reference, no material utility-ranking loss,
aggregate non-inferiority to ALNS, and no catastrophic subgroup collapse. High
measured drift is not relabelled; it passes only if the frozen mitigation remains
calibrated and solver-stable on CAL-HOLDOUT.

The final decision is exactly one of `PROCEED`, `MODEL_REVISION`, or `HOLD`.
Final evaluation sets remain locked in every case.
