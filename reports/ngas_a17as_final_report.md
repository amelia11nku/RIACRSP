# NGAS A1.7A-S supplemental diagnostic closure

## Terminal classification

`NGAS_A1_7AS_PASS_SUPPLEMENTAL_CLOSURE`

A1.7A-S is an audit-only development closure over 27 frozen clean non-R12
trajectory states. It does not establish final generalization and does not change
the production algorithm.

## Required conclusions

1. **Clean critic ranking remains weak across search stages.** For U0, valid-state
   mean Spearman rho is 0.2516 in EARLY, 0.0643 in MIDDLE, and 0.0753 in LATE.
   The critic Top-1 hits the best positive-U0 action in 0/8, 0/4, and 0/4
   informative states, respectively.
2. **U0 support is sparse and stage dependent.** U0 is informative in 16/27
   states overall: S 4/9, M 6/9, L 6/9; EARLY 8/9, MIDDLE 4/9, LATE 4/9.
3. **U1 recovers limited support absent in U0.** Among 11 U0-constant states, U1
   becomes nonconstant in 2/11 and has a positive best action in the same 2/11.
   The recovery occurs in one EARLY and one MIDDLE state; none occurs in LATE.
4. **U3 exposes robustness variation whenever U0 is flat.** U3 is nonconstant in
   11/11 U0-constant states and in all 27 states overall. This is ranking signal,
   while only 1/27 states has positive-best U3 under the frozen definition.
5. **U2 does not materially reorder U1.** Corrected U1/U2 agreement is 27/27
   same Top-1 across all states and 18/18 across pair-informative states. Mean
   valid-state rho is 0.999989; mean Top-5 and Top-10 overlap are 0.9778 and
   0.9889.
6. **The earlier U1/U2 mismatch was a tie-handling defect in a duplicate report
   path.** Python `max()` inherited action-list order for tied U1 values, while the
   canonical pairwise path used utility-descending/action-ID-ascending ordering.
   One clean state was affected. The positive-U1 clean value changes from 4/5 to
   5/5; raw outcomes and solver behavior are unchanged.
7. **A1.7B remains justified as a separately frozen development pilot.** Sparse
   immediate support and utility-dependent signal strengthen the case for testing
   whether adaptive trial allocation can preserve useful evidence at lower cost.
   These results do not validate adaptive racing or authorize its implementation.
8. **A1.7C should remain regime-aware and multi-task.** Scale/stage heterogeneity
   and the different support of U0, U1, and U3 strengthen that design rationale.
   They do not authorize C1-v2 training in this phase.
9. **No production change is justified now.** The evidence does not support a
   change to refresh horizon, portfolio, encoder architecture, exploration,
   acceptance, repair, candidate-bank construction, or decoder.

## Execution and integrity

The deterministic selection contains 27/27 states, with exactly two TRAIN and one
VALIDATION state in every S/M/L by EARLY/MIDDLE/LATE cell. The formal full-bank
audit completed 9,560 unique actions, 76,480 ordered direct trials, and 152,960
continuation decoder evaluations. Completion and boundary checks pass 21/21.

The U1/U2 correction is generated from centralized per-state rows and propagated
to its summary, Markdown report, figure source CSV, and corrected A1.7A-R figure.
Current production wording names the encoder `compact_relational`; the sole
retained historical-encoder match is a legitimate historical terminal name.

R12 remains `DEVELOPMENT_EXPOSED`. R13 remains `LOCKED_FINAL_EVAL`; R14 remains
`LOCKED_GENERALIZATION_EVAL`; CORE45 remains external-baseline-only. None was used
for supplemental evaluation, label generation, or model selection, and Gurobi was
not run.

## Evidence locations

- Frozen protocol: `artifacts/ngas_a17as/supplemental_protocol_manifest.json`
- Selection evidence: `reports/ngas_a17as_state_selection.md`
- Full-bank audit: `outputs/ngas_a1/trajectory_utility_a17as_v1/audit/full_bank_completion_audit.json`
- Derived diagnostics: `outputs/ngas_a1/trajectory_utility_a17as_v1/derived/`
- Metric correction: `reports/ngas_a17as_metric_consistency_fix.md`
- Terminology audit: `reports/ngas_a17as_terminology_audit.md`
- Figures and QA: `reports/figures/ngas_a17as/`
- Final decision: `outputs/ngas_a1/trajectory_utility_a17as_v1/final_decision.json`
- Result manifest: `outputs/ngas_a1/trajectory_utility_a17as_v1/result_manifest.json`
