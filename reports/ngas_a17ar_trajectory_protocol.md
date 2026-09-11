# NGAS A1.7A-R clean trajectory protocol

Protocol revision **3** is frozen before formal collection. It supersedes rejected revision 1 (`aa79611757f037394763755f15a56c8d0fbf87089e1134c414ee4f6e2e83d1ce`). It does not train C1-v2 or change the production solver.

- TRAIN: **81 instances**, one balanced selection across all 81 scale×CF×RI×TI cells.
- VALIDATION: **27 instances**, balanced across S/M/L, CF1/2/3, RI1/2/3, and TI1/2/3.
- Each instance contributes five snapshots at normalized budget fractions `[0.1, 0.3, 0.5, 0.7, 0.9]` for **540 expected clean states**.
- Every trajectory uses the frozen `PERSISTENT_FIXED_REFRESH` C1 solver, refresh interval 20, five repairs, and `candidate_trials=8` stochastic realizations per selected joint action.
- Budget: `2 * |O|` seconds per trajectory; nominal serial budget is 7.28 hours before setup and final audit work.
- Candidate-bank hash: `e64c7a1cdfbe64495a2e3175fe7060dcea745f094ea6e3f1b3839fb48367bd07`.
- R12 is development-exposed and audit-only. R13/R14 remain locked with no solver access.
- RCIAS-CB1-CORE45 is external-baseline-only and excluded from both training and model selection.

Instance IDs and file-content SHA256 values are mutually disjoint between TRAIN and VALIDATION and from R12, R13, R14, and RCIAS-CB1-CORE45. The full per-instance list, generation seeds, trajectory seeds, budgets, checkpoint hash, collection-code hashes, and audit subset are in `artifacts/ngas_a17ar/trajectory_protocol_manifest.json`.
