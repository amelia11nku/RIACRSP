# NGAS A1.7A-R prior-staleness protocol

Revision 2 freezes the fixed-refresh staleness audit before outcomes. It uses
all 36 already governed full-bank states: 18 clean non-R12 development states and
18 R12 development-exposed audit-only states. At offsets 0, 5, 10, 15, and 19,
the persistent base bank is scored against full U0 counterfactual evidence. A
hypothetical fresh production refresh supplies semantic rank/score drift and a
matched fresh-top-1 reference; it does not change the archived solver trajectory.

- Refresh interval: 20 iterations.
- Trials: eight matched repair/decode realizations per persistent action.
- Checkpoint: `448b0aaf871f0629dec2d94bad63c888fbdaf71c113eb5ade58c4228647c8560`.
- Freeze source commit: `41168b4bc10cbf3839536bc4fefeb707cc941d81`.
- R13/R14 remain locked; CORE45 is external-only; Gurobi is not run.
- Adaptive refresh and C1-v2 training remain disabled.
- Rejected v1 is preserved at `artifacts/ngas_a17ar/prior_staleness_protocol_manifest_rejected_v1.json`; the formal full-bank
  execution path is unchanged.
