# Frozen tiny validation suite

- `tiny_01.json`: two 3-operation automotive products, nonlinear DAGs, one W and one F vehicle. This is the exact/replay/CSV/Gantt reference instance.
- `tiny_02.json`: five-operation FJSP-family boundary instance with two islands and one W/F vehicle.

- `tiny_03.json`: four operations on four fixed islands with two W-AGVs and two F-AGVs; used for native Gurobi MILP versus CP-SAT exact/replay comparison.

Both use fixed seeds and the unchanged RCIAS F-kit semantics. Regenerate only with `python scripts/generate_tiny_suite.py`.
