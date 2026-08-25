# tiny_03 Native Exact Validation

## Instance

`instances/tiny/tiny_03.json` is a deterministic RCIAS-2.0 validation case
with two products, four operations, four assembly islands, three
configurations, two W-AGVs, and two F-AGVs. Each operation has one fixed
eligible island and each island processes one operation. This deliberately
isolates multi-vehicle routing, component-kit contention, product precedence,
configuration readiness, and synchronization for an auditable comparison of
two native exact formulations.

## Environment

- Python 3.11.8
- Gurobi/gurobipy 13.0.3
- OR-Tools CP-SAT 9.11.4210
- NumPy 1.24.4 and Pandas 2.0.3 (selected to remain compatible with the
  existing Anaconda SciPy/Numba stack)

The local Gurobi license is bound to Windows user `ASUS`; Gurobi validation was
therefore executed in that host-user context. CP-SAT also runs in the ordinary
project environment.

## Formulations

Both formulations minimize makespan and model the same profile:

- fixed operation-island processing and initial configuration readiness;
- fixed product-chain predecessor and W release time;
- exactly one W and one F assignment per operation;
- sequence-dependent W empty repositioning and loaded transport;
- non-overlapping F warehouse round trips;
- processing start synchronized with product, W, F, and configuration readiness.

Gurobi uses a MILP with binary W routing arcs and F disjunctive ordering. CP-SAT
uses optional F intervals with `NoOverlap` and a per-W `Circuit`. Both extracted
assignments are converted to production decoder actions, replayed, and checked
independently. The existing exhaustive active-schedule solver supplies a third
reference.

## Results

| Backend | Version | Status | Objective | Best bound | Gap | Solver runtime |
|---|---|---|---:|---:|---:|---:|
| Gurobi MILP | 13.0.3 | OPTIMAL | 36 | 36 | 0 | 0.0130 s |
| OR-Tools CP-SAT | 9.11.4210 | OPTIMAL | 36 | 36 | 0 | 0.0187 s |
| Exhaustive active-schedule BnB | internal | OPTIMAL | 36 | 36 | 0 | 0.1816 s |

Both native results replay to makespan 36 and pass the independent feasibility
checker. Gurobi and CP-SAT choose the same W/F assignments. Their extracted
operation action ordering may differ because multiple optimal topological
orders exist; both replay to the same resource schedule and objective.

Formal artifacts are under `outputs/validation/tiny_03/run_1/`, including both
native solution JSON files, the exhaustive reference, comparison CSV/JSON,
one byte-identical agreed replay resource schedule with Gantt PNG/PDF files,
and objective/runtime figures.

`TINY_03_EXACT_VALIDATED = TRUE`
