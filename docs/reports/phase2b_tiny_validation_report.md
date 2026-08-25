# Phase 2B — Tiny Exact Schedule Validation

## Frozen validation suite

- `instances/tiny/tiny_01.json`: six operations, two products, three islands,
  one W-AGV, and one F-AGV. It covers a nonlinear DAG, incomparable
  operations, alternative islands, configuration changes, same-configuration
  processing, same- and cross-island product movement, W empty repositioning,
  and F-AGV contention.
- `instances/tiny/tiny_02.json`: secondary edge-case instance generated from
  the FJSP-family fixture.
- The former demo and four legacy Tiny/Small JSON files were removed after all
  references were migrated to the two frozen files.

## Exact solution and deterministic replay

`tiny_01` was solved with the exhaustive active-schedule branch-and-bound
backend because Gurobi and CP-SAT were unavailable. The solver returned
`OPTIMAL` with makespan **157.0**, zero optimality gap, and 8,942 explored
nodes. The exact action sequence was replayed through the production insertion
decoder. Exact and replayed makespans are identical.

The independent feasibility checker reports all 13 required constraint
categories as `PASS`. The operation table, resource table, W/F task tables,
and Gantt chart are derived from the same replayed `Schedule` object through
`ResourceTimelineExporter`.

Formal artifacts are under `outputs/validation/tiny_01/`:

- `solution.json`
- `operation_schedule.csv`
- `resource_timeline.csv`
- `w_agv_tasks.csv`
- `f_agv_tasks.csv`
- `gantt.png` and `gantt.pdf`
- `feasibility_report.json`
- `validation_summary.md`

## Verification

- Full regression after path migration: **36 passed**.
- The official Gantt was visually inspected for per-resource legibility and
  clearly distinguishes island reconfiguration/processing, W empty/loaded
  travel, and F outbound/return activities.
- Repository audit removed cache artifacts and found no duplicate-content
  groups or legacy schema files.

`TINY_EXACT_VALIDATED = TRUE`
