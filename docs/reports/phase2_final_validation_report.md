# Phase 2 Final Validation Report

## Benchmark

The frozen public suite contains **130** instances:

- Brandimarte Mk01–Mk10: 10;
- Hurink edata la01–la40: 40;
- Hurink rdata la01–la40: 40;
- Hurink vdata la01–la40: 40.

Hurink sdata is intentionally excluded. The existing files under
`FJSP-benchmark-main/` were reused as read-only source data; no public `.fjs`
content was modified or downloaded again. Every generated operation preserves
its source eligible-machine set and processing-time map exactly, with only
machine `k -> Mk` symbol mapping.

`generation_config.json` freezes generator version 2.1.0, configuration and
capability repair, sparse non-total DAG construction, reconfiguration
time/cost, Manhattan layout, two W/two F fleet sizing, travel-time rounding,
and logistics rates. Products with one or two operations have no forced
precedence; products with at least three operations are acyclic and retain an
incomparable pair.

The seed rule is the first 32 big-endian bits of
`SHA256("RCIAS-2.0::family::instance_name")`. JSON is UTF-8/LF with sorted keys
and fixed two-space indentation. `checksums.sha256` and source/generated hashes
are recorded in the manifests.

Final verification regenerated every instance in memory and compared bytes:

```text
verified 130/130
CANONICAL_BENCHMARK_READY = TRUE
```

## Tiny exact validation

The frozen suite now also includes `tiny_03`, a four-operation case with four
assembly islands, two W-AGVs, and two F-AGVs for native-solver comparison.

`tiny_01` was solved by the exhaustive active-schedule branch-and-bound backend.
The final exact run returned:

- solver status: `OPTIMAL`;
- objective: makespan **157.0**;
- optimality gap: **0.0**;
- explored nodes: **8,942**;
- runtime: approximately **2.8 s**.

The six exact actions were replayed through the deterministic insertion
decoder. Exact and replay makespans are equal. The independent checker marks
all 13 required categories `PASS`, including DAG/product/island,
reconfiguration, W same/cross-island and empty travel, F contention and
synchronization, and objective recomputation.

Auditable outputs are under `outputs/validation/tiny_01/`:

- `solution.json` and `feasibility_report.json`;
- `operation_schedule.csv` and `resource_timeline.csv`;
- `w_agv_tasks.csv` and `f_agv_tasks.csv`;
- `gantt.png`, `gantt.pdf`, and `validation_summary.md`.

All CSVs and the Gantt derive from the same replayed `Schedule` via
`ResourceTimelineExporter`. Same-island action `o21` uses `W_required=False`,
`W_AGV=NONE`, and creates no fictitious zero-duration W task. The PNG was
visually inspected at original resolution.

For `tiny_03`, independent native formulations were run with Gurobi MILP
13.0.3 and OR-Tools CP-SAT 9.11.4210. Both returned `OPTIMAL`, objective and
best bound **36**, gap **0**, and feasible decoder replay makespan **36**. The
exhaustive active-schedule solver independently confirmed the same optimum
after 825 explored nodes. Formal comparison artifacts are under
`outputs/validation/tiny_03/run_1/`.

## Graph

Non-ready operations no longer invoke dynamic decoder/timeline probes. They
retain static processing, DAG, capability, eligibility, and lower-bound data,
with `dynamic_feature_valid=0` and `is_actionable=0`. Ready candidates use four
explicit interfaces for O, M, W, and F.

For every ready operation-island pair, W and F candidates are probed
independently and the machine is probed once. Complexity is therefore
`O(|W|+|F|)`, not `O(|W||F|)`. All seven requested profile instances meet the
instrumented linear bound. The largest case, BR_Mk10, produced 279 nodes,
2,726 edges, 1,682 hierarchical choices, and 1,575/1,575 allowed probes in
about 0.05 seconds of graph construction.

`FeatureNormalizer` centrally scales time, distance, cost, load, and count.
Tests reject NaN/Inf, enforce stable dimensions and numeric-only features, and
verify configuration/island/W/F relabeling invariance. Under a consistent
permutation, objective, graph semantics, masks, and the neural policy action
change only by the same permutation.

## Neural

`rcias_clgri/nn/` contains the tensorizer, RT-HGT encoder, autoregressive
policy, value head, and integrated model. The default config is 128 dimensions,
four heads, and two layers. The encoder implements type-specific projections,
relation-specific key/message transformations, edge-aware attention bias and
gating, residual connections, layer normalization, and feed-forward blocks.

The policy factorization is:

```text
P(o|s) P(m|o,s) P(w|o,m,s) P(f|o,m,w,s)
```

Every stage uses a hard mask and candidate features. There is no flattened
O×M×W×F action head; same-island movement admits only the `NONE` W action.

The demonstration archive contains exact/H1/H2/H3 episodes for both Tiny
instances, including every pre-action graph, masks, action, and candidate
features. Exact `tiny_01` was selected for the overfit run. The deterministic
64-dimensional validation configuration stopped at epoch 26 with:

- total mean loss: **0.11791679**;
- O/M/W/F accuracy: **100% / 100% / 100% / 100%**;
- joint exact-action reproduction: **100%**;
- independent rollout feasibility: **100%**;
- policy/exact makespan: **157.0 / 157.0**;
- exact six-action sequence equality: **true**.

The residual operation loss reflects ID-invariant symmetric candidates; adding
an arbitrary ID embedding to force lower loss would violate the permutation
requirement. Training curves and full metrics are in
`outputs/bc_validation/run_1/`. No PPO, preference policy, Critical
Synchronization Graph, or neural destroy/repair was used.

## Repository

The final formal implementation has one canonical generator pipeline, one
exhaustive Tiny solver, one paired native-formulation module, one graph
builder, and one neural package. Added entry points are documented in
`README.md`; dependencies are frozen at the package level in
`requirements.txt`.

Removed after reference/import checks:

- two root demo JSON files;
- four superseded Tiny/Small JSON files;
- obsolete `graph/features.py` and `graph/graph_state.py` files;
- two stale root validation-result JSON files;
- cache, bytecode, pytest cache, and safe temporary artifacts.

No mathematical model, algorithm specification, user documentation, canonical
instance, Tiny formal instance, or public raw benchmark was moved or deleted.
The final audit reports no legacy schema, identical-content duplicates,
unexpected Tiny files, root demos, versioned/module-era Python, duplicate exact
solver, duplicate graph builder, or candidate redundant Python file.

Final command results:

```text
python -m compileall .                                  PASS
pytest -q                                                50 passed
python scripts/generate_canonical_benchmarks.py --verify-only
                                                         130/130 verified
python scripts/run_small_validation.py                   PASS
python scripts/run_native_tiny_validation.py             TINY_03_EXACT_VALIDATED = TRUE
python scripts/profile_graph_builder.py                  GRAPH_PROFILE_VALID = TRUE
python scripts/run_bc_validation.py                      BC_VALIDATED = TRUE
python scripts/audit_repo_structure.py                   repository_clean = true
```

## Final gates

```text
CANONICAL_BENCHMARK_READY = TRUE
TINY_EXACT_VALIDATED = TRUE
GRAPH_SEMANTICS_VALIDATED = TRUE
BC_VALIDATED = TRUE
REPOSITORY_CLEAN = TRUE
READY_FOR_PPO = TRUE
```

`READY_FOR_PPO` means the requested prerequisites are satisfied; PPO itself is
not implemented or executed in this revision.
