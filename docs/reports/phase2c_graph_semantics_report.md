# Phase 2C — Graph Semantic Hardening

## Semantic corrections

- Dynamic timeline probes are now restricted to unscheduled, topologically
  ready operations. Non-ready operations expose static processing, DAG,
  capability, eligibility, and lower-bound information only, with
  `dynamic_feature_valid = 0` and `is_actionable = 0`.
- Candidate extraction is split into operation, island, W, and F interfaces.
  Each ready operation-island pair probes W candidates and F candidates
  independently and performs one machine probe. No W-by-F Cartesian probing
  remains.
- Operation, product, island, W, F, relation, and candidate feature values are
  numeric. Configuration, island, and vehicle identifiers occur only as graph
  keys and relation endpoints, never as arbitrary numeric/categorical feature
  values.
- `FeatureNormalizer` centrally defines time, distance, cost, load, and count
  scales. Runtime checks reject NaN, infinity, and nonnumeric feature values.

## Verification

The tests cover non-ready semantics, the linear probe bound, finite normalized
features, stable per-type dimensions, and simultaneous configuration/island/
W/F relabeling. After a consistent relabeling, objective values, numeric node
and edge semantics, and hard action masks are identical under the same
permutation.

Full regression: **39 passed**.

## Profiling

Formal measurements are stored in
`outputs/profiling/graph_builder_profile.json`.

| Instance | Nodes | Edges | Ready ops | Hierarchical choices | Graph time (s) | Candidate time (s) | Probes / bound |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny_01 | 13 | 68 | 3 | 30 | 0.0011 | 0.0007 | 27 / 27 |
| BR_Mk01 | 75 | 480 | 32 | 367 | 0.0096 | 0.0051 | 335 / 335 |
| BR_Mk05 | 129 | 734 | 64 | 619 | 0.0136 | 0.0080 | 555 / 555 |
| BR_Mk10 | 279 | 2726 | 107 | 1682 | 0.0492 | 0.0283 | 1575 / 1575 |
| HU_E_la01 | 69 | 328 | 33 | 238 | 0.0064 | 0.0030 | 205 / 205 |
| HU_R_la01 | 69 | 400 | 33 | 348 | 0.0072 | 0.0044 | 315 / 315 |
| HU_V_la01 | 69 | 498 | 30 | 440 | 0.0105 | 0.0053 | 410 / 410 |

Times are lightweight single-process wall-clock measurements and are intended
as complexity diagnostics rather than hardware-independent benchmarks.

`GRAPH_SEMANTICS_VALIDATED = TRUE`
