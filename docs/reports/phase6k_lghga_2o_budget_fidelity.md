# Phase 6K LG_HGA-2O budget fidelity

Implementation and smoke checks pass. Formal eligibility remains conditional
on a complete Phase 6K preregistration, shared solver accounting and the
deployment gate. No new formal comparator outcome was produced.

The canonical method is `LG_HGA-RIACRSP-v2-N4M`, established by the existing
pre-Core v2 amendment and `scripts/run_advanced_baseline_v2.py`. Its separate
config and nine scale-by-CF DTR bundles are used. The old implementation and
100-generation results remain unchanged.

`N` in the existing runner is `instance.num_operations`, so the new budget is
exactly `2|O|` seconds. The new `rcias_clgri/search/lghga_2o.py` retains the
canonical initializer, RNG, genetic operators, local search, population
replacement, decoder and inner-loop deadline checks. It removes only the
outer 100-generation stop and preserves source-domain DTR input `generation /
100`. Completed generation bookkeeping still occurs when the inner deadline
is reached, as in the canonical solver.

All nine frozen tree bundles were inspected at generations 1, 50, 99, 100,
101, 200, 1,000 and 1,000,000. All tree thresholds are at most 1, and every
post-100 prediction equals its generation-100 prediction. No cap, refit or
threshold change is needed. Training instance IDs and content hashes are
disjoint from all 54 CAUR instances; the check uses holdout manifest hashes
without parsing holdout instances.

Tests exercise both inactive and active local-search gates using a controlled
clock. Before generation 100, canonical and 2O results, complete traces,
decoder counts, actions, diagnostics and timer accounting are exactly equal.
Repeated 2O runs are deterministic. When the old cap binds, the canonical
solver stops at 100, the adapter continues, the first 100 generation records
remain identical and DTR inputs continue as 1.00, 1.01, 1.02.

A real-clock tiny-instance smoke used unchanged canonical parameters and the
frozen S_CF1 DTRs. It ran 1,731 generations and 69,358 decoder evaluations in
12.006227 s against a 12 s budget. The 6.227 ms overshoot is reported; the
outer timer measured 12.006356 s. The returned schedule is independently
feasible and the trace is monotone. This smoke is not solver-quality evidence.

Artifacts:

- `configs/lghga_2o_baseline.json`
- `scripts/run_phase6k_lghga_2o_smoke.py`
- `tests/test_phase6k_lghga_2o.py`
- `outputs/phase6k_runtime_v1/audit/boundary_and_baseline_audit.json`
- `outputs/phase6k_runtime_v1/audit/lghga_2o_artifacts_and_leakage.json`
- `outputs/phase6k_runtime_v1/audit/lghga_2o_real_clock_smoke.json`

The artifact audit records 52 exact file hashes. Canonical implementation
manifest SHA-256 is
`9dbf3c5e468d749831d2cef05aec10023373bdedc437bf3584175f7d67975223`;
the frozen knowledge content hash is
`8f316ca918d3709f237205edf7ea9466da54006689b056c3866cf35d8e4cc515`.
Existing source-gap labels remain applicable to this budget adaptation.
