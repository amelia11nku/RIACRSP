# Phase 2A — Canonical Benchmark Freeze

- Public source root: `FJSP-benchmark-main/` (existing local files; no download performed).
- Included: Brandimarte Mk01–Mk10; Hurink edata/rdata/vdata la01–la40.
- Excluded from the primary suite: Hurink sdata.
- Canonical instance count: 130.
- Schema: RCIAS-2.0.
- Generator version: 2.1.0.
- Seed: first 32 SHA256 bits of `RCIAS-2.0::family::instance_name`.
- Source machine eligibility and processing times: exact value-by-value preservation.
- DAG: source job chains discarded; 2-operation products have no edge; products with at least 2 operations retain an incomparable pair.
- Validation: 130/130 valid.
- Byte-level deterministic regeneration: 130/130 verified.

Generated metadata:

- `instances/canonical/RCIAS-2.0/manifest.csv`
- `instances/canonical/RCIAS-2.0/manifest.json`
- `instances/canonical/RCIAS-2.0/validation_report.csv`
- `instances/canonical/RCIAS-2.0/checksums.sha256`
- `instances/canonical/RCIAS-2.0/generation_config.json`

`CANONICAL_BENCHMARK_READY = TRUE`
