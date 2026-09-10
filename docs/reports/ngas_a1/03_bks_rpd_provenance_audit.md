# Raw evidence, BKS and RPD provenance audit

Imported 216 frozen comparator runs and 54 Phase 6P diagnostic runs, with current
file hashes verified against the starting boundary. Each record identifies instance,
source version/commit, checkpoint/config, seed, budget/initialization accounting,
makespan, feasibility, total work/runtime, last-best time/evals, and original path/hash.
Historical last-best iteration is explicitly null because raw evidence did not record it.

BKS v001 uses exactly the historical Phase 6P 270-run scope and 18 instances. The
derived manifest names the BKS version, content hash and path. A new BKS creates the
next immutable manifest; unchanged comparators are never rerun to update their RPD.
Exclusive-create writes reject replacement of an existing BKS file.

Tests show ALNS raw makespan 100 remains unchanged after an NGAS makespan 80;
derived ALNS RPD becomes 25%, while the original manifest still reproduces 0%.
Hash drift and overwriting an existing manifest are rejected.

Artifacts: `outputs/ngas_a1/provenance/raw_run_registry.json`,
`bks_manifest_v001.json`, and `derived_metric_manifest_v001.json`.
