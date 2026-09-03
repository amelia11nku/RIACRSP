# Advanced baseline reproduction handoff

## Current state

The implementation and pre-experiment gates for `DABC-RIACRSP` and `LG_HGA-RIACRSP` are complete. The starting repository commit is `7c448000be69084511af181f52cad0a8db71479a`. Formal knowledge generation completed on 2026-09-02, and the preregistered three-seed CB1-Core comparison is in progress. No Phase 6H/6I-MR output was modified by the advanced-baseline work.

The full repository regression suite passes: `227 passed`. Focused advanced-baseline tests cover the generalized event DAG, POX/UX, DABC defaults/search accounting/Theorems 1–10, LG_HGA four neighborhoods/DTR gate/local budgets, frozen manifests, and knowledge/Core leakage.

The formal knowledge matrix contains all 180 expected runs and 72,000 rows. Its canonical row-content hash, input-freeze hash, four model hashes, and empty Core ID/content-hash overlap sets were independently rechecked after training. The four training RMSE values are 7.9083, 7.8627, 6.1969, and 6.6397 percentage points for N1--N4. A complete 100-generation prediction scan found a maximum predicted improvement rate of 42.5556%, so the frozen strict `>50` online gate activates in zero generations. This is a frozen-data outcome, not a reason to tune the threshold after opening formal evaluation.

## Implemented artifacts

- Shared search: `rcias_clgri/search/operators.py`, `rcias_clgri/search/dabc_chdg.py`.
- DABC: `rcias_clgri/search/dabc.py`, `rcias_clgri/search/dabc_chdg_rules.py`.
- LG_HGA: `rcias_clgri/search/lghga.py`, `rcias_clgri/search/lghga_neighborhoods.py`, `rcias_clgri/search/lghga_learning.py`.
- Frozen configs/manifests: `configs/baselines/`.
- Offline knowledge runner: `scripts/run_lghga_knowledge.py`.
- Matched online runner: `scripts/run_advanced_baselines.py`.
- Source audits: `docs/reports/dabc_source_fidelity.md`, `docs/reports/lghga_source_fidelity.md`.

## Verified smoke runs

These are implementation checks, not algorithm-comparison results.

1. LG_HGA-KB: one real training instance, seed `540101`, one generation. It completed 160 common-decoder evaluations and generated exactly 20 samples for each of N1–N4; the final schedule was independently feasible.
2. DABC runner: one CB1-Core instance, seed `530101`, `0.001 × 2N` pilot budget. It completed the mandatory 120-individual initialization and returned an independently feasible schedule.
3. LG_HGA runner: the same Core pilot key using a smoke-only DTR bundle trained from the one-generation smoke rows. It verified model serialization/loading and online metadata. Its model hash is explicitly `SMOKE_ONLY_NOT_FORMAL` and cannot pass the formal-run freeze check.

Pilot artifacts are under `outputs/baselines/` and are ignored by Git.

## Frozen experiment boundary

- Primary suite: 45 CB1-Core instances, freeze hash `3d97e6a3f84d930213c424a633fa9e8192b16ec4eb9c2f169f65c0b48e691f0a`.
- Formal seeds: `530101`–`530110`, shared with Phase 5C.
- Budget: `2 × number_of_operations` seconds per method-instance-seed.
- Expected online records: `45 × 10 × 2 = 900`.
- Knowledge set: nine training-only R01 instances, balanced over `3 scales × 3 CF`, fixed at RI2/TI2.
- Knowledge seeds: `540101`–`540120`; expected offline runs: `9 × 20 = 180`.
- The formal LG_HGA runner refuses to start without a complete leakage-audited `knowledge_manifest.json` and matching model hashes.

## Next execution sequence

Generate the complete knowledge matrix (resumable and externally parallel):

```bash
conda run -n gnn311 python scripts/run_lghga_knowledge.py generate --workers <safe_cpu_workers>
```

After all 180 run files exist, train and freeze the four DTRs:

```bash
conda run -n gnn311 python scripts/run_lghga_knowledge.py train
```

Then run a preregistered three-seed preliminary comparison before the ten-seed study:

```bash
conda run -n gnn311 python scripts/run_advanced_baselines.py \
  --seeds 530101 530102 530103 --workers <safe_cpu_workers>
```

Finally omit `--seeds` to execute all ten frozen seeds. The runners skip existing result keys and write atomically, so interruption/resumption does not duplicate completed runs.

## Reporting still pending

Formal result reports, paired statistics, and anytime plots must be created only after the frozen DTR bundle and matched run matrix exist. In particular, the smoke objectives above must not be copied into manuscript comparison tables.
