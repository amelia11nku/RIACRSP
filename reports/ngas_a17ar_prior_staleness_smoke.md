# NGAS A1.7A-R prior-staleness CUDA smoke

Revision-2 smoke status: **PASS**.

- State: `CLEAN|CB1_TRAIN_S_CF1_RI1_TI1_R01:seed817000054:iteration2760`.
- Offsets: 0 and 5.
- Persistent actions evaluated per offset: 6. The scope contains the first three
  actions, the stale neural Top-1, and every offset-specific archived selected
  action; it is excluded from formal metrics.
- All archived repair/decode trials used to reach offset 5 replayed exactly.
- Offset-0 semantic bank Jaccard and representation graph Jaccard were both 1.0.
- At offset 5, no move had been accepted, so compact operation features and typed
  graph were unchanged. The hypothetical fresh bank nevertheless had semantic
  Jaccard 0.155738 with the persistent bank because bank construction is tied to
  the refresh state ID. Common-action percentile ranks remained highly stable
  (Spearman 0.998278).
- Every stored numerical value is finite. R13/R14 remained locked, CORE45 remained
  excluded, and no Gurobi run occurred.

Evidence: `outputs/ngas_a1/trajectory_utility_a17ar_v1/smoke/prior_staleness_smoke.json`
(SHA256 `454f1b3beda86fb4ab31e84a0ca51ceae7f2229ad244ff4c16dbe031bdf93f55`).
