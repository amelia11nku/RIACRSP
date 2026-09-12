# NGAS A1.7A-R prior-staleness audit

The frozen audit completed 36 states at offsets 0, 5, 10, 15, and 19 before the
production refresh at iteration 20. It evaluated all persistent base-bank actions
with eight matched repair/decode trials: 180 state-offsets and 63,875 action-offset
evaluations. Clean non-R12 and R12 development-exposed rows remain separate.

| origin | offset | info | rho* | regret+ | bank J | same top1 | top5 | rank drift | feature L2 | edge J | stale/fresh U0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CLEAN_NON_R12_DEVELOPMENT | 0 | 4/18 | 0.0484 | 10.2500 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.7222/0.7222 |
| CLEAN_NON_R12_DEVELOPMENT | 5 | 8/18 | 0.0397 | 14.7500 | 0.1413 | 0.1111 | 0.1000 | 0.0286 | 0.0000 | 1.0000 | 0.0000/0.0000 |
| CLEAN_NON_R12_DEVELOPMENT | 10 | 7/18 | 0.0241 | 23.8571 | 0.1272 | 0.0556 | 0.1333 | 0.0303 | 0.0422 | 0.9675 | 0.0000/0.0000 |
| CLEAN_NON_R12_DEVELOPMENT | 15 | 5/18 | 0.0497 | 23.4000 | 0.1237 | 0.0556 | 0.0778 | 0.0320 | 0.0422 | 0.9675 | 0.0000/0.0000 |
| CLEAN_NON_R12_DEVELOPMENT | 19 | 4/18 | 0.0631 | 20.2500 | 0.1260 | 0.0556 | 0.0778 | 0.0312 | 0.0422 | 0.9675 | 0.0000/0.0000 |
| R12_DEVELOPMENT_EXPOSED | 0 | 5/18 | 0.1065 | 53.8000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000/0.0000 |
| R12_DEVELOPMENT_EXPOSED | 5 | 6/18 | 0.1178 | 40.5000 | 0.1325 | 0.2222 | 0.1333 | 0.0344 | 0.0272 | 0.9745 | 0.8889/0.0000 |
| R12_DEVELOPMENT_EXPOSED | 10 | 7/18 | 0.0892 | 30.4286 | 0.1342 | 0.1111 | 0.1222 | 0.0290 | 0.0277 | 0.9677 | 1.8333/0.0000 |
| R12_DEVELOPMENT_EXPOSED | 15 | 4/18 | 0.0881 | 20.0000 | 0.1307 | 0.1111 | 0.1333 | 0.0359 | 0.0484 | 0.9325 | 0.0000/0.0000 |
| R12_DEVELOPMENT_EXPOSED | 19 | 5/18 | 0.1061 | 31.4000 | 0.1260 | 0.1111 | 0.1111 | 0.0322 | 0.0484 | 0.9325 | 0.0000/0.5556 |

`rho*` includes only nonconstant-U0 states. `regret+` includes only states with a
positive best U0 action. All-zero states remain in denominators and are not credited
as ranking success. Top-5 overlap is normalized by five.

## Instance-cluster uncertainty

| origin | offset | informative rate (95% CI) | bank Jaccard (95% CI) | rank drift (95% CI) |
|---|---|---|---|---|
| CLEAN_NON_R12_DEVELOPMENT | 0 | 0.2222 [0.0556, 0.4444] | 1.0000 [1.0000, 1.0000] | 0.0000 [0.0000, 0.0000] |
| CLEAN_NON_R12_DEVELOPMENT | 5 | 0.4444 [0.2222, 0.6667] | 0.1413 [0.1355, 0.1474] | 0.0286 [0.0233, 0.0341] |
| CLEAN_NON_R12_DEVELOPMENT | 10 | 0.3889 [0.1667, 0.6111] | 0.1272 [0.1088, 0.1419] | 0.0303 [0.0226, 0.0392] |
| CLEAN_NON_R12_DEVELOPMENT | 15 | 0.2778 [0.1111, 0.5000] | 0.1237 [0.1069, 0.1368] | 0.0320 [0.0235, 0.0427] |
| CLEAN_NON_R12_DEVELOPMENT | 19 | 0.2222 [0.0556, 0.4444] | 0.1260 [0.1066, 0.1424] | 0.0312 [0.0247, 0.0385] |
| R12_DEVELOPMENT_EXPOSED | 0 | 0.2778 [0.1667, 0.3333] | 1.0000 [1.0000, 1.0000] | 0.0000 [0.0000, 0.0000] |
| R12_DEVELOPMENT_EXPOSED | 5 | 0.3333 [0.1667, 0.5000] | 0.1325 [0.1137, 0.1470] | 0.0344 [0.0215, 0.0520] |
| R12_DEVELOPMENT_EXPOSED | 10 | 0.3889 [0.2222, 0.5556] | 0.1342 [0.1118, 0.1534] | 0.0282 [0.0218, 0.0331] |
| R12_DEVELOPMENT_EXPOSED | 15 | 0.2222 [0.1111, 0.3333] | 0.1307 [0.1085, 0.1507] | 0.0347 [0.0257, 0.0414] |
| R12_DEVELOPMENT_EXPOSED | 19 | 0.2778 [0.1111, 0.4444] | 0.1260 [0.1070, 0.1436] | 0.0317 [0.0254, 0.0384] |

Intervals use 10,000 deterministic percentile resamples of unweighted instance
means. These are exploratory development diagnostics; no formal hypothesis test is
declared and no solver-level generalization inference is made.

## Interpretation boundary

The fresh candidate bank's semantic overlap falls immediately after offset zero,
including states whose compact operation features and typed graph are unchanged.
Candidate construction incorporates the refreshed state identifier, so this bank
churn is not evidence that the compact relational representation or RT-HGT encoder
itself became stale. Among semantic actions common to both banks, percentile-rank
drift stays small. U0 is sparse and neither origin shows monotonic degradation of
the stale critic's informative-state rank correlation through offset 19.

Fresh neural Top-1 does not provide a consistent U0 advantage over stale Top-1.
The result does not support changing the frozen refresh interval inside A1.7A-R;
any adaptive-refresh design requires a separately frozen development protocol.
Scale- and search-stage rows are retained in
`outputs/ngas_a1/trajectory_utility_a17ar_v1/derived/prior_staleness_summary.csv`.

Machine-readable sources: `outputs/ngas_a1/trajectory_utility_a17ar_v1/derived/prior_staleness_state_metrics.csv`,
`outputs/ngas_a1/trajectory_utility_a17ar_v1/derived/prior_staleness_summary.csv`, and
`outputs/ngas_a1/trajectory_utility_a17ar_v1/derived/prior_staleness_analysis.json`.
