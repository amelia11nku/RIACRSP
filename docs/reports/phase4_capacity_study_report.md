# Phase 4 Model Capacity Pilot

All candidates used the same 20 S / 5 M / 1 L synthetic demonstration seeds,
best-of-H1/H2/H3 teachers, one BC epoch, and the same ten held-out S/M/L
validation instances. Canonical instances were not used for training or model
selection.

| Candidate | Parameters | BC train s | Validation normalized makespan | Validation s | Peak GPU MiB | Feasible |
|---|---:|---:|---:|---:|---:|---:|
| D32-L2 | 161,229 | 22.33 | 0.197128 | 16.76 | 11.72 | 100% |
| D64-L2 | 630,893 | 21.50 | 0.192703 | 15.84 | 13.51 | 100% |
| D64-L3 | 903,873 | 24.03 | 0.187133 | 17.83 | 14.58 | 100% |
| D128-L3 | 3,584,193 | 24.31 | 0.183982 | 18.33 | 25.77 | 100% |

D128-L3 is selected for the full-data BC pilot. It has the best held-out score,
adds only 0.50 seconds over D64-L3 for the complete validation set, and remains
far below the 8 GiB GPU-memory limit. This is a pilot selection and does not use
the canonical 130 benchmark.

The 100 S / 100 M / 50 L data-generation probe completed all 250 teachers and
one D64-L2 epoch. It showed that full-data BC is feasible but that keeping all
tensorized graph states resident consumes about 13.5 GiB CPU RAM and makes
per-graph host-to-device transfer the dominant practical cost. The selected
full-data D128-L3 run therefore starts with three epochs; its validation trend
will determine whether a longer BC budget is justified.
