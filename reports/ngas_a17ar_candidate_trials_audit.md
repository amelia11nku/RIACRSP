# NGAS A1.7A-R candidate-trials audit

Every full-bank action retained exactly eight ordered repair/decode outcomes.
Quality loss is the best candidate makespan under a prefix cap minus the best of
all eight; positive-gain loss additionally clips at the captured incumbent.

| trial cap | marginal best reduction | P(best changes later) | quality loss vs 8 | positive-gain loss | no-new-best rate |
|---|---|---|---|---|---|
| 1 | 0.0000 | 0.8807 | 187.8078 | 0.0564 | 0.0000 |
| 2 | 86.3678 | 0.7472 | 101.4400 | 0.0442 | 0.4825 |
| 3 | 37.9677 | 0.6133 | 63.4723 | 0.0416 | 0.6538 |
| 4 | 20.3461 | 0.4973 | 43.1263 | 0.0353 | 0.7578 |
| 5 | 15.9011 | 0.3654 | 27.2251 | 0.0214 | 0.7930 |
| 6 | 11.0178 | 0.2434 | 16.2073 | 0.0204 | 0.8413 |
| 7 | 9.3849 | 0.1131 | 6.8224 | 0.0155 | 0.8508 |
| 8 | 6.8224 | 0.0000 | 0.0000 | 0.0000 | 0.8869 |

## Frozen subgroup diagnostics at cap 4

| dimension | group | actions | cap-4 loss | trial variance |
|---|---|---|---|---|
| repair | greedy | 2555 | 42.0751 | 23700.4626 |
| repair | reconfiguration_aware | 2555 | 43.5159 | 21198.2584 |
| repair | regret2 | 2555 | 45.7530 | 24046.1233 |
| repair | regret3 | 2555 | 46.6301 | 24497.3473 |
| repair | transport_aware | 2555 | 37.6571 | 17603.0197 |
| scale | L | 4290 | 49.1105 | 28349.7168 |
| scale | M | 4295 | 45.6165 | 21236.1332 |
| scale | S | 4190 | 34.4465 | 16919.1021 |
| search_stage | 0-20% | 2130 | 46.7540 | 24778.5390 |
| search_stage | 40-60% | 8520 | 42.3572 | 21389.5641 |
| search_stage | 80-100% | 2125 | 42.5736 | 22919.1252 |

Caps 1, 2, and 4 can be read directly from the corresponding rows. The
`no-new-best` rate is the wasted-trial proxy: after trial 1, it records the fraction
of trials that fail to improve the current prefix-best candidate. Repair, scale,
search-stage, and origin breakdowns are stored in the source CSV. This audit can
inform a separately frozen A1.7B racing protocol; it does not implement racing.

Machine-readable source: `outputs/ngas_a1/trajectory_utility_a17ar_v1/derived/candidate_trial_curve.csv`.
