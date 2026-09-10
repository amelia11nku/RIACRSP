# A1.3R deterministic GPU profile

Device: NVIDIA GeForce RTX 4060 Ti. Precision: FP32. AMP and TF32 disabled.
Each timing covers one complete state encoding followed by batched scoring of
all joint actions.

| Variant | State | Mean (ms) | p50 (ms) | p90 (ms) | p99 (ms) |
|---|---|---:|---:|---:|---:|
| C0 | S | 1.120085 | 1.101399 | 1.142618 | 1.377634 |
| C0 | M | 1.141385 | 1.126322 | 1.171497 | 1.386957 |
| C0 | L | 1.146865 | 1.128253 | 1.190400 | 1.331549 |
| C0 | MAX | 1.154713 | 1.138119 | 1.225607 | 1.322899 |
| C1 | S | 1.962446 | 1.947357 | 1.966457 | 2.335117 |
| C1 | M | 2.011707 | 1.995875 | 2.056765 | 2.222105 |
| C1 | L | 1.998202 | 1.986461 | 2.017942 | 2.324566 |
| C1 | MAX | 1.980554 | 1.974603 | 1.991448 | 2.043136 |
| R1 | S | 46.864619 | 46.695364 | 47.634778 | 48.559489 |
| R1 | M | 47.776177 | 47.521314 | 48.756619 | 49.485069 |
| R1 | L | 48.757029 | 48.612217 | 49.825778 | 50.458314 |
| R1 | MAX | 48.863042 | 48.686365 | 49.572365 | 50.292048 |

| Variant | Parameters | Inference peak reserved memory | Formal-training peak reserved memory | Mean training states/s | Mean training actions/s |
|---|---:|---:|---:|---:|---:|
| C0 | 65,922 | 60,817,408 | not retained; preformal smoke 94,371,840 | 47.265 | 4,244.041 |
| C1 | 261,762 | 94,371,840 | 150,994,944 | 39.214 | 3,521.123 |
| R1 | 1,457,730 | 100,663,296 | 188,743,680 | 5.766 | 517.697 |

C1 remains well below the frozen 30 ms p90 budget. R1 exceeds it at every
representative scale. The immutable historical C0 checkpoint was profiled
read-only after selection to complete the three-way resource table; this did
not affect the frozen C1/R1 selection. Its original output tree was not
modified. Smoke tests and the C0 read-only profile established finite output
and bitwise-equal repeated inference on the maximum state.
