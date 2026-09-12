# NGAS A1.7A-R clean trajectory coverage

## Collection integrity

`PASS`

The frozen clean non-R12 collection contains **108/108 runs**
and **540/540 states**. Every raw file was independently
hashed and validated against the frozen revision-3 protocol. The single formal
session released its lock with `SUCCESS`; all 108 starts and completions occurred
exactly once. R13 and R14 remained locked, CORE45 remained excluded, and no Gurobi
run occurred.

The archived states retain the full compact-relational feature payload, full unique
joint-action identities and critic scores. Every selected action has exactly eight
stochastic repair/decode trial records. Candidate-bank size ranges from
**330** to
**360** actions.

## Balance

- TRAIN: S/M/L = `{"L": 135, "M": 135, "S": 135}`.
- VALIDATION: S/M/L = `{"L": 45, "M": 45, "S": 45}`.
- Capture targets: `{"0.1": 108, "0.3": 108, "0.5": 108, "0.7": 108, "0.9": 108}`.
- Medium and Large each contribute **180**
  and **180** states, respectively.

Each state can have multiple condition tags, so condition counts below are not
mutually exclusive. Search stage uses the observed normalized budget fraction and
the frozen 20%-wide bins from the A1.7A-R manual.

## Scale × stage × condition

| role | scale | stage | condition | states |
|---|---|---|---|---:|
| TRAIN | L | 0-20% | bottleneck_or_critical_transition | 24 |
| TRAIN | L | 0-20% | high_action_entropy | 17 |
| TRAIN | L | 0-20% | immediately_post_improvement | 3 |
| TRAIN | L | 0-20% | improving | 13 |
| TRAIN | L | 0-20% | low_action_entropy | 10 |
| TRAIN | L | 0-20% | plateau_or_stagnating | 14 |
| TRAIN | L | 20-40% | bottleneck_or_critical_transition | 4 |
| TRAIN | L | 20-40% | high_action_entropy | 15 |
| TRAIN | L | 20-40% | improving | 1 |
| TRAIN | L | 20-40% | low_action_entropy | 12 |
| TRAIN | L | 20-40% | plateau_or_stagnating | 26 |
| TRAIN | L | 40-60% | bottleneck_or_critical_transition | 2 |
| TRAIN | L | 40-60% | high_action_entropy | 18 |
| TRAIN | L | 40-60% | low_action_entropy | 9 |
| TRAIN | L | 40-60% | plateau_or_stagnating | 27 |
| TRAIN | L | 60-80% | high_action_entropy | 16 |
| TRAIN | L | 60-80% | low_action_entropy | 11 |
| TRAIN | L | 60-80% | plateau_or_stagnating | 27 |
| TRAIN | L | 80-100% | high_action_entropy | 15 |
| TRAIN | L | 80-100% | low_action_entropy | 12 |
| TRAIN | L | 80-100% | plateau_or_stagnating | 27 |
| TRAIN | M | 0-20% | bottleneck_or_critical_transition | 16 |
| TRAIN | M | 0-20% | high_action_entropy | 12 |
| TRAIN | M | 0-20% | immediately_post_improvement | 1 |
| TRAIN | M | 0-20% | improving | 6 |
| TRAIN | M | 0-20% | low_action_entropy | 15 |
| TRAIN | M | 0-20% | plateau_or_stagnating | 21 |
| TRAIN | M | 20-40% | high_action_entropy | 13 |
| TRAIN | M | 20-40% | low_action_entropy | 14 |
| TRAIN | M | 20-40% | plateau_or_stagnating | 27 |
| TRAIN | M | 40-60% | bottleneck_or_critical_transition | 1 |
| TRAIN | M | 40-60% | high_action_entropy | 21 |
| TRAIN | M | 40-60% | low_action_entropy | 6 |
| TRAIN | M | 40-60% | plateau_or_stagnating | 27 |
| TRAIN | M | 60-80% | bottleneck_or_critical_transition | 2 |
| TRAIN | M | 60-80% | high_action_entropy | 17 |
| TRAIN | M | 60-80% | low_action_entropy | 10 |
| TRAIN | M | 60-80% | plateau_or_stagnating | 27 |
| TRAIN | M | 80-100% | bottleneck_or_critical_transition | 1 |
| TRAIN | M | 80-100% | high_action_entropy | 18 |
| TRAIN | M | 80-100% | immediately_post_improvement | 1 |
| TRAIN | M | 80-100% | improving | 1 |
| TRAIN | M | 80-100% | low_action_entropy | 9 |
| TRAIN | M | 80-100% | plateau_or_stagnating | 26 |
| TRAIN | S | 0-20% | bottleneck_or_critical_transition | 8 |
| TRAIN | S | 0-20% | high_action_entropy | 18 |
| TRAIN | S | 0-20% | low_action_entropy | 9 |
| TRAIN | S | 0-20% | plateau_or_stagnating | 27 |
| TRAIN | S | 20-40% | bottleneck_or_critical_transition | 1 |
| TRAIN | S | 20-40% | high_action_entropy | 11 |
| TRAIN | S | 20-40% | low_action_entropy | 16 |
| TRAIN | S | 20-40% | plateau_or_stagnating | 27 |
| TRAIN | S | 40-60% | bottleneck_or_critical_transition | 2 |
| TRAIN | S | 40-60% | high_action_entropy | 21 |
| TRAIN | S | 40-60% | low_action_entropy | 6 |
| TRAIN | S | 40-60% | plateau_or_stagnating | 27 |
| TRAIN | S | 60-80% | high_action_entropy | 12 |
| TRAIN | S | 60-80% | low_action_entropy | 15 |
| TRAIN | S | 60-80% | plateau_or_stagnating | 27 |
| TRAIN | S | 80-100% | bottleneck_or_critical_transition | 3 |
| TRAIN | S | 80-100% | high_action_entropy | 19 |
| TRAIN | S | 80-100% | improving | 1 |
| TRAIN | S | 80-100% | low_action_entropy | 8 |
| TRAIN | S | 80-100% | plateau_or_stagnating | 26 |
| VALIDATION | L | 0-20% | bottleneck_or_critical_transition | 7 |
| VALIDATION | L | 0-20% | high_action_entropy | 4 |
| VALIDATION | L | 0-20% | immediately_post_improvement | 1 |
| VALIDATION | L | 0-20% | improving | 3 |
| VALIDATION | L | 0-20% | low_action_entropy | 5 |
| VALIDATION | L | 0-20% | plateau_or_stagnating | 6 |
| VALIDATION | L | 20-40% | bottleneck_or_critical_transition | 1 |
| VALIDATION | L | 20-40% | high_action_entropy | 6 |
| VALIDATION | L | 20-40% | low_action_entropy | 3 |
| VALIDATION | L | 20-40% | plateau_or_stagnating | 9 |
| VALIDATION | L | 40-60% | bottleneck_or_critical_transition | 1 |
| VALIDATION | L | 40-60% | high_action_entropy | 6 |
| VALIDATION | L | 40-60% | low_action_entropy | 3 |
| VALIDATION | L | 40-60% | plateau_or_stagnating | 9 |
| VALIDATION | L | 60-80% | high_action_entropy | 5 |
| VALIDATION | L | 60-80% | low_action_entropy | 4 |
| VALIDATION | L | 60-80% | plateau_or_stagnating | 9 |
| VALIDATION | L | 80-100% | high_action_entropy | 6 |
| VALIDATION | L | 80-100% | low_action_entropy | 3 |
| VALIDATION | L | 80-100% | plateau_or_stagnating | 9 |
| VALIDATION | M | 0-20% | bottleneck_or_critical_transition | 6 |
| VALIDATION | M | 0-20% | high_action_entropy | 5 |
| VALIDATION | M | 0-20% | improving | 2 |
| VALIDATION | M | 0-20% | low_action_entropy | 4 |
| VALIDATION | M | 0-20% | plateau_or_stagnating | 7 |
| VALIDATION | M | 20-40% | bottleneck_or_critical_transition | 2 |
| VALIDATION | M | 20-40% | high_action_entropy | 4 |
| VALIDATION | M | 20-40% | low_action_entropy | 5 |
| VALIDATION | M | 20-40% | plateau_or_stagnating | 9 |
| VALIDATION | M | 40-60% | bottleneck_or_critical_transition | 1 |
| VALIDATION | M | 40-60% | high_action_entropy | 4 |
| VALIDATION | M | 40-60% | low_action_entropy | 5 |
| VALIDATION | M | 40-60% | plateau_or_stagnating | 9 |
| VALIDATION | M | 60-80% | high_action_entropy | 7 |
| VALIDATION | M | 60-80% | low_action_entropy | 2 |
| VALIDATION | M | 60-80% | plateau_or_stagnating | 9 |
| VALIDATION | M | 80-100% | high_action_entropy | 7 |
| VALIDATION | M | 80-100% | low_action_entropy | 2 |
| VALIDATION | M | 80-100% | plateau_or_stagnating | 9 |
| VALIDATION | S | 0-20% | high_action_entropy | 1 |
| VALIDATION | S | 0-20% | low_action_entropy | 8 |
| VALIDATION | S | 0-20% | plateau_or_stagnating | 9 |
| VALIDATION | S | 20-40% | bottleneck_or_critical_transition | 1 |
| VALIDATION | S | 20-40% | high_action_entropy | 8 |
| VALIDATION | S | 20-40% | low_action_entropy | 1 |
| VALIDATION | S | 20-40% | plateau_or_stagnating | 9 |
| VALIDATION | S | 40-60% | high_action_entropy | 5 |
| VALIDATION | S | 40-60% | low_action_entropy | 4 |
| VALIDATION | S | 40-60% | plateau_or_stagnating | 9 |
| VALIDATION | S | 60-80% | bottleneck_or_critical_transition | 1 |
| VALIDATION | S | 60-80% | high_action_entropy | 7 |
| VALIDATION | S | 60-80% | low_action_entropy | 2 |
| VALIDATION | S | 60-80% | plateau_or_stagnating | 9 |
| VALIDATION | S | 80-100% | high_action_entropy | 6 |
| VALIDATION | S | 80-100% | low_action_entropy | 3 |
| VALIDATION | S | 80-100% | plateau_or_stagnating | 9 |

Machine-readable source data: `reports/ngas_a17ar_trajectory_coverage.csv`. The clean origin is
development evidence; this report does not access or claim evidence from R13, R14,
or CORE45.
