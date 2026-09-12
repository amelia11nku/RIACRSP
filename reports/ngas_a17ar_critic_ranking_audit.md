# NGAS A1.7A-R critic ranking audit

The audit enumerated all 12,775 unique joint actions from 36 frozen states. The
clean non-R12 and R12 development-exposed origins are reported separately.
Only states with nonconstant realized utility enter Spearman summaries. Metrics
marked `*` use only states where a positive best action exists; all-zero utility
states are retained in the state count and are not credited as substantive hits.

| origin | utility | states | informative | positive-best | mean rho | mean regret* | hit@1* | hit@5* |
|---|---|---|---|---|---|---|---|---|
| CLEAN_NON_R12_DEVELOPMENT | U0_IMMEDIATE | 18 | 4 | 4 | 0.0484 | 10.2500 | 0.0000 | 0.0000 |
| CLEAN_NON_R12_DEVELOPMENT | U1_SHORT_HORIZON | 18 | 5 | 5 | 0.0244 | 9.2000 | 0.0000 | 0.0000 |
| CLEAN_NON_R12_DEVELOPMENT | U2_COST_NORMALIZED | 18 | 5 | 5 | 0.0244 | 62.1642 | 0.0000 | 0.0000 |
| CLEAN_NON_R12_DEVELOPMENT | U3_STOCHASTIC_ROBUSTNESS | 18 | 18 | 0 | 0.3535 | NA | NA | NA |
| R12_DEVELOPMENT_EXPOSED | U0_IMMEDIATE | 18 | 5 | 5 | 0.1065 | 53.8000 | 0.0000 | 0.0000 |
| R12_DEVELOPMENT_EXPOSED | U1_SHORT_HORIZON | 18 | 6 | 6 | 0.1363 | 65.5000 | 0.0000 | 0.1667 |
| R12_DEVELOPMENT_EXPOSED | U2_COST_NORMALIZED | 18 | 6 | 6 | 0.1363 | 283.6403 | 0.0000 | 0.1667 |
| R12_DEVELOPMENT_EXPOSED | U3_STOCHASTIC_ROBUSTNESS | 18 | 18 | 0 | 0.4954 | NA | NA | NA |

## Scale and stage

| breakdown | group | states | informative | mean rho | mean regret* |
|---|---|---|---|---|---|
| dataset_origin+scale | CLEAN_NON_R12_DEVELOPMENT|L | 6 | 0 | NA | NA |
| dataset_origin+scale | CLEAN_NON_R12_DEVELOPMENT|M | 6 | 3 | 0.0529 | 12.3333 |
| dataset_origin+scale | CLEAN_NON_R12_DEVELOPMENT|S | 6 | 1 | 0.0349 | 4.0000 |
| dataset_origin+scale | R12_DEVELOPMENT_EXPOSED|L | 6 | 2 | 0.1436 | 87.5000 |
| dataset_origin+scale | R12_DEVELOPMENT_EXPOSED|M | 6 | 2 | 0.0819 | 42.0000 |
| dataset_origin+scale | R12_DEVELOPMENT_EXPOSED|S | 6 | 1 | 0.0812 | 10.0000 |
| dataset_origin+search_stage | CLEAN_NON_R12_DEVELOPMENT|40-60% | 18 | 4 | 0.0484 | 10.2500 |
| dataset_origin+search_stage | R12_DEVELOPMENT_EXPOSED|0-20% | 6 | 5 | 0.1065 | 53.8000 |
| dataset_origin+search_stage | R12_DEVELOPMENT_EXPOSED|40-60% | 6 | 0 | NA | NA |
| dataset_origin+search_stage | R12_DEVELOPMENT_EXPOSED|80-100% | 6 | 0 | NA | NA |

## Instance-cluster uncertainty

| origin | informative rate (95% CI) | U0 regret given positive best (95% CI) | rate instances | regret instances |
|---|---|---|---|---|
| CLEAN_NON_R12_DEVELOPMENT | 0.2222 [0.0556, 0.4444] | 10.2500 [5.5000, 16.7500] | 18 | 4 |
| R12_DEVELOPMENT_EXPOSED | 0.2778 [0.1667, 0.3333] | 53.8000 [23.2000, 86.2000] | 6 | 5 |

Intervals use 10,000 deterministic percentile resamples of unweighted instance
means. They are exploratory uncertainty summaries; no formal hypothesis test or
multiplicity family is declared.

The immediate-utility signal is sparse: the critic's predicted top action did not
hit the best positive U0 action in any informative origin-specific state. Positive
rank correlations are weak for U0/U1 and materially stronger for U3 signed
robustness. These are state-level development diagnostics, not solver-level
generalization evidence. R12 remains development-exposed.

Machine-readable sources: `outputs/ngas_a1/trajectory_utility_a17ar_v1/derived/full_bank_state_metrics.csv`,
`outputs/ngas_a1/trajectory_utility_a17ar_v1/derived/critic_ranking_summary.csv`, and
`outputs/ngas_a1/trajectory_utility_a17ar_v1/derived/critic_category_breakdowns.csv`. The latter supplies the
required neighborhood-size, candidate-family, and repair breakdowns.
