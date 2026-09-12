# NGAS A1.7A-R critic/portfolio interaction

| origin | states | top1 change rate | top5 overlap | final/neural disagreement | portfolio-neural U0 | conditional U0 change | selected-neural U0 |
|---|---|---|---|---|---|---|---|
| CLEAN_NON_R12_DEVELOPMENT | 18 | 0.0556 | 4.8333 | 1.0000 | -0.7222 | -13.0000 | -0.7222 |
| R12_DEVELOPMENT_EXPOSED | 18 | 0.0556 | 4.9444 | 1.0000 | 0.0000 | 0.0000 | 3.7778 |

## Scale and stage dependence

| dimension | origin | group | states | change rate | portfolio-neural U0 |
|---|---|---|---|---|---|
| scale | CLEAN_NON_R12_DEVELOPMENT | L | 6 | 0.0000 | 0.0000 |
| scale | CLEAN_NON_R12_DEVELOPMENT | M | 6 | 0.1667 | -2.1667 |
| scale | CLEAN_NON_R12_DEVELOPMENT | S | 6 | 0.0000 | 0.0000 |
| scale | R12_DEVELOPMENT_EXPOSED | L | 6 | 0.0000 | 0.0000 |
| scale | R12_DEVELOPMENT_EXPOSED | M | 6 | 0.0000 | 0.0000 |
| scale | R12_DEVELOPMENT_EXPOSED | S | 6 | 0.1667 | 0.0000 |
| search_stage | CLEAN_NON_R12_DEVELOPMENT | 40-60% | 18 | 0.0556 | -0.7222 |
| search_stage | R12_DEVELOPMENT_EXPOSED | 0-20% | 6 | 0.1667 | 0.0000 |
| search_stage | R12_DEVELOPMENT_EXPOSED | 40-60% | 6 | 0.0000 | 0.0000 |
| search_stage | R12_DEVELOPMENT_EXPOSED | 80-100% | 6 | 0.0000 | 0.0000 |

The portfolio changes the deterministic neural top-1 rarely. The archived final
selection differs from neural top-1 in every audited state because production
selection samples from the adjusted distribution; its utility comparison therefore
includes both portfolio weighting and frozen exploration. Conditional results are
reported separately so a rare intervention is not averaged into an apparent broad
benefit. Scale/stage rows and rank promotions/demotions are retained in the source.

Machine-readable source: `outputs/ngas_a1/trajectory_utility_a17ar_v1/derived/portfolio_interaction.csv`.
