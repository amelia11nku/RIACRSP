# NGAS A1.7A-S utility-ranking alignment

All-state same-Top1 uses all 27 states. Conditional same-Top1 and Spearman use
only pair-informative states, with separate denominators shown. Ties are resolved
by utility descending and action ID ascending.

| pair | all n | rho valid n | mean rho | same top1 all | same top1 informative | top5 overlap | top10 overlap |
|---|---|---|---|---|---|---|---|
| U0_IMMEDIATE/U1_SHORT_HORIZON | 27 | 16 | 0.9261 | 23/27 (0.8519) | 14/16 (0.8750) | 0.9111 | 0.9296 |
| U0_IMMEDIATE/U2_COST_NORMALIZED | 27 | 16 | 0.9261 | 23/27 (0.8519) | 14/16 (0.8750) | 0.9111 | 0.9370 |
| U0_IMMEDIATE/U3_STOCHASTIC_ROBUSTNESS | 27 | 16 | 0.1960 | 4/27 (0.1481) | 4/16 (0.2500) | 0.1556 | 0.1481 |
| U1_SHORT_HORIZON/U2_COST_NORMALIZED | 27 | 18 | 1.0000 | 27/27 (1.0000) | 18/18 (1.0000) | 0.9778 | 0.9889 |
| U1_SHORT_HORIZON/U3_STOCHASTIC_ROBUSTNESS | 27 | 18 | 0.1784 | 4/27 (0.1481) | 4/18 (0.2222) | 0.1333 | 0.1333 |
| U2_COST_NORMALIZED/U3_STOCHASTIC_ROBUSTNESS | 27 | 18 | 0.1784 | 4/27 (0.1481) | 4/18 (0.2222) | 0.1333 | 0.1333 |
