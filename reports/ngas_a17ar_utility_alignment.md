# NGAS A1.7A-R utility alignment

| origin | states | informative U0/U1 | mean rho | same top1 | mean top5 overlap | U1/U2 same top1* |
|---|---|---|---|---|---|---|
| CLEAN_NON_R12_DEVELOPMENT | 18 | 4 | 0.9330 | 0.9444 | 4.8333 | 0.8000 |
| R12_DEVELOPMENT_EXPOSED | 18 | 5 | 0.7664 | 0.8333 | 4.4444 | 1.0000 |

`*` conditions on a state having positive U1 utility. U0 and U1 often share the
same top action because most state banks contain no positive action; the
conditional rank correlation is therefore the more useful diagnostic. U2 uses
U1 improvement per measured repair-and-decode second and can reorder actions with
the same raw horizon gain. No C1-v2 fitting or solver behavior change occurred.

Machine-readable source: `outputs/ngas_a1/trajectory_utility_a17ar_v1/derived/utility_alignment.csv`.
