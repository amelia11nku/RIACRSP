# NGAS A1.7A-S clean stage critic audit

All results below use the 27 frozen clean non-R12 states. A state is informative
only when its finite action utilities differ by more than `1e-12`; tied/all-zero
banks receive no substantive ranking credit.

| dimension | group | states | informative | positive best | valid rho n | mean rho | mean norm rank | regret n | mean regret | hit@1 | hit@5 | hit@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| overall | ALL | 27 | 16 | 16 | 16 | 0.1607 | 0.0208 | 16 | 63.6250 | 0.0000 | 0.1250 | 0.1875 |
| scale | L | 9 | 6 | 6 | 6 | 0.1924 | 0.0277 | 6 | 75.8333 | 0.0000 | 0.3333 | 0.5000 |
| scale | M | 9 | 6 | 6 | 6 | 0.1967 | 0.0249 | 6 | 86.0000 | 0.0000 | 0.0000 | 0.0000 |
| scale | S | 9 | 4 | 4 | 4 | 0.0591 | 0.0043 | 4 | 11.7500 | 0.0000 | 0.0000 | 0.0000 |
| stage | EARLY | 9 | 8 | 8 | 8 | 0.2516 | 0.0345 | 8 | 112.1250 | 0.0000 | 0.1250 | 0.2500 |
| stage | LATE | 9 | 4 | 4 | 4 | 0.0753 | 0.0077 | 4 | 18.7500 | 0.0000 | 0.0000 | 0.0000 |
| stage | MIDDLE | 9 | 4 | 4 | 4 | 0.0643 | 0.0064 | 4 | 11.5000 | 0.0000 | 0.2500 | 0.2500 |
| scale_stage | L|EARLY | 3 | 3 | 3 | 3 | 0.3082 | 0.0469 | 3 | 133.0000 | 0.0000 | 0.3333 | 0.6667 |
| scale_stage | L|LATE | 3 | 2 | 2 | 2 | 0.0660 | 0.0056 | 2 | 15.5000 | 0.0000 | 0.0000 | 0.0000 |
| scale_stage | L|MIDDLE | 3 | 1 | 1 | 1 | 0.0974 | 0.0141 | 1 | 25.0000 | 0.0000 | 1.0000 | 1.0000 |
| scale_stage | M|EARLY | 3 | 3 | 3 | 3 | 0.3078 | 0.0423 | 3 | 153.0000 | 0.0000 | 0.0000 | 0.0000 |
| scale_stage | M|LATE | 3 | 1 | 1 | 1 | 0.1323 | 0.0167 | 1 | 40.0000 | 0.0000 | 0.0000 | 0.0000 |
| scale_stage | M|MIDDLE | 3 | 2 | 2 | 2 | 0.0624 | 0.0028 | 2 | 8.5000 | 0.0000 | 0.0000 | 0.0000 |
| scale_stage | S|EARLY | 3 | 2 | 2 | 2 | 0.0823 | 0.0042 | 2 | 19.5000 | 0.0000 | 0.0000 | 0.0000 |
| scale_stage | S|LATE | 3 | 1 | 1 | 1 | 0.0368 | 0.0028 | 1 | 4.0000 | 0.0000 | 0.0000 | 0.0000 |
| scale_stage | S|MIDDLE | 3 | 1 | 1 | 1 | 0.0349 | 0.0058 | 1 | 4.0000 | 0.0000 | 0.0000 | 0.0000 |

The machine-readable summary also retains NDCG and Top-1/5/10 utility gaps with
their denominators. These are development diagnostics and do not establish final
generalization.
