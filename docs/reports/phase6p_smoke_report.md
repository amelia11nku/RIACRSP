# Phase 6P P2 smoke report

Status: **PASS**.

The real frozen Phase 6N OOF critic was restored in FP32 on `cuda` and routed to fold 0 for `CB1_CAUR_S_CF1_RI2_TI2_R12_C02`. The fixed H1 state produced 24 proposals, 22 unique targets, and 2 deduplicated duplicates. Every unique target was scored by all three routed seed models. Repeating the same realized state produced exactly identical seed scores, ensemble ranking, and top-6.

The six-iteration live search exercised neural iterations 0 and 5, completed eight repair trials in every iteration, returned only feasible decoded candidates, and kept best-so-far monotone. Initial makespan was 1940.000000; smoke best was 1899.000000. This is an invariant smoke result, not a development quality comparison.

The primary source contains no Phase 6O critic, historical frozen-score scorer, or Gurobi dependency. R13/R14 access ledgers remain absent. Machine-readable evidence: `outputs/phase6p_adaptive_portfolio_v1/smoke/result.json`.
