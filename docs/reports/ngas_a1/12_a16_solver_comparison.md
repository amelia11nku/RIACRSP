# NGAS A1.6 Frozen 2|O| Solver Comparison

## Terminal decision

**`NGAS_A1_6_INVALID_COMPARISON`**

The 54 intended NGAS raw files completed and passed replay, finite-value, hash,
feasibility, and atomic-budget checks. The formal development comparison is
nevertheless invalid because two formal workers overlapped on the first
instance-seed run for at least 262.002 seconds. This violated the frozen
`execution_concurrency = 1` boundary and exposed that wall-clock run to CPU/GPU
contention. The affected raw is preserved and will not be rerun.

The frozen finalizer originally returned `NGAS_A1_PASS_DEVELOPMENT` because its
raw-level audit did not inspect process concurrency. That output is preserved
under `audit/frozen_finalizer_*`; all quality numbers below are **diagnostic
only** and cannot support an A1.6 pass claim.

## Integrity

- NGAS raw scope: 54/54; raw manifest: 54/54
- NGAS feasible and exact replay: 54/54
- Frozen comparators: four sets of 54, all registry/manifests/raw hashes intact
- BKS v001 and C1 checkpoint: unchanged
- Protocol SHA-256: `0b9e7d97440f0d24b84390b4c1aca8bf85f25d4ae76e84ec54f77e92c38d1d37`
- Budget overshoot: median 0.002956 s, maximum 0.057474 s; atomic starts after deadline: 0
- Regression: 498 passed
- Concurrency: FAIL on the first run
- R13/R14: locked and untouched
- Gurobi: not run

## Diagnostic final quality

NGAS diagnostic mean/median RPD to BKS v001 is 1.4701%/1.4503%.
Its five-algorithm diagnostic average rank is 1.5556.

- versus PHASE6N_TOP1: mean instance-level gain 3.4298%, W/T/L 17/0/1;
  Holm p = 0.000152588, rank-biserial effect = 0.9649.
- versus PHASE6H: gain 3.3568%, W/T/L 16/0/2;
  Holm p = 0.00257874, effect = 0.8129.
- versus ALNS: gain 2.7335%, W/T/L 17/0/1;
  Holm p = 0.000434875, effect = 0.9181.
- versus LG_HGA_2O: gain 1.7366%, W/T/L 12/0/6;
  Holm p = 0.154045, effect = 0.3918.

The diagnostic Friedman test gives chi-square(4) =
24.933333, p = 5.18866e-05.
Average ranks are NGAS 1.5556, LG_HGA_2O 2.6667, ALNS 3.2222,
PHASE6H 3.7222, and PHASE6N_TOP1 3.8333.

## Required interpretation

1. **PHASE6N_TOP1:** Diagnostic NGAS gain is 3.4298% with W/T/L 17/0/1,
   but the invalid execution boundary prevents a formal superiority claim.
2. **Anytime:** Diagnostic gains versus PHASE6N_TOP1 are 3.6772%, 3.5995%,
   3.4345%, 3.2841%, and 3.3905% at 10/25/50/75/100% budget. The signal is
   visible early rather than appearing only at termination.
3. **Scale:** Instance-block gains versus PHASE6N_TOP1 are L 1.5320%,
   M 3.4992%, and S 5.2581%. The observed advantage decreases with scale.
4. **Other solvers:** Diagnostic terminal gains favor NGAS over ALNS, Phase6H,
   and LG_HGA_2O, although the LG_HGA comparison is not Holm-significant.
5. **Refresh cost:** Neural refresh consumes a mean 1.8958% of the solver budget.
6. **Reuse per refresh:** Mean guided iterations per critic call is 19.9561,
   consistent with the frozen 20-iteration horizon.
7. **Progress per evaluation:** NGAS records 10.5791 H1-to-final makespan units
   per 1000 decoder evaluations, versus 12.3074 for PHASE6N_TOP1, 15.3045 for
   Phase6H, 7.6552 for ALNS, and 8.5970 for LG_HGA_2O. This does not isolate a
   causal critic effect.
8. **Concentration:** Diagnostic gain versus PHASE6N_TOP1 is positive for every
   seed (2.5842% to 3.9372%), scale, CF level (2.5790% to 4.0674%), and C01/C02
   group. The largest scale gain occurs on S, not L.
9. **Feasibility:** All 54 NGAS schedules replay exactly and are feasible.
10. **Refresh tails:** Among 11853 refreshes, 78
    exceed 30 ms, 76 exceed 50 ms, and 1
    exceeds 100 ms. p99 is 26.590 ms and maximum is
    198.629 ms; the single >100 ms event remains visible.

## Evidence

- Execution-concurrency audit: `outputs/ngas_a1/solver_comparison_v1/audit/execution_concurrency_audit.json`
- Corrected completion audit: `outputs/ngas_a1/solver_comparison_v1/audit/completion_audit.json`
- Corrected terminal decision: `outputs/ngas_a1/solver_comparison_v1/final_decision.json`
- Diagnostic tables: `outputs/ngas_a1/solver_comparison_v1/derived/`
- Raw manifest: `outputs/ngas_a1/solver_comparison_v1/raw_manifest.json`

No raw result, comparator, BKS entry, source, checkpoint, seed, or search
parameter was modified. No rescue or favorable rerun is authorized after this
integrity failure.
