# NGAS A1.6 Frozen 2|O| Solver Comparison

## Decision and integrity

- Terminal decision: **`NGAS_A1_PASS_DEVELOPMENT`**
- Formal scope: 54/54 NGAS runs plus four immutable 54-run comparators
- Feasibility and replay: 54/54 NGAS; all reused comparator payloads remain feasible
- Budget audit: median overshoot 0.002956 s; maximum 0.057474 s; atomic starts after deadline = 0
- Regression: 498 tests passed
- R13/R14: R13/R14 are eligible only for a separately preregistered next stage; they were not accessed here.
- Gurobi: not run

## Final quality

NGAS mean RPD is 1.4701% and median RPD is
1.4503%. Against PHASE6N_TOP1, its mean instance-level
gain is 3.4298% with W/T/L
17/0/1. Against LG_HGA_2O, the
corresponding result is 1.7366% and
12/0/6.

Average ranks are NGAS_A1_6=1.556, PHASE6N_TOP1=3.833, PHASE6H=3.722, ALNS=3.222, LG_HGA_2O=2.667. The Friedman result is
`chi2(4)=24.933333`,
`p=5.18866e-05`. Holm-corrected comparisons are: PHASE6N_TOP1: p_holm=0.000152588, effect=0.965; PHASE6H: p_holm=0.00257874, effect=0.813; ALNS: p_holm=0.000434875, effect=0.918; LG_HGA_2O: p_holm=0.154045, effect=0.392.

## Required interpretation

1. **PHASE6N_TOP1:** NGAS paired gain is 3.4298% with W/T/L 17/0/1.
2. **When quality appears:** Paired gains at normalized checkpoints are 10%=3.677%, 25%=3.600%, 50%=3.434%, 75%=3.284%, 100%=3.391%; these causal checkpoints use only incumbents already present at each deadline.
3. **Scale:** Paired NGAS gains versus PHASE6N_TOP1 are S=5.258%, M=3.499%, L=1.532%. This shows whether improvement strengthens or collapses with scale without extrapolating beyond the 18 instances.
4. **Other solvers:** The pairwise table at `derived/pairwise_comparisons.csv` reports matched NGAS comparisons with ALNS, Phase6H, and LG_HGA_2O under the same 2|O| budget.
5. **Refresh budget:** Mean neural-refresh share is 1.896%.
6. **Guidance reuse:** Mean guided iterations per critic call is 19.956.
7. **Progress per evaluation:** `derived/runtime_summary.csv` reports decoder evaluations/second, new-best events/1000 evaluations, and improvement/1000 evaluations. These are descriptive associations, not isolated causal effects of the critic.
8. **Concentration:** Scale, CF, C01/C02, seed, and checkpoint tables are retained in `derived/`; conclusions should follow those strata rather than a single aggregate.
9. **Feasibility:** NGAS feasibility and exact schedule replay remain 100%.
10. **Refresh tails:** Across 11853 refreshes, 78 exceeded 30 ms, 76 exceeded 50 ms, and 1 exceeded 100 ms; p99 is 26.590 ms and maximum is 198.629 ms. Their practical impact is bounded by the measured refresh share and reported without hiding the A1.5R tail observation.

## Evidence

- Protocol: `outputs/ngas_a1/solver_comparison_v1/preregistration/protocol.json`
- Completion audit: `outputs/ngas_a1/solver_comparison_v1/audit/completion_audit.json`
- Decision: `outputs/ngas_a1/solver_comparison_v1/final_decision.json`
- Raw manifest: `outputs/ngas_a1/solver_comparison_v1/raw_manifest.json`
- Derived and manuscript-ready tables: `outputs/ngas_a1/solver_comparison_v1/derived/`

BKS v001 remains unchanged. No comparator was rerun, no post-outcome tuning was
performed, and this report does not authorize automatic access to R13 or R14.
