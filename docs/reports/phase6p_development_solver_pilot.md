# Phase 6P development solver pilot

## Decision

**`ADAPTIVE_PORTFOLIO_PILOT_NO_GO`**. The detached P3 campaign completed all
270 frozen tasks and the completion audit passes. P1 nevertheless misses the
mandatory development condition that overall quality be better than Phase 6N
deterministic top-1. P1 mean makespan is
2938.296 versus
2936.204, and mean RPD is
5.0876% versus
5.0784%. Lower is better for both metrics.

The rescue field was frozen as `null` before P3 outcomes. No post-outcome
rescue, runtime qualification, five-seed formal R12, R13, or R14 run is
scientifically authorized in this Phase 6P attempt.

## Result integrity and comparison basis

The audit validates all 270 protocol hashes, instance hashes, task identities,
2|O| budgets, feasibility replays, monotone incumbent traces, and comparator
run manifests. Every method has 54 results (18 instances x 3 seeds), every
schedule is feasible, and initialization plus live overhead is charged to the
wall-clock budget. The four comparator registry entries are frozen canonical;
no favorable rerun was made.

RPD uses the current development BKS: the lowest final makespan observed for
each instance across all five methods and all three frozen seeds. W/T/L is
computed over the 18 paired instance means. The three-seed stage is descriptive;
formal significance is not claimed.

| method | mean_final_makespan | median_final_makespan | mean_rpd_percent | median_rpd_percent | bks_hits | average_instance_rank | mean_decoder_evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ALNS | 2921.833 | 2943.500 | 4.341 | 4.100 | 4.000 | 2.500 | 40083.130 |
| LG_HGA_2O | 2899.333 | 2914.500 | 3.322 | 2.461 | 8.000 | 2.444 | 37236.667 |
| P1_CSG_ADAPTIVE_PORTFOLIO | 2938.296 | 2903.500 | 5.088 | 4.863 | 1.000 | 3.333 | 19755.963 |
| PHASE6H | 2924.963 | 2889.000 | 5.037 | 5.241 | 4.000 | 3.333 | 16652.759 |
| PHASE6N_DETERMINISTIC_TOP1 | 2936.204 | 2901.500 | 5.078 | 4.722 | 2.000 | 3.389 | 19979.981 |

| comparator | W/T/L | mean_relative_improvement_percent |
| --- | --- | --- |
| PHASE6N_DETERMINISTIC_TOP1 | 9/0/9 | -0.009 |
| ALNS | 4/0/14 | -0.730 |
| PHASE6H | 9/0/9 | -0.065 |
| LG_HGA_2O | 8/0/10 | -1.778 |

P1 is within 0.730% of ALNS and 0.065% of Phase 6H by mean paired relative
improvement, but it loses to ALNS on 14/18 instance means and splits Phase 6H
at 9 wins and 9 losses. Against Phase 6N top-1 it is exactly 9/0/9, with a -0.009% mean paired
relative improvement and a 2.093 higher overall mean makespan. This is a small
but directionally failed result under the pre-registered conjunctive gate.

## Scale and search behavior

Relative to Phase 6N top-1, P1 mean quality changes by
S +0.420%, M -0.317%,
and L -0.106%. There is no material scale collapse
under the descriptive 1% audit margin. P1 averages
19756.0 decoder evaluations, and
48/54 runs strictly improve their H1 initializer, so the
failure is not caused by an absence of search.

Across 26,707 neural decisions, all six neural ranks, all five
origin families, and 23/24 rules are selected.
The largest rank/family shares are 40.86%
and 35.11%; selection therefore does
not collapse to one rank or family. Safe fallback count is zero.

Destroy weights change the portfolio probability vector in
45.82% of
neural decisions. The mean L1 distance from the pure 1/r prior is
0.0258 (total variation
0.0129). Thus the ALNS
weights measurably alter early sampling, but only modestly on average; the
checkpoint table also records their later convergence toward the 0.1 floor.

Weighted iteration-level rates are acceptance
5.739%, current improvement
2.983%, and new global best
0.324%.

## Runtime accounting inside P3

P1 recorded 12949.160 aggregate iteration seconds.
The included critic consumed 5046.671 s
(38.97%),
including 63.299 s candidate-bank work,
588.859 s tensorization/transfer,
4389.281 s three-seed inference,
and 5.232 s ranking. Mean critic time is
188.964 ms per neural decision.
Decoder work consumed 7814.417 s. These P3 aggregate
observer timings diagnose overhead; they are not the formal p50/p90/p99 runtime
qualification, which is prohibited after the quality gate fails.

## Artifacts

- Completion audit: `outputs/phase6p_adaptive_portfolio_v1/development/completion_integrity_audit.json`
- Audited runs and hashes: `outputs/phase6p_adaptive_portfolio_v1/development/audited_run_summary.csv`, `result_hash_manifest.csv`
- Method/pair/scale/CF/anytime summaries: `outputs/phase6p_adaptive_portfolio_v1/development/`
- Portfolio and weight diagnostics: `portfolio_diagnostics.json`, `selection_distribution.csv`, `adaptive_weight_summary.csv`
- Terminal decision: `outputs/phase6p_adaptive_portfolio_v1/final/final_decision.json`
