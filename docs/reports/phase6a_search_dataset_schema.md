# Phase 6A Search Dataset Schema

The raw tables are Parquet files under `outputs/phase6a/raw_logs`. All time and makespan fields use the instance's native time unit. IDs and categorical values are strings; counts are integers; flags are booleans; ratios are floating-point values.

## `run_summary.parquet`

Run identity fields (`run_id`, `instance_id`, `suite`, `scale`, `CF_level`, `seed`) are static diagnostic metadata. `time_limit`, `budget_scale`, and `logging_enabled` describe the run configuration. `best_makespan`, `best_found_time`, `runtime`, `decoder_evaluations`, `iterations`, and `feasible` are label/diagnostic outputs. RSS fields measure process memory. `convergence_trace` and `diagnostics` are JSON strings and diagnostic-only.

## `transition_log.parquet`

| Fields | Source / meaning | Availability class |
|---|---|---|
| run identity, iteration, elapsed time, decoder evaluations | runner/search counters | `DIAGNOSTIC_ONLY` (scale/CF may be intentionally exposed later) |
| current/best makespan before | decoded pre-action state | `POLICY_INPUT_CANDIDATE` |
| destroy/repair operator, fraction/count, destroyed IDs | chosen action | `DIAGNOSTIC_ONLY` for analysis; action representation for later learning |
| candidate makespan, immediate delta, relative improvement | post-repair decode | `LABEL_ONLY` |
| accepted, acceptance type, after makespans, new global best | post-action outcome | `LABEL_ONLY` |
| operator weights before | ALNS state before action | `POLICY_INPUT_CANDIDATE` |
| operator weights after | adaptive update | `LABEL_ONLY` |
| repair counts, decoder evaluations, runtime | naturally executed repair diagnostics | `DIAGNOSTIC_ONLY` |
| temperature before | search state | `POLICY_INPUT_CANDIDATE` |
| bottleneck type | deterministic pre-action proxy | `POLICY_INPUT_CANDIDATE` |
| critical/resource overlaps, means, and diversities | aggregation of pre-action destroyed-target features | `POLICY_INPUT_CANDIDATE` after a target set is proposed |
| search stage | normalized iteration position, added offline | `DIAGNOSTIC_ONLY`; a causal stage feature must use the known budget |
| future-best-within-20 | offline look-ahead label | `LABEL_ONLY` |

`number_of_candidate_insertions_evaluated` is null because current repair does not enumerate insertion alternatives. `candidate_trials_completed` and `repair_decoder_evaluations` report the naturally generated candidates; no extra candidates are created for logging.

## `destroy_target_log.parquet`

Each row is one destroyed operation at one iteration. Static fields are operation/product/configuration and precedence/eligibility counts. Dynamic pre-action fields are assigned island, processing/start/end/slack, critical flags/score, island load, local reconfiguration, W/F/synchronization delay proxies, and product/island/W/F chain positions. These are `POLICY_INPUT_CANDIDATE`. `move_immediate_delta`, `move_accepted`, and `move_new_global_best` are `LABEL_ONLY`.

Exact per-field classifications, including identifiers and derived columns, are generated in `outputs/phase6a/diagnostics/information_leakage_audit.csv`. Future policy work must use only rows marked `POLICY_INPUT_CANDIDATE`; identifiers require an explicit encoding decision and must not be memorized as instance labels.
