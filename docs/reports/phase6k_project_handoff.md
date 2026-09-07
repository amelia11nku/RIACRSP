# Phase 6K project handoff

## Terminal state

Phase 6K is complete with **`MODEL_REVISION_RUNTIME`**. The current authority is
`outputs/phase6k_runtime_v1/final/final_decision_v2.json`; the older
`final_decision.json` remains immutable evidence of the preprocessing HOLD that
preceded the user-approved protocol amendment.

E4R is runtime-equivalent: 288/288 R12 states, 6,809/6,809 candidates and 150
repeated-realization pairs passed, with maximum absolute output error
4.7683716e-6 and 100% decision identity. Formal neural p90 is 25.980939 ms and
passes 30 ms. Formal full-live p90 is 109.494985 ms and fails 100 ms. The
formal stream contains all 864 warmup and 1,440 measured rows; its measurement
SHA-256 is
`2b4458945490229c6e9d65ea0043b2de3c1f93b3a515a5dcad82f38b1d0125d6`.

CSG-NI v1 is **not frozen**. There is no `csgni_v1_freeze.json` and therefore no
v1 defining bundle hashes. The R12 solver gate was not run. R13/R14 remain
locked and unaccessed. Do not run those gates with E4R, relax the 30/100 ms
caps, filter formal latency rows, change historical preprocessing, or add
another candidate under the completed Phase 6K protocol.

## What happened

Historical Phase 6J CUDA FP16-autocast preprocessing is nondeterministic because
CUDA `index_add_` reductions vary even with identical input operands. The
approved amendment retained that exact historical path and compared E0 with
optimized implementations on the same freshly realized input. Live latency
continued to execute fresh historical preprocessing on every repetition.

The first compiled E3 formal stream exited with CUDA OOM after 170 complete
states because dynamic `reduce-overhead` CUDA Graph pools accumulated across
shapes. This was a PyTorch process failure, not a Codex quota or timeout. E3S
disabled dynamic CUDA Graph retention and passed a continuous 288-state memory
audit. E4S hoisted relation Q/K/V projections; E4R additionally reused graph
and membership tensors only inside the same live decision. These changes
reached the neural cap but could not reduce the full historical live path below
100 ms.

The L-state full-live median/p90 are 108.202/111.942 ms. Formal historical
preprocessing alone has overall p50/p90 60.540/77.335 ms and L p50/p90
76.121/79.105 ms. This persistent size effect, rather than only the retained
p99 outliers, causes the final failure.

## Evidence

- Final report: `docs/reports/phase6k_runtime_v1_final_report.md`
- Final decision: `outputs/phase6k_runtime_v1/final/final_decision_v2.json`
- Completion audit:
  `outputs/phase6k_runtime_v1/runtime/formal_latency_e4r/completion_audit.json`
- Formal result:
  `outputs/phase6k_runtime_v1/runtime/formal_latency_e4r/result.json`
- Raw formal measurements:
  `outputs/phase6k_runtime_v1/runtime/formal_latency_e4r/measurements.jsonl`
- E4R equivalence:
  `outputs/phase6k_runtime_v1/runtime/e4r_equivalence/result.json`
- Protocol and amendments:
  `docs/reports/phase6k_runtime_v1_preregistered_protocol.md`
- Preprocessing root cause:
  `docs/reports/phase6k_preprocessing_root_cause.md`
- LG_HGA-2O fidelity:
  `docs/reports/phase6k_lghga_2o_budget_fidelity.md`
- R12 solver go/no-go: `docs/reports/phase6k_r12_go_no_go.md`

Important hashes:

| Artifact | SHA-256 |
| --- | --- |
| E4R config | `55c5f5b928f96a47ac1ef45174f0cfd49cfa4b20100ae3cc853c0a7e38e3e3f2` |
| E4R equivalence | `06ab59e0b002fb1819ba3f39cbae3e591da9f95ef68868696c28835d309d4642` |
| Formal result | `3c7765631d6d139c0b3688d36da266c312e12bc74e3817f84357a22ec7d19f73` |
| Completion audit | `c129dbddf130c4c7acee1bb1a207d8dfa95703986d76ceb25342f1aceffd4a16` |
| Final decision v2 | `d3b506aa830767886f5875238619241d07fa35029cb5c226564312c775cd13fe` |

Protected predecessor evidence: 6,636 files, all unchanged. Final regression:
**342 passed in 16.45 s**. No Phase 6K process is running.

## Next admissible work

Any continuation requires a separately named and preregistered model/runtime
revision with a new scientific boundary. It may use the Phase 6K reports for
engineering diagnosis, but must not tune from R13/R14 or claim E4R as CSG-NI
v1. R13/R14 remain untouched for a future candidate that first passes its own
runtime, deployment-freeze and R12 solver gates.

Copyable continuation prompt:

> Read `docs/reports/phase6k_runtime_v1_final_report.md` and
> `outputs/phase6k_runtime_v1/final/final_decision_v2.json`. Verify their hashes,
> the 342-test regression and the zero R13/R14 access audit. Phase 6K is terminal
> MODEL_REVISION_RUNTIME: do not add another Phase 6K candidate or open solver/
> holdout gates. If a new revision is authorized, preregister it under a new
> name and preserve all Phase 6K evidence unchanged.
