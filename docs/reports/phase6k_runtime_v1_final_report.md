# Phase 6K runtime-equivalent optimization final report

## Final decision

Phase 6K is complete with **`MODEL_REVISION_RUNTIME`**. The terminal E4R
implementation is runtime-equivalent to frozen Phase 6J J1 and passes the
30 ms neural gate, but its formal full-live p90 is **109.495 ms**, above the
frozen 100 ms cap. CSG-NI v1 is not frozen. No deployable-bundle manifest was
created, the R12 solver gate was not started, and R13/R14 remain locked and
unaccessed.

The authoritative decision is
`outputs/phase6k_runtime_v1/final/final_decision_v2.json`. It preserves the
earlier preprocessing HOLD record rather than replacing it. That HOLD was
superseded scientifically by the user-approved paired realized-input amendment,
then the amended runtime protocol reached this later terminal gate.

## Interruption diagnosis

The interrupted original E3 formal run was not caused by a Codex quota, tool
timeout, or user interruption. Python exited with `torch.OutOfMemoryError` after
170 complete states and two warmups of state 171. Dynamic `reduce-overhead`
CUDA Graph pools accumulated across input sizes until the 8 GiB GPU was
exhausted. Its 1,362 flushed records remain immutable under
`runtime/formal_latency/`; no percentile was accepted from that incomplete run.

E3S retained `mode="reduce-overhead"`, `fullgraph=True`, and `dynamic=True`,
while setting `cudagraph_skip_dynamic_graphs=True`. The full 288-state stream
then remained stable, with maximum reserved memory 207,618,048 bytes. This
established the OOM root cause and removed it without changing J1 arithmetic.

## Approved equivalence reference

Historical preprocessing continues to execute the original CUDA FP16-autocast
path with default nondeterministic reductions. J1, the ensemble and every new
optimization remain FP32. No deterministic preprocessing, archive-value
rerolling, normalization refit, candidate change, mixed precision or
cross-decision cache was introduced.

For runtime equivalence, one fresh historical preprocessing realization is
materialized once and shared between E0 and the candidate. This compares two
implementations on the same input. Live timing separately executes the entire
historical preprocessing path afresh on every repetition.

The terminal E4R paired audit passed:

| Check | Result |
| --- | ---: |
| Full R12 states | 288 / 288 |
| Candidate rows | 6,809 / 6,809 |
| Repeated-realization pairs | 150 / 150 |
| Maximum three-seed output error | 4.7683716e-6 |
| Candidate identity/order | 100% |
| Support mask | 100% |
| Neural preferred candidate | 100% |
| Fallback candidate | 100% |
| Intervention decision | 100% |
| Final selected action | 100% |
| Peak reserved CUDA memory | 226,492,416 bytes |

The maximum output error satisfies the preregistered `atol=rtol=1e-5` bound.
The earlier observed 0.0077123 difference arose from different legal
preprocessing realizations and is not an E0-versus-E4R error on a shared input.

## Fixed optimization ladder

| Candidate | Diagnostic neural p90 | Diagnostic live p90 | Outcome |
| --- | ---: | ---: | --- |
| E1 eager reconstruction | 40.43 ms | 144.26 ms | Equivalent; over caps |
| E2 vectorized seed heads | 37.10 ms | 119.15 ms | Equivalent; over caps |
| E3 compiled, corrected live scope | 10.02 ms | 97.26 ms | Formal stream invalidated by CUDA OOM |
| E3S, dynamic CUDA Graph retention disabled | 31.49 ms | 120.09 ms | Stable; over caps |
| E4S, relation-layer Q/K/V projection hoist | 25.50 ms | 115.70 ms | Neural pass; live fail |
| E4R, same-decision tensor reuse | 24.98 ms | 109.01 ms | Equivalent and stable; sent to formal gate |

E4R reuses only the tensor graph, projected membership and CUDA graph tensors
already constructed inside the same fresh historical preprocessing call. It
sorts the existing membership groups into the frozen target-set-ID order. It
does not regenerate a candidate, reproject an operation ID, skip historical
preprocessing, or retain data across decisions.

## Formal latency result

The formal E4R stream ran all 288 R12 states in one process. Each state used
three warmups followed by five retained measurements, producing 864 warmup and
1,440 measured rows. Every repetition freshly executed CSG construction,
24-rule proposal generation and deduplication, historical tensorization and
transfer, frozen-score inference and calibration, source features, J1, three
seed aggregation, calibration/gate logic and final action extraction. No row
was removed or rerun.

| Scale | Neural p50 | Neural p90 | Neural p99 | Full-live p50 | Full-live p90 | Full-live p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 24.895 | **25.981** | 27.158 | 91.299 | **109.495** | 548.632 |
| S | 24.995 | 26.302 | 27.656 | 77.342 | 82.210 | 89.684 |
| M | 25.019 | 26.001 | 27.068 | 91.284 | 94.757 | 222.492 |
| L | 24.736 | 25.425 | 26.491 | 108.202 | 111.942 | 695.287 |

The neural cap passes with 4.019 ms margin. Full-live misses its cap by
9.495 ms. This is not caused only by rare p99 interruptions: L-state full-live
median is already 108.202 ms and its p90 is 111.942 ms.

The formal overall historical-preprocessing total has p50 60.540 ms and p90
77.335 ms. For L states it has p50 76.121 ms and p90 79.105 ms. E4R's added
same-decision reuse work is only about 1.21 ms at p50 and 1.29 ms at p90;
J1 neural work is already below its independent cap. The remaining full-live
limit is therefore dominated by the frozen historical path and its
size-dependent work. Changing that path would violate the Phase 6J provenance
contract approved for this phase.

## Completion audit and boundary

The terminal audit independently parsed all 2,304 JSONL records, required
exact repetitions `-3..4` for every state, verified 480 measured rows per
scale, checked finite timings and present decisions, and recomputed every
p50/p90/p99 exactly. The measurements SHA-256 is
`2b4458945490229c6e9d65ea0043b2de3c1f93b3a515a5dcad82f38b1d0125d6`.

All 6,636 protected Phase 6I-MR/6J evidence files remain byte-identical. No
R13/R14 outcome-access artifact exists. The first device-boundary trace was
retained as invalid instrumentation; the replacement CUDA profiler audit
passed. The final complete repository regression is **342 passed in 16.45 s**.

The final defining evidence hashes are:

| Artifact | SHA-256 |
| --- | --- |
| E4R config | `55c5f5b928f96a47ac1ef45174f0cfd49cfa4b20100ae3cc853c0a7e38e3e3f2` |
| E4R equivalence | `06ab59e0b002fb1819ba3f39cbae3e591da9f95ef68868696c28835d309d4642` |
| Formal result | `3c7765631d6d139c0b3688d36da266c312e12bc74e3817f84357a22ec7d19f73` |
| Completion audit | `c129dbddf130c4c7acee1bb1a207d8dfa95703986d76ceb25342f1aceffd4a16` |
| Final decision v2 | `d3b506aa830767886f5875238619241d07fa35029cb5c226564312c775cd13fe` |

There are no CSG-NI v1 defining hashes because v1 was not frozen. The next
scientifically admissible step is a separately named and preregistered
model/runtime revision. The failed E4R candidate cannot proceed to the R12
solver gate, and R13/R14 cannot be opened under this Phase 6K result. The
explicit R12 decision is recorded in `phase6k_r12_go_no_go.md`.
