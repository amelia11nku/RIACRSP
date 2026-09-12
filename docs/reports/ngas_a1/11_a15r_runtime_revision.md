# NGAS A1.5R runtime revision

## Terminal decision

`NGAS_A15R_RUNTIME_PASS`

The frozen primary gate passed: every representative-state complete live single-C1
refresh p90 and the pooled p90 are at most 30 ms. A1.6 is now eligible for a
separately authorized/frozen next stage. R13, R14, and Gurobi remain untouched.

## Frozen boundary

- Historical A1.5 remains immutable at commit `77c369a81127642aef9ade60cf8001ea93fc315d`
  with terminal state `NGAS_A1_REVISE_RUNTIME`.
- A1.5R formal protocol SHA-256: `240d1b5380973421bc44f169899c5f3ab3013c11d4b117a4f5eacdfbffd7d717`.
- C1 checkpoint SHA-256: `448b0aaf871f0629dec2d94bad63c888fbdaf71c113eb5ade58c4228647c8560`.
- Representative-state SHA-256: `e49f7a777ff398609b981fb35499550e4058d1ab95a960ffeabf7bfa7d995273`.
- Device: `NVIDIA GeForce RTX 4060 Ti`; normal GC; 30 warmups
  and 200 measured complete refreshes per state; linear percentile definition.

## Root cause and implementation

The development diagnosis found a collection in all 300 normal-GC samples. Its
M/L/L_MAX p90 values were `58.029`,
`79.927`, and
`84.821 ms`; disabling
GC reduced tails but left L/L_MAX p90 above 50 ms. Profiling therefore supported a
structural rewrite rather than treating GC suppression as a qualification result.

`ProductionRefreshRuntime` is now the only neural refresh authority used by both
formal timing and the actual NGAS search. It caches immutable per-instance indices,
uses reusable contiguous workspaces, constructs compact CSG/event/CPM arrays,
shares schedule features across all three candidate sizes, computes target features
once per unique target, creates tensor views directly, and runs the unchanged C1
compact-relational checkpoint and frozen prior/ranking rules. Event monitoring reuses the compact
critical analyzer. Historical Phase 6P source hashes remain valid.

## Formal latency

| 状态 | A1.5 p90 (ms) | A1.5R p50 (ms) | A1.5R p90 (ms) | A1.5R p99 (ms) | 判定 |
|---|---:|---:|---:|---:|---|
| S | 16.815 | 11.021 | 11.579 | 13.209 | PASS |
| M | 55.341 | 17.900 | 19.269 | 20.324 | PASS |
| L | 75.569 | 23.701 | 24.449 | 27.833 | PASS |
| L_MAX | 77.705 | 24.688 | 26.512 | 26.896 | PASS |

Pooled A1.5R complete-refresh p90 is
`24.938 ms` versus historical A1.5
`76.335 ms`.

- L: compact state p90 `11.873 ms`, three-size bank p90 `6.239 ms`, action/tensor-view p90 `2.195 ms`.
- L_MAX: compact state p90 `13.039 ms`, three-size bank p90 `6.918 ms`, action/tensor-view p90 `2.328 ms`.

The preceding development run also met its stricter 27 ms headroom target for all
four states; L/L_MAX medians were
`23.137` and
`24.247 ms`.

## Semantic and search integration evidence

- The 40-row equivalence matrix passed with maximum external error
  `0` for advantages
  and exact feature tensors, action payloads, critical identities, prior rankings,
  and fixed-seed selections. It covers four frozen representatives, four H1 states,
  20 accepted states spanning all five repair operators, 20 resource-order changes,
  and four A→B→A sequences.
- The real CUDA search replay passed every integration check: deterministic decisions,
  100% final replay feasibility, shared-runtime timing presence, and absence of the
  old duplicate neural feature/tensor components.
- Full regression passed `492` tests.

## Distinct-state transition trace

| 状态 | distinct states | p50 (ms) | p90 (ms) | p99 (ms) | critical changes | resizes |
|---|---:|---:|---:|---:|---:|---:|
| S | 20 | 11.612 | 12.231 | 152.435 | 19 | 0 |
| M | 20 | 18.339 | 19.285 | 24.075 | 19 | 0 |
| L | 20 | 24.130 | 24.554 | 24.893 | 19 | 0 |
| L_MAX | 20 | 25.021 | 25.270 | 25.386 | 19 | 0 |

All 80 timed states had distinct hashes within scale, every transition changed the
candidate, all five repairs were covered, all optimized outputs matched the reference,
all static-context accesses hit, and no workspace resized. The single S p99 tail is
reported as observed and is not used to replace the preregistered representative-state
primary gate.

## Evidence

- `outputs/ngas_a1/runtime_revision_v1/preregistration/protocol.json`
- `outputs/ngas_a1/runtime_revision_v1/formal/raw/`
- `outputs/ngas_a1/runtime_revision_v1/equivalence/equivalence_matrix.json`
- `outputs/ngas_a1/runtime_revision_v1/transition_trace/transition_trace.json`
- `outputs/ngas_a1/runtime_revision_v1/audit/completion_audit.json`
- `outputs/ngas_a1/runtime_revision_v1/result_manifest.json`

A1.6 was not started in this stage. R13/R14 were not accessed and Gurobi was not run.
