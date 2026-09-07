# Phase 6K historical preprocessing equivalence: HOLD

Phase 6K stopped at the historical-input integrity gate, before runtime
optimization or holdout access. **CSG-NI v1 is not frozen.**

## Scope and evidence

Starting regression passed 300 tests in 13.31 s. A read-only Phase 6J terminal
audit reproduced the OOF/checkpoint/normalization/latency evidence. All 6,636
Phase 6I-MR/6J files, totaling 1,763,645,455 bytes, were hashed and verified
unchanged. All 54 CAUR instance byte hashes match their manifest. R13/R14
payloads were not parsed and their access ledgers remain absent.

The GPU is RTX 4060 Ti, driver 575.64.03; the validated environment is Python
3.11.15, PyTorch 2.11.0+cu128, CUDA 12.8. Sandboxed GPU initialization failed
in one diagnostic command; actual GPU audits ran in the verified host
environment. That failed command generated no GPU result.

## Why FP16 must remain in historical preprocessing

The existing `score_frozen_candidate_bank` uses CUDA FP16 autocast, then
casts score output to FP32. These scores determine
`normalized_frozen_score_rank`, other source features and role selection.
J1 itself uses FP32.

A diagnostic using the same cached R12 tensors compared historical FP16
against FP32: 19/288 states and 38/6,809 candidate rows changed score rank;
top-1 and fallback identities did not change in that comparison. This is
evidence against silently converting preprocessing to FP32, not a runtime
candidate or a selection experiment. The user explicitly authorized retaining
the historical FP16 boundary and requiring exact full-R12 preprocessing parity.

## Full original-path replay

The decisive audit reconstructed each R12 source schedule and called the
unchanged historical preprocessor, all 24 rules, deduplication, role selection
and source-feature builder. It used the historical launch settings:
`PYTHONHASHSEED=0`, all CPU thread limits 1, CUDA FP16 autocast and default
`deterministic_algorithms=False`. Scores were compared against the archived
`frozen_raw_score` values, never continuation outcomes.

| Check | Result |
| --- | --- |
| Complete states / candidate rows | 288 / 6,809 |
| Candidate identities and original generation order | 288/288 exact |
| Fallback candidate | 288/288 exact |
| Frozen score values | 218/288 exact; 70 mismatches |
| Largest absolute score discrepancy | 0.001953125 |
| Derived source features | 286/288 exact; 2 mismatches |
| Final J1 selected action | Not run after input-equivalence failure |
| Audit wall time, excluding setup | 17.134 s |

The two derived-feature failures are exclusively
`normalized_frozen_score_rank`:

- `CB1_CAUR_L_CF3_RI2_TI2_R12_C02__seed691202__it0000046`
- `CB1_CAUR_S_CF3_RI2_TI2_R12_C02__seed691201__it0000054`

Earlier diagnostic contexts produced a different pair of affected states.
Those raw records are retained. Tests of deterministic algorithms on/off and
original live candidate order did not establish exact reproduction. The root
cause is **not established**; small FP16 differences alone do not prove a
specific CUDA, reduction-order or environment cause. No score was rounded,
patched, replaced from cache or accepted under a relaxed tolerance.

## Decision and remaining work

Decision: **HOLD**, for failed historical-input reproduction. This is not a
new neural-latency failure and not a scientific rejection of J1 continuation
quality. Runtime gates have not been measured for Phase 6K.

The next admissible work is to resolve the existing preprocessing
reproducibility failure while preserving the now-explicit input contract.
Compare archived per-state scores and graph/action tensors with original-path
replay and establish a reproducible cause before changing any implementation.
Do not bypass the failure with cached feature injection in full-live timing,
score rounding, new tolerances, changed labels or another holdout.

If exact reproduction cannot be recovered under the unchanged contract,
retain HOLD and request a separate scientific revision; do not continue the
runtime-only v1 promotion claim.

## Reproduction and artifacts

Audit scripts refuse to overwrite their recorded outputs. The following
command verifies the completed stop decision and protected evidence without
rerunning GPU inference:

```bash
/home/liulei/miniconda3/envs/gnn311/bin/python scripts/finalize_phase6k_preprocessing_hold.py
```

- `outputs/phase6k_runtime_v1/audit/starting_audit.json`
- `outputs/phase6k_runtime_v1/audit/protected_evidence.json`
- `outputs/phase6k_runtime_v1/audit/phase6j_terminal_reaudit.json`
- `outputs/phase6k_runtime_v1/audit/boundary_and_baseline_audit.json`
- `outputs/phase6k_runtime_v1/audit/source_precision_audit.json`
- `outputs/phase6k_runtime_v1/audit/live_source_order_audit.json`
- `outputs/phase6k_runtime_v1/audit/source_determinism_audit.json`
- `outputs/phase6k_runtime_v1/audit/preprocessing_equivalence.json`
- `outputs/phase6k_runtime_v1/final/final_decision.json`

No stage exceeded five minutes; all launched work completed while the
interactive session remained active. No background job or ETA is pending.

Final regression: **310 passed in 14.25 s**, including four budget-adapter
tests and six fail-closed preprocessing-summary tests.
