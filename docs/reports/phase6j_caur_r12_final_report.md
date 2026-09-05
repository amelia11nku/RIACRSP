# Phase 6J R12 terminal decision

## Outcome

**MODEL_REVISION — stopped before R13.** Integrity PASS; no family remains
eligible for R13. No CSG-NI v1 artifact was selected or promoted, and Phase 6H
remains the manuscript/production reference. R13 and R14 remain locked and
unaccessed. No solver-translation or final holdout experiment was run.

The J1 deployment worker completed normally at 2026-09-05 03:08:04 UTC
(11:08:04 CST), exit code zero. A successful process is not a successful
scientific gate: its terminal status is
`CACHED_LATENCY_GATE_FAILED_BEFORE_R13`.

## Family rejection evidence

| Family | R12 data gates | Deployment gate | Outcome |
| --- | --- | --- | --- |
| J1 | PASS; 6 retained gates, winner ECE 0.047758 | Neural p90 32.716758 ms > 30 ms | Rejected before R13 |
| J2 | No gate passes lift/support requirements | Not run after data rejection | Rejected before R13 |
| J3 | No retained gate; ECE 0.114795 > 0.10 | Not run after data rejection | Rejected before R13 |

J1's OOF gated continuation lift is 0.004976 with grouped-bootstrap lower
bound 0.001926, and 94 interventions (S/M/L: 25/35/34). Its Spearman 0.175995
and pairwise accuracy 0.562392 pass the applicable essential ranking rule
but miss preferred targets. These are continuation-label results, not
evidence of improved final solver makespan or promotion readiness.

## Deployment completeness and latency

- All three full-R12 seed checkpoints are complete, with fixed 5/16/4 epochs
  for seeds 696101/696102/696103. Hashes, exact trainable keys, finite values,
  full-R12 normalization and frozen base identity pass reload verification.
- All 288 R12 full-bank states were checked. Shared frozen-encoder inference
  matched the three independent models and their final gate decisions
  exactly, maximum absolute output difference zero.
- All 864 timed decisions are present: three repetitions of each state.
  State/scale identities and repetition keys are exact and unique; timing
  values are finite and positive. The repetitions are not independent new
  states.
- Per-seed parameters: 5,332,991 total and 36,863 trainable, within the
  frozen per-seed caps. Physical shared ensemble: 5,406,717 parameters.

| Scope | Neural p90 (ms), cap 30 | Cached-total p90 (ms) |
| --- | ---: | ---: |
| Overall | 32.716758 | 38.031468 |
| S | 32.237169 | 37.523452 |
| M | 32.868555 | 38.146195 |
| L | 32.761459 | 38.128557 |

The overall neural p90 exceeds its cap by 2.716758 ms (9.0559%). All three
scale summaries also exceed 30 ms; the observation is not confined to large
instances. No component-level profiler was run, so the specific bottleneck
has **not** been established.

Cached-total p90 is below 100 ms, but that measurement excludes live CSG
construction, proposal generation and frozen-score feature generation. It
does not establish a full online total-latency PASS and cannot override the
separate neural cap. Further online validation was stopped once the required
neural gate failed. No slower samples were excluded, no favorable rerun was
selected and no threshold or model was changed after measurement.

## Audit and authoritative artifacts

The CPU-only terminal audit reproduces both existing OOF completion audits,
verifies the collection/feasibility evidence chain, reloads final-fit
checkpoints, checks code/input/protocol hashes, validates the worker log and
recomputes p50/p90/p99 directly from the original timing CSV. It does not
train, run inference timing again or access R13/R14. The recorded all-state
parity result is backed by assertions in the hash-frozen worker; the terminal
audit verifies that provenance rather than calling a new timing run.

- Final decision: `outputs/phase6j_caur/final/final_decision.json`
- Final status: `outputs/phase6j_caur/final/final_status.json`
- Completion audit:
  `outputs/phase6j_caur/deployment/j1_full_r12/completion_integrity_audit.json`
- Original timing report/CSV and seed checkpoints remain under
  `outputs/phase6j_caur/deployment/j1_full_r12/`, unchanged.
- Separate read-only byte checks confirm all six registered Phase 6I-MR
  evidence files retain their original hashes and sizes. No outcome contents
  were interpreted or used for selection.

Reproduce the terminal audit without modifying prior evidence:

```bash
/home/liulei/miniconda3/envs/gnn311/bin/python scripts/finalize_phase6j_caur_r12.py
```

Existing identical terminal JSONs are left untouched; changed terminal
evidence is rejected instead of overwritten. Integrity errors abort before
publishing a `MODEL_REVISION` result.

Final regression: `300 passed in 13.60s`, including thirteen new terminal
audit tests. No existing frozen training or deployment implementation was
modified by this closure.

## One proposed next revision — not launched

If authorized, preregister a single **runtime-only J1 implementation revision**
in a new output/protocol namespace. Preserve the three trained checkpoints,
full candidate bank, feature definitions, normalization, calibrator, gate,
solver semantics and 30/100 ms caps. First profile the inference components
to establish the actual bottleneck; then implement only an exact-equivalent
execution change supported by that profile, with no retraining or threshold
search. Require output and decision equivalence before a new, separately
recorded latency assessment, including full online total latency before any
R13 eligibility claim.

This is a proposal, not permission to overwrite or relabel the failed run.
Do not unlock R13/R14, reduce ensemble seeds, relax latency caps, launch an
architecture sweep, or use Core/Sensitivity/Legacy as replacement evidence.
If exact-equivalent runtime work is insufficient, return for a new explicit
research decision instead of accumulating unregistered changes.
