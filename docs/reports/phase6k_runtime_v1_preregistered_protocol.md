# Phase 6K runtime protocol: blocked before preregistration freeze

Status: **HOLD_BEFORE_RUNTIME_PREREGISTRATION**. This document records the
authorized precision amendment and the failed prerequisite; it is not a
formal runtime authorization or a deployable-bundle freeze.

The requested lineage is verified: local `main`, fetched `upstream/main`, and
HEAD are `3db49d5c347ba278a3da0d16d1f6d0e7f3f59a8b`. The initial worktree was
clean. No existing implementation, checkpoint, configuration, or Phase 6I-MR /
Phase 6J output was edited. No push was performed.

## User-authorized precision amendment

The prohibition of FP16/BF16/mixed precision applies only to the Phase 6K J1
neural inference path and to newly introduced runtime optimizations. The
historical Phase 6J frozen-score preprocessing must preserve its original
CUDA FP16 execution semantics exactly, because it is part of the frozen
feature-generation contract. Phase 6K must not convert this preprocessing to
FP32 merely for consistency.

- FP16 remains confined to the historical frozen-score preprocessing.
- Its operators, dtype, candidate ordering, lexical tie-breaking and output
  cast remain unchanged. The original score extraction is
  `output.scores.detach().float().cpu().numpy()`: FP16 output is cast to FP32
  before source-feature construction. The J1 numeric-feature transform also
  explicitly produces `np.float32`.
- J1 encoder, all three heads, ensemble, calibrator and gate cannot acquire
  new FP16/BF16 execution. The frozen calibrator's existing numerical
  semantics must also be retained in its tensor equivalent.
- The initial compile boundary is the J1 neural/ensemble subgraph. Historical
  preprocessing stays outside it. Expanding that boundary requires full parity.
- All 288 R12 states must pass exact frozen-score, candidate-order and
  derived-feature checks, followed by identical final selected actions.

This amendment came from the user's explicit clarification during this run.
It supersedes an interpretation that would force historical preprocessing to
FP32. It does not authorize tolerances, feature changes or replacement scores.

## Unchanged planned progression

The primary incumbent is Phase 6H, with ALNS as internal anchor and the
separately named LG_HGA-2O budget adaptation. Iterative budgets are
`2 * instance.num_operations` seconds, with initialization included.

After the historical-input gate passes, the runtime preregistration must
freeze the component profiler, E0/E1/E2 ladder, one bounded
`torch.compile(mode="reduce-overhead")` feasibility audit, fallback rule,
warmup, thread settings, runtime-only selection rule and implementation hashes.
No runtime variant has been selected or formally measured in this run.

The learned three-seed J1 ensemble, epochs 5/16/4, normalization, calibrator,
gate, H=4 labels, 24-rule bank and eight repair trials per target stay frozen.
Formal assessment still requires five measurements per R12 state, neural p90
at most 30 ms and complete live p90 at most 100 ms. No slow-sample filtering or
favorable rerun selection is allowed.

Only a frozen, equivalent, runtime-eligible bundle can enter R12 solver
go/no-go. R13 follows a passing R12 gate, and R14 follows passing R13. The
original manual's quality, efficiency, integrity and baseline gates apply.
R13/R14 are still unopened; their instance files were only byte-hashed.

## Current blocker

The original unmodified FP16 preprocessor, replayed under its historical
launch environment, fails the user's exact source-feature contract. See
`phase6k_preprocessing_equivalence_hold.md` and
`outputs/phase6k_runtime_v1/audit/preprocessing_equivalence.json`.

Consequently, E1/E2, component latency profiling, compilation, formal timing,
deployable freeze and solver/holdout runs remain unstarted. A faster J1 path
cannot establish the missing historical-input equivalence.

## Approved amendment: paired realized-input equivalence

**PROTOCOL_AMENDMENT_APPROVED — proceed with Phase 6K runtime-equivalent
optimization under paired realized-input equivalence.** The user explicitly
approved this amendment after reviewing the reduction-localization evidence.
The preceding HOLD text remains a historical record, superseded only in its
equivalence reference and authorization status by this appended amendment.
The immutable old configuration remains `configs/phase6k_runtime_v1.json`;
the active configuration is `configs/phase6k_runtime_v1_amended.json`.
`outputs/phase6k_runtime_v1/amendment/approval.json` links both evidence chains.

Historical preprocessing provenance and runtime implementation equivalence
are separate gates. Preserve the original Phase 6J code, CUDA FP16 autocast,
operators, candidate identities and generation order, sorting/ties, field
definitions and FP32 output cast. Keep default, nondeterministic algorithms
for that preprocessing. Do not convert it to FP32, refit normalization, or
repeat until an archived realization happens to match. Archive score and
derived-score discrepancies are disclosed, not used as an implementation
parity failure. Deterministic structure and support must remain consistent.

For runtime equivalence, execute the historical preprocessor once per
realization, then give E0 and each candidate the same materialized input
tensors. Check all 288 states and 6,809 candidates, all three seed outputs,
finite values, support, lexical ordering, neural preference, fallback,
intervention and final target-set identity. E1 raw outputs must be bit exact;
E2/E3 use `atol=1e-5, rtol=1e-5`. All categorical decisions must match 100%.
Additionally, reuse the five state IDs fixed in the earlier diagnostic
protocol, with 30 fresh realizations per state; every pair must pass.
Cross-realization variation is allowed and never used to select a candidate.
The earlier maximum difference 0.0077123 compares different inputs and is
not an E0-versus-optimized runtime error.

J1 neural computation and new acceleration stay FP32, without TF32 or
autocast. Frozen calibration and scalar gate retain their original double
arithmetic when moved to GPU; this preserves existing numerical semantics
and introduces no reduced precision. J1 deterministic inference is scoped
separately from the unchanged nondeterministic preprocessing. Compilation
cannot encompass historical preprocessing.

Before E1 implementation, profile the lexically first state in each of the
nine S/M/L × CF cells. Use three warmups and three uninstrumented diagnostic
repetitions; separately record CUDA-event and wall component timings.
Fix CPU/inter-op threads at one, PYTHONHASHSEED=0, TF32 off and
CUBLAS_WORKSPACE_CONFIG=:4096:8. This environment is shared across variants;
archive-realization mismatch counts are not compared as controlled trends
against earlier collection environments.

Execute E1, one E2 head-vectorization strategy, then one bounded E3 attempt:
`torch.compile(mode="reduce-overhead", fullgraph=True, dynamic=True)`,
240-second process timeout, variable shapes across the nine cells. Record
cold compile, graph breaks, recompilations, warm latency and parity. An
unsupported, timed-out or non-equivalent compiler path is ineligible;
retain eligible eager paths without a compiler-mode or TorchScript sweep.
E4 is disabled in this preregistration. Implementation hashes are frozen
at each eligibility checkpoint and again before formal timing.

After completing the fixed ladder, choose the first equivalent variant in
E1/E2/E3 order with diagnostic neural p90 ≤27 ms and live p90 ≤100 ms;
if none qualifies, use the same order with neural ≤30 ms. Never select by
solution quality. If none qualifies, close MODEL_REVISION_RUNTIME.
Formal timing uses three per-state warmups followed by five measured
repetitions for every R12 state; retain all samples and report overall and
S/M/L p50/p90/p99. Each live repetition freshly executes original
preprocessing, source features, tensorization, transfer, J1 and final gate.
Shared inputs are an equivalence-test technique, not a live timing cache.

R13/R14 remain locked until the unchanged runtime, deployment-freeze and
preceding solver gates pass. This approval changes the reference object,
not the model, data partitions, caps or promotion criteria.

## E3 stability amendment after formal-stream OOM

The first E3 formal stream terminated inside PyTorch with CUDA OOM after 170
complete states and two warmups of the next state. This was not a Codex quota,
tool timeout, or user interruption. The process returned exit code 1 with a
`torch.OutOfMemoryError`; 1,362 measurements had already been flushed. The
compiler had warned that `reduce-overhead` was recording CUDA Graphs for many
distinct dynamic sizes. The failed stream and selection records remain
immutable under `runtime/formal_latency/` and `runtime/selection_v2.json`.

The user subsequently instructed Codex to diagnose the interruption, execute
a feasible fix, and continue Phase 6K. Before any replacement assessment,
`configs/phase6k_runtime_v1_stability_amended.json` freezes one mechanism-led
E3 correction: keep `mode="reduce-overhead"`, `fullgraph=True`, and
`dynamic=True`, and set
`torch._inductor.config.triton.cudagraph_skip_dynamic_graphs=True`. This keeps
the compiled FP32 graph but prevents a distinct dynamic input size from
retaining another CUDA Graph pool. It is one stability correction, not a
compiler-mode sweep; no TorchScript or E4 alternatives are tested.

Call this candidate E3S. Before a new formal stream, E3S must repeat the full
288-state/6,809-candidate paired audit, the fixed five-state × 30-realization
audit, and a continuous 288-state memory-stability pass. All earlier numerical
and decision gates remain unchanged. The replacement formal stream is valid
only if those checks pass and it completes all 288 states in one process,
without chunking, restarts, cache clearing, dropped samples, or favorable
reruns. R13/R14 remain locked.

## E4S single-cleanup amendment

E3S completed its 288-state continuous memory stream with all paired checks
passing and maximum reserved memory 207,618,048 bytes. It therefore resolves
the OOM mechanism. Its diagnostic neural p90 was 31.49 ms and full-live p90
was 120.09 ms, so it is still runtime-ineligible; this result is frozen in
`runtime/selection_v4.json`.

The original E0 component profile located the dominant work in the two
relation layers and showed repeated Q/K/V projection for each relation. The
user's instruction to execute a feasible correction authorizes continuing
within the manual's optional E4 boundary. Before implementation,
`configs/phase6k_runtime_v1_e4s.json` freezes exactly one cleanup: compute each
node type's query/key/value projection once per relation layer, then use the
unchanged relation indices to select rows. E4S retains the E2 vectorized heads
and E3S compiler settings. No other E4 optimization, compiler mode, shape
padding/bucketing, TorchScript, precision, weight, feature, calibration, gate,
or candidate change is allowed.

E4S must pass the same 288-state/6,809-candidate paired check, fixed five-state
× 30-realization audit, continuous memory stream and 30/100 ms formal stream.
Any numerical or decision mismatch, OOM, or cap failure closes
MODEL_REVISION_RUNTIME. R13/R14 remain locked throughout.

## E4R same-decision tensor-reuse amendment

E4S passed the complete paired-equivalence and memory-stability audits. Its J1
neural p90 was 25.50 ms, below the 30 ms cap, while full-live p90 remained
115.70 ms and failed the 100 ms cap. The full-live rows show persistent
size-dependent overhead in the L cells rather than a single removable outlier.
The result is frozen in `runtime/e4s_equivalence/result.json`.

Inspection of the live integration found that the unchanged historical
preprocessor already tensorizes the graph and candidate membership and transfers
its batch to CUDA. After that path returns, the E4S integration tensorizes the
same graph and membership again and transfers the graph tensors again for J1.
Before changing this path, `configs/phase6k_runtime_v1_e4r.json` freezes one
integration cleanup. E4R captures the already materialized graph, membership and
CUDA batch from that same historical-preprocessing call, reorders only the
existing membership tensor groups into the frozen target-set-ID order, and
reuses the graph tensors for J1. It must not regenerate candidates, reproject
destroyed-operation IDs, or cache any tensor across live decisions or
repetitions.

This reuse is inside one fresh live decision. Every measured repetition still
executes the full original CUDA FP16-autocast preprocessing, including graph
construction, proposal generation, tensorization, transfer, neural scoring,
output cast and calibration. Historical preprocessing operators and numerical
semantics remain unchanged; J1 and all added work remain FP32. E4R retains the
E4S model and stable compiler settings. Concurrent streams, another kernel
cleanup, compiler alternatives, shape padding/bucketing and mixed precision are
outside this amendment.

Before formal timing, E4R must prove that its reused graph and reordered
membership equal a fresh independent reconstruction, then pass the same full
288-state/6,809-candidate paired audit, five-state × 30-realization robustness
audit and continuous memory stream. Formal timing remains three warmups plus
five measured fresh live decisions for each of all 288 R12 states, in one
process, with the 30 ms neural and 100 ms full-live p90 gates. If E4R fails any
equivalence, stability or formal latency gate, Phase 6K closes as
MODEL_REVISION_RUNTIME without another optimization candidate. R13/R14 remain
locked until all preceding gates pass.

## Terminal execution record

E4R passed the full 288-state/6,809-candidate paired audit and all 150
repeated-realization pairs, with maximum absolute error 4.7683716e-6 and no
decision mismatch. The formal stream then completed all 288 states in one
process with 864 warmups and 1,440 retained measurements. Neural p90 was
25.980939 ms and passed the 30 ms cap. Full-live p90 was 109.494985 ms and
failed the unchanged 100 ms cap.

The preregistered terminal rule therefore closes Phase 6K as
**`MODEL_REVISION_RUNTIME`**. No additional optimization candidate is allowed
under this protocol. CSG-NI v1 was not frozen, no deployable bundle was
created, the R12 solver gate was not run, and R13/R14 remain locked and
unaccessed. Authoritative closure is recorded in
`outputs/phase6k_runtime_v1/final/final_decision_v2.json`; the earlier HOLD
record remains preserved as historical evidence.
