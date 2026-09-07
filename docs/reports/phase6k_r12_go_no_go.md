# Phase 6K R12 solver go/no-go

Decision: **NO-GO — runtime prerequisite failed**.

The user-approved paired realized-input protocol established E4R runtime
equivalence on all 288 R12 states and 6,809 candidates, plus 150 repeated
realization pairs. Its maximum absolute three-seed output error was
4.7683716e-6 and every candidate, support, fallback, intervention and final
action decision matched E0.

The formal latency stream completed 864 warmups and 1,440 measurements. Neural
p90 was 25.980939 ms and passed the 30 ms prerequisite. Full-live p90 was
109.494985 ms and failed the frozen 100 ms prerequisite. The authoritative
decision is therefore `MODEL_REVISION_RUNTIME`.

The Phase 6K manual requires runtime equivalence and both latency gates before
freezing a deployable bundle or running the R12 solver sanity gate. Consequently:

- no deployable CSG-NI v1 bundle was frozen;
- no Phase 6K R12 solver comparison was started;
- no solver outcome was used to select or revise E4R;
- R13/R14 remain locked and unaccessed;
- LG_HGA-2O remains an audited comparator implementation but was not run on the
  blocked solver/promotion stages.

Evidence:

- `outputs/phase6k_runtime_v1/final/final_decision_v2.json`
- `outputs/phase6k_runtime_v1/runtime/e4r_equivalence/result.json`
- `outputs/phase6k_runtime_v1/runtime/formal_latency_e4r/result.json`
- `outputs/phase6k_runtime_v1/runtime/formal_latency_e4r/completion_audit.json`
- `docs/reports/phase6k_runtime_v1_final_report.md`

Opening the solver or holdout gates with E4R would contradict the
preregistered sequence. A future attempt requires a separately named and
preregistered model/runtime revision that first passes its own runtime gates.
