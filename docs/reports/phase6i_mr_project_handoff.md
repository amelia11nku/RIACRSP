# Phase 6I-MR project handoff

## Current gate

- Decision: **`MODEL_REVISION`**
- CSG-NI v1 frozen: **false**
- R11: 288/288 complete; completion/replay audit PASS
- Frozen candidate: `U2_MIXED_OLD_NEW` three-seed ensemble
- Source commit at handoff: `6a8ac523adc2186c7cb359c2902c207d90184b5b`
- Gurobi executed: false

## Why the phase stops here

The solver remains feasible and within the frozen +1% ALNS non-inferiority
margin, and decoder evaluations are 51.59% lower. Nevertheless, only 9/900
diagnostic states intervene, selected realized utility and fallback-relative
lift are negative, and forced abstention evidence shows useful actions were
missed. These failures trigger `MODEL_REVISION`; R11 retuning is forbidden.

## Authoritative evidence

- `docs/reports/phase6i_mr_live_utility_revision_report.md`
- `outputs/phase6i_mr/final/final_status.json`
- `outputs/phase6i_mr/r11_validation/final_decision.json`
- `outputs/phase6i_mr/r11_validation/promotion_gate_table.csv`
- `outputs/phase6i_mr/r11_validation/completion_integrity_audit.json`
- `outputs/phase6i_mr/r11_validation/r11_result_hash_manifest.csv`
- `outputs/phase6i_mr/r11_validation/r11_ranking_calibration.json`
- `outputs/phase6i_mr/r11_validation/r11_anytime_runtime.json`

## Frozen boundaries

- Do not alter or rerun the existing R11 results.
- Do not retune thresholds, normalization, calibration, losses, or model choice
  using R11.
- Do not label the current artifact CSG-NI v1.
- Do not replace the manuscript Phase6H Core45 column with Phase6I-MR.
- Do not open Core/Sensitivity/Legacy for this failed promotion attempt.
- Do not begin the Phase6I-MR v2 backlog; it requires a prior
  `PROCEED_FREEZE_V1` decision.

## Corrected post-processing boundary

The frozen runner omitted `decoder_seconds` only from its derived
`validation_run_summary.csv`; all 288 immutable run JSONs contained the value.
`repair_phase6i_mr_r11_summary.py` restored that projection, recorded before
and after hashes, and did not alter scientific result payloads. The independent
completion audit then replayed every stored action sequence and validated every
trace.

## Next authorized work

The only scientifically valid continuation is to design a new, explicitly
versioned utility/gating revision with fresh selection and holdout splits. It
should address the immediate-versus-continuation target mismatch, candidate
label truncation, and support-aware cross-state abstention before any new long
experiment. That design must be frozen before accessing new selection data.
