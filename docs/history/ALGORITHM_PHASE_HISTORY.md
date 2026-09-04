# RI-ACRSP algorithm phase history

This index is a navigation layer, not a replacement for frozen protocols,
machine-readable gates, or final validation reports. The listed commit is the
first/last authoritative repository checkpoint containing the cited final
report; older material remains recoverable from Git history.

| Phase | Purpose | Final status | Key conclusion | Authoritative report | Historical commit | Successor |
|---|---|---|---|---|---|---|
| 2 | Canonical generator, decoder, feasibility, exact-tiny, graph and BC foundations | `READY_FOR_PPO=TRUE` | Canonical and decoder foundations passed their gates. | [`phase2_final_validation_report.md`](../reports/phase2_final_validation_report.md) | `2fc85b2e56f5cc875c1bfec1f70f6a8d5b7625e3` | Phase 3 |
| 3 | Constructive PPO validation | `READY_FOR_CSG=TRUE` at the phase-local gate | PPO was viable for continued research, without establishing global superiority. | [`phase3_ppo_validation_report.md`](../reports/phase3_ppo_validation_report.md) | `c62f1087dcdeadeff91d376b2d31472f26af56f6` | Phase 4 |
| 4 | Model-capacity pilot | `D128-L3_SELECTED` | D128-L3 was selected for the full-data BC pilot under the isolated capacity study. | [`phase4_capacity_study_report.md`](../reports/phase4_capacity_study_report.md) | `f4f5c8bf99e6fdf3bb93437aaeb07991297991ec` | Phase 5A |
| 5A | Stable PPO validation and drift diagnosis | `PHASE5A_CANONICAL_GATE=FAIL` | Development gains did not justify replacing BC; operation-stage drift remained blocking. | [`phase5a_stable_ppo_validation_report.md`](../reports/phase5a_stable_ppo_validation_report.md) | `d6c4eb28062c6d1189fb078fd6fb626c8999638e` | Phase 5B |
| 5B | Hierarchical constructive policy | `CONSTRUCTIVE_POLICY_READY=TRUE` | Freezing BC operation/island decisions and learning downstream W/F decisions passed the phase gate. | [`phase5b_hierarchical_policy_validation_report.md`](../reports/phase5b_hierarchical_policy_validation_report.md) | `42afb6d293ac3598bbc429ed42f7c31ce47142bb` | Phase 5C |
| 5C | Controlled CB1 benchmark and baselines | `CORE_STRUCTURAL_GATES_PASSED=TRUE` | DEV, Core and paired Sensitivity boundaries were frozen and kept distinct from Legacy-130. | [`phase5c_controlled_benchmark_report.md`](../reports/phase5c_controlled_benchmark_report.md) | `e9aa48418bf91143f10537ac5256003a06b2ff60` | Phase 6A |
| 6A | ALNS implementation audit and observational search diagnosis | `DIAGNOSIS_COMPLETE` | Instrumentation preserved search behavior; observational logs motivated counterfactual labels. | [`phase6a_alns_search_diagnosis_report.md`](../reports/phase6a_alns_search_diagnosis_report.md) | `e7538ad790e9e8823565f760cde9e6611f0c4cbd` | Phase 6B |
| 6B | Counterfactual target-set pilot | `SCALE_WITH_REVISED_ARM_DESIGN` | Set-level signal was promising, while operation labels required a revised scaled design. | [`phase6b_counterfactual_data_pilot_report.md`](../reports/phase6b_counterfactual_data_pilot_report.md) | `78c6640ba10592da44ba2dfe0e898e46f13ba681` | Phase 6C |
| 6C | Scaled counterfactual dataset | `PROCEED_TO_CSG_DEFINITION` | Set/operator signal passed; direct operation scalar labels did not. | [`phase6c_scaled_counterfactual_dataset_report.md`](../reports/phase6c_scaled_counterfactual_dataset_report.md) | `113831bcc51765e33f554338663cb6259085e25a` | Phase 6D |
| 6D | CSG-1.0 formal definition and validation | `STRUCTURAL_ACCEPTANCE_PASS` | Deterministic graph construction and exact semantic mappings passed; model value remained unevaluated. | [`phase6d_csg_validation_report.md`](../reports/phase6d_csg_validation_report.md) | `14dde2171efbfe13bfe5c847f548532a56dd4844` | Phase 6E |
| 6E | First supervised neural intervention model | `REVISE_MODEL` | Offline selection signal improved, but absolute utility and M/L latency blocked live integration. | [`phase6e_supervised_ni_validation_report.md`](../reports/phase6e_supervised_ni_validation_report.md) | `33f6ad2c7c3a4badbd9e2fd556968d2f346fb423` | Phase 6F |
| 6F | Utility-aware compact-model revision | `PHASE6G_RECOMMENDATION=PROCEED_TO_LIVE_NI_SOLVER_INTEGRATION` | The compact single model and latency gate passed on the fresh revision holdout. | [`phase6f_utility_aware_model_revision_report.md`](../reports/phase6f_utility_aware_model_revision_report.md) | `0dbf0b9f3e81a71ce59f74253569eb98b3817a38` | Phase 6G |
| 6G | Live CSG-NI integration | `PHASE6H_RECOMMENDATION=REVISE_CALIBRATION` | DEV performance passed, but calibration instability and high state drift blocked final evaluation. | [`phase6g_live_csgni_integration_report.md`](../reports/phase6g_live_csgni_integration_report.md) | `87ef6837c13dcad034ecd5062b76eeb3ef836872` | Phase 6H |
| 6H | Leakage-safe live calibration revision | `MODEL_REVISION` | Solver performance passed, but utility calibration/drift failed; CSG-NI v1 was not frozen and Core45 evidence remained the manuscript result. | [`phase6h_live_calibration_report.md`](../reports/phase6h_live_calibration_report.md) | `7c448000be69084511af181f52cad0a8db71479a` | Phase 6I-MR |
| 6I-MR | Live-utility model revision with frozen R11 holdout | `MODEL_REVISION` | R11 completed 288/288 with integrity PASS; `U2_MIXED_OLD_NEW` was not promoted and R11 is permanently closed to tuning. | [`phase6i_mr_live_utility_revision_report.md`](../reports/phase6i_mr_live_utility_revision_report.md) | `fa97fec6cac2a94cd1bdc39cbb8c0001f411fa2c` | Phase 6J |

## Preserved boundaries

- Phase 6H Core45 remains the manuscript evidence; Phase 6I-MR does not replace it.
- Phase 6I-MR R11 is final holdout evidence only and must not be reused for
  tuning, feature design, threshold selection, model selection, or rescue
  analysis.
- The production proposal bank remains 24 deterministic rules followed by
  target-set deduplication. `candidate_trials=8` is a decoder/repair trial
  count per target, not an eight-arm candidate bank.
- Removed active-tree artifacts are listed with hashes and recovery commands
  in [`archive_manifest.csv`](archive_manifest.csv).
- Compact ignored-output preservation for Phase 6I-MR is indexed by
  [`phase6i_mr_local_evidence_manifest.csv`](phase6i_mr_local_evidence_manifest.csv).
