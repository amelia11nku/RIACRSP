# Phase 6L 项目交接

当前终态为 **`MODEL_REVISION_QUALITY`**。Phase 6L 已在 L4 结束；不要继续 L5/L6、solver、R13 或 R14。

## 当前边界

- Starting commit：`c52a37f64573d3fa822282960a6a9cbf6222f375`
- L0：`15d466fc539f2e9db0d72ffaef95f153d235bed4`
- L1：`875e76b2a931646b8569fa3a7752f5aca70a183b`
- L2：`a8adc02`
- L3 launcher：`48ec809b3c1dd430cd741c19faa0bc9af275d06a`
- L3 evidence：`45de6a8`
- Terminal bundle 前 HEAD：`45de6a8528d1d42b4a50ac94de4959e35950ea17`
- Ending evidence commit：包含本交接与 terminal JSON 的本地 commit；用 `git log -1 --oneline` 获取其不可自引用哈希。

## 决策依据

primary raw essential checks 全通过，但 retained gate 为 0。失败项是 `retained_gate_with_scale_coverage`、`positive_gated_lift_lcb`、`nonnegative_gated_scale_lift`。不得用 raw lift 为正覆盖 gate 失败，也不得调整冻结阈值后续跑。

关键证据：

- `outputs/phase6l_legacy_score_decoupling_v1/training/completion_integrity_audit.json`
- `outputs/phase6l_legacy_score_decoupling_v1/quality/development_quality.json`
- `outputs/phase6l_legacy_score_decoupling_v1/quality/ablation_comparison.json`
- `outputs/phase6l_legacy_score_decoupling_v1/quality/phase6j_j1_comparison.json`
- `outputs/phase6l_legacy_score_decoupling_v1/final/final_decision.json`

## 锁定与环境

回归为 352 passed in 16.27 s。保护审计验证 6,636 + 21,365 predecessor output files 和 139 个 tracked files。PASS_LOCKED_NOT_ACCESSED。PID 36703 已退出，没有后台任务、checkpoint/resume 或 ETA。

后续只能从新的科学假设与新的预注册边界开始。建议重点研究 score-free winner support/calibration，而不是降低 coverage 门槛；本轮所有 R12 OOF 和失败 gate 结果必须保持冻结。

## 新会话开场提示

```text
阅读 docs/reports/phase6l_project_handoff.md 与 outputs/phase6l_legacy_score_decoupling_v1/final/final_decision.json。Phase 6L 已以 MODEL_REVISION_QUALITY 在 L4 终止，R13/R14 锁定。先审计当前 git status、保护 manifests 与终态哈希，再分析一个全新且需单独预注册的 score-free confidence/support 模型修订；不要复用 Phase 6L 结果调 gate 或打开 holdout。
```
