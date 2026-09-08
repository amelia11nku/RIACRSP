# Phase 6N 项目交接

当前终态为 **`MODEL_REVISION_REPRESENTATION`**。Phase 6N 已在 N5 结束；不得继续 N6–N9、R13 或 R14。

## Commit 与运行状态

- Starting commit：`8efe396e6cca44493a18c07720c4effaf2add7c1`
- N0_architecture_data_audit: `f38725a61e6340bd3047401d2e570fe88fcb7493`
- N1_preregistration: `bd0b245`
- N2_data_implementation: `c662750f5c5dc6d2986ca1a1faae8dfe07edbad7`
- N2_data_completion: `21d5b57`
- N3_model_implementation: `101309cfb0683626ef6e7f12be34b8087a2eee34`
- N3_smoke_qualification: `8e96d5634fefeed3c0175b06e7d7e9b5e17acb46`
- N4_training_and_N5_representation: `c2e7718b019aa221dabd4e5ba97e5035a18654b9`
- Terminal bundle 前 HEAD：`c2e7718b019aa221dabd4e5ba97e5035a18654b9`
- Ending evidence commit：包含本交接与 terminal JSON 的本地 commit；用 `git log -1 --oneline` 获取。

N2 worker PID 67432 与 N4 worker PID 88724 均已退出，没有活动 Phase 6N job。完整回归为 398 passed in 18.81 s。Phase 6L/6M 的 208 个冻结文件及更早保护链均复核通过。PASS_LOCKED_NOT_ACCESSED。

## 决策依据

N4 的 9-run outer OOF 工件完整。N5 common Spearman 相对 Phase 6L 提高 0.033946，但 paired raw-lift 改善为 -0.001105，18-instance bootstrap LCB 为 -0.003703。这是唯一失败硬项，协议要求直接判为 `MODEL_REVISION_REPRESENTATION`。

关键证据：

- `outputs/phase6n_candidate_conditioned_csg_v1/training/completion_integrity_audit.json`
- `outputs/phase6n_candidate_conditioned_csg_v1/quality/raw_representation_gate.json`
- `outputs/phase6n_candidate_conditioned_csg_v1/final/final_decision.json`
- `docs/reports/phase6n_representation_quality_report.md`
- `docs/reports/phase6n_final_report.md`

下一轮只能从新的科学假设和新的预注册边界开始。优先分析相关性与 top-selection utility 的错配、L-scale 负向 paired delta，以及 pooling/训练目标是否把排序信号转化为错误的 top candidate；不得对 Phase 6N 的 gate、seed 或 bootstrap 做事后修改。

## 新会话开场提示

```text
阅读 docs/reports/phase6n_project_handoff.md 与 outputs/phase6n_candidate_conditioned_csg_v1/final/final_decision.json。Phase 6N 已以 MODEL_REVISION_REPRESENTATION 在 N5 终止，N6-N9 与 R13/R14 均未运行。先复核 git status、终态哈希及保护 manifests，再为 ranking-to-selection utility 错配诊断定义一个全新预注册边界；不要调整 Phase 6N 的冻结 gate。
```
