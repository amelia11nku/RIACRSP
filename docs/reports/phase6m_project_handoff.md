# Phase 6M 项目交接

当前终态为 **`MODEL_REVISION_QUALITY`**。Phase 6M 已在 M4 结束；不得继续 M5/M6、bundle、solver、R13 或 R14。

## Commit 与运行状态

- Starting commit：`8f0d37397af82e6a8bfdb437c88459712cdf5ec7`
- M0_failure_attribution: `6f99e2d`
- M1_preregistration: `1c42559`
- M2_implementation: `f99cdc4`
- M3_training: `be74fe2`
- M4_quality: `4af628e`
- Terminal bundle 前 HEAD：`4af628e5d9229c7cf37432765c6b853c361a6287`
- Ending evidence commit：包含本交接与 terminal JSON 的本地 commit；用 `git log -1 --oneline` 获取。

M3 worker PID 52577 已退出，没有活动 job。完整回归为 372 passed in 16.28 s。Phase 6L 的 77 个冻结文件和前代 Phase 6I/6J、Phase 6K manifests 均复核通过。PASS_LOCKED_NOT_ACCESSED。

## 决策依据

Raw score-free ranker 的 Spearman、所有 scale Spearman、selected lift 与 bootstrap LCB 均为正。修订 support 达到 candidate 99.192%、winner 100%。但预注册 gate 0/18 retained，最宽松 selector LCB 最大值 -0.012257，正式 intervention 为 0/288。因此 quality gate 失败，不能以 raw ranking 或 support 改善替代 intervention readiness。

关键证据：

- `outputs/phase6m_selective_confidence_v1/training/completion_integrity_audit.json`
- `outputs/phase6m_selective_confidence_v1/quality/development_quality.json`
- `outputs/phase6m_selective_confidence_v1/quality/gate_grid.csv`
- `outputs/phase6m_selective_confidence_v1/final/final_decision.json`
- `docs/reports/phase6m_final_report.md`

下一步只能从新的科学假设与新的预注册边界开始。优先诊断 selector scale、fold shift 和 immediate-utility head；不得对 Phase 6M 的 18 个 gate 做事后扩展或降低门槛。

## 新会话开场提示

```text
阅读 docs/reports/phase6m_project_handoff.md 与 outputs/phase6m_selective_confidence_v1/final/final_decision.json。Phase 6M 已以 MODEL_REVISION_QUALITY 在 M4 终止，R13/R14 锁定。先复核 git status、终态哈希与保护 manifests，再为下一轮 selector scale/fold shift、immediate-utility 或不可晋级 offline-teacher 诊断定义一个全新预注册边界；不要调整 Phase 6M 的冻结 gate。
```
