# Phase 6L Legacy-Score Decoupling 最终报告

## 最终决策

**`MODEL_REVISION_QUALITY`**，停止边界为 **L4、早于 development runtime 与新的 R12 qualification**。

唯一 primary `L1_SCORE_FREE_CONT_FROZEN` 完成 3 seeds × 3 outer folds，完整性审计通过。它在不使用历史 score 输入的条件下保留了正的 raw 排序与 utility 信号：Spearman 0.177723，selected lift 0.007516，grouped-bootstrap LCB 0.002866，ECE 0.010884。但是冻结 gate 网格 retained 数为 0，无法形成满足每 scale coverage 与 gated-lift 条件的干预策略，因此不具备晋级资格。

## 证据边界

- 继承的 Phase 6K：E4R neural p90 25.980939 ms 通过，完整 live p90 109.494985 ms 失败；Phase 6K 终态为 `MODEL_REVISION_RUNTIME`。
- Phase 6L 新证据：288 states、6,809 candidates、三 seed OOF、单 seed不可选择消融、full-bank/support/origin/分层与 Phase 6J J1 对照均已完成。
- L5/L6 runtime：未授权执行，因为 L4 quality 失败。
- solver：未运行；deployable bundle 未创建。
- R13/R14：PASS_LOCKED_NOT_ACCESSED。

冻结 Phase 6J J1 与 score-free primary 的 raw 指标接近；J1 保有覆盖 S/M/L 的 retained gate，而 score-free 模型没有。诊断性 canonical-fallback 重评分仍保留 J1 gate，说明差异不能只归因于 10 个状态的标签重锚定。下一轮应作为全新、预注册的模型修订研究 score-free confidence/support 表征；不得在本轮调 gate 网格、seed 或 coverage 下限。

## 完整性与复现

完整回归：**352 passed in 16.27 s**。保护审计通过：Phase 6I/6J 6,636 files，Phase 6K 21,365 files，tracked predecessor 139 files。L3 worker PID 36703 已退出；无仍在运行的 Phase 6L job。

起始 commit：`c52a37f64573d3fa822282960a6a9cbf6222f375`。生成终局证据前的最新冻结 stage commit：`45de6a8528d1d42b4a50ac94de4959e35950ea17`。最终 evidence commit 为包含本报告与 `outputs/phase6l_legacy_score_decoupling_v1/final/final_decision.json` 的后续本地 commit。
