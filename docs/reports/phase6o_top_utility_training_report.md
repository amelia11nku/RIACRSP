# Phase 6O Top-Utility 训练报告

状态：**COMPLETE，完整性审计 `PASS`；等待独立 OOF quality gate**。

Route B 完成 3 seeds × 3 whole-instance outer folds，共 9 runs。每个 run 的两个 inner 方向均完整运行 50 epochs，并按冻结词典序选择 epoch，再在两个 outer-training folds 上重训。选中 epochs 为 `[4, 6, 6, 5, 16, 9, 2, 11, 2]`。完整训练耗时 9603.68 秒。

外层 OOF 共 61,323 seed-candidate rows，ensemble 为 20,441 candidates/864 states。候选身份与顺序、更新后的 continuation truth、full-bank、feasibility、fold isolation、checkpoint 哈希和有限数值全部通过。所有 checkpoint 仅含 667,838 个 pooler/fusion/head 参数，不含 frozen encoder 参数。

Expanded selected lift 为 0.00634727，95% grouped LCB 为 0.00388466；common original selected lift 为 0.00503233，selection regret 为 0.03606904。这些是训练摘要，晋级由独立预注册 OOF gate 判定。

historical scorer calls 为 0；未运行 Gurobi；未访问 R13/R14。
