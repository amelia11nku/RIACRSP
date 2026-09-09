# Phase 6O Neural Shortlist Screening and Top-Utility Alignment 最终报告

## 最终决策

**`MODEL_REVISION_TOP_UTILITY`**，停止边界为 **Route B OOF quality gate**。

O0 使用冻结 Phase 6L/6N 证据否决了 k≤6 的 Route A，并在新预注册下执行 Route B。定向重标注完成 864 states、4,786 candidates 和 14,358 additional seed rows；随后完成冻结全 Phase 6F encoder 的 3 seeds × 3 whole-instance OOF。训练完整性审计为 PASS：9/9 runs、61,323 seed-candidate predictions、20,441 ensemble candidates、所有检查点哈希、candidate identity/order、full-bank、feasibility、fold isolation 和 FP32 score-free 边界均通过。

| 范围 | Spearman | Pairwise | NDCG@1 | Raw lift | Lift LCB | Regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 6O common 288 | 0.158695 | 0.556355 | 0.602179 | 0.005032 | 0.001242 | 0.036069 |
| Phase 6O expanded 864 | 0.212809 | 0.575892 | 0.617148 | 0.006347 | 0.003885 | 0.035518 |

Common raw lift 必须不低于冻结 Phase 6L 的 0.007516，实际为 0.005032；common regret 必须不高于 0.033866，实际为 0.036069。两项均失败。Expanded 正 lift/LCB、所有规模非负 lift 和五类 selected-origin diversity 均通过，说明本轮目标仍保留可测信号，但没有把它转化为优于冻结 Phase 6L 的 common top selection。

在同一 Phase 6O targeted-relabel truth 上，Phase 6L 的 lift 为 0.005360，Phase 6O paired delta 为 -0.000328，18-instance bootstrap 95% CI 为 [-0.004391, 0.003716]。这一诊断没有改变预注册常数门槛。

## 完整性与停止边界

- 正式训练 wall time：9603.68 秒；后台任务已完成并退出。
- 完整回归：**425 passed in 20.00 s**。
- 受保护 Phase 6N 证据及其 Phase 6L/6M/predecessor chain 复核通过，共 2504 个 Phase 6N 文件。
- historical scorer calls = 0；Gurobi 未运行；R13/R14 未访问。
- direct decision、development pilot、runtime、formal R12 均因 OOF gate fail 而未运行。

本阶段不能继续增加 selector、放宽 lift/regret 门槛或从当前 OOF 事后挑选 seed/epoch。任何后续模型修订都需要新的科学假设与新预注册边界。
