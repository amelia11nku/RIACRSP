# NGAS-A1 持久神经先验搜索报告

## A1.4 结论

A1.4 终态为 **`NGAS_A1_4_PASS_C1_FIXED_REFRESH`**。生产求解器固定为 C1 enhanced compact critic、持久缓存先验、独立在线组合器和 20 次迭代固定刷新。R1 仅保留解释性角色。A1.5 完整在线刷新延迟资格测试已解锁；A1.6、R13 和 R14 尚未解锁。

## 搜索机制选择

小规模 R12 开发协议覆盖 S/M/L、CF1/CF2/CF3、C01/C02，使用三个固定种子。六种模式各运行 9 次，共 54 次。相对于 `ONLINE_PORTFOLIO_ONLY`：

| 模式 | 最终配对增益 | anytime AUC 增益 | 胜/平/负 | critic 影响中位数 | 引导迭代/critic 调用中位数 |
|---|---:|---:|---:|---:|---:|
| `NEURAL_PRIOR_ONLY` | 0.2603% | 0.8144% | 3/0/6 | 1.0000 | 1068.0000 |
| `NEURAL_X_PORTFOLIO` | -0.9785% | -0.0496% | 3/0/6 | 1.0000 | 1052.0000 |
| `ONE_SHOT_TOP1` | -1.2032% | -0.2049% | 1/1/7 | 0.000923 | 1.0000 |
| `PERSISTENT_FIXED_REFRESH` | **0.9127%** | **0.6735%** | **6/0/3** | **1.0000** | **19.8310** |
| `PERSISTENT_EVENT_REFRESH` | 0.3982% | 0.4912% | 6/0/3 | 1.0000 | 6.5902 |

固定刷新和事件刷新均通过预注册资格门。固定刷新在第一排序指标最终配对增益上胜出，因此被直接选中。一次性 top-1 的持续影响接近于零，说明有效集成需要缓存先验持续参与后续动作分布。

全部 54 次运行通过内容哈希、有限数值、真实 1.0 终止检查点和最终排程精确回放。总计执行 469674 次 decoder evaluation、58722 次搜索迭代、1604 次 critic 调用和 2118 次动作库刷新。

## C1/R1 求解器级解释性消融

机制冻结后，在相同实例、种子、held fold 0、动作库、修复、刷新、在线组合器、接受逻辑、RNG 命名空间和墙钟预算下执行 9 对 C1/R1 运行。

R1 相对 C1 为 7 胜 0 平 2 负；平均最终增益为 `-0.0735%`，即整体最终质量略差；平均 anytime AUC 增益为 `0.6081%`。该结果没有形成足以重新开启表示选择的大幅优势，且协议明确禁止借此晋升 R1。

| 指标 | C1 | R1 |
|---|---:|---:|
| 平均最终 makespan/H1 | 0.932599 | 0.933007 |
| 平均 anytime AUC/H1 | 0.954999 | 0.948893 |
| 平均 decoder evaluations | 8208.11 | 7789.56 |
| 平均 critic 调用 | 51.89 | 49.22 |
| 平均神经开销 | 4.2744 s | 5.9060 s |
| 观察到的完整刷新 p90 | 211.12 ms | 249.05 ms |

C1 仍是唯一生产表示。18/18 raw 的 checkpoint 哈希、单变量边界、墙钟预算、遥测与终态重放全部通过审计。

## A1.5 边界

A1.4 中观察到的 C1 完整刷新延迟已经明显高于 30 ms，且随规模从 S 到 L 增长：S/M/L p90 分别约为 65.44/139.65/223.33 ms。主要平均开销来自重复的 critical/CSG 构造、动作特征和张量化，而 C1 模型前向约为 2–3 ms。

A1.5 必须在独立冻结协议下测量并优化完整路径，不能用模型前向或部分缓存计时替代。若语义等价优化后的正式完整刷新 p90 仍超过 30 ms，终态应为 `NGAS_A1_REVISE_RUNTIME`，不得进入 A1.6。

机器可读证据：

- `outputs/ngas_a1/search_integration_v1/audit/completion_audit.json`
- `outputs/ngas_a1/search_integration_c1_r1_v1/summary.json`
- `outputs/ngas_a1/search_integration_c1_r1_v1/audit/completion_audit.json`
- `outputs/ngas_a1/search_integration_c1_r1_v1/result_manifest.json`，SHA256 为 `4a0e90213e5f74c5b770702c6309ea45737a57b262023519f6e990ad57d1a1a8`
