# NGAS-A1.4 主试验结果与机制冻结

## 冻结结论

54 次小规模 R12 开发运行全部完成，独立完成审计状态为 **PASS**。预注册选择规则将 `PERSISTENT_FIXED_REFRESH` 冻结为 A1.4 搜索机制。下一道且唯一获准的门是匹配 C1/R1 解释性求解器消融；A1.5 尚未解锁。

## 主结果

相对于内部对照 `ONLINE_PORTFOLIO_ONLY`：

| 生产候选 | 平均最终增益 | 平均 anytime AUC 增益 | 胜/平/负 | critic 影响中位数 | 每次 critic 调用引导迭代中位数 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `PERSISTENT_FIXED_REFRESH` | 0.9127% | 0.6735% | 6/0/3 | 1.0000 | 19.8310 | 合格并入选 |
| `PERSISTENT_EVENT_REFRESH` | 0.3982% | 0.4912% | 6/0/3 | 1.0000 | 6.5902 | 合格，排序第二 |

固定刷新在预注册的第一排序指标“平均配对最终增益”上更优，因此不需要启用第二或第三排序条件。该选择不是按单个实例或有利种子作出的。

作为机制诊断，一次性 top-1 的 critic 影响中位数仅为 `0.000923`，每次调用只引导 1 次迭代；固定刷新则保持全程神经影响并在调用间复用约 19.83 次迭代。这验证了 A1.4 的核心改变来自持续搜索集成，而不是单次动作推荐。

## 完整性与运行边界

- 正式 raw：54/54，全部内容哈希匹配，范围无缺失或额外文件。
- 最终排程：54/54 通过冻结 decoder 重放，makespan 精确一致。
- 数值：所有 JSON 数值有限。
- 总 decoder evaluations：469674。
- 总搜索迭代：58722。
- 总 critic 调用：1604。
- 总动作库刷新：2118。
- 墙钟上限最小超越：0.000120 秒；最大超越：0.075128 秒，均来自允许完成的原子调用。
- 遥测：每次运行均包含 0.10、0.25、0.50、0.75 和真实终止 1.0 检查点。

首次 launcher 启动的 worker 实际一直在宿主侧运行。一次因沙箱进程视图不可见而启动的 systemd 备用 worker，在发现目标 raw 已存在时由不可变 `open('x')` 写入保护立即失败退出；它没有覆盖或改变任何正式结果。最终只有原 PID 完成全部 54 次运行，raw 清单与重放审计均通过。

机器可读证据：

- `outputs/ngas_a1/search_integration_v1/primary_summary.json`
- `outputs/ngas_a1/search_integration_v1/mechanism_decision.json`
- `outputs/ngas_a1/search_integration_v1/raw_manifest.json`
- `outputs/ngas_a1/search_integration_v1/audit/completion_audit.json`
- `outputs/ngas_a1/search_integration_v1/result_manifest.json`，SHA256 为 `5d9c138eb0f908f01eca85cfdf30585421ecaccc268545087addfdb66ba95c4d`

R1 仍为解释性消融分支，无论下一步求解结果如何都不具备生产资格。R13、R14 与 Gurobi 保持锁定。
