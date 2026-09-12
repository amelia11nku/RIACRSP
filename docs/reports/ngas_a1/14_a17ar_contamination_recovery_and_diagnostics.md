# NGAS A1.7A-R 污染恢复与轨迹诊断

终态为 `NGAS_A1_7AR_PASS_CONTAMINATION_RECOVERED`。程序化重建确认：最终 C1 的 72 个状态、6,465 个
joint-action 标签来自全部 18 个 R12 算例，且 A1.6R 也使用相同的 18 个
算例与三个 Native16 seed。因此 R12 永久归类为 `DEVELOPMENT_EXPOSED`，
A1.6R 只保留开发阶段工程与性能证据，不再承担独立测试或未见泛化证据。

数据治理已经机器化：81 个 TRAIN 算例与 27 个 VALIDATION 算例按 ID 和
内容哈希与 R12、R13、R14、CORE45 完全隔离；R13/R14 继续锁定。本阶段完成
108 条干净轨迹、540 个状态、36 状态的 12,775-action 全候选池诊断，以及
36 状态 × 5 offset 的 staleness 审计。生产 C1、搜索策略、20-iteration
refresh 和 `candidate_trials=8` 均未改变。

诊断表明，U0/U1 有效状态稀少且 critic Top-1 未命中正效用最佳 action；
U1 与成本归一化 U2 排序几乎相同；portfolio 的确定性 Top-1 调整很少且未
显示系统收益；少 trial cap 会造成明显原始候选质量损失，但正效用损失很小。
因此建议分别预注册 A1.7B adaptive trial racing 和 A1.7C trajectory-aware
C1-v2 小试验，本阶段不自动启动。

完整 21 项结论见 `reports/ngas_a17ar_final_report.md`；机器终态与全量哈希见
`outputs/ngas_a1/trajectory_utility_a17ar_v1/final_decision.json` 和
`outputs/ngas_a1/trajectory_utility_a17ar_v1/result_manifest.json`。
