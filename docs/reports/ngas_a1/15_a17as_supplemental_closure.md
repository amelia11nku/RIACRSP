# NGAS A1.7A-S 补充诊断闭环

终态为 `NGAS_A1_7AS_PASS_SUPPLEMENTAL_CLOSURE`。本阶段从冻结的 A1.7A-R 干净轨迹池中确定性选择
27 个非 R12 状态，严格覆盖 S/M/L × EARLY/MIDDLE/LATE，每格 2 个 TRAIN
和 1 个 VALIDATION。全候选池审计完成 9,560 个动作、76,480 次有序直接试验
和 152,960 次续算解码；完成、哈希与边界检查通过 21/21。

U0 仅在 16/27 状态提供非恒定信号，且 critic 对正 U0 最优动作的 Top-1
命中为 0/16。U0 平坦时，U1 在 2/11 状态恢复非恒定且正改进支持，U3 在
11/11 状态恢复非恒定稳健性信号。U1/U2 修正后全状态 Top-1 一致 27/27，
有效状态平均 rho 为 0.999989，成本归一化没有实质改变短期效用排序。

旧 U1/U2 数值不一致由重复报告路径的并列值处理造成：`max()` 继承动作
列表顺序，而规范路径按效用降序、动作 ID 升序确定性解并列。修正值从
4/5 变为 5/5，只更新诊断派生数据、报告和图件，原始结果与求解器未变。

证据继续支持将 A1.7B 作为独立冻结的 adaptive trial racing 开发小试验，
并支持 A1.7C 保持 regime-aware/multi-task 方向；两者均未在本阶段启动。
生产 `compact_relational` C1、refresh=20、portfolio、candidate_trials=8、
repair、candidate bank、decoder 及搜索逻辑全部保持不变。

R12 仍为 `DEVELOPMENT_EXPOSED`；R13/R14 保持锁定；CORE45 未用于本阶段；
未运行 Gurobi。完整结论和机器证据分别见
`reports/ngas_a17as_final_report.md`、终态 JSON 与结果清单。
