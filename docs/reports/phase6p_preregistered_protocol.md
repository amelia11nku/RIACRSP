# Phase 6P P1 预注册协议

状态：**`FROZEN_BEFORE_ANY_PHASE6P_SOLVER_QUALITY_OUTCOME`**。本协议冻结于任何 Phase 6P solver-quality outcome 之前，P0 commit 为 `f18896fdf05884a5507174392aa166ef65350058`。R13/R14 保持锁定，禁止 Gurobi 和新神经训练。

## 主策略

`P1_CSG_ADAPTIVE_PORTFOLIO` 沿用 H1 初始化与 Phase 6H 的确定性 20% neural eligibility；非 eligible iteration 保持原 ALNS。每次神经决策生成完整 24-rule bank 并按 destroyed-operation set 去重，对所有 unique targets 使用冻结 Phase 6N critic 评分，然后按预测 continuation advantage 排序，target ID 作为固定 tie-break，保留 top-6。

候选权重为 `1/r × mapped destroy weight`。P0 在 864 states 中发现 2 个跨 destroy-operator 去重 target，因此 mapped weight 冻结为全部唯一来源 destroy 权重的算术均值；target 产生真实结果后，每个唯一来源 operator 各执行一次原 ALNS 更新。repair 独立按五个当前权重 roulette，保留 8 trials、模拟退火、5/1/0.1-floor 奖励和 `reaction_factor=0.2`。

canonical related fallback 始终保留在 critic 的 fallback-relative 上下文中；它在 top-6 时正常参与采样，评分不完整、非有限或在采样前异常时作为安全回退。主策略没有 confidence、immediate-utility 或 support hard gate。

## 冻结 critic

Phase 6N critic 由 **9 个 whole-instance OOF checkpoints** 组成。实例按 `Scale × CF` 路由到 held fold，再对三个训练 seed 的 `predicted_continuation_advantage` 取算术平均。模型严格 FP32，历史 scorer 与 Phase 6O critic 调用均为 0。该对象只作为冻结 OOF critic 证据使用；Phase 6N 的 N5 失败及未创建 deployable bundle 的结论不变。

## 开发与正式边界

开发范围冻结为全部 **18 个 R12 CAUR-FIT instances**，三 seeds 为 `[746101, 746102, 746103]`；所有方法使用包含初始化和全部 live overhead 的 `2*|O|` 总 wall-clock。正式 R12 使用同一实例集与五 seeds `[746101, 746102, 746103, 746104, 746105]`，仅在开发和 runtime gates 通过后运行。

任何 comparator 执行前必须先建立 standing `outputs/frozen_2o_baselines/` registry 并产出 reuse audit。已存在结果只有在实例、代码、checkpoint、配置、seed、预算、计时边界、decoder、指标和环境全部相同时才可复用；禁止有利重跑。

P2 只能做 tiny smoke 与不变量验证。P2 通过后直接进入 P3，不增加 surrogate/OOF quality gate。runtime 只在 P3 promising 后执行，门槛仍为 neural p90 ≤30 ms、complete-live p90 ≤100 ms。

机器冻结证据位于 `outputs/phase6p_adaptive_portfolio_v1/preregistration/`。
