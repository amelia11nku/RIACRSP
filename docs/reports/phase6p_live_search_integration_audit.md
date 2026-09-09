# Phase 6P P0 在线搜索集成审计

状态：**`PASS`**。起始 HEAD 与说明书要求一致，Phase 6O 终态及哈希保持不变；R13/R14 仍锁定，未运行 Gurobi 或任何新神经训练。

## 候选生成与多来源结论

对 Phase 6N 已授权的全部 **864 个 R12 CAUR-FIT states** 进行了当前候选生成器的真实回放，共重建 20,736 个原始 proposals、20,441 个去重 target。每个 state 均生成 24 条规则，去重后数量分布为 `{'21': 8, '22': 38, '23': 195, '24': 623}`；候选 ID、生成顺序、destroyed-operation set 以及存档聚合来源均与回放证据一致。

跨 destroy-operator 的去重 target 数为 **2**，涉及 2 个 states。检测到跨 destroy-operator 去重，P2 实现必须暴露全部来源，并以这些来源当前权重的均值作为候选 destroy 权重。

完整逐 target 来源表见 `outputs/phase6p_adaptive_portfolio_v1/audit/multi_origin_target_audit.csv`。

## 搜索语义

冻结 ALNS 仍使用 7 个 destroy 与 5 个 repair 算子，二者独立按当前权重 roulette；`destroy_fraction=0.15`、`candidate_trials=8`、`reaction_factor=0.2`。奖励为新全局最优 5、接受非最优 1、拒绝 0 并在更新时设 0.1 floor；权重更新为 `(1-0.2)w + 0.2*max(score,0.1)`。候选先按模拟退火更新 current，严格改善才更新 best，最终返回 best。

Phase 6H 的真实入口是 `FrozenLiveInference.decide` 加 `solve_csgni`，destroy count 由 `solve_csgni` 使用继承 ALNS 比例计算，8 trials 同样来自冻结 ALNS 配置。

Phase 6N 当前只有数据生成和 OOF 推理入口，没有正式 live solver 或 deployable bundle。可复用对象是 9 个冻结 whole-instance OOF checkpoints：按 structural held fold 路由，再对三个训练 seed 集成。Phase 6N 自身的 N5 失败结论不变，Phase 6P 的 P1 必须将这项复用边界及全部 checkpoint 哈希预注册。

LG_HGA_2O 的 `solve_lghga_2o` 与 `operation_budget` 可运行，既有真实时钟 smoke 通过，预算从 population initialization 前开始并使用 `2*|O|`。当前还没有 standing canonical baseline registry，需在 P3 comparator audit 中判定并冻结，不能把 tiny smoke 当成正式 comparator 结果。

## P0 判定

机器可读检查见 `live_search_integration.json` 与 `candidate_bank_semantics.json`。P0 通过，可以进入 P1 预注册。
