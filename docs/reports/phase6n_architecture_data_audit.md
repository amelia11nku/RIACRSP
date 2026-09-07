# Phase 6N 架构与数据审计

## N0 结论

**`N0_COMPLETE — PROCEED_TO_PREREGISTRATION`**。

Phase 6M 终态 commit `8efe396e6cca44493a18c07720c4effaf2add7c1` 与 `MODEL_REVISION_QUALITY` 已核验。R13/R14 未访问，未运行 Gurobi；Phase 6I-MR/6J/6K 的既有保护链重新通过，Phase 6L/6M 的 208 个文件（36,230,480 bytes）已在 N0 起点逐文件冻结。

审计支持进入一个新的、唯一可晋级的 candidate-conditioned CSG critic 预注册。这里的科学假设比“Phase 6L 没有候选身份”更精确：Phase 6L 已使用真实 destroy target 的 OP membership 做 mean/max/attention pooling；缺口是 target 边界关系、critical/bottleneck 子集和 critical synchronization boundary 没有被显式保留。

## 实际模型路径

冻结基座是 `outputs/phase6f/training/final_seeds/seed_660301/checkpoint_best.pt`（SHA256 `f1ccceb607b0e453dfb74e7aa7a946616001db8ec08c12dc9900a66c6f165fc7`），配置为 hidden=128、2 个 `FULL_CSG` relation blocks、4 heads、edge features、FP32。CSG tensorizer包含 20 个 canonical 与 20 个机械 reverse relations。encoder 在每个 state 上运行一次，并返回全部 8 类 node embeddings 与 graph embedding；node embeddings 在 global pooling 前后仍可供候选索引使用。

Phase 6L checkpoint `outputs/phase6l_legacy_score_decoupling_v1/training/oof/L1_SCORE_FREE_CONT_FROZEN/seed_706101/fold_0.pt` 明确引用同一基座。其 `batch.target_operation_indices`/`target_action_index` 先选出每个候选的 OP nodes，再做 target mean、max 和 graph-query attention；随后拼接 fallback action、差值及 score-free cheap context。当前聚合不显式表示：哪些 typed relations 穿过 destroy-set boundary、边界方向/关系族、candidate∩critical、candidate∩bottleneck、critical synchronization chain boundary。Phase 6N 应补这些量，并保持一次 state encoder、全 bank 批量 pooling。

## Immediate 与 continuation

冻结 R12 CAUR-FIT 共 6,809 candidates，其中 3,011 个 continuation-positive；2,006 个同时具有负 realized immediate utility，1,846 个低于 `-0.005`。

旧 Phase 6L predicted-immediate hard gate 会阻断 5,395 个 candidates（79.23%），其中 2,380 个实际 continuation-positive。被阻断集合的 positive rate 为 44.11%，平均 continuation advantage 为 -0.002823。realized immediate 与 continuation 的 Pearson/Spearman 为 0.346003/0.351314；旧 immediate head prediction 与 continuation 仅为 0.024798/0.007244。

因此旧 immediate head 只保留为诊断，不作为 6N 晋级硬门槛。负的“blocked mean”说明不能把全部 blocked candidates 描述成有益；有效结论是该门槛同时删除了大量有益 continuation actions，且其预测量不适合作为 continuation 的代理。

## Cheap-context collision

碰撞距离固定为六个自然落在 `[0,1]` 的 cheap features 的 RMS，且只在同 state、同 origin family 内找确定性最近邻。阈值为 distance `<= 0.10`；material/strong continuation gap 分别为 `>= 0.01`/`>= 0.02`。

6,807 个可比较 candidates 中，5,465 个具有 cheap-similar 最近邻；material collisions 为 3,968，strong collisions 为 2,656。在 288 个有同-family peer 的 state-best candidates 中，218 个按该门槛 poorly separated，占全部 288 states 的 75.69%。0.1-width 条件桶中有 633 个 multi-candidate buckets，candidate-weighted within-bucket variance 为 0.00026425。

这些是 representation-necessity 的支持证据，不是未来模型必然提升的证明。CSV 保留每个 anchor 的同-family nearest neighbor、target Jaccard、cheap values 与真实 outcome gap，避免只摘录有利案例。

## 有效样本与标签噪声

数据包含 6,809 candidate means、13,618 raw CRN rows、288 states、18 instances；每实例恰好 16 states。三个 structural folds 各有 6 instances、96 states，candidates 为 2,270/2,269/2,270。可用于 instance-grouped qualification 的独立顶层单位是 18 个 instances，不能把 6,809 行当成 6,809 个独立 state examples。

两个 CRN seeds 的 outcome Pearson 为 0.531043，平均/中位 absolute gap 为 0.023316/0.015557，p90/p99 为 0.058303/0.113307；符号不一致率为 26.51%。这支持增加等量、按进度分层的 R12 states，并继续按 whole instance/cell cross-fit。

## 搜索安全

生产路径无需修复 best-so-far。`decode_candidate` 每次调用 `check_schedule` 并对 infeasible 结果失败；`solve_csgni` 独立维护 `current` 与 `best`，只在严格 makespan 改善时更新 `best`。确定性 tiny 行为审计运行 20 iterations，观察到 19 次“接受更差 current”，最大 accepted current makespan 70.0，而返回 best 保持 39.0；两次相同 seed 的非计时轨迹完全一致，所有 candidate/current/best 均 feasible。

N1 必须在任何新 rollout 或 optimizer step 前冻结唯一 primary family、candidate boundary 索引、controlled fine-tuning、whole-instance folds、数据生成数量、四项 loss、empirical residual calibration 与 direct decision gates。R13/R14 继续锁定。

## 证据

- `outputs/phase6n_candidate_conditioned_csg_v1/audit/architecture_data_audit.json`
- `outputs/phase6n_candidate_conditioned_csg_v1/audit/immediate_vs_continuation.csv`
- `outputs/phase6n_candidate_conditioned_csg_v1/audit/candidate_representation_collision.csv`
- `outputs/phase6n_candidate_conditioned_csg_v1/audit/protected_phase6l_phase6m_evidence.json`
