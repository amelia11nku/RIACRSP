# Phase 6E Supervised NI Validation Report

## 1. Executive conclusion

Phase 6E v2 已完成冻结边界内的离线监督学习验证。一次性内部 holdout 覆盖 20,000 个状态、472,452 个候选 target set；三种子 Full CSG 集成在 solver-relevant selected-action utility 上显著优于全部预注册对照，但绝对平均效用仍为 -1.5749%。三模型集成的 M/L 推理时延超过冻结预算，因此 Phase 6F **不得直接进入 live solver integration**，结论为 `REVISE_MODEL`。

v1 holdout 因 B2/B3 baseline alias 缺陷被正式作废且未用于结论；本文只使用 `phase6e-holdout-v2`。

```text
TENSORIZER_VALIDATED = TRUE
TARGET_SET_SCORER_TRAINED = TRUE
THREE_MODEL_SEEDS_COMPLETE = TRUE
NO_LABEL_LEAKAGE = TRUE
FULL_CSG_BEATS_RANDOM = TRUE
FULL_CSG_BEATS_RELATED = TRUE
FULL_CSG_BEATS_BEST_FIXED_OPERATOR = TRUE
FULL_CSG_BEATS_TABULAR_BASELINE = TRUE
FULL_CSG_BEATS_FLAT_SET_MODEL = TRUE
REALIZED_SYNCHRONIZATION_TOPOLOGY_ADDS_VALUE = TRUE
EDGE_FEATURES_ADD_VALUE = TRUE
MODEL_STABLE_ACROSS_SEEDS = TRUE
MODEL_STABLE_ACROSS_STRUCTURAL_REGIMES = TRUE
INFERENCE_COST_ACCEPTABLE_FOR_SOLVER_INTEGRATION = FALSE
PHASE6F_RECOMMENDATION = REVISE_MODEL
```

## 2. Frozen Phase 6C/6D boundary

- Phase 6C 数据版本：`phase6c-v1`；冻结 hash：`695307ac6193ecbbeb0f73e81a94ea20672ffd47aa9419bb37610be9d3161437`。
- Phase 6D 表示：`CSG-1.0`；schema SHA-256：`c76f5e0af972f31e2e2b0f53c8f2b1fff1afa48c08481357db3c32ac9c0a6c69`。
- 冻结提交：`14dde2171efbfe13bfe5c847f548532a56dd4844`。
- 环境冻结检查全部通过；未修改 Phase 6C 标签、Phase 6D schema 或调度/搜索语义。
- holdout 在 checkpoint、baseline 和 evaluation plan 全部冻结后才打开；`checkpoint_selection_after_open = false`。

## 3. Tensorization and data pipeline

采用 native PyTorch heterogeneous tensorizer，保留 8 类节点、20 类 canonical forward relations、显式 edge numeric features，并机械派生 reverse relations；派生边不改变 CSG-1.0 语义。实体 ID 仅用于 lookup/mapping，不进入数值 tensor。

冻结策略为 `A_PRETENSORIZED_SHARDED_CACHE`。缓存审计覆盖 405 个 source/cache shards、100,000 个状态、2,362,722 个动作、18.56 GiB；所有 source hash、cache hash、split count、schema hash 与 partial-file 检查均通过。训练按 state batching：每个状态只编码一次 CSG，再一次性评分该状态的全部候选动作。

## 4. Model architecture

冻结主模型为 `H128_L3_LR1E4`：hidden dimension 128、3 层、4 heads、dropout 0.1、7,705,857 parameters。每类节点有独立 input projection；relation-specific key/value projection 消费 edge attributes；type-specific mean pooling 与合法 graph-level features 形成 state embedding。实现为紧凑的 relation-aware heterogeneous attention encoder，不依赖 PyG。

## 5. Target-set action encoder

action encoder 对 target OP embeddings 使用 permutation-invariant mean、max、state-conditioned attention pooling，并拼接 state embedding 和 normalized target size。destroy origin operator 不作为主模型输入；base graph 与候选 target set 分离，不执行 action-specific graph rebuild。

## 6. Training objective

目标为 `L = 1.0 * pairwise_logistic_rank_loss + 0.25 * weighted_BCE`。pairwise 样本在 state 内确定性构造；BCE positive weight 只从 TRAIN 计数导出。训练使用 TRAIN 梯度，TRAIN_VALIDATION 只用于预注册的 `0.5 * pairwise_accuracy + 0.5 * NDCG` checkpoint/config selection，holdout 不参与任何选择。

## 7. Baselines

预注册对照包括：B0 exact random expectation、B1 fixed related、B2 validation-frozen best original operator（`related`）、B3 legal Phase 6C tabular diagnostic、flat-set neural control、static-CSG control、no-edge-features control。B3 v2 使用真实 target-set robust labels 与合法 pre-action features；已修复导致 v1 作废的 baseline alias 问题。

## 8. Sanity/leakage tests

mandatory sanity status 为 `PASS`：tiny overfit、label shuffle degradation、graph-state shuffle degradation、target-mask shuffle degradation 全部通过。正确映射的 tiny-set PR-AUC 为 0.9909，label-shuffle PR-AUC 降至 0.1544，target-mask-shuffle PR-AUC 降至 0.0782。这些结果支持“未检测到标签/target mapping 泄漏”，但不应解释为对所有潜在泄漏的形式化证明。

## 9. Validation model selection

小规模开发研究只比较 3 个预注册候选，最终冻结 `H128_L3_LR1E4`。三个独立最终种子均完成 6 epochs，最佳 checkpoint 均在 epoch 6；未报告或选择“最佳 holdout seed”。

| Seed | Validation objective | Holdout ROC | Holdout pairwise | Holdout NDCG | Holdout utility |
| --- | --- | --- | --- | --- | --- |
| 660201 | 0.7135 | 0.7646 | 0.5832 | 0.8579 | -1.5833% |
| 660202 | 0.7044 | 0.7499 | 0.5727 | 0.8540 | -1.7305% |
| 660203 | 0.7100 | 0.7590 | 0.5807 | 0.8566 | -1.6490% |

## 10. Internal-holdout predictive results

| Method | ROC-AUC | PR-AUC | Pairwise | Spearman | NDCG | Top-1 | Top-3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full CSG ensemble | 0.7624 | 0.6474 | 0.5832 | 0.2330 | 0.8586 | 0.1187 | 0.2874 |
| Phase6C tabular | 0.7714 | 0.6701 | 0.5820 | 0.2311 | 0.8586 | 0.1031 | 0.2571 |
| Flat set | 0.7154 | 0.5711 | 0.5588 | 0.1653 | 0.8472 | 0.0970 | 0.2376 |
| Static CSG | 0.7308 | 0.5857 | 0.5494 | 0.1390 | 0.8456 | 0.0980 | 0.2503 |
| No edge features | 0.7707 | 0.6626 | 0.5815 | 0.2274 | 0.8562 | 0.1069 | 0.2607 |

Full ensemble 的 ROC-AUC=0.7624、PR-AUC=0.6474、pairwise=0.5832、Spearman=0.2330、NDCG=0.8586。B3/no-edge 的 global ROC/PR 略高，说明 Full CSG 的优势主要体现在 state-conditioned ranking/selection，而不是全局二分类分离度。

## 11. Selected-action utility

| Method | Mean utility | Median | Positive | Top-3 | Mean regret |
| --- | --- | --- | --- | --- | --- |
| Full CSG ensemble | -1.5749% | -0.0998% | 49.09% | 28.43% | 4.0097% |
| Phase6C tabular | -1.6037% | -0.1361% | 48.71% | 25.78% | 4.0385% |
| Flat set | -1.7607% | -0.2806% | 47.37% | 24.99% | 4.1955% |
| Static CSG | -1.7938% | -0.2855% | 47.36% | 25.09% | 4.2286% |
| No edge features | -1.8328% | -0.1115% | 48.88% | 26.61% | 4.2676% |
| Fixed related | -2.1286% | -0.4971% | 44.11% | 11.73% | 4.5634% |
| Best fixed original | -2.1286% | -0.4971% | 44.11% | 11.73% | 4.5634% |
| Random expectation | -2.9728% | -1.3615% | 37.00% | 12.71% | 5.4077% |

状态配对 Wilcoxon 与 BH-FDR 结果：

| Comparator | Mean paired delta | W/T/L | BH-FDR p | Significant |
| --- | --- | --- | --- | --- |
| Random expectation | +1.3979% | 14505/2/5493 | 0 | TRUE |
| Fixed related | +0.5537% | 11464/136/8400 | 4.81e-113 | TRUE |
| Best fixed original | +0.5537% | 11464/136/8400 | 4.81e-113 | TRUE |
| Phase6C tabular | +0.0288% | 7880/4544/7576 | 0.0217 | TRUE |
| Flat set | +0.1857% | 7818/5277/6905 | 6.17e-15 | TRUE |
| Static CSG | +0.2189% | 7363/6233/6404 | 1.79e-16 | TRUE |
| No edge features | +0.2578% | 7563/5284/7153 | 1.61e-07 | TRUE |

Full CSG 对 B3 的平均增益仅 +0.0288%（BH-FDR p=0.02174），属于统计显著但实际幅度很小的优势。Full 选择动作的绝对平均效用为负，说明离线预测器尚不能保证单次动作平均改善。

## 12. Full CSG vs flat representation

Full 对 flat 的 paired selected-utility delta 为 +0.1857%（BH-FDR p=6.17e-15），并同时提高 ROC/PR、pairwise、Spearman、NDCG、top-1/top-3 和 selected-positive fraction。该 controlled comparison 支持 relational message passing 在离线 target-set 选择任务上提供增量价值，但增益规模仍不足以绕过 runtime gate。

## 13. Synchronization-topology ablation

Full 对 static-CSG 的 paired utility delta 为 +0.2189%（BH-FDR p=1.79e-16），主要 ranking 指标也一致更高，因此 realized synchronization/resource topology adds value。

Full 对 no-edge-features 的 selected-utility delta 为 +0.2578%（BH-FDR p=1.61e-07）。edge features 改善 solver-relevant utility、pairwise、Spearman、NDCG 与 top-k；但 no-edge 的 ROC/PR 更高，因此结论限定为“对 state-conditioned selection 有价值”，不宣称所有预测指标均改善。

## 14. Structural-regime robustness

| Dimension | Regime | States | Pairwise | NDCG | Selected utility | Positive selected |
| --- | --- | --- | --- | --- | --- | --- |
| scale | L | 6669 | 0.6152 | 0.8776 | +1.2505% | 67.00% |
| scale | M | 6669 | 0.5806 | 0.8573 | -0.7118% | 50.26% |
| scale | S | 6662 | 0.5538 | 0.8410 | -5.2674% | 29.98% |
| CF_level | CF1 | 6669 | 0.5884 | 0.8631 | -0.9483% | 52.59% |
| CF_level | CF2 | 6669 | 0.5831 | 0.8580 | -0.9750% | 49.51% |
| CF_level | CF3 | 6662 | 0.5782 | 0.8548 | -2.8028% | 45.15% |
| RI_level | RI1 | 6668 | 0.5866 | 0.8603 | -0.4830% | 52.52% |
| RI_level | RI2 | 6666 | 0.5842 | 0.8594 | -1.1418% | 50.29% |
| RI_level | RI3 | 6666 | 0.5788 | 0.8562 | -3.1003% | 44.45% |
| TI_level | TI1 | 6667 | 0.5704 | 0.8498 | -6.1462% | 26.50% |
| TI_level | TI2 | 6667 | 0.5895 | 0.8617 | +0.1751% | 54.67% |
| TI_level | TI3 | 6666 | 0.5898 | 0.8644 | +1.2466% | 66.08% |

所有主要 Scale/CF/RI/TI 子组的 pairwise accuracy 均不低于 0.5538，NDCG 均不低于 0.8410，且每个子组 selected utility 均高于 fixed related。S 与 TI1 的绝对效用明显偏弱，但对应 oracle/随机效用也低，属于困难分布而非相对 baseline 崩溃。相对 B3 则不是全子组一致领先，不能声称 universal dominance。

## 15. Seed stability

| Seed | Validation objective | Holdout ROC | Holdout pairwise | Holdout NDCG | Holdout utility |
| --- | --- | --- | --- | --- | --- |
| 660201 | 0.7135 | 0.7646 | 0.5832 | 0.8579 | -1.5833% |
| 660202 | 0.7044 | 0.7499 | 0.5727 | 0.8540 | -1.7305% |
| 660203 | 0.7100 | 0.7590 | 0.5807 | 0.8566 | -1.6490% |

三个 holdout seed 的 ROC range 为 0.0147，selected-utility range 为 0.0015；满足冻结稳定性规则。ensemble 改善 NDCG/top-k/selected utility，但不是所有 global AUC 的最佳单 seed。

## 16. Runtime and memory

| Scale | p90 single | p90 ensemble | Shared speedup | GPU peak | CPU RSS peak | 1,000 decisions |
| --- | --- | --- | --- | --- | --- | --- |
| S | 79.1 ms | 142.0 ms | 9.88x | 53.9 MiB | 1505.4 MiB | 142.0 s |
| M | 139.7 ms | 203.6 ms | 6.18x | 54.9 MiB | 1567.4 MiB | 203.6 s |
| L | 216.1 ms | 279.4 ms | 4.86x | 55.8 MiB | 1602.7 MiB | 279.4 s |

共享 state encoding 相对 naïve repeated graph encoding 获得 4.86x–9.88x median speedup，GPU inference peak 仅 53.9–55.8 MiB，memory gate 通过。冻结的 150 ms p90 ensemble budget 只在 S 通过；M/L 分别为 203.6/279.4 ms，1,000 次决策预算亦失败。CPU RSS 是包含 Python、数据与已加载模型的进程峰值，不等同于单次推理增量。

## 17. Failure cases

1. S 和 TI1 分层的 absolute selected utility 较低；需要进一步改善 utility-aware calibration/objective。
2. Full 与 B3 的总体 utility 差距很小，并在若干结构子组反转。
3. no-edge 的 ROC/PR 高于 Full，提示 edge features 对全局分类有 trade-off。
4. 三模型 sequential ensemble 在 M/L 不满足 live iterative-search 时延预算。
5. v1 holdout baseline alias 缺陷证明 baseline identity/hash 审计不可省略；v1 已作废，不进入任何科学结论。

## 18. Scientific interpretation

Q1：Full CSG 在 solver-relevant utility 上显著优于最强合法非图 B3，但 global ROC/PR 不占优，效应量很小。Q2：Full 显著优于 flat，支持 relational topology 的增量价值。Q3：Full 显著优于 static，支持 realized synchronization/resource relations。Q4：Full 对 random、related、best fixed 和 B3 均有正的配对 utility delta。Q5：主要结构分层未相对 fixed-related 崩溃，但困难分层和对 B3 的局部反转必须保留。Q6：当前 ensemble inference 不满足未来 live integration 预算。

## 19. Phase 6F recommendation

`PHASE6F_RECOMMENDATION = REVISE_MODEL`。

建议保留冻结 CSG-1.0 和当前 action-set 语义，优先做聚焦修订：以 selected utility/regret 为直接验证目标改进 objective/calibration，并将三种子 ensemble 蒸馏或压缩为单模型/共享编码部署路径。修订后必须重新冻结阈值并在未见数据上复验；在 latency gate 通过前，不进入 live ALNS/NI integration。

## 20. Reproducibility checklist

- [x] Phase 6C dataset hash 与 Phase 6D schema hash 冻结并验证。
- [x] tensor schema、405-shard cache、100,000 states/2,362,722 actions 全量 hash 审计。
- [x] tiny overfit 与三类 shuffle sanity 通过。
- [x] config/checkpoint 仅由 TRAIN_VALIDATION 选择。
- [x] 三个独立 seed 的 best/last/optimizer/config/schema metadata 保留。
- [x] v2 holdout 一次性访问记录与全部输出 hash 保留。
- [x] state-paired Wilcoxon、BH-FDR、结构分层结果保留。
- [x] 10 组图均由 `scripts/generate_phase6e_report.py` 生成 PNG/PDF。
- [x] inference S/M/L 分组件、memory 与 100/500/1,000 decisions 投影保留。
- [ ] live solver integration：按 Phase 6E stop condition 明确禁止，等待模型修订与人工评审。
