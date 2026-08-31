# Phase 6F Utility-Aware Model Revision Report

## 1. Executive conclusion

Phase 6F 已在冻结边界内完成 utility-aware CSG-NI 的离线修订验证。新的 R06 revision holdout 覆盖 81 个全新实例、8,100 个状态和 191,416 个候选 target set；部署 checkpoint 在 R06 标签开封前按 `TRAIN_VALIDATION` 冻结为 seed `660301`，SHA-256 为 `f1ccceb607b0e453dfb74e7aa7a946616001db8ec08c12dc9900a66c6f165fc7`。

部署策略在 R06 上的平均 selected utility 为 -2.1164%，Phase 6E 三模型集成为 -2.2572%，状态配对增益为 **+0.1408 percentage points**（Wilcoxon p=0.00172，Holm p=0.00349）。该增益严格低于 U1 的 +0.15 pp 门槛 0.0092 pp，因此 `U1=FAIL`；但质量优于而非劣于旧集成，同时单模型推理成本降低约 3.27 倍，S/M/L p90 模型决策时延均低于 24 ms，因此 `U2=PASS`，总部署门禁为 `PASS`。

`PHASE6G_RECOMMENDATION = PROCEED_TO_LIVE_NI_SOLVER_INTEGRATION`，但这只是下一阶段建议。本阶段未修改 ALNS、未执行 live integration，也未在完整调度轨迹上宣称优于 ALNS/GA/Gurobi。

## 2. Frozen Phase 6E evidence

Phase 6E 的 20,000-state internal holdout 已被打开，因此不能再作为新的无偏选择集。其 Full CSG ensemble 在旧 holdout 上达到 pairwise accuracy 0.5832、NDCG 0.8586、selected utility -1.5749%，优于 flat、fixed-related 和 tabular 等 solver-utility 对照，但三模型顺序推理的 M/L 端到端开销超过当时冻结预算。Phase 6F 因此保留 CSG-1.0 与 Phase 6C 标签语义，只修订训练目标、校准/介入策略和部署模型体量。

## 3. Why model revision was required

旧目标主要优化通用 pairwise ranking 与二分类；它没有直接增加大 utility/regret gap 样本的权重，也没有预测 calibrated utility。旧部署候选还是三模型 ensemble，推理成本与 solver 内高频调用不匹配。Phase 6F 的工程目标是：提高 state-paired selected utility、降低 regret、提供冻结 fallback policy，并用单 checkpoint 满足每个尺度 150 ms 的硬门禁。

## 4. Fresh R06 revision-holdout protocol

R06 使用与训练 R01–R05 隔离的新实例后缀，覆盖 `3 scales × 3 CF × 3 RI × 3 TI = 81` 个结构单元。每单元固定抽取 100 个 outcome-blind 状态，并在五个搜索阶段各保留 20 个。状态身份冻结后才按 Phase 6C v1 语义生成标签：destroy fraction 0.15、`transport_aware` repair、每 target set 三个 repair seeds。

sealed label freeze hash 为 `da37f046aec83654c6fd4262b9b14262ff5dfc96e961e03d8626915cb721b121`。模型 freeze hash 为 `51aa7e757171c4d6d2e58a19a56dfd7494a2743e593d681342c6edd83c5782d3`。开封审计 9/9 检查通过，明确记录 `revision_holdout_labels_opened_after_model_freeze = true`。张量缓存的 81 shards、8,100 states、191,416 actions、source/cache/schema hash 全部通过审计。

## 5. Utility-aware objective

开发研究只比较三个预注册目标：

| Objective | Validation utility | Regret | Pairwise | NDCG |
| --- | ---: | ---: | ---: | ---: |
| O1 Phase6E reference | -3.3377% | 5.3281% | 0.5483 | 0.8400 |
| O2 regret-weighted ranking | -3.1220% | 5.1124% | 0.5621 | 0.8464 |
| O3 utility-aware multitask | **-3.0861%** | **5.0765%** | 0.5619 | 0.8463 |

O3 同时优于 O1 的 utility 和 regret，并略优于 O2 的 utility，因此 `UTILITY_AWARE_OBJECTIVE_ADDS_VALUE = TRUE`。选择只使用 TRAIN 与 TRAIN_VALIDATION；R06 未参与目标选择。

## 6. Calibration and selective intervention

在 TRAIN_VALIDATION 上比较 Platt 与 isotonic 后冻结 isotonic probability calibrator、utility calibrator 和三个 intervention thresholds。R06 的 Brier score 为 0.1905，ECE 为 0.0359。冻结策略相对 fixed-related 增益 +0.4763 pp，selected utility 从 -2.5927% 提升到 -2.1164%，regret 从 4.8115% 降到 4.3352%。

需要明确限制：R06 intervention coverage 为 **100%**，即冻结阈值没有在该分布上触发 fallback。它满足“有用覆盖、正增量、低 regret”门禁，但尚未验证真实 live search 中的有效 abstention 行为。因此 Phase 6G 应继续记录 fallback 触发率与校准漂移。

## 7. Compact architecture

三个紧凑候选均不超过 Phase 6E 模型体量。最终选择 `C1_H128_L2`：hidden size 128、2 layers、4 heads、dropout 0.1、5,329,410 parameters，并保留 full CSG relations/edge features。它在 TRAIN_VALIDATION 上的 utility 为 -3.0703%，优于 C2_H96_L2 的 -3.0861% 与 C3_H96_L3 的 -3.1174%。

## 8. Distillation

ensemble-to-student distillation 仅比较冻结候选 weight 0 与 0.1。weight 0.1 的 validation utility 为 -3.0954%，劣于 weight 0 的 -3.0703%，因此最终未使用 distillation：`DISTILLATION_USED = FALSE`。

## 9. Validation-only model selection

最终训练 seeds 为 660301、660302、660303，均完成 6 epochs。部署规则在 R06 打开前冻结为“TRAIN_VALIDATION hybrid utility 最高，其次 regret 最低，再其次 seed 最小”，由此选择 660301。R06 上另外两个 seed 的结果仅用于稳定性报告，不能替换部署 checkpoint。

| Seed | TRAIN_VALIDATION hybrid utility | R06 raw utility | R06 positive | R06 regret |
| --- | ---: | ---: | ---: | ---: |
| 660301 (deployed) | -2.6817% | -2.1177% | 45.78% | 4.3365% |
| 660302 | -2.7344% | -2.0618% | 46.15% | 4.2806% |
| 660303 | -2.8224% | -2.0291% | 46.90% | 4.2479% |

三个 seed 均优于 R06 旧集成，utility span 为 0.0886 pp；`MODEL_STABLE_ACROSS_SEEDS = TRUE`。不能据 R06 结果改选 660303。

## 10. Revision-holdout evaluation

所有历史模型与新模型在完全相同的 R06 状态/动作上评分。主要动作级指标如下：

| Model | ROC-AUC | PR-AUC | Pairwise | NDCG | Top-3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase6E ensemble | 0.7465 | 0.5801 | 0.5715 | 0.8509 | 24.52% |
| Phase6C tabular | **0.7595** | **0.6016** | 0.5751 | 0.8550 | 23.74% |
| Phase6F seed 660301 | 0.7441 | 0.5857 | **0.5785** | 0.8547 | **26.30%** |

全局 ROC-AUC 不是主门禁；Phase6C tabular 的 ROC/PR 更高，但部署判断使用 state-paired selected utility 和 regret。

## 11. Selected-action utility

| Policy | Mean utility | Median | Positive | Top-3 | Regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase6F frozen hybrid | **-2.1164%** | -0.4684% | **45.78%** | **26.31%** | **4.3352%** |
| Phase6E ensemble | -2.2572% | -0.5545% | 45.30% | 24.52% | 4.4760% |
| Phase6E deployable single | -2.2083% | -0.6461% | 44.72% | 25.22% | 4.4272% |
| Phase6C tabular | -2.2040% | -0.4742% | 45.49% | 23.74% | 4.4229% |
| Flat set | -2.5947% | -0.7038% | 43.43% | 21.98% | 4.8136% |
| Fixed related | -2.5927% | -0.8105% | 41.79% | 11.86% | 4.8115% |
| Random expectation | -3.5135% | -1.6731% | 34.05% | 12.70% | 5.7323% |

平均 utility 仍为负，不能被 ROC 或显著性掩盖。该离线模型改善了候选选择的相对质量，但不保证单步 move 为正，也不等价于完整调度结果更优。

## 12. Regret analysis

对 Phase 6E ensemble，Phase 6F 的 mean regret 降低 0.1408 pp；对 fixed-related 降低 0.4763 pp。由于每个状态的 oracle utility 固定，state-paired utility 增益与 regret 降幅数值相等、符号相反。该一致性是本阶段最直接的 solver-relevant 改进证据。

## 13. Structural-regime robustness

按 S/M/L、CF1–3、RI1–3、TI1–3、search stage 与 bottleneck 分组后，最差 utility delta 是搜索早期 `0-20%` 的 -0.0565 pp，仍在冻结 U2 的 0.10 pp 容差内。重点困难组均未下降：S +0.2008 pp、TI1 +0.0229 pp、CF3 +0.4085 pp、RI3 +0.3459 pp。因此 `MODEL_STABLE_ACROSS_STRUCTURAL_REGIMES = TRUE`。

## 14. Comparison with Phase 6E model

部署策略对 Phase 6E ensemble 的状态配对结果为 2,053 wins / 4,182 ties / 1,865 losses，mean delta +0.1408 pp，Holm-corrected p=0.00349。对 Phase6E deployable single 的 delta 为 +0.0919 pp；对 Phase6C tabular 为 +0.0876 pp（不显著）；对 fixed-related 为 +0.4763 pp；对 flat-set 为 +0.4783 pp。

`COMPACT_MODEL_BEATS_PHASE6E_ENSEMBLE_UTILITY = TRUE` 指正 delta 且经校正显著；它不等于 U1。U1 要求至少 +0.15 pp，本结果只达到 +0.1408 pp，所以必须保留 `U1=FAIL`。

## 15. Inference optimization

Phase 6F 用一个 2-layer checkpoint 取代三个 3-layer checkpoints 顺序执行，并保持“一状态一次 CSG encoding、一次性评分所有 target sets”。R06 全量 artifact 计时中，旧三模型合计 105.60 s，部署单模型 32.33 s，测得 reduction factor 3.27×。没有弱化 CSG-1.0 关系或 edge features 来换取速度。

## 16. End-to-end deployment cost

冻结协议在 NVIDIA GeForce RTX 4060 Ti 上，每尺度 12 states、5 warmups、30 timed repetitions：

| Scale | p90 model decision | p90 end-to-end | 150 ms gate |
| --- | ---: | ---: | --- |
| S | 22.71 ms | 32.87 ms | PASS |
| M | 22.92 ms | 43.66 ms | PASS |
| L | 23.54 ms | 96.34 ms | PASS |

L 的 end-to-end 尾延迟仍主要来自 CSG build/state reconstruction，而不是模型 forward。硬门禁定义为 model decision p90，因此 `LATENCY_GATE_PASSED = TRUE`；报告同时保留 end-to-end 数字，避免隐藏集成成本。

## 17. Failure cases

- U1 以 0.0092 pp 的差距失败；Phase 6G 不应把 U2 描述为“明确大幅提升”。
- R06 calibrated policy 的 coverage=100%，没有实际 fallback 状态；abstention 泛化仍未知。
- Phase6C tabular 的 global ROC/PR 仍高于新模型，且两者 selected utility 差异不显著。
- 平均 selected utility 仍为负；Phase 6F 没有测量完整 ALNS trajectory 或最终 schedule objective。
- L end-to-end p90 为 96.34 ms，虽低于 150 ms，但远高于纯模型决策，需要在 Phase 6G 监控累计调用成本。

## 18. Phase 6G recommendation

冻结规则允许 U1 或 U2。U2、三尺度 latency、positive incremental utility、lower regret、结构稳定、mandatory leakage sanity 均通过，因此建议进入 **受控的 Phase 6G live NI solver integration**。Phase 6G 初始阶段应保留 `transport_aware` repair 与 destroy fraction 0.15，记录 intervene/fallback、accepted move、完整轨迹 objective 和累计 overhead；若真实轨迹中校准漂移或 fallback 永不触发，应优先修订 calibration，而不是静默扩大策略权限。

## 19. Reproducibility checklist

- [x] 81 个 R06 实例及结构/可行性审计
- [x] 8,100 outcome-blind state identities 冻结
- [x] sealed labels 在模型冻结后才开封
- [x] 81-shard tensor cache source/hash/schema 审计
- [x] 3 objectives、3 compact models、2 distillation settings，未扩大搜索
- [x] tiny overfit、label/state/target-mask shuffle sanity 通过
- [x] 3 final seeds，部署 seed 仅由 TRAIN_VALIDATION 选择
- [x] 历史模型与新模型同状态 R06 配对评估
- [x] Wilcoxon、win/tie/loss、BH 与 Holm correction
- [x] S/M/L model-decision 与 end-to-end latency
- [x] 10 张图同时输出 PNG/PDF
- [x] 完整仓库 regression/audit（报告生成后的严格步骤 21 已通过并写入 completion gate）

```text
REVISION_HOLDOUT_CREATED = TRUE
REVISION_HOLDOUT_STATE_COUNT = 8100
REVISION_HOLDOUT_UNTOUCHED_UNTIL_MODEL_FREEZE = TRUE
UTILITY_AWARE_OBJECTIVE_ADDS_VALUE = TRUE
CALIBRATION_VALIDATED = TRUE
SELECTIVE_INTERVENTION_VALIDATED = TRUE
DISTILLATION_USED = FALSE
COMPACT_SINGLE_MODEL_READY = TRUE
COMPACT_MODEL_BEATS_PHASE6E_ENSEMBLE_UTILITY = TRUE
COMPACT_MODEL_PRESERVES_PHASE6E_UTILITY = TRUE
MODEL_DECISION_P90_S_MS = 22.711150
MODEL_DECISION_P90_M_MS = 22.915476
MODEL_DECISION_P90_L_MS = 23.542038
LATENCY_GATE_PASSED = TRUE
MODEL_STABLE_ACROSS_SEEDS = TRUE
MODEL_STABLE_ACROSS_STRUCTURAL_REGIMES = TRUE
PHASE6G_RECOMMENDATION = PROCEED_TO_LIVE_NI_SOLVER_INTEGRATION
```

Phase 6F 到此停止；本报告不授权在本阶段实现 live CSG-NI search。
