# Phase 6L L4 开发质量与消融报告

## 判定

L4 的正式判定为 **`MODEL_REVISION_QUALITY`**。primary 的 raw ranking/utility essential checks 全部通过，但 18 个冻结 gate 组合没有一个满足 Phase 6J intervention-readiness 门槛。Phase 6L 因此在 L4 终止，不进入 L5 开发运行时、L6 单次 R12 资格、bundle、solver、R13 或 R14。

这里把两个层级明确分开：raw 模型存在有效排序信号；可部署 gate 资格失败。不能用前者替代后者。

## Primary 三 seed outer OOF

| 指标 | Phase 6L |
| --- | ---: |
| States / candidates | 288 / 6809 |
| Overall Spearman | 0.177723 |
| Pairwise accuracy | 0.563103 |
| NDCG@1 | 0.627466 |
| Raw selected lift | 0.007516 |
| Raw selected lift 95% LCB | 0.002866 |
| Selection regret | 0.033866 |
| Selected-winner ECE | 0.010884 |
| Candidate / winner support | 72.21% / 70.49% |

S/M/L mean Spearman 为 0.165456 / 0.145667 / 0.222045，均未发生符号反转。三项 preferred 诊断阈值（0.25 Spearman、0.60 pairwise、0.75 NDCG@1）均未达到；它们不单独决定 essential 资格。

## Gate 与 action frequency

冻结网格为 3 个 `p_min` × 2 个 `lambda` × 3 个 `delta_min`，共 18 个组合，retained 数为 **0**。因此 `selected_gate=null`，正式可部署 intervention 为 0/288；若保持 fail-closed 行为，288/288 均回退。神经 argmax 本身选择 fallback 2 次、非 fallback 286 次，这不代表通过 gate。

`p_min=0.55` 的组合产生 42 次 intervention，S/M/L 为 19/13/10，低于每 scale 20 的冻结下限，且 M lift 为 -0.000233。`p_min=0.65` 产生 20 次，S/M/L 为 11/5/4；lift 非负但 coverage 仍失败。`p_min=0.75` 仅 2 次且总体 lift 为负。所有 forced-abstention exception 均失败。

失败项为：`retained_gate_with_scale_coverage`、`positive_gated_lift_lcb`、`nonnegative_gated_scale_lift`。完整网格保存在 `outputs/phase6l_legacy_score_decoupling_v1/training/gate_grid.csv`。

## 与冻结 Phase 6J J1 对照

| 指标 | Phase 6J J1 | Phase 6L score-free |
| --- | ---: | ---: |
| Overall Spearman | 0.175995 | 0.177723 |
| Pairwise accuracy | 0.562392 | 0.563103 |
| NDCG@1 | 0.618195 | 0.627466 |
| Raw selected lift | 0.007156 | 0.007516 |
| Raw selected lift LCB | 0.003222 | 0.002866 |
| Retained gates | >=1 | 0 |

Phase 6L 的 raw 指标与 J1 相近或略高，但冻结 J1 有一个 94-intervention retained gate，S/M/L 为 25/35/34，gated lift 0.004976、LCB 0.001926。将既有 J1 OOF 预测仅按 Phase 6L canonical fallback 标签重评分的诊断仍得到 6 个 retained gate，最佳组合 84 次 intervention。该诊断没有重跑 J1；10 个 fallback 改变状态仍保留历史 J1 输入上下文，因此只用于说明 gate 差异并非单纯由标签重锚定解释。

## 预注册 feature-removal 消融

单 seed 706101、不可选择的 `L1_NO_FALLBACK_CONTEXT_ABLATION` 完成 3 folds。它将 `fallback_overlap_fraction`、`fallback_jaccard`、`is_fallback` 三个模型输入置零，但保留 fallback identity 的标签和决策语义。

| 指标 | Primary seed 706101 | 移除 fallback context | 差值 |
| --- | ---: | ---: | ---: |
| Spearman | 0.184380 | 0.172752 | -0.011628 |
| Pairwise | 0.565504 | 0.561105 | -0.004399 |
| NDCG@1 | 0.620523 | 0.626190 | 0.005667 |
| Selected lift | 0.006841 | 0.007419 | 0.000578 |

fallback context 对 Spearman 与 pairwise 有小幅正贡献，但并不能解释全部 gate coverage 缺失。该消融耗时 342.75 秒，严格保持不可晋级。

## 完整性与分层证据

完整 bank 审计覆盖 288 states、6,809 个唯一 state/candidate identity；每状态请求 24 条规则，去重后候选数 21–24，全部 archived repair 均可行，每状态恰有一个 canonical fallback。S/M/L、CF1/2/3 与 search-stage 分层指标在 `quality/stratified_metrics.csv`；support 与 origin 分布分别在 `support_by_regime.csv`、`origin_selection_by_scale.csv`。

保护审计重新验证 Phase 6I/6J 6,636 个文件、Phase 6K 21,365 个文件和 139 个 tracked predecessor 文件。R13/R14 未访问。
