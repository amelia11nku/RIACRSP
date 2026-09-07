# Phase 6M M4 开发集质量报告

状态：**MODEL_REVISION_QUALITY**。M4 已完成，M5/M6 runtime、solver、R13 和 R14 均未运行。

## 结论

Phase 6M 精确保留了 Phase 6L 的 score-free 排名结果：Spearman 0.177723、pairwise accuracy 0.563103、NDCG@1 0.627466，raw selected lift 0.007516，grouped-bootstrap LCB 0.002890。S/M/L raw lift 分别为 0.003877、0.007739、0.010933。

新的 selective-risk 层未恢复可部署干预。18 个预注册 gate 全部为 0 次干预，retained gate 为 0。最宽松 LCB 条件 `selector_mean - 0.5 * total_scale > 0` 的通过数为 0 / 288；其最大值为 -0.012257。同一组合中 `p>=0.55` 仅 14 / 288，通过 frozen immediate-harm floor 的为 84 / 288。失败主要由 selector 的保守下界整体为负导致，不能通过降低已冻结阈值修正本轮结果。

## Confidence 与 support

Winner confidence 的 ECE 为 0.090809，Brier 为 0.248920，AUROC 为 0.578258，AUPRC 为 0.633791，Brier resolution 为 0.006475。概率标准差为 0.068851，范围 [0.189413, 0.587301]；`p>=0.55` 仅覆盖 4.861%。相比 Phase 6L，分辨率和 sharpness 没有改善到足以产生 selective intervention 的程度。

修订后的 hard support 候选通过率为 99.192%，winner 为 100.000%。55 个 unsupported candidates 全部包含 `max|robust-z|>12` 数值边界原因；未知 high-level family/operator 与 fine-rule OOV 均为 0，unsupported winner 为 0。support 修订消除了 Phase 6L 的 winner support 阻塞，但没有改变 LCB 失败。

## 门槛判定与冻结边界

Raw ranking 的四项预注册要求全部通过，candidate identity/order、24-rule full bank、fallback identity、feasibility 与 origin diversity 也通过。正式 readiness 失败项为：`retained_cross_scale_gate`、`positive_gated_lift`、`positive_grouped_bootstrap_lcb` 和 `cross_scale_coverage`。因此按协议终止为 `MODEL_REVISION_QUALITY`，不得进入 M5/M6、bundle、R12 solver gate、R13 或 R14。

主要证据：

- `outputs/phase6m_selective_confidence_v1/quality/development_quality.json`
- `outputs/phase6m_selective_confidence_v1/quality/gate_grid.csv`
- `outputs/phase6m_selective_confidence_v1/quality/stratified_metrics.csv`
- `outputs/phase6m_selective_confidence_v1/quality/support_by_regime.csv`
- `outputs/phase6m_selective_confidence_v1/quality/selective_risk_curve.csv`

本结论使用全部 288 个 R12 CAUR-FIT states、6,809 个候选和全部三 selector seeds。historical score online forward calls 为 0；R13/R14 access ledgers 不存在；未运行 Gurobi。
