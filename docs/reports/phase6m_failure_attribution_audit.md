# Phase 6M M0：冻结失败归因审计

状态：**M0_COMPLETE — READY_TO_PREREGISTER_M1**。本阶段只读取已冻结的 Phase 6L/6J 证据，没有训练新模型、没有运行 Gurobi，也没有访问 R13/R14。

## 结论

Phase 6L 的失败位于部署选择层，而不是已验证的原始排序路径。5,184 条状态×门控复算与冻结 `gate_grid.csv` 完全一致。最宽松门控 `p_min=0.55, lambda=0.5, delta=0` 的 244 个非 fallback 拒绝状态中，155 个至少触发 probability 或 support，比例为 63.525%。这不支持 probability/support 单独主导的预设；immediate-harm 条件覆盖了 82.787% 的这些拒绝状态，是当前最强的单项约束。

逐项绕过实验进一步定位了原因：只绕过 probability 可新增 20 次干预，只绕过 support 新增 0 次，而只绕过 immediate-harm 可新增 89 次。Phase 6L immediate-utility head 只有 84/288 通过 `-0.005`，冻结 J1 为 214/288；Phase 6L 该预测与真实 immediate utility 的 Spearman 仅 0.010108。M1 的 selective-risk 表征必须纳入并审计这一路信号，同时保持最终 immediate-harm floor 不变。

固定 `p_min` 后，`lambda` 和 `delta_min` 的 6 个组合均只有一个唯一干预集合：`p_min=0.55/0.65/0.75` 分别为 42、20、2 次干预。该结论是逐状态集合哈希的精确相等，不是近似判断。`p_min=0.55` 的 Medium gated lift 为负，而更高阈值又无法达到每个规模至少 20 次直接干预；原 18 个 gate 因而全部拒绝。

Phase 6L winner probability 的 ECE 为 0.010884，但 AUROC 仅 0.596025、AUPRC 为 0.653675、Brier resolution 为 0.007793、概率标准差为 0.099062。冻结 J1 对应值为 ECE 0.047758、AUROC 0.581207、AUPRC 0.594548、resolution 0.016228、概率标准差 0.115168。这些量把全局校准与选择分辨率分开；M1 不应继续仅用 288 个 winner 的低 ECE 作为置信度充分证据。

Phase 6L 候选 support 为 72.213%，winner support 为 70.486%。共 1892 个候选、85 个 winner 不支持；未见类别原因计数为 0，数值 robust-z 原因计数为 2070，多原因候选为 138。最高频原因是 `numeric_robust_z:critical_overlap_fraction`=1588、`numeric_robust_z:is_fallback`=288、`numeric_robust_z:origin_rule_count`=92、`numeric_robust_z:bottleneck_overlap_fraction`=68、`numeric_robust_z:origin_family_count`=34。其中二值 `is_fallback` 在训练折 IQR 为零时被当作连续异常值，导致全部 288 个 fallback 候选 support 失败；`critical_overlap_fraction` 同样受零/极小 IQR 影响。这是硬 support 表示的具体缺陷。M1 应保留高层语义类别 fail-closed，将二值特征从连续距离中分离，并把其余数值 support 改为训练折内连续距离及明确硬边界。

三 seed 不确定性表覆盖全部 288 个状态。以原 `p_min=0.55, lambda=0.5, delta=0` 为参照，复现 42 次干预、17 个非正收益干预和 138 个正收益弃权。winner 均值/标准差、top1-top2 margin、winner-fallback margin、seed vote/rank disagreement、候选分数熵/离散度和 support 比例现已形成 M1 特征选择的冻结依据。

## 方法边界

- Gate 审计按原 Phase 6L 顺序复算 fallback、probability、LCB、support 和 immediate-harm 五个局部条件；overall lift、grouped-bootstrap LCB、对应 scale lift 与 coverage 作为聚合条件单列。
- Brier 分解采用 10 个等宽概率箱，是离散近似；JSON 同时保存 identity residual，避免把分箱近似写成精确恒等式。
- Support 原因由每个 held fold 的两个训练折重新拟合原 median/IQR 与类别词表后反演，并与冻结 OOF 的每个 `supported` 位逐项一致。
- 不确定性只使用现有三 seed Phase 6L OOF 预测与已存 R12 continuation outcome；没有生成新的 qualification output。

## 证据

- `outputs/phase6m_selective_confidence_v1/audit/failure_attribution.json`
- `outputs/phase6m_selective_confidence_v1/audit/gate_rejection_waterfall.csv`
- `outputs/phase6m_selective_confidence_v1/audit/confidence_diagnostics.csv`
- `outputs/phase6m_selective_confidence_v1/audit/support_rejection_breakdown.csv`
- `outputs/phase6m_selective_confidence_v1/audit/support_rejection_summary.csv`
- `outputs/phase6m_selective_confidence_v1/audit/candidate_selection_uncertainty.csv`
- `outputs/phase6m_selective_confidence_v1/audit/protected_phase6l_evidence.json`

M1 只能在上述归因基础上冻结一个 promotable family：`M1_SCORE_FREE_SELECTIVE_RISK`。R13/R14 继续锁定；最终规模覆盖、正 grouped-bootstrap LCB、各规模非负 lift 及后续延迟门槛保持不变。
