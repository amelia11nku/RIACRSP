# Phase 6M M1：Score-Free Selective Risk 预注册协议

状态：**FROZEN BEFORE PHASE 6M OUTER-OOF QUALIFICATION OR OPTIMIZER STEP**。

## M0 对原工作假设的修订

M0 保留了 Phase 6L 的核心解释：score-free continuation ranker 的原始排序证据仍有效，Phase 6M 不改排名架构或损失。但机器证据不支持“实际门控仅由 probability/support 主导”：在最宽松 gate 下，probability/support 只覆盖 63.525% 的非 fallback 拒绝状态，而 immediate-harm 条件覆盖 82.787%；单独绕过 immediate-harm 会增加 89 次干预，单独绕过 probability 仅增加 20 次，绕过 support 不增加干预。Phase 6L immediate-utility 预测与真实 immediate utility 的 Spearman 为 0.010108。

因此 primary family 仍是唯一可晋级的 `M1_SCORE_FREE_SELECTIVE_RISK`，但 selector 特征明确包含 frozen ranker 的 immediate-utility 均值/seed 标准差，最终 gate 继续原样执行 `predicted immediate utility >= -0.005`。这使新 selector 可以学习其风险含义，同时不能绕过或降低既有 harm 标准。

## 固定模型

Phase 6L `L1_SCORE_FREE_CONT_FROZEN` 架构、训练目标与三 ranker seeds 全部继承。每个 outer held fold 的正式 ranker 输出直接复用已冻结、严格 held-fold 的 Phase 6L OOF；winner 继续取全量去重 24-rule bank 上最大 ensemble mean advantage，并用 `target_set_id` 做稳定 tie-break。

Selector 是候选级三 seed MLP：`Linear(d,32)-GELU-Dropout(0.1)-Linear(32,16)-GELU`，输出 continuation mean、positive scale 和 seed-positive logit。训练标签来自每个候选已有的两条 CRN continuation advantage；目标函数为 Gaussian NLL、逐 seed 正收益 BCE 和 candidate-mean Huber 的固定加权和。三个 selector 输出形成 predictive mean、直接 seed-positive probability，以及同时包含 selector aleatoric/ensemble 与 ranker ensemble disagreement 的 total predictive scale。

没有 Platt 或 isotonic 后处理。置信度是三 selector 的平均 seed-positive probability，避免再次只在 288 个 winner 上寻找低 ECE 解。

## 严格嵌套交叉拟合

每个 outer fold 的 selector 只能看到另外两个 structural folds。训练 selector 所需的 ranker 特征也必须对状态 OOF，并排除 outer held fold：在两个允许 folds 之间执行双向 ranker cross-fit，每个方向使用三条继承 ranker seed。由于只有三个冻结 structural folds，inner ranker 每次只训练一个 fold；epoch 固定为冻结 Phase 6L 九次 best epoch 的全局中位数 12，不依据任何 Phase 6M held output 调整。

Selector epoch selection 用 `(held+2)%3` 训练、`(held+1)%3` 验证，按总验证损失早停；随后按选定 epoch 在两个 outer-training folds 的 OOF ranker 特征上重新拟合。每个 held outer outcome 只在最终 OOF quality 汇总时使用一次，不选择特征、架构、loss、support 或 gate。

## 固定特征与泄漏边界

精确特征列表保存在 `configs/phase6m_selective_confidence_v1.json`。其来源限于：score-free ranker 三 seed 预测、当前候选 bank 内 margin/vote/rank/entropy/dispersion、训练折 support 距离、Phase 6L 原有 score-free 结构特征、scale 与当前 search/target progress。输入禁止 continuation/repair/future-search outcome、decoded makespan、历史 frozen score/rank 和 R13/R14 元数据。

候选级训练扩大了 selective-risk 的监督量，但 outcome 仅作为 loss target，不进入特征。所有 normalization、类别词表和 support 统计只拟合允许的训练 folds。

## Support 修订

M0 表明 1,892 个 unsupported candidates 全部由数值 robust-z 触发，没有任何 category OOV。`critical_overlap_fraction` 贡献 1,588 次原因；二值 `is_fallback` 因 IQR=0 被当作连续异常，导致全部 288 个 fallback candidates 不支持。

M1 将 `is_fallback` 从连续距离中移出。连续列尺度固定为 `max(IQR, 1.4826*MAD, 0.05*(max-min), 0.001)`，输出最大/RMS robust-z、`|z|>4` 比例、训练范围外比例和连续 support score。fine `primary_origin_rule` OOV 使用显式 `UNK` 和 indicator，不能单独导致拒绝；未知 family、未知 destroy operator、非有限数值或 `max|z|>12` 仍 fail-closed。所有阈值在 outer OOF 前固定。

## Gate 与晋级标准

仍使用 18 个组合：`p_min={0.55,0.65,0.75}`、`lambda={0.5,1.0}`、`delta_min={0,0.0025,0.005}`。LCB 改为与新输出语义一致的 `selector_mean - lambda * total_predictive_scale`。其他条件不变：hard support 必须通过、ranker immediate utility 必须不低于 `-0.005`、每个规模至少 20 次直接干预或满足原 forced-abstention exception。

可晋级 gate 必须同时满足 overall gated lift > 0、grouped-bootstrap LCB > 0、S/M/L lift 全部非负、规模覆盖以及 winner ECE <= 0.1。M4 若没有 retained cross-scale gate，终止为 `MODEL_REVISION_QUALITY`；不运行 runtime、solver、R13 或 R14。

## 冻结边界

- 唯一 promotable family：`M1_SCORE_FREE_SELECTIVE_RISK`；不运行 teacher ablation。
- H1、CSG、24-rule generation/dedup、候选身份顺序、canonical `operator_related` fallback、8 repairs、decoder 与 feasibility 不变。
- online historical scorer forward 必须为零；FP32 only。
- R13/R14 保持锁定；不运行 Gurobi。
- M1 后的实现只能忠实实现本协议。任何模型、target、feature、support 或 gate 变更都需要新的版本和重新预注册，不能覆盖本证据。
