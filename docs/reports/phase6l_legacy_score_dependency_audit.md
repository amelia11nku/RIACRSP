# Phase 6L 历史分数依赖审计

## 阶段结论

L0 结论为 **PASS — PROCEED_TO_L1_PREREGISTRATION**。历史 Phase 6J 分数模型不改变 24 条规则生成、目标集合去重、候选身份或候选生成顺序，因此无需修改 action space，也不触发 `HOLD_DEPENDENCY_AUDIT`。Phase 6L 可以在独立命名空间中构建无分数的 continuation-value ranker。

本报告仅使用已授权的 `R12_CAUR_FIT` 开发/OOF 证据来追踪既有依赖，没有执行新的 R12 资格比较，没有读取 R13/R14 内容。

## 起始边界

- 起始提交：`c52a37f64573d3fa822282960a6a9cbf6222f375`。
- Phase 6L 编辑前工作树：clean。
- 起始回归：`342 passed in 16.58 s`。
- Phase 6I-MR/6J 受保护文件：6,636 个，现有清单逐文件复核通过；清单 SHA-256 为 `9d4203b7542fa9b95dd99fdc188c79b4698bfad34c1132bbf88b52f28abb4ec5`。
- Phase 6K 输出：21,365 个文件、638,852,260 bytes，新增逐文件清单 SHA-256 为 `cc88386a524d9bf211d029d3cefa13dfd471a0d3c55b8605af48c37287895662`。
- 受保护的已跟踪 Phase 6I/6J/6K 文件：139 个，清单 SHA-256 为 `5c0b9136abb7d8fe067067dfe09c8dfbe535480f96e6bc258da34a301ef65d76`。
- Phase 6J 输出中不存在 R13/R14 访问路径或 ledger；两者仍为 locked。

完整机器证据位于 `outputs/phase6l_legacy_score_decoupling_v1/audit/starting_audit.json`、`protected_phase6k_evidence.json` 与 `protected_tracked_predecessor_files.json`。

## 候选集合与历史模型的边界

静态调用链为：`score_frozen_candidate_bank` 先构造 CSG，再调用 `build_live_proposal_bank -> generate_revised_target_arms` 完成 24 条规则、按排序后的 operation tuple 去重及 `target_set_id` 生成，之后才张量化并调用历史模型。候选生成接口不接收 policy、模型或 frozen score。

动态追踪从 S/M/L 各固定一个既有 `R12_CAUR_FIT` 状态，重放当前 schedule，并直接执行两次候选生成。三个状态均满足：

- 请求规则数为 24；
- 两次 `ArmGenerationResult` 完全一致；
- 候选身份及顺序与 Phase 6J 存档 `full_bank_target_ids` 完全一致；
- 候选生成期间历史模型 forward 次数为 0；
- 改变或反转存档分数只影响后续角色与特征，不改变候选集合。

全部 288 个状态共 6,809 个唯一候选，每状态由 24 个请求规则产生 21–24 个去重候选。每个状态都恰好有一个 provenance 包含 `operator_related` 的候选，且其 operation set 等于生成器的 `canonical_related_target`。

因此 Phase 6L 保留原 `generate_revised_target_arms`、去重、候选身份和顺序，不另设候选空间。

## fallback 的实际间接依赖

历史角色选择先按 frozen raw score 选取互异的 Top-1/Top-2，再选择 `ALNS_RELATED_FALLBACK`。fallback 虽优先 `operator_related`，但角色不得复用同一 target：在 10/288 个状态中，canonical `operator_related` 已被 Top-1/Top-2 占用，历史 fallback 因而退到 `related_variant_1`。其余 278 个状态的历史 fallback 等于 canonical `operator_related`。

这项依赖不改变 action space，但会影响以下内容：fallback 身份、`fallback_overlap_fraction`、`fallback_jaccard`、`is_fallback`、fallback-relative continuation advantage、`beats_fallback`、归一化、support、校准及 gate。

Phase 6L 的确定性替代固定为：**选择去重候选中唯一包含 `operator_related` provenance 且等于 `canonical_related_target` 的候选**。若候选不存在、不唯一或 operation set 不一致，新接口立即失败。该选择发生在任何历史或 Phase 6L 神经推理之前。

现有 raw seed label 保存了每个状态、每个候选和两个 CRN seed 的 continuation 结果。因此 10 个差异状态可从既有结果重新锚定到 canonical fallback；无需新 decoder/continuation 运行。派生标签将在 L2 中进入新的数据集，原 Phase 6J 文件保持不变。

## 依赖处置

| 接口/字段 | 处置 | Phase 6L 规则 |
| --- | --- | --- |
| 历史 scorer online forward | `REMOVE` | live path 不暴露历史 policy/model 参数 |
| `frozen_raw_score` 及历史校准输出 | `OFFLINE_TEACHER_ONLY` | primary model 不使用 teacher；online schema 排除 |
| 24-rule 生成、去重、身份、顺序 | `KEEP` | 保留 Phase 6C 实现与 target-ID tie-break |
| frozen Top-1/Top-2 和 reduced top-8 roles | `REMOVE` | 正式训练与 live 决策使用完整去重 bank |
| `ALNS_RELATED_FALLBACK` | `REPLACE` | 固定 canonical `operator_related` |
| fallback overlap/Jaccard、`is_fallback` | `REPLACE` | 相对新确定性 fallback 重算 |
| `best_frozen_score_jaccard` | `REMOVE` | 从输入 schema 删除 |
| `normalized_frozen_score_rank` | `REMOVE` | 从输入 schema 删除 |
| `normalized_diversity_rank` | `KEEP` | 只依赖 target-set Jaccard 与 target ID |
| fallback-relative 标签 | `REPLACE` | 用已存 continuation 结果重新锚定 |
| normalization、support、模型参数 | `REPLACE` | 只在新 schema 的授权训练 fold 拟合 |
| calibrator、gate 阈值选择 | `REPLACE` | 只从新模型 grouped OOF 预测重做 |
| gate 公式和候选 winner tie-break | `KEEP` | 沿用 Phase 6J 公式与阈值网格 |
| repair、8 trials、decoder、feasibility | `KEEP` | 不改求解语义 |

机器依赖表列出 20 个已知 live 接口及静态位置，见 `legacy_score_dependency_map.json`。

## 新的无分数接口

`rcias_clgri.analysis.phase6l_legacy_score` 已建立独立 source-feature builder。它仅接收已生成 bank、状态 ID、operation 数量和确定性的 critical/bottleneck operation 集合；签名不能接收 policy、model 或 frozen scores。在线数值特征由原 12 项减为 10 项，删除两个 frozen-score 派生字段；其余 fallback 特征相对 canonical fallback 重算。

接口校验完整候选 bank、生成顺序、唯一 fallback、结果字段泄漏、历史分数字段及两项被删除特征。当前定向回归为 **44 passed in 4.21 s**，其中新增测试覆盖依赖表、fail-closed fallback provenance、候选顺序和在线接口参数边界。

L1 将在任何新的 R12 资格运行前冻结数据边界、单一 primary 架构、三 seed、重新锚定规则、训练/OOF、质量 gate 和正式 runtime 协议。R13/R14 继续保持锁定。
