# Phase 6O 定向重标注恢复修订

状态：**VALIDATION-ONLY RECOVERY**。本修订保留原 O1 预注册协议、已完成 outcome 和失败证据，不覆盖原始运行记录。

正式 worker 在 288/864 states 后、处理第 289 个 state 前 fail closed：重新生成的 canonical `operator_related` fallback 与 Phase 6J replay 中保存的历史 fallback 不同。失败 state 没有 raw shard 或 status；已完成的 288 个 state 各有一个 raw shard 和一个 status，576 个文件均通过冻结 SHA-256 清单复核。

根因是验证引用错误。Phase 6J 的角色唯一性会在 canonical candidate 已被 Top-1/Top-2 使用时选择 `related_variant_1` 作为历史 fallback。Phase 6L 已预先识别这些 10/288 original states，并将标签重锚到 canonical `operator_related`；Phase 6N 的 combined training truth 与 Phase 6O 冻结 union 继承了该 canonical fallback。原 Phase 6O worker 的 continuation 计算也已经使用 canonical fallback，但额外将它与旧 replay fallback 比较，导致首个已知重锚状态停止。

恢复修订只将 fallback 校验对象改为冻结 Phase 6O union，并对历史 replay 差异进行显式 provenance 审计：

- frozen union 必须恰有一个 fallback，且必须等于重新生成的 canonical `operator_related`；
- NEW states 的 replay fallback 仍必须直接等于 canonical fallback；
- ORIGINAL states 只有诊断中预先枚举的 10 个精确 `(state, historical fallback, canonical fallback)` 三元组可以不同；任何额外差异继续 fail closed；
- repair namespace、candidate trials、continuation seeds/namespace、H=4、CRN 及 advantage 公式不变；
- 288 个旧 shard 继续按原 implementation commit/hash 验证，新 shard 记录 amended commit/hash；最终汇总接受且披露这两个实现边界。

冻结证据位于 `outputs/phase6o_neural_shortlist_v1/relabeling/runtime_recovery_diagnosis.json` 与 `pre_recovery_completed_shard_manifest.csv`。恢复时生成 `runtime_recovery_amendment.json`，其中锁定旧/新 worker 哈希、amended commit 与诊断哈希。

本修订没有调用 historical scorer，没有运行 Gurobi，没有访问 R13/R14。完整定向重标注、O2 训练与后续 gate 仍需按原 O1 协议完成。
