# NGAS-A1.4 起点审计

审计时间：2026-09-11（实验记录中的 UTC 时间为 `2026-09-10T18:50:31.726778+00:00`）。

## 结论

状态为 **PASS**。A1.4 可在当前提交 `d10181a6d2ccbf402b80622080ce522ebe09dfa7` 上开始实现，并仅使用 A1.3R 生产分支 C1。A1.3R、A1.2 与历史 C0 证据保持冻结；R1 只允许在 A1.4 机制冻结后做一次解释性消融；R13、R14 与 Gurobi 均保持禁用。

机器可读证据位于 `outputs/ngas_a1/search_integration_v1/audit/starting_state.json`。

## 已核验边界

- A1.3R 终态：`NGAS_A1_3R_PASS_COMPACT`。
- 生产表示：C1 compact relational encoder。
- 生产 checkpoint：`outputs/ngas_a1/critic_training_rthgt_v2/production/revised_joint_critic.pt`。
- checkpoint SHA256：`448b0aaf871f0629dec2d94bad63c888fbdaf71c113eb5ade58c4228647c8560`。
- A1.3R 完整结果清单：78 个文件全部匹配，无额外文件。
- C1 最坏代表性 p90：`2.056765020824969 ms`，具备 A1.4 集成资格。
- R1 最坏代表性 p90：`49.8257779981941 ms`，超过 30 ms 上限，只保留解释性角色。
- A1.2 标签身份：6465 个动作、58185 次重复、174555 个配对步，错配数为 0。
- 历史 C0：41 个文件，树哈希仍为 `4239e9236cb5a5d9837def6acc70baa9155b32cb2282a85efb604a57a4b3a9f9`。
- CUDA 审计环境：Python 3.11.15、PyTorch 2.11.0+cu128、CUDA 可用。
- 实现前回归：`40 passed in 1.90s`。

## A1.4 实施边界

A1.4 新代码只能消费冻结 checkpoint 并改变搜索集成机制，不得重新训练、重标注或改写历史实验。小规模 R12 开发子集必须在查看结果前固定实例、种子、预算、六组消融、选择规则和停止规则。正式运行前还必须通过短程 CUDA 烟测、排程回放、遥测计数与确定性复现检查。

任何候选若不能形成可度量的持续神经影响，或不能在内部对照上给出可信搜索行为，A1.4 应终止为 `NGAS_A1_REVISE_SEARCH_INTEGRATION`，不得进入 A1.5。
