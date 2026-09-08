# Phase 6N Candidate-Conditioned CSG 最终报告

## 最终决策

**`MODEL_REVISION_REPRESENTATION`**，停止边界为 **N5、早于 empirical calibration 与 direct-decision gate**。

Phase 6N 完成了架构/数据审计、预注册、576 个扩展状态采集、864-state candidate-conditioned CSG 训练及完整 outer OOF。N4 审计为 `PASS`：9/9 runs、864 states、20,441 candidates、61,323 outer seed-candidate rows 和相同数量的 inner-validation rows 均完整；候选身份、truth、feasibility、whole-instance isolation 与工件哈希全部通过。

## N5 科学结果

| 范围 | Spearman | Pairwise | NDCG@1 | Raw lift | Lift LCB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase 6N common 288 | 0.211669 | 0.574069 | 0.617505 | 0.006411 | 0.002760 |
| Phase 6N expanded 864 | 0.253297 | 0.589622 | 0.628987 | 0.007120 | 0.004509 |
| Phase 6L common 288 | 0.177723 | 0.563103 | 0.627466 | 0.007516 | 0.002936 |

Candidate-conditioned representation 将 common Spearman 提高 0.033946，且 Spearman 与 pairwise 两项优于 Phase 6L；expanded Spearman 达到 0.25 preferred target。它没有证明主要选择效用优于 Phase 6L：paired raw-lift 改善均值 -0.001105，95% 区间 [-0.003703, 0.001367]。唯一失败硬项为 `paired_phase6l_lift_improvement_lcb_positive`。

这说明当前表示改善了候选排序相关性，但其 top-ranked candidate 没有在冻结 common evidence 上形成可验证的增量 utility。后续研究需建立新的预注册边界，针对 ranking surrogate 与 top-selection utility 的错配、L-scale utility 回落以及 candidate-conditioned pooling/训练目标进行诊断；不得调整本轮 bootstrap、硬门槛或事后选择 repetition。

## 完整性与停止边界

- N2：576/576 新状态、13,632 candidates，wall time 9329.89 秒。
- N4：9/9 runs，wall time 2783.66 秒；training PID 已退出。
- 完整回归：**398 passed in 18.81 s**。
- Phase 6L/6M 保护：208 files / 36,230,480 bytes，逐文件复核通过；前代保护链通过。
- historical score online forward calls：0；Gurobi：未运行；PASS_LOCKED_NOT_ACCESSED。
- N6 calibration/decision、N7 ablations、N8 runtime、N9 solver、R13、R14：均未运行。

起始 commit：`8efe396e6cca44493a18c07720c4effaf2add7c1`。终局证据生成前 HEAD：`c2e7718b019aa221dabd4e5ba97e5035a18654b9`。最终 evidence commit 是包含本报告及 terminal JSON 的后续本地 commit。
