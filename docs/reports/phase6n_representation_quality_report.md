# Phase 6N N5 原始表示质量报告

终态：**`MODEL_REVISION_REPRESENTATION`**。

N4 的 9-run outer OOF 完整性审计通过后，按预注册规则在 common original 288 states 上与冻结 Phase 6L 做 paired comparison，并独立报告 expanded 864-state OOF。没有执行 empirical calibration、direct-decision gate、ablation、runtime 或 solver qualification。

| 范围 | Spearman | Pairwise | NDCG@1 | Raw selected lift | Grouped LCB | Regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 6N common 288 | 0.211669 | 0.574069 | 0.617505 | 0.006411 | 0.002760 | 0.034971 |
| Phase 6N expanded 864 | 0.253297 | 0.589622 | 0.628987 | 0.007120 | 0.004509 | 0.034595 |
| Phase 6L common 288 | 0.177723 | 0.563103 | 0.627466 | 0.007516 | 0.002936 | 0.033866 |

表内 Phase 6L LCB 按 N5 的 5,000 次、seed 727001 口径重算；配置中冻结的 Phase 6L 原始 LCB 0.002866 使用其 Phase 6L 2,000 次、seed 707001 口径。二者均为正，点指标与冻结 reference 精确一致，这一 bootstrap 口径差异不参与 paired improvement 判定。

Phase 6N 在 common evidence 上的 Spearman 提升 0.033946，Spearman 与 pairwise 两项严格优于 Phase 6L；expanded Spearman 达到 preferred 0.25。候选来源没有坍缩，full-bank、feasibility、finite predictions、whole-instance isolation、zero historical scorer 和 R13/R14 lock 均通过。

阻断项是预注册的 paired utility improvement：Phase 6N common raw selected lift 为 0.006411，Phase 6L 为 0.007516，均值差为 -0.001105；18-instance paired bootstrap 95% 区间为 [-0.003703, 0.001367]。LCB 未严格大于 0，因此 `paired_phase6l_lift_improvement_lcb_positive=FALSE`。

这表明显式 candidate-conditioned CSG representation 改善了相关性，但没有在同一 288-state evidence 上证明优于 Phase 6L 的实际 raw selection utility。按冻结 gate，N5 必须判为 `MODEL_REVISION_REPRESENTATION`，并停止 calibration、decision、runtime、solver 及 R13/R14。
