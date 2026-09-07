# Phase 6L L3 正式训练报告

## 结论

L3 为 **COMPLETE / INTEGRITY PASS**。唯一预注册 primary `L1_SCORE_FREE_CONT_FROZEN` 的 3 seeds × 3 outer folds 共 9 个 run 全部完成。每个 checkpoint 与 held-fold prediction 的 SHA-256 均匹配 run record；20,427 个 OOF 行恰好覆盖 6,809 个 state/candidate identity × 3 seeds。

后台 worker PID 36703 已正常退出。worker elapsed 为 456.335 s，各 run 记录 runtime 合计 455.277 s。没有 R13/R14 access ledger。

## Primary OOF 结果

| 指标 | 结果 |
| --- | ---: |
| Overall Spearman | 0.177723 |
| Pairwise accuracy | 0.563103 |
| NDCG@1 | 0.627466 |
| Selected lift | 0.007516 |
| Selected lift 95% LCB | 0.002866 |
| Selection regret | 0.033866 |
| S/M/L mean Spearman | 0.165456 / 0.145667 / 0.222045 |
| Selected-winner ECE | 0.010884 |

未 gated 的 ranking/argmax 指标均为正，且三种规模没有 Spearman 符号反转。与冻结 Phase 6J J1 相比，overall Spearman、pairwise accuracy、NDCG@1 和 raw selected lift 均相近或略高；这只是开发/OOF 对照，不是 predecessor equivalence 声明。

## Gate 预警

18 个冻结 gate 组合的 retained 数为 0，因此当前 primary 没有可部署的 calibrated intervention gate。`p_min=0.55` 时有 42 次干预，但 S/M/L 仅 19/13/10，未满足每 scale 至少 20；M scale lift 为 -0.000233。`p_min=0.65` 时各 scale lift 非负，但 S/M/L 干预仅 11/5/4。forced-abstention exception 也均未通过。

该结果由 L4 按预注册 essential gate 正式判定，不允许调整阈值网格、coverage 规则、模型或 seed。L4 仍执行预注册的单 seed、不可晋级 fallback-context 消融，之后终止或继续完全由冻结 gate 决定。
