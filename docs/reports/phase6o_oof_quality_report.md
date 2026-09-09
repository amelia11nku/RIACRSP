# Phase 6O OOF 质量报告

终态：**`MODEL_REVISION_TOP_UTILITY`**。

正式训练完整性审计通过后，本审计按冻结 Route B 门槛评估 3-seed ensemble OOF。没有拟合 direct-decision calibration，没有运行 development solver pilot、runtime、formal R12、Gurobi、R13 或 R14。

| 范围 | Spearman | Pairwise | NDCG@1 | Raw selected lift | Grouped LCB | Regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 6O common 288 | 0.158695 | 0.556355 | 0.602179 | 0.005032 | 0.001242 | 0.036069 |
| Phase 6O expanded 864 | 0.212809 | 0.575892 | 0.617148 | 0.006347 | 0.003885 | 0.035518 |
| Frozen Phase 6L common 288 | 0.177723 | 0.563103 | 0.627466 | 0.007516 | n/a | 0.033866 |

Common 的冻结硬门槛要求 raw selected lift ≥ 0.00751635、regret ≤ 0.03386557。实际分别为 0.00503233 和 0.03606904，两项均失败。Expanded lift 与 grouped LCB 为正，common/expanded 的 S/M/L lift 全部非负，origin diversity、candidate identity/order、full bank、feasibility、whole-instance isolation 和 finite prediction 均通过。

为避免混淆 reference truth，本报告另外把 Phase 6L 排序应用到 Phase 6O 的同一份 targeted-relabel truth。其 lift 为 0.00535990，Phase 6O 的 paired delta 为 -0.00032757，18-instance bootstrap 95% CI 为 [-0.00439149, 0.00371603]。该 CI 是诊断；冻结点值和 regret 门槛仍使用原 Phase 6L truth 上预注册的常数。

按协议，Route B 在 point/top-selection 硬要求失败时必须终止为 `MODEL_REVISION_TOP_UTILITY`，不得创建新 selector，也不得进入 solver pilot。R13/R14 保持锁定。
