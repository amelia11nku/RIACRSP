# Phase 6M M3 训练与完整性报告

状态：**M3_COMPLETE — INTEGRITY_PASS — READY_FOR_M4**。

正式 worker 已退出，连续进程耗时 246.69 秒。18/18 个 nested inner-ranker runs 与 9/9 个 selector runs 均通过文件存在性、记录状态、implementation hash、checkpoint hash、prediction hash、fold/seed identity 和有限数值检查。合并输出为 20,427 条 selector-seed OOF rows、6,809 个唯一 candidates 和 288 个 states。

## Cross-fit 与输入边界

每个 inner ranker 只在一个允许的 structural fold 上拟合并预测另一个 fold；对应 outer held fold 未参与。ranker transform 已从实际 training fold 重新计算并与 checkpoint 逐字段相等。每个 selector 的 inner fit、inner validation 和 outer held folds 恰好构成 `{0,1,2}`；outer support 与 selector normalization 已从两个 outer-training folds 的 OOF ranker features 重新计算并与 checkpoint 一致。

保存的 selector checkpoint 在 CPU 上以 `atol=rtol=1e-5` 重放全部 held predictions。aggregate candidate identity/order、ranker ensemble advantage 和 288 个 winner identity 均与冻结 Phase 6L 精确边界一致。historical score online forward calls 为 0；R13/R14 未访问。

## 数值与稳定性观察

所有 checkpoint 与输出均为有限值。新 hard support 的 candidate rate 为 99.192%，selected-winner rate 为 100.000%。winner selector probability 范围为 [0.189413, 0.587301]，total predictive scale 范围为 [0.040043, 1.651710]。

九个 selector best epochs 范围为 3–98，具体为 `[52, 88, 96, 98, 3, 3, 5, 54, 16]`。这表明 outer folds/seeds 的收敛差异较大；它不构成完整性失败，但必须在 M4 的 discrimination、calibration、scale lift 和 retained-gate 审计中显式评估，不能挑选单个 seed 或 fold。

## 证据

- `outputs/phase6m_selective_confidence_v1/training/completion_integrity_audit.json`
- `outputs/phase6m_selective_confidence_v1/training/progress.json`
- `outputs/phase6m_selective_confidence_v1/training/oof_summary.json`
- `outputs/phase6m_selective_confidence_v1/training/oof_predictions.parquet`
- `outputs/phase6m_selective_confidence_v1/training/ensemble_oof.parquet`
- `outputs/phase6m_selective_confidence_v1/training/selected_winners.parquet`
- `outputs/phase6m_selective_confidence_v1/training/formal_training_20260907T141046Z.log`

本报告只确认 M3 数据与协议完整性，不宣称质量 gate 通过。M4 必须使用全部三 seeds、全部 288 states 和预注册 18-gate grid；若没有 retained cross-scale gate，终止为 `MODEL_REVISION_QUALITY`。
