# Phase 6N 扩展数据生成报告

状态：**`PASS`**。

按 N1 冻结计划完成 72 条 H1-seeded frozen ALNS source trajectories 和 576 个新增 R12 CAUR-FIT states。新增 candidate means 为 13,632，two-CRN rows 为 27,264；与原始 Phase 6L 数据合并后为 864 states、20,441 candidates、40,882 raw rows。18 个 instances 每个恰好 48 states。

每个 state 均生成 frozen 24-rule full bank、按原规则 dedup，并保留 generator order；canonical `operator_related` fallback 每 state 恰好一个。所有 forced candidates 通过 deterministic decoder 与 `check_schedule`。新增 source sampler、candidate features 和 labels 的 historical scorer calls 均为 0。

原始 288 states 标记为 `ORIGINAL_PHASE6J_CAUR`，新增 576 states 标记为 `NEW_ALNS_EXPANSION`。后续 common-original 与 expanded OOF 指标必须分开报告。

累计 forced-repair decoder evaluations 为 109,056，continuation decoder evaluations 为 872,448，state labeling wall time 合计 7165.87 秒。R13/R14 未访问，未运行 Gurobi。

完整哈希与 composition 位于 `outputs/phase6n_candidate_conditioned_csg_v1/data/data_integrity.json` 和 `new_r12_collection/collection_shard_manifest.csv`。
