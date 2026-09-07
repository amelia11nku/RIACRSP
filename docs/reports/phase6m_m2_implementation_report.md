# Phase 6M M2 实现与训练前审计

状态：**M2_COMPLETE — IMPLEMENTATION_V2_FROZEN_BEFORE_FIRST_OPTIMIZER_STEP**。

## 实现结果

新增 `rcias_clgri/ni/phase6m_selective_risk.py`，实现训练折限定的 hierarchical support、候选/候选 bank 不确定性特征、84 维 selector 编码、三 head selective-risk MLP 和 whole-state-balanced distribution loss。训练入口实现 18 个严格 nested inner-ranker runs、9 个 outer selector runs、逐 run checkpoint/hash、可恢复跳过和最终 OOF identity audit；启动器以独立 session 持久运行并记录 PID、日志、输出、resume 路径和 ETA。

实现会逐文件验证 M0 保存的全部 77 个 Phase 6L artifacts、Phase 6L 的更早 predecessor manifests、M1 preregistration、schema amendment、M2 code/input hashes和 R13/R14 access ledgers。selector online feature frame 只保留预注册字段；任何 outcome、decoded makespan 或 historical score/rank 字段进入 feature frame 都会 fail-closed。每个状态必须保持 requested 24 rules、完整 deduplicated bank、唯一且 ID 一致的 canonical fallback，以及三 ranker seeds 的 candidate identity/order。

## 训练前 schema fail-closed 与修正

首次 CPU preflight 在 optimizer step 前发现 `bottleneck_proxy` 实际为四值 categorical，而原 M1 表误列为 numeric。错误、无效 v1 implementation hashes 和零 optimizer/zero OOF 边界保存在 `implementation/preflight_failure_v1.json`。M1 amendment 只把该字段移入 one-hot categorical；没有改变模型、target、loss、support、cross-fit、gate 或 promotion criteria。有效实现单独冻结为 `feature_schema_v2.json` 与 `implementation_protocol_v2.json`，没有覆盖 v1 证据。

## 验证

修正后 CPU preflight：

- states：96；candidates：2,270；
- selector input dimension：84；
- held-fold hard-support rate：0.994273；
- historical score online forward calls：0；
- R13/R14 accessed：false/false。

自动化测试覆盖 support 二值隔离、fine-rule OOV、高层类别 fail-closed、margin/vote/rank feature、feature determinism、outcome 隔离、24-rule/full-bank/fallback identity、inner cross-fit fold isolation、distribution loss、schema amendment 和冻结 implementation boundary。完整仓库回归为 **367 passed in 16.48 s**。

## 冻结证据

- M1 amendment SHA256：`1c951c124740b9aeb7673031cc701129c66dac8eadbcb147ee063a071591b8ec`
- invalid v1 implementation SHA256：`6bc3542258e0e2b0342aed2e6a9fe1e867a3eaab1cf0deb01ba989b2aadf553b`
- valid v2 feature schema SHA256：`1914f733915884b505aa1aaf7f2eab34589ef410ff020aedd3d2c766bcf1eba2`
- valid v2 implementation protocol SHA256：`ccfcd786898cc59debd1332217c737b2c6d8ad159f25c9c82bff60fcac235244`

M3 只能从提交后的 v2 implementation 启动。当前没有 Phase 6M optimizer output、qualification output、后台训练或 R13/R14 访问。
