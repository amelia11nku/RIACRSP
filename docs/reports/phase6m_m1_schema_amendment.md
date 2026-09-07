# Phase 6M M1 schema amendment 1：`bottleneck_proxy` 字段类型修正

状态：**FROZEN_SCHEMA_CORRECTION_BEFORE_FIRST_OPTIMIZER_STEP**。

M2 的首次 CPU preflight 在任何 optimizer step 或 outer-OOF 生成前 fail-closed。原 M1 config 把 `bottleneck_proxy` 列入 `structural_numeric`，但冻结 Phase 6L R12 数据中该字段是四值类别：`CROSS_RESOURCE_SYNCHRONIZATION`、`F_LOGISTICS`、`RECONFIGURATION`、`W_LOGISTICS`。直接转为 float 因而失败。

本 amendment 只修正字段编码：

- 从 `structural_numeric` 移除 `bottleneck_proxy`；
- 将 `bottleneck_proxy` 加入 `categorical_one_hot`，沿用训练折词表、显式 `UNK` 的既定编码；
- 其余 ranker、selector、target、loss、support、cross-fit、gate 和 promotion criteria 完全不变。

禁止用任意 ordinal 数值映射替代该修正。原 preregistration、首次无效 implementation protocol、feature schema 和 preflight 错误均保留；新实现使用带 `_v2` 后缀的冻结协议。此次修正在第一步优化前完成，不包含任何 Phase 6M qualification outcome，也没有访问 R13/R14。
