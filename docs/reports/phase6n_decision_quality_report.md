# Phase 6N 决策质量报告

状态：**未运行，N5 表示质量门槛终止。**

N5 得出 `MODEL_REVISION_REPRESENTATION`。按预注册协议，只有 raw critic quality 通过后才能执行 N6 empirical residual calibration 与 direct-decision gate。因此没有拟合 residual quantile、没有选择 decision threshold、没有生成 intervention coverage 或 gated-lift 结果，也没有以 Phase 6M selector 替代该步骤。

N7 的四组 non-promotable representation ablations 同样未运行。它们不能用于事后改选已冻结 primary，也不能覆盖 N5 的 paired utility 硬失败。

R13/R14 保持锁定。
