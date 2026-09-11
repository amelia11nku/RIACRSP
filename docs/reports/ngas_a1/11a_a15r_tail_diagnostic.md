# NGAS A1.5R 尾延迟诊断

本报告是开发诊断，不是正式资格结果。A1.5 历史结果保持不变。

| 状态 | 条件 | wall p50 / p90 / p99 (ms) | GC samples | voluntary ctx samples | involuntary ctx samples | residual p50 (ms) |
|---|---|---:|---:|---:|---:|---:|
| M | NORMAL_GC | 34.355 / 58.029 / 59.110 | 100 | 100 | 12 | 2.989 |
| M | GC_DISABLED_DURING_REFRESH | 33.287 / 33.605 / 33.958 | 0 | 100 | 12 | 2.908 |
| L | NORMAL_GC | 76.621 / 79.927 / 83.554 | 100 | 100 | 25 | 3.661 |
| L | GC_DISABLED_DURING_REFRESH | 51.738 / 52.551 / 54.627 | 0 | 100 | 21 | 3.711 |
| L_MAX | NORMAL_GC | 81.070 / 84.821 / 87.533 | 100 | 100 | 26 | 3.930 |
| L_MAX | GC_DISABLED_DURING_REFRESH | 55.064 / 55.828 / 57.873 | 0 | 100 | 23 | 4.028 |

## 解释

- M: normal GC p90 58.029 ms, GC-disabled p90 33.605 ms; normal samples containing GC collection 100/100.
- L: normal GC p90 79.927 ms, GC-disabled p90 52.551 ms; normal samples containing GC collection 100/100.
- L_MAX: normal GC p90 84.821 ms, GC-disabled p90 55.828 ms; normal samples containing GC collection 100/100.
- Every normal-GC sample (300/300) contained at least one collection, so an within-condition no-collection mean is unavailable.
- Stable median cost remains structural even when GC is disabled; the implementation must reduce CSG/event, bank, action-feature, and tensor materialization work.

GC-disabled 数据只用于归因，不用于正式门槛判定。
