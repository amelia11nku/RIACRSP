# Phase 6N N3 实现与 smoke 报告

状态：**`PASS_TO_N4_FORMAL_TRAINING`**。

## 输入与 tensor cache

N2 完成后使用冻结的组合标签构建 Phase 6N tensor cache。原始 288 states 直接复用 Phase 6J 已验证的 CSG tensors；新增 576 states 从 N2 replay 重建 schedule 和 CSG。最终为 18 shards、864 states、20,441 candidates，tensor schema 单一。候选身份和按 `target_set_id` 的模型顺序逐 state 对齐；缓存完整性为 `PASS`，历史 scorer calls 为 0，R13/R14 未访问。构建耗时 17.06 秒。

## 冻结实现

唯一 promotable family 为 `N1_CANDIDATE_CONDITIONED_CSG`。Phase 6F 两层 `FULL_CSG` encoder 每 state 执行一次；input projections、relation block 0 和 graph projection 冻结，relation block 1 可训练。

每个候选显式使用 target mean/max/graph-query attention、六个关系族的 incoming/outgoing 边界池化、candidate-critical 与 candidate-bottleneck 交集池化、critical synchronization 外部边界池化、全局图表示及十个 score-free cheap features。关系边界按 `(candidate, heterogeneous external node)` 去重；synchronization 通道仅接受 `binding_indicator == 1`。critic 仅含 continuation-advantage 与 beats-fallback 两个输出头，不含 immediate、heteroscedastic scale 或 selector head。

参数总数为 5,766,462，可训练参数 3,060,926，其中最后 relation block 为 2,393,088，新模块为 667,838；均低于 8M/4M 上限。所有 outer folds 为 6 held instances/288 held states 和 12 training instances/576 training states，instance overlap 为 0；所有 inner/outer fit 的 categorical sizes 均为 `(25, 8, 6)`。

实现、输入、参数边界、pooling 语义及训练规则已在 `outputs/phase6n_candidate_conditioned_csg_v1/training/training_protocol.json` 中于首次 smoke optimizer step 前冻结。

## CUDA smoke

使用 seed 726101、held fold 0，inner 与 outer 各运行 1 epoch。总耗时 40.73 秒。第一批包含 4 states/94 candidates，loss 3.832924 且有限；inner epoch loss 3.666193，outer epoch loss 3.587963。checkpoint、outer-held predictions 和 inner-validation predictions 均已写入并由 run record 的 SHA256 验证。

inner validation 的 raw selected lift 为 0.003248、grouped-bootstrap LCB 为 0.000960。该结果仅证明数据、梯度、确定性 CUDA 路径与工件闭环可运行，不用于选择架构、超参数或宣告 N5 representation gate 通过。

完整回归在实现隔离修复后为 **395 passed**。测试同时验证 frozen layer 边界、candidate target indexing、relation-boundary 去重、eval determinism、outcome tensor 不被 forward 读取、whole-instance isolation 和 N2/N3 锁定证据。
