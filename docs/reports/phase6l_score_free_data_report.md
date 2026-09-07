# Phase 6L L2 无分数数据与模型接口报告

## 结论

L2 为 **PASS**。已从受保护的 Phase 6J `R12_CAUR_FIT` continuation 结果派生独立的 score-free 数据集，未修改任何前代文件，未执行新的 decoder/continuation rollout，未调用历史 frozen-score 模型，未读取 R13/R14。

## 数据完整性

- 288 个 state、6,809 个 deduplicated candidate、13,618 个 candidate/CRN-seed 行完整。
- 每个 state 恰好一个 canonical `operator_related` fallback；576 个 fallback seed 行的 advantage 均严格为 0。
- 10 个 state 按 L0 结论重新锚定 fallback-relative 标签。
- 其余 278 个 state 的 `continuation_advantage_mean` 与 Phase 6J 原值最大绝对差为 0。
- 新 grouped dataset SHA-256：`461e644c2b2d924c2ab17d9fd7ce6d03787712ef18f26eec01dd4a761d178daa`。
- 新 raw-seed dataset SHA-256：`74c13c47077f0e904f27a3acc825e71d729c5e1ef89272584a61a5ff559aa584`。

在线 schema 已删除 `best_frozen_score_jaccard`、`normalized_frozen_score_rank` 及所有历史 raw/calibrated score 字段。normalization manifest 分别冻结每个 inner/outer training fold 的 vocabulary、median 和 IQR；held fold 不进入对应 transform。

## live builder 一致性

使用全部 288 个存档 schedule 重建 24-rule bank，并比较 6,809 个候选的 13 项 online input。candidate identity、lexical model-input order及所有 categorical/numeric feature 均通过；生成/比较路径的历史模型 forward 次数为 0。

Phase 6L 新接口不接受 policy、model 或 frozen-score 参数，并对以下偏差 fail closed：非 24-rule bank、dedup count 不闭合、candidate order 错位、canonical fallback 缺失或非唯一、历史 score 字段进入 online rows、continuation/repair outcome 泄漏。

## 模型和 optimizer 边界

唯一 primary `L1_SCORE_FREE_CONT_FROZEN` 保留冻结的 Phase 6F/6H encoder，使用 3 个 categorical embedding 和 10 个 numeric input。实际总参数为 5,332,927，可训练参数为 36,799，均低于 5.35M/0.50M 上限。

optimizer 前 training protocol 已冻结。重新锚定不会改变 state 内候选两两 label gap，因此 pair-gap scale 仍为 `0.02131782945736434`；immediate 标签未变，Huber delta 为 `0.06125535891414354`。

冻结后在 CUDA FP32 上执行一个 outer fold、两 epoch 的 smoke。inner/outer 训练、反向传播、checkpoint、held-fold 预测与 tensor action 对齐全部完成，运行 7.424 s；这是接口 smoke，不是质量选择证据。

完整仓库回归为 **350 passed in 16.29 s**。下一步仅运行预注册的 3 seeds × 3 outer folds 正式训练；smoke checkpoint 不可进入 ensemble 或 promotion。
