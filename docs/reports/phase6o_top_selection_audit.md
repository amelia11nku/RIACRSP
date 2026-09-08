# Phase 6O O0 Top-Selection 审计

## 路线决策

**`PROCEED_TOP_UTILITY_RETRAIN`**。

冻结 Phase 6N critic 在 k≤6 内未达到强 shortlist recall。Route A 的 oracle evaluation 潜力很高，但真实最优候选经常不在 shortlist 中；用 solver outcomes 调整 k 会违反冻结边界。因此 O1 应预注册 Route B：保留 candidate-conditioned pooling，冻结整个 Phase 6F encoder，以 top-utility-aligned、source-balanced、noise-aware objective 重训。

| 范围 | k | Exact recall | Near-best@0.005 | Oracle lift | Regret | Positive-opportunity presence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| common_original_288 | 1 | 7.986% | 11.111% | 0.006411 | 0.034971 | 59.567% |
| common_original_288 | 2 | 15.972% | 22.569% | 0.017885 | 0.023497 | 77.256% |
| common_original_288 | 3 | 22.222% | 29.167% | 0.022597 | 0.018785 | 83.755% |
| common_original_288 | 4 | 30.208% | 38.889% | 0.027242 | 0.014140 | 88.087% |
| common_original_288 | 6 | 37.847% | 47.569% | 0.030873 | 0.010509 | 93.863% |
| common_original_288 | 8 | 44.097% | 53.819% | 0.032591 | 0.008791 | 96.390% |
| expanded_864 | 1 | 9.722% | 13.889% | 0.007120 | 0.034595 | 59.353% |
| expanded_864 | 2 | 18.866% | 24.653% | 0.019048 | 0.022668 | 76.739% |
| expanded_864 | 3 | 26.273% | 32.986% | 0.024571 | 0.017144 | 84.892% |
| expanded_864 | 4 | 31.829% | 40.162% | 0.028294 | 0.013422 | 89.448% |
| expanded_864 | 6 | 40.741% | 50.810% | 0.031998 | 0.009718 | 94.604% |
| expanded_864 | 8 | 50.116% | 59.954% | 0.034253 | 0.007462 | 96.643% |

没有 k≤6 同时满足 common/expanded exact recall 80%、ε=0.005 near-best recall 90% 及 scale-collapse 条件。k=6 的 common exact/near recall 分别为 37.847%/47.569%；expanded 分别为 40.741%/50.810%。

## Top-selection 与标签噪声

common 288 中，Phase 6N 的 state-level pairwise accuracy 改善、但 top candidate regret 同时变差的状态为 51。paired selected-lift delta 为 -0.001105；最差 25 个负向状态贡献全部负 deficit 的 50%，最差 10 个贡献 25.738%。按 scale 的 delta 为 `{'L': -0.0030125423051119236, 'M': 2.26521428044662e-05, 'S': -0.0003257171573794263}`。

两条 CRN continuation seeds 在 501/864 个状态上给出不同 top candidate（57.986%）。这是显著的 top-label instability，O1 应在任何新标签生成前冻结 targeted high-fidelity relabeling 的 candidate union、额外 seeds 与停止边界；原 Phase 6L/6N truth 不得覆盖。

## 目标函数与来源偏移

Phase 6N 每 state 平均有 536.4 个 ordered pairs，但 true-best-vs-rest 仅 22.6 个；其 pairwise-loss contribution 均值占比 13.526%。ListNet 给 best/top-3/top-6 的平均 target mass 为 19.398%/40.705%/60.609%。现有 pairwise + standard-z ListNet objective 因而主要优化 broad ordering，而非 top-1 expected utility。

Phase 6N 训练以 state 等权处理 288 个 ORIGINAL_PHASE6J_CAUR 与 576 个 NEW_ALNS_EXPANSION，实际 aggregate weight 为 1/3 与 2/3。反事实恢复 Phase 6F relation block 后，fine-tuned block 引起的 mean absolute prediction drift 在 ORIGINAL 为 0.153709，NEW 为 0.152908；ORIGINAL 仅高 0.000802（比值 1.0052），没有观察到大的 source-specific drift 差距。完整 feature/origin/advantage/error/composition 分布见 `source_shift_metrics.csv`。

## 边界

O0 没有 optimizer step、live solver、Gurobi、R13 或 R14 访问。Phase 6I-MR/6J/6K/6L/6M/6N 保护链通过。路线在 O1 配置与协议冻结前不得开始训练或 targeted relabeling。
