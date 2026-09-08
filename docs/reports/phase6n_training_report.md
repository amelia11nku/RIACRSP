# Phase 6N N4 训练报告

状态：**`PASS`**。

正式 whole-instance nested OOF 已完成 3 seeds × 3 held folds，共 9 runs。完整训练 wall time 为 2783.66 秒；每个 held fold 包含 6 instances、288 states，训练侧包含 12 instances、576 states，instance overlap 为 0。best epochs 为 `[9, 3, 5, 4, 9, 9, 4, 5, 6]`。

外层 OOF 共 61,323 seed-candidate rows，inner-validation 亦为 61,323 rows；两者对每个 state/candidate 均恰好覆盖三个 seeds。ensemble 为 20,441 candidates/864 states，身份、顺序和 continuation truth 与冻结组合数据完全一致。全部 checkpoint、outer prediction、inner-validation prediction 与 run record 哈希通过。

Expanded OOF：Spearman 0.253297，pairwise accuracy 0.589622，NDCG@1 0.628987，raw selected lift 0.007120，grouped-bootstrap LCB 0.004509。

Common original 288：Spearman 0.211669，pairwise accuracy 0.574069，NDCG@1 0.617505，raw selected lift 0.006411，LCB 0.002760。

历史 scorer calls 为 0；所有候选标签均为 feasible；R13/R14 未访问。N4 仅证明训练和 OOF 工件完整，是否进入 calibration 由独立 N5 raw representation gate 决定。
