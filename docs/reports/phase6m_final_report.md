# Phase 6M Score-Free Selective Confidence 最终报告

## 最终决策

**`MODEL_REVISION_QUALITY`**，停止边界为 **M4、早于 development runtime**。

唯一 promotable family `M1_SCORE_FREE_SELECTIVE_RISK` 已完成 18 个 inner ranker runs、9 个 selector runs 和完整 nested outer OOF。它精确保留 Phase 6L 的候选、winner 与 state-level raw ranking 结果：Spearman 0.177723、pairwise accuracy 0.563103、NDCG@1 0.627466、raw lift 0.007516。Phase 6M 预注册 bootstrap seed 得到 raw-lift LCB 0.002890；Phase 6L 原 seed 的存档 LCB 为 0.002866，两者均为正。

新的 support 表示达到了候选 99.192%、winner 100.000%，消除了旧 support 对 winner 的拒绝。新的 confidence/scale 没有恢复选择能力：ECE 0.090809、AUROC 0.578258、AUPRC 0.633791、resolution 0.006475；只有 4.861% winner 达到 `p>=0.55`，且最宽松 selector LCB 的 288 个值全部为负。18 个 gate 因而全部 0 intervention、0 retained。

## 科学解释

Phase 6L 已证明 score-free continuation ranking 信号存在，Phase 6M 进一步证明单独改造当前 selector/support 方案仍不足以形成可部署的 cross-scale intervention。Support 修订有效，但不是当前阻塞点。主要阻塞是预测 scale 相对 mean 过大，使预注册 lower bound 全部为负；probability sharpness 也从 Phase 6L 的 0.099062 降到 0.068851，AUROC 未提高。

下一轮必须作为新的预注册模型修订。应先诊断 outer-fold scale shift、两条 CRN outcome 对 aleatoric scale 的可识别性、selector scale 与实际误差的覆盖关系，以及仍只有 84/288 通过的 frozen immediate-utility head。若研究 archived historical score 信息，只能采用说明书允许的独立、不可晋级 offline-teacher/distillation 诊断；不得回头调本轮 gate、seed、LCB 系数或 coverage 下限。

## 完整性与停止边界

- M3 连续 worker 耗时 246.69 秒，PID 52577 已退出，无后台 Phase 6M 任务。
- 完整回归：**372 passed in 16.28 s**。
- Phase 6L 保护 manifest：77 files / 16,687,216 bytes，全部通过。
- 前代保护：Phase 6I/6J 6,636 files，Phase 6K 21,365 files，tracked predecessor 139 files。
- historical score online forward calls：0；Gurobi：未运行；PASS_LOCKED_NOT_ACCESSED。
- M5/M6 runtime、bundle、R12 solver、R13、R14：均未运行。

起始 commit：`8f0d37397af82e6a8bfdb437c88459712cdf5ec7`。终局证据生成前 HEAD：`4af628e5d9229c7cf37432765c6b853c361a6287`。最终 evidence commit 是包含本报告与 `outputs/phase6m_selective_confidence_v1/final/final_decision.json` 的后续本地 commit。
