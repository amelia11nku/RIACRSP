# Phase 6N Candidate-Conditioned CSG 预注册协议

## 状态与唯一主模型

本协议在任何 Phase 6N 新 continuation rollout、optimizer step、outer OOF 或 qualification 结果生成前冻结。唯一可晋级 family 为 **`N1_CANDIDATE_CONDITIONED_CSG`**。Phase 6M selector、heteroscedastic scale、total predictive scale、旧 winner-only selector、historical scorer 和 immediate-utility hard gate 均不进入可晋级路径。

Phase 6M 终态为 `MODEL_REVISION_QUALITY`，起始 commit 为 `8efe396e6cca44493a18c07720c4effaf2add7c1`；N0 commit 为 `f38725a`。R13/R14 继续锁定，不运行 Gurobi。

## 数据边界

原始 18 个 R12 CAUR-FIT instances、288 states、6,809 candidate means 全部保留。新增数据固定为每个 instance 四条 trajectory、每条八个按 `[0.05, 0.18, 0.32, 0.45, 0.58, 0.72, 0.85, 0.95]` 分层的 states，共新增 72 条 source trajectories、576 states；合并后每 instance 恰好 48 states、总计 864 states。新增 seeds 固定为 `721201..721204`。

新增 source sampler 为 frozen H1-seeded ALNS、预算 `0.25N`，不运行 neural/scorer。每个 state 使用 frozen 24-rule full bank、原 dedup/order、八次 deterministic transport-aware repair、canonical `operator_related` fallback、H=4 continuation 和 CRN seeds 695101/695102。不得依据模型质量减少数量、换 seed 或挑选 state。基础 288-state 正式采集耗时 4,793.51 秒、490,248 decoder evaluations、41,549,687 bytes；线性估计新增任务约 9,587 秒（2.66 小时）、980,496 evaluations、83,099,374 bytes。冻结前磁盘可用约 56 GiB，资源允许采用说明书推荐的 48 states/instance。

新增 sampler 与原 288 states 的 Phase 6H source policy 不同，因此报告必须把 common-original-288 与 expanded-864 指标分开。新增 ALNS states 的作用是扩大当前调度分布和独立 state 数；不得把二者混写成完全同分布复制。

## Candidate-conditioned 表示

Phase 6F checkpoint 初始化一个两层、hidden 128 的 FULL_CSG encoder。每个 state 只编码一次。input projections、relation block 0 和 graph projection 冻结；只允许 relation block 1、candidate pooler、comparative fusion 与两个 heads 训练。

每个 candidate 使用目标 OP 的 mean/max/graph-query attention 与 normalized size。对六类边界分别保留 incoming/outgoing channels：precedence、island/resource、reconfiguration、W-AGV transport、F-AGV delivery 和 binding synchronization。每个 channel 对 target 外侧的一跳 node embeddings 做 mean/max/count pooling；空集合为零。另做 candidate∩critical、candidate∩bottleneck、candidate-critical binding synchronization boundary 三个池。所有候选从同一份 node/edge tensor 批量索引，不允许每候选重跑图网络。

候选 embedding 同 frozen canonical fallback embedding 及二者差值再融合，直接输出 fallback-relative continuation advantage 与 beats-fallback logit。没有 immediate head、Gaussian scale 或 selector。在线特征只能来自当前 CSG、candidate identity/provenance、canonical fallback 和十个既有 score-free cheap features；任何 continuation/repair/after-state outcome 都禁止作为输入。

## Whole-instance nested OOF

沿用三个 structural folds：fold 0=`S_CF1/M_CF2/L_CF3`，fold 1=`S_CF2/M_CF3/L_CF1`，fold 2=`S_CF3/M_CF1/L_CF2`，每个 cell 的 C01/C02 instance 均不可拆分。每个 outer held fold 为 6 instances/288 expanded states，训练侧为 12 instances/576 states。inner fit=`(held+2)%3`，inner validation/calibration=`(held+1)%3`。禁止 candidate-row split。

三个固定 seeds 为 726101/726102/726103。每个 seed 的 epoch 由 inner-validation instance-grouped raw selected-lift LCB 决定，tie-break 为较低 joint loss、再较早 epoch；之后从相同初始化在两个 outer-training folds 上按固定 epoch refit。FP32、deterministic algorithms、AdamW；new modules LR `3e-4`，最后 relation block LR `5e-5`。

Loss 权重固定为 pairwise logistic 1.0、ListNet 0.75、Huber advantage 0.5、beats-fallback BCE 0.25。ranking terms 在 state 内标准化；gap weight 与 Huber delta 只由允许的 training fold 按配置公式计算。

## Raw representation gate

先在 common original 288 states 对 frozen Phase 6L 作 paired comparison，再单独报告 expanded 864-state OOF。硬门槛为：overall 与 S/M/L selected lift 全正；overall Spearman 正且无 scale Spearman 反转；common 与 expanded 的 instance-bootstrap lift LCB 均正；common paired lift-improvement LCB 正；common Spearman 至少超过 Phase 6L 0.02；Spearman/pairwise/NDCG@1 至少两项严格优于 Phase 6L；winner 至少来自三个 origin families 且单族不超过 75%；full-bank、finite、feasibility、fold isolation 全通过。

0.25/0.60/0.75 仍分别是 Spearman、pairwise、NDCG@1 的 preferred diagnostics，单项窄幅未达不独立触发失败。任一硬门槛失败即 `MODEL_REVISION_REPRESENTATION`，停止 calibration、decision、runtime 和 solver。

## Empirical calibration 与 direct decision

只有 raw gate 通过才运行。每个 outer fold 使用 inner critic ensemble 在 inner-validation state 上选出的 winners，计算 `predicted - realized`，冻结 numpy linear empirical 0.90 quantile；`empirical_LCB = prediction - quantile`。beats probability 用同一 inner-validation winners 做 Platt；若缺少任一类别则 fail closed。Outer held fold 只应用一次并报告 overall/instance/scale coverage。

Winner 是 ensemble mean 最大 candidate，按 `target_set_id` lexical tie-break。只有 winner 非 fallback、empirical LCB>0、Platt probability>=0.55，且 identity/full-bank/index/finite/decoder semantic support 全通过时才 intervene；否则用 canonical fallback。没有 distance gate、Phase 6M selector 或 immediate gate。

Direct-decision gate 要求 overall gated lift 与 instance-bootstrap LCB 均正、S/M/L lift 非负、每个 scale 至少十次 direct interventions、至少三个 winner origin families 且单族不超过 75%，并通过 full-bank/feasibility。失败为 `MODEL_REVISION_DECISION_CALIBRATION`。

## 后续门槛

Primary architecture 冻结后才运行四项不可晋级 ablation：cheap only、global+cheap、target pool、full target+relation boundary。不得依据 ablation 改选主模型。

只有 raw 与 direct-decision 都通过才进入 runtime。正式门槛仍为 neural p90<=30 ms、complete-live p90<=100 ms，FP32，一次 graph encoding，全 bank batched pooling，不使用 CUDA graphs 或 favorable rerun。只有 exact deployable bundle 与 runtime 通过才可按 `2N` 秒做 R12 solver gate；R13/R14 的解锁顺序不变。
