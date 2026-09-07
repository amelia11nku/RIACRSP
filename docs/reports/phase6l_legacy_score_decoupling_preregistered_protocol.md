# Phase 6L Legacy-Score Decoupling 预注册协议

状态：**PREREGISTERED BEFORE PHASE 6L DATASET DERIVATION, OPTIMIZER STEP, OR NEW QUALIFICATION OUTPUT**

## 科学问题与继承状态

Phase 6K 以 `MODEL_REVISION_RUNTIME` 终止。E4R 已证明冻结 J1 的 FP32 神经计算可在同输入上满足实现等价及 30 ms neural p90，但包含历史 Phase 6J source-score 前处理的完整 live p90 为 109.495 ms，超过 100 ms。Phase 6L 检验一个新的模型假设：continuation-value policy 是否能在在线决策中完全移除历史 scorer，同时保持有用的候选排序、gate 行为和求解改进。

这不是 E4R 数值等价优化。Phase 6L 不声称复现 J1 分数；它是独立模型修订。Phase 6I-MR、6J、6K 文件保持只读，R13/R14 保持锁定。

## 数据边界

沿用 `phase6j_instance_manifest.csv` 中已有名称：R12=`CAUR_FIT`、R13=`CAUR_SELECT`、R14=`CAUR_HOLDOUT`。仓库没有另一份与 Phase 6J 同定义的 continuation-value 训练 split，因此不虚构“pre-R12” split。Phase 6L 只复用已经冻结的 R12 full-bank continuation 结果，通过同一 whole-instance/structural-cell nested OOF 协议训练与资格评估；R13 是首个未见 selection split。

R12 共 18 个 instance、288 个状态、6,809 个去重候选。每个状态保存两个 CRN seed、H=4 continuation 结果。L2 只从这些不可变结果派生 score-free 数据，不执行新 continuation rollout，不重新生成历史 score。原聚合标签与 raw seed 标签的 SHA-256 分别为 `616a0f20ea76eaf88b5f0c51d5bdb023aada4c42bada416b4d81d739bea637af` 和 `d821cf426da82eacbea2a501f0a4f87acb1089e82b32314a076a919d860fe86e`。

三折固定为：

- fold 0：S_CF1、M_CF2、L_CF3；
- fold 1：S_CF2、M_CF3、L_CF1；
- fold 2：S_CF3、M_CF1、L_CF2。

对每个 outer held fold `h`，inner training 为 `(h+2)%3`，inner validation 为 `(h+1)%3`；inner validation 只选择 epoch 数。随后使用两个非 held folds 按该 epoch 数重新拟合，并只对 outer held fold 预测。每个状态/候选/seed 只能产生一个 outer OOF 预测。架构、特征、seed、校准候选和 gate 网格不得从这些结果改变。

L4 对训练稳定性、完整性和冻结 outer OOF 质量做停止判断；L6 只允许对这个唯一 bundle 汇总正式 R12 科学 gate 并执行一次连续 formal runtime。不能在 R12 比较多个可晋级模型后择优。R13/R14 不得参与训练、归一化、校准、阈值或实现选择。

## 候选、fallback 与标签

保留 H1 初始化、CSG、`generate_revised_target_arms` 的 24 条规则、按排序 operation tuple 的插入序去重、target-set identity、8 次 deterministic repair、decoder 与 feasibility 语义。正式训练和 live 决策始终使用完整去重 bank；top-8 仅属前代审计，不进入 Phase 6L。

fallback 固定为去重 bank 中唯一包含 `operator_related` provenance 且 operation set 等于 `canonical_related_target` 的候选。缺失、非唯一或集合不一致立即失败。该规则不调用历史网络。

Phase 6J 的角色唯一性使 10/288 个状态的存档 fallback 不是 canonical candidate。L2 使用已保存的每候选、每 seed continuation best makespan，将全部标签重新锚定为：

`A_H = (V_H(canonical operator_related fallback) - V_H(candidate)) / pre-action incumbent makespan`。

`beats_fallback`、均值、标准差和 fallback continuation 字段同步重新派生。原文件不修改，也不挑选重复 realization。

## 在线特征与泄漏边界

categorical 输入保持 `primary_origin_rule`、`origin_destroy_operator`、`origin_family`。numeric 输入固定为：`origin_rule_count`、`origin_family_count`、`destroy_target_cardinality`、`destroy_target_fraction`、相对 canonical fallback 的 overlap fraction/Jaccard、critical overlap、bottleneck overlap、`normalized_diversity_rank` 和 `is_fallback`。

删除 `best_frozen_score_jaccard` 与 `normalized_frozen_score_rank`。历史 raw score、校准概率、历史 utility 既不进入 online feature，也不进入 primary training objective；本轮不启用 offline teacher。continuation、repair、future-search outcome、candidate decoded makespan 和 holdout metadata 均禁止作为在线特征。

每个 inner/outer training fold 单独拟合 categorical vocabulary、numeric median/IQR，IQR 下限为 `1e-6`。support 要求 category 在训练 fold 出现，且裁剪前 robust z 全在 [-8,8]；网络输入裁剪到 [-8,8]。任何 held fold 或 holdout 统计都不能进入 transform。

## 唯一 primary 模型

唯一晋级候选为 `L1_SCORE_FREE_CONT_FROZEN` 三 seed ensemble，seed 为 706101、706102、706103。它冻结 Phase 6F/6H graph-state-action encoder，只训练新的 10-numeric candidate context 与 continuation advantage、beats-fallback、immediate utility 三个 head。context 和 head 宽度、dropout 与 Phase 6J J1 相同；总参数上限 5.35M、可训练参数上限 0.50M。全部新训练与推理为 FP32。

训练保持 AdamW、lr `3e-4`、weight decay `1e-4`、每 batch 8 个完整 state group、gradient norm 1.0、maximum 120 epochs、patience 12。loss 权重原样保持：pairwise logistic 1.0、ListNet 0.75、advantage Huber 0.5、beats-fallback BCE 0.25、immediate Huber 0.10。pair gap scale 与 immediate Huber delta 在 optimizer 第一步前从完整重新锚定数据按 Phase 6J 规则计算并冻结。

不存在可晋级 fallback model。强制的一项非选择性消融 `L1_NO_FALLBACK_CONTEXT_ABLATION` 仅用 seed 706101，去除 fallback overlap/Jaccard/indicator，用于估计确定性 replacement 的贡献；其结果不能改变 primary 或被晋级。

## 校准、support 与 gate

使用三个 seed 的 outer OOF ensemble。校准只使用每状态 argmax-selected OOF winner。候选为 Platt 与 isotonic；仅当每个 cross-fit 拟合侧至少 200 个 winner 时允许 isotonic。按 ECE、Brier、Platt 简单性依次选择。

winner 为 ensemble mean continuation advantage 最大者，`target_set_id` 打破并列。gate 网格保持 `p_min={0.55,0.65,0.75}`、`lambda={0.5,1.0}`、`delta={0,0.0025,0.005}`，immediate harm floor 为 -0.005。干预必须同时通过 probability、LCB、support 和 harm。每 scale 至少 20 个直接干预；或至少 40 个 forced-abstention 且其 lift 95% UCB 非正。

gate 筛选顺序原样保持：overall selected lift 与 grouped-bootstrap LCB 均为正；各 scale lift 非负；coverage/exception 通过；随后最大化 lift LCB、最小化 regret、最小化 selected-winner ECE、最大化 supported intervention 数，再按较小的 p、lambda、delta 字典序打破并列。bootstrap 单位为 instance mean、2,000 次、seed 707001。

## 科学质量 gate

Phase 6J 的 essential gate 原值复制如下：replay/feasibility=`PASS`；full-bank completeness=`PASS`；overall Spearman >0；每个 scale mean Spearman >=0；selected lift >0；selected grouped-bootstrap LCB >0；selected-winner ECE <=0.10；candidate-origin collapse 必须为 false。

preferred 指标保持 overall Spearman >=0.25、pairwise accuracy >=0.60、NDCG@1 >=0.75，只作为同原协议一致的 preferred 诊断，不替代 essential gate。任何 essential quality、完整性、泄漏或 provenance 失败都终止为 `MODEL_REVISION_QUALITY`，不得为诊断打开 R13/R14。

L4 还报告 selected utility/lift、regret、support-aware coverage、fallback/intervention frequency、S/M/L、CF/search-stage 分层、三 seed stability、与冻结 Phase 6J J1 的既有 OOF 对照，以及预注册的单 seed fallback-context 消融。

## Runtime 协议

primary runtime 使用 E4R 风格的显式 FP32 vectorized ensemble arithmetic，不使用 `torch.compile` 或 CUDA graphs，不引入 FP16/BF16。每次 live 决策真实执行 CSG、24-rule generation/dedup、score-free features、tensorization/transfer、Phase 6L ensemble、calibration/gate 与 action extraction；历史 frozen-score forward 必须精确为 0。允许同一决策内复用已经构造的 tensor，禁止跨决策缓存。

L5 开发目标为 neural p90 <=25 ms、complete-live p90 <=85 ms，用于给正式 hard cap 留出余量；不改变正式门槛。正式 L6 使用全部 288 个 R12 状态，在一个连续进程中每状态 3 次 warmup、5 次保留测量。每个 repetition 都执行 fresh complete live decision，保留全部样本，不过滤 slow row，不选择有利重跑；报告 overall 与 S/M/L 的 p50/p90/p99。

正式门槛保持 neural p90 <=30 ms、complete-live p90 <=100 ms。quality 通过但 runtime 失败时终止 `MODEL_REVISION_RUNTIME`；quality 失败时终止 `MODEL_REVISION_QUALITY`。只有两者均通过，才可冻结 deployable bundle 并按已有共同计时口径执行 R12 solver gate；其预算为 `2 * instance.num_operations` 秒。solver gate 失败为 `NO_GO_R12_SOLVER`。只有 bundle 与 solver gate 均通过才可 `PASS_TO_R13`。

## 冻结与执行顺序

机器配置位于 `configs/phase6l_legacy_score_decoupling_v1.json`。L1 冻结记录将保存配置、本协议、L0 dependency map、输入数据、前代 protocol、base checkpoint、现有 tensor cache、代码和 access boundary 的 SHA-256。

执行顺序固定为 L2 数据/feature manifests、L3 三 seed primary、L4 quality/ablation、L5 development runtime、L6 单一 bundle R12、条件式 L7 bundle/solver、条件式 L8 R13 后 R14。不得跳过前置 gate。
