# RCIAS 最终算法路线设计总结：H1 初始化 + CSG-NI 神经改进框架

## 1. 当前研究结论与路线调整

经过 Phase 3–5B 的 constructive BC/PPO 研究以及 Phase 5C 在原生 RCIAS-CB1 基准上的强基线比较，当前已经获得几个关键结论：

1. **H1 是当前最可靠的构造式初始解方法。** 相比 H2、BC 与 Phase 5B 三随机种子 PPO，H1 在原生 RCIAS 场景中表现更稳定、推理速度显著更快，且不存在明显的随机种子崩溃问题。
2. **Phase 5B constructive PPO 不适合作为最终初始解生成器。** 当前 PPO 存在训练分布偏移、operation branch 冻结形成性能上限、不同随机种子波动大、高柔性场景下灾难性退化等问题。
3. **ALNS-H1 是当前最强的传统求解基线。** 在 45 个 RCIAS-CB1 Core 实例上，ALNS 从 H1 初始解出发仍能进一步获得约 5% 的平均 makespan 改善，说明当前真正有价值的优化空间主要位于“高质量初始解之后的解改进阶段”。
4. 因此后续主线从

\[
\text{BC} \rightarrow \text{constructive PPO}
\]

正式调整为

\[
\boxed{
\text{H1 Initialization}
\rightarrow
\text{Critical Synchronization Graph}
\rightarrow
\text{Neural Improvement}
}
\]

即：**不再让强化学习从空白状态构造整个调度，而是让学习模型基于一个已经较强的 H1 可行解，判断“哪里值得改、改多少、改哪些工序以及如何修复”。**

---

# 2. 最终算法总体框架

暂定算法概念名称：

> **CSG-NI: Critical Synchronization Graph based Neural Improvement**

总体流程：

\[
I
\xrightarrow{H1}
S_0
\xrightarrow{CSG}
G_0
\xrightarrow{NI}
a_0
\xrightarrow{Destroy/Repair}
S_1
\xrightarrow{CSG}
G_1
\rightarrow \cdots \rightarrow
S^*
\]

其中：

- \(I\)：RCIAS 问题实例；
- \(S_0\)：H1 生成的初始可行调度；
- \(G_t\)：由当前完整调度 \(S_t\) 构造的 Critical Synchronization Graph；
- \(a_t\)：Neural Improvement policy 生成的改进动作；
- \(S_{t+1}\)：执行 destroy/repair 后得到的新可行调度；
- \(S^*\)：给定计算预算下获得的最好调度。

核心学习问题由原先的：

> “如何从空白状态逐步构造完整调度？”

改为：

> **“给定一个已经较好的完整调度，当前 makespan 的主要结构性瓶颈在哪里，以及最值得进行哪种局部/大邻域改进？”**

---

# 3. 核心创新点设计

## 3.1 创新点一：Critical Synchronization Graph（CSG）

CSG 不是传统意义上的静态 instance graph，也不是只描述 operation-machine eligibility 的 disjunctive graph，而是一个**基于当前完整可行调度构造的动态 solution-state graph**。

CSG 需要显式描述 RCIAS 中已经实现的同步结构，包括：

\[
E =
E^{prec}
\cup
E^{island}
\cup
E^{reconf}
\cup
E^{W}
\cup
E^{F}
\cup
E^{sync}
\]

其中：

- \(E^{prec}\)：装配工序 DAG precedence；
- \(E^{island}\)：同一装配岛上的实际执行链；
- \(E^{reconf}\)：配置转换关系；
- \(E^{W}\)：W-AGV 实际任务链；
- \(E^{F}\)：F-AGV 实际任务链；
- \(E^{sync}\)：加工开始前的多资源同步关系。

CSG 的重点不是回答“一个工序可以去哪加工”，而是回答：

> **“当前 makespan 为什么这么大？”**

因此必须包含动态特征，例如：

- start/end time；
- slack；
- criticality；
- waiting；
- blocking；
- reconfiguration delay；
- W/F transportation delay；
- resource load；
- synchronization delay；
- 当前实际资源链位置。

CSG 应成为论文中最重要的问题特定表示创新之一。

---

## 3.2 创新点二：Hierarchical Neural Improvement

Neural Improvement 不应只做“ALNS 算子选择”。

最终目标动作建议分解为：

\[
a_t =
(a_t^{bottleneck},
 a_t^{size},
 a_t^{target},
 a_t^{repair})
\]

包括：

1. **Bottleneck decision**：当前最值得优化的是哪一类瓶颈；
2. **Destroy/neighborhood size**：本次扰动规模；
3. **Destroy target selection**：真正应该被重新优化的工序集合；
4. **Repair decision/ranking**：可选的修复策略或候选插入排序。

其中最重要的学习任务预计是：

\[
\boxed{\text{Learn where to search}}
\]

即直接学习每个工序/事件的 improvement potential，而不是仅仅学习：

\[
\text{which heuristic operator to call}
\]

这能使 CSG-NI 与普通 RL-guided ALNS/operator-selection 方法形成明显区分。

---

## 3.3 创新点三：Search-supervised Neural Learning

现有 ALNS 不只是 baseline，也将成为 teacher 与数据生成器。

训练实例首先通过：

\[
I \rightarrow H1 \rightarrow ALNS
\]

产生搜索轨迹：

\[
S_t \rightarrow S_{t+1}
\]

记录：

\[
(G_t,a_t,\Delta C_t)
\]

其中：

\[
\Delta C_t = C_{\max}(S_t)-C_{\max}(S_{t+1})
\]

通过这些轨迹，NI 可以先学习：

- 什么状态下哪些 destroy operator 有效；
- 哪些工序更值得被重新优化；
- 什么 destroy size 更有效；
- 什么 repair 方式更可能产生改善。

这样避免再次采用 cold-start PPO。

---

## 3.4 创新点四：Counterfactual Neighborhood Learning

仅模仿 ALNS 实际选择的动作是不够的，因为 ALNS 本身具有随机性和启发式偏差。

对选定状态 \(G_t\)，应额外采样多个候选改进动作：

\[
a_1,a_2,\dots,a_m
\]

分别真实解码得到：

\[
\Delta C_1,\Delta C_2,\dots,\Delta C_m
\]

从而建立：

\[
(G,a,\Delta C)
\]

或 pairwise ranking：

\[
a_i \succ a_j
\quad \text{if}\quad
\Delta C_i > \Delta C_j
\]

训练神经网络学习：

\[
Q_{imp}(G,a)
\]

即动作在当前同步结构下的真实 improvement potential。

这一机制比传统 behavior cloning 更强，因为模型学习的不是“teacher 做了什么”，而是：

> **“在同一个调度状态下，什么动作客观上更有效？”**

---

## 3.5 创新点五：RL Self-improvement Beyond the Teacher

在 supervised NI 已经稳定以后，再考虑 PPO/actor-critic fine-tuning。

此时 PPO 的角色发生根本变化：

- 不再是完整调度构造器；
- 不再承担 O→M→W→F 的长时域组合决策；
- 仅作为短时域 improvement policy 的训练优化器。

MDP 可以定义为：

\[
s_t = CSG(S_t)
\]

\[
a_t = (type,size,target,repair)
\]

即时 reward 可定义为：

\[
r_t =
\frac{C_{\max}(S_t)-C_{\max}(S_{t+1})}
{C_{\max}(S_t)}
\]

也可加入 decoder/evaluation cost：

\[
r_t = improvement - \lambda \cdot search\_cost
\]

最终保留 RL 的前提是：

\[
CSG\text{-}NI_{RL}
>
CSG\text{-}NI_{supervised}
\]

如果 RL 没有显著增益，则最终算法不必为了复杂度而强行保留 PPO。

---

# 4. 每个核心步骤的输入、输出、目的与预期

| 步骤 | 输入 | 核心处理 | 输出 | 目的 | 预期 |
|---|---|---|---|---|---|
| H1 Initialization | RCIAS instance \(I\) | 冻结 H1 | \(S_0\) | 提供强、快、稳定初始解 | 明显优于当前 PPO initialization |
| Schedule Decomposition | \(I,S_t\) | 提取实际资源链、等待、同步与重构 | structured state | 恢复 makespan 的真实形成机制 | 得到完整解结构 |
| CSG Construction | structured state | 构造 heterogeneous solution graph | \(G_t\) | 将当前调度转成可学习表示 | 暴露关键同步瓶颈 |
| CSG Encoding | \(G_t\) | HGT/heterogeneous GNN | node/graph embeddings | 建立上下文化调度表示 | 区分关键与非关键区域 |
| Bottleneck Decision | embeddings | 分类/排序 | bottleneck type | 判断改哪里 | 降低无效搜索 |
| Destroy Size | state embeddings | 分类/回归 | \(q_t\) | 控制搜索强度 | 状态自适应扰动 |
| Destroy Target | operation embeddings | node scoring/ranking | \(D_t\) | 直接学习值得重新优化的工序 | 替代固定 CP/random target rule |
| Structure-aware Repair | \(S_t,D_t\) | 现有强 repair | \(S'_t\) | 保证可行并控制学习难度 | 高成功率修复 |
| Decoder Evaluation | \(S'_t\) | frozen decoder/checker | makespan/reward | 获得真实改进信号 | 即时可靠反馈 |
| Search Acceptance | \(S_t,S'_t\) | SA/ALNS acceptance | \(S_{t+1}\) | 保留 diversification | 避免局部最优 |
| CSG Update | \(S_{t+1}\) | 重建/增量更新 | \(G_{t+1}\) | 形成闭环 | policy 随当前解动态变化 |
| Termination | search history | time/evaluation budget | \(S^*\) | 与 ALNS 公平对比 | 更好质量或更低搜索成本 |

---

# 5. 建议的训练体系

## 5.1 TRAIN / DEV / TEST 严格分离

现有：

- 18 个 CB1-DEV：仅用于模型选择、early stopping、hyperparameter selection；
- 45 个 CB1-Core：最终主测试集；
- 45 个 CB1-Sensitivity：最终 RI/TI 机制分析；
- 130 个 Legacy：external/OOD/stress test。

神经网络训练必须额外建立：

> **RCIAS-CB1-TRAIN**

建议通过 controlled generator 程序化生成：

\[
800\sim1500
\]

个 base training instances，覆盖：

\[
S/M/L
\times
CF1/CF2/CF3
\times
RI
\times
TI
\]

TRAIN 的随机种子必须与 DEV/Core/Sensitivity/Legacy 完全隔离。

---

## 5.2 数据规模

每个训练实例通过：

\[
H1 \rightarrow ALNS
\]

可产生大量 search states。

最终监督数据目标可以达到：

\[
10^5\sim10^6
\]

级：

- state-action samples；
- node-target samples；
- counterfactual action comparisons。

不需要把所有 graph tensor 永久落盘，可以保存可重建的 schedule/search state，通过 deterministic CSG dataset loader 在线构图。

---

# 6. 为什么下一步必须先分析 ALNS

目前只知道：

> ALNS-H1 在 CB1-Core 上平均比 H1 进一步改善约 5%。

但尚不知道：

- 哪些 destroy operator 真正贡献最大；
- 哪些 repair operator 最有效；
- 什么 destroy size 更有效；
- 成功操作主要发生在 critical path 还是其他同步瓶颈；
- CF1/CF2/CF3 下 operator behavior 是否不同；
- S/M/L 下模式是否不同；
- 改善主要来自 OS/island/reconfiguration/W/F 哪一层；
- accepted worsening move 是否具有后续价值；
- ALNS 是否能产生足够稳定的 node-level supervision。

因此**不能直接开始写 CSG 网络**。

正确下一步是：

\[
\boxed{
\text{Phase 6A: ALNS Search Behavior Diagnosis and NI Data Readiness Audit}
}
\]

该阶段只做 instrumentation、日志收集和行为分析，不改变 ALNS 搜索逻辑。

---

# 7. 后续 Codex 执行阶段目录

建议建立：

```text
docs/codex_execution/
│
├── 00_master_roadmap.md
├── 01_phase6a_alns_diagnosis.md
├── 02_phase6b_training_distribution.md
├── 03_phase6c_search_dataset_generation.md
├── 04_phase6d_csg_definition.md
├── 05_phase6e_csg_validation.md
├── 06_phase6f_supervised_ni_v1.md
├── 07_phase6g_counterfactual_learning.md
├── 08_phase6h_neural_destroy_policy.md
├── 09_phase6i_repair_policy_extension.md
├── 10_phase6j_rl_finetuning.md
├── 11_phase6k_full_solver_integration.md
├── 12_phase6l_ablation_and_validation.md
└── 13_phase6m_final_evaluation.md
```

---

# 8. 分阶段目标与 Gate

## Phase 6A — ALNS Search Diagnosis

目标：分析当前最强 ALNS 为什么有效，以及哪些信息值得学习。

输出：

- operator success/improvement statistics；
- destroy size effect；
- target criticality/overlap analysis；
- S/M/L、CF1/CF2/CF3 分组行为；
- repair contribution；
- improvement timing；
- NI data readiness conclusion。

Gate：明确第一版 NI 最应该学习什么。

当前预期：

\[
\boxed{destroy\ target\ learning}
\]

最有潜力，operator selection 为辅助，neural repair 暂缓。

---

## Phase 6B — Training Distribution

目标：设计 RCIAS-CB1-TRAIN。

输出：训练分布、seed 隔离、实例生成协议。

Gate：训练分布覆盖主要 RCIAS 结构且不污染 test。

---

## Phase 6C — Search Dataset Generation

目标：使用 H1 + ALNS 生成真实改进轨迹。

输出：search dataset。

Gate：数据具有足够的 positive/neutral/negative action diversity。

---

## Phase 6D — Formal CSG Definition

目标：正式定义 node/edge/features 和动态图语义。

Gate：CSG 能完整表示 makespan-relevant realized synchronization structure。

---

## Phase 6E — CSG Validation

目标：证明图结构正确、无信息泄漏、可重建关键链。

Gate：CSG generation deterministic and structurally correct。

---

## Phase 6F — Supervised NI v1

目标：先学习 bottleneck/operator + destroy-target scoring。

repair 保持 heuristic。

Gate：优于 random/CP heuristic target selection。

---

## Phase 6G — Counterfactual Learning

目标：对同一 state 的多个候选动作进行真实 evaluation，并学习 improvement ranking。

Gate：counterfactual model 在 DEV 上能够正确区分高价值和低价值动作。

---

## Phase 6H — Neural Destroy Policy

目标：正式替代 ALNS 的 handcrafted destroy targeting。

Gate：同 repair、同搜索预算下优于 standard ALNS destroy mechanism。

---

## Phase 6I — Repair Policy Extension

仅当 neural destroy 已证明有效后进行。

目标：学习 repair candidate ranking，而不是直接生成非法解。

Gate：显著增加 neural destroy 的收益。

---

## Phase 6J — RL Fine-Tuning

仅当 supervised NI 稳定后进行。

目标：让 policy 超越 search teacher。

Gate：

\[
CSG\text{-}NI_{RL}
>
CSG\text{-}NI_{sup}
\]

如果不成立，则不保留 RL。

---

## Phase 6K — Full Solver Integration

完整流程：

\[
H1
\rightarrow
CSG
\rightarrow
NI
\rightarrow
Destroy/Repair
\rightarrow
Decoder
\rightarrow
CSG
\rightarrow\cdots
\]

与 ALNS 使用完全相同 stopping budget。

---

## Phase 6L — Ablation

至少比较：

- H1；
- ALNS；
- CSG + heuristic selection；
- NI without CSG；
- supervised CSG-NI；
- + counterfactual；
- + RL（若保留）。

目标：证明每项创新真实贡献。

---

## Phase 6M — Final Evaluation

冻结架构后最终评估：

- 45 CB1-Core；
- 45 CB1-Sensitivity；
- 130 Legacy。

最终核心比较：

\[
\boxed{CSG\text{-}NI\quad vs.\quad ALNS}
\]

而不是 PPO vs H1。

---

# 9. 顶刊创新性判断

如果最终算法仅是：

> H1 + GNN 选择一个 ALNS 算子

则创新性不足以支撑 TCYB 目标。

建议最终至少形成三层真正的方法贡献：

### Contribution 1

**Critical Synchronization Graph**：针对重构装配岛 + 双物流 + DAG 同步耦合的完整解动态图表示。

### Contribution 2

**Critical-structure-aware Hierarchical Neural Improvement**：学习 bottleneck、destroy size 与 destroy targets，而非简单 heuristic selection。

### Contribution 3

**Search-supervised + Counterfactual + Self-improvement Learning**：

\[
Search\ Demonstration
\rightarrow
Counterfactual\ Ranking
\rightarrow
Policy\ Improvement
\]

让模型不仅模仿 ALNS，而是最终尝试超越 ALNS。

---

# 10. 最终算法成功标准

最终算法不能只超过 H1，因为 ALNS 已经比 H1 强约 5%。

真正目标必须设为：

\[
\boxed{CSG\text{-}NI > ALNS}
\]

可接受的成功形式包括：

1. **同等计算预算下平均 makespan 明显更低**；
2. **相同 makespan 质量下需要更少 decoder evaluations / 更短时间**；
3. **在 Core、规模与 capability flexibility 分组中具有更稳定的泛化**；
4. **在 Legacy/OOD 场景中不发生类似 constructive PPO 的 catastrophic collapse**。

如果相对 ALNS 能进一步取得约 1.5%–3% 的稳定平均改善并通过统计检验，将具有较强的论文说服力。

---

# 11. 当前正式研究路线

当前建议正式冻结：

```text
Initializer:
H1

Strong conventional baseline:
ALNS-H1

Historical learning baseline:
BC / Phase 5B constructive PPO

Final research direction:
H1 + CSG + Neural Improvement

Learning strategy:
ALNS/search demonstrations
→ supervised NI
→ counterfactual ranking
→ optional RL fine-tuning
```

下一步不应直接写神经网络，而应先完成：

> **Phase 6A — ALNS Search Behavior Diagnosis and NI Data Readiness Audit**

以真实 ALNS 搜索数据决定第一版 Neural Improvement 应学习什么。
