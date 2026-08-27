# Codex 算法撰写与实现执行说明书

## 面向 RCIASP-DL 的 Capability–Logistics Coupled Graph Reinforcement and Improvement Framework

> **工作算法名**：CLGRI（Capability–Logistics Graph Reinforcement and Improvement）
> **目标定位**：不是“普通 GNN + PPO 的调度应用”，而是针对 RCIAS-2.0 中 **DAG 拓扑线性化—能力重构—W/F 双层物流同步** 三类内生关系设计结构感知学习与神经改进算法。
> **目标期刊层级**：算法设计按 IEEE Transactions on Cybernetics / IEEE TNNLS / IEEE TEVC / CIE / JMS 等高水平期刊所需的“方法贡献 + 问题结构利用 + 泛化验证”标准组织。
> **重要说明**：本说明书提供的是高水平研究设计与工程执行规范，不代表仅凭算法名称或组件堆叠即可保证特定期刊录用。

---

# 1. 文献基线与为什么不能只做 HGNN + PPO

截至 2025–2026 年，以下路线已经出现：

1. RMC 动态调度中使用人工状态特征 + DQN：Zheng et al., IJPR 2025, DOI `10.1080/00207543.2025.2497961`；
2. 动态 IPPS-RMC 使用 heterogeneous multi-agent PPO：Zheng et al., JMS 2026, DOI `10.1016/j.jmsy.2026.01.004`；
3. FJSP + AGV 中使用 heterogeneous disjunctive graph + PPO：EAAI 2025, DOI `10.1016/j.engappai.2025.111356`；
4. FJSPT 中使用 heterogeneous graph + Graph Transformer + evolutionary imitation + RL：CIE 2025, DOI `10.1016/j.cie.2025.111465`；
5. 2026 年已经出现 FJSP + multi-AGV 的 HGNN + PPO 端到端方法。

因此，以下方案**禁止作为最终论文主算法**：

```text
heterogeneous graph -> GAT/HGT -> PPO -> operation-machine-AGV tuple
```

因为其算法结构已经高度同质化。

CLGRI 必须至少体现三个问题特定创新：

1. **Capability–Logistics Coupled Graph (CLCG)**：显式编码配置状态、DAG、主工件实际位置、W/F 同步关系，而不是普通 O-M-AGV 图；
2. **Preference-conditioned constrained autoregressive policy**：按问题因果关系分解 `operation -> island -> W -> F`，而不是平铺组合动作；
3. **Critical-Synchronization Graph Neural Improvement (CSG-NI)**：利用实现调度中的关键同步图进行神经化 destroy–repair，而不是只生成一次构造解。

推荐再加两个训练层面的增强，但不要作为主要创新点吹大：

4. small-instance exact/strong-heuristic demonstrations 预训练；
5. scale + reconfiguration/logistics-intensity curriculum。

---

# 2. 算法总框架

最终算法由四个主模块构成：

```text
RCIAS-2.0 instance
      |
      v
Dynamic Capability-Logistics Coupled Graph (CLCG)
      |
      v
Relation-aware Temporal Heterogeneous Graph Transformer (RT-HGT)
      |
      v
Preference-conditioned Masked Autoregressive Construction Policy (PC-MAP)
      |
      v
Feasible Insertion Decoder
      |
      v
Initial schedule
      |
      v
Critical Synchronization Graph (CSG)
      |
      v
Neural Destroy Selector + policy-based Repair
      |
      v
Improved nondominated schedule set
```

推荐论文算法名暂用：

> **CLGRI: Capability–Logistics Coupled Graph Reinforcement and Improvement Algorithm**

不要在代码中把名字写死到每个类名中，以方便后续改名。

---

# 3. 总体软件目录

Codex 按以下目录实现，不允许把所有逻辑堆在一个脚本中：

```text
rcias_clgri/
├── data/
│   ├── instance.py
│   ├── loader.py
│   └── schema.py
├── env/
│   ├── state.py
│   ├── schedule.py
│   ├── timelines.py
│   ├── insertion_decoder.py
│   └── rcias_env.py
├── graph/
│   ├── builder.py
│   ├── features.py
│   ├── relations.py
│   └── critical_sync_graph.py
├── model/
│   ├── embeddings.py
│   ├── rt_hgt.py
│   ├── policy_heads.py
│   ├── value_head.py
│   ├── destroy_head.py
│   └── clgri_model.py
├── rl/
│   ├── rollout.py
│   ├── ppo.py
│   ├── preference.py
│   ├── curriculum.py
│   ├── imitation.py
│   └── self_improvement.py
├── search/
│   ├── destroy.py
│   ├── repair.py
│   └── neural_lns.py
├── eval/
│   ├── metrics.py
│   ├── pareto.py
│   ├── baselines.py
│   └── statistical_tests.py
├── tests/
│   ├── test_loader.py
│   ├── test_dag.py
│   ├── test_decoder.py
│   ├── test_graph.py
│   ├── test_masks.py
│   ├── test_reward.py
│   └── test_reproducibility.py
├── train.py
├── evaluate.py
├── infer.py
└── config.yaml
```

Python >= 3.11；PyTorch；PyTorch Geometric 可用，但 graph builder 不应依赖 PyG 才能完成逻辑校验。

---

# 4. 数据结构

## 4.1 必须直接读取 RCIAS-2.0

不得在算法代码里重新定义另一套符号。

实例字段：

```text
sets.products              -> J
sets.operations            -> O
sets.islands               -> M
sets.configurations        -> C
sets.agvs_w                -> W
sets.agvs_f                -> F
products[J].precedence     -> DAG A_j
operations[o].product
operations[o].required_configuration
operations[o].eligible_islands
operations[o].processing_time
islands[m].supported_configurations
islands[m].initial_configuration
reconfiguration.time
reconfiguration.cost
logistics.distance
logistics.W.loaded_time
logistics.W.empty_time
logistics.F.outbound_time
logistics.F.return_time
```

## 4.2 `Instance` dataclass

至少包含：

```python
@dataclass(frozen=True)
class OperationData:
    op_id: str
    product_id: str
    required_config: str
    eligible_islands: tuple[str, ...]
    processing_time: dict[str, int]

@dataclass(frozen=True)
class ProductData:
    product_id: str
    operations: tuple[str, ...]
    precedence: tuple[tuple[str, str], ...]

@dataclass(frozen=True)
class IslandData:
    island_id: str
    supported_configs: tuple[str, ...]
    initial_config: str

@dataclass(frozen=True)
class Instance:
    ...
```

加载时预计算：

```text
pred[o], succ[o]
transitive_pred_count[o]
transitive_succ_count[o]
ready_initial[j]
config_index[c]
product_of[o]
eligible[o]
processing_time[o,m]
```

禁止在每个环境 step 中重复做 DAG transitive closure。

---

# 5. 构造式调度环境

## 5.1 为什么采用“构造 + 可行插入”而不是简单 event dispatch

这是静态离线调度。算法应允许学习全局组合结构，而不是只在机器释放事件上选择 dispatching rule。

一次 episode 逐步把每道工序加入当前部分解。

每一步动作：

$$
a_t=(o_t,m_t,w_t,f_t),
$$

但采用自回归条件分解：

$$
P(a_t|s_t)
=
P(o_t|s_t)
P(m_t|o_t,s_t)
P(w_t|o_t,m_t,s_t)
P(f_t|o_t,m_t,w_t,s_t).
$$

其中：

- `w_t = NONE` 当主工件不需要跨岛运输；
- F 始终需要选择一辆 F-AGV。

## 5.2 Ready operation mask

工序 `o` 可被选择，当且仅当：

1. `o` 尚未被加入产品实际拓扑顺序；
2. `pred[o]` 中所有 DAG 前驱已经被选择。

即标准拓扑 ready set。

**注意**：

“已选择”不等于“已完成”。环境在构造离线调度，因此 DAG ready mask 基于拓扑线性化是否已经确定，而实际时间可行性由 decoder 保证。

## 5.3 产品实际前序

对于产品 `j`：

```python
last_product_op[j]
last_product_island[j]
```

表示目前拓扑线性化中最后选择的工序。

当选择新工序 `o` 时：

```text
actual_predecessor(o) = last_product_op[product_of[o]]
```

若不存在，则 `o` 是产品第一实际工序。

该关系用于生成 W 物流任务。

---

# 6. Feasible Insertion Decoder

这是整个工程中最先写、最先测试的模块。网络训练前 decoder 必须完全正确。

## 6.1 Schedule 数据结构

维护：

```python
machine_timeline[m]: list[OperationInterval]
w_timeline[w]: list[WTask]
f_timeline[f]: list[FTask]
product_sequence[j]: list[op_id]
operation_schedule[o]: OperationSchedule
```

区间均按 start 排序。

### `OperationSchedule`

```python
@dataclass
class OperationSchedule:
    op_id: str
    product_id: str
    island_id: str
    config_id: str
    start: float
    end: float
    w_task_id: str | None
    f_task_id: str
```

### `WTask`

```python
@dataclass
class WTask:
    task_id: str
    product_id: str
    predecessor_op: str | None
    destination_op: str
    pickup: str       # WH or Mx
    delivery: str     # Mx
    release: float
    agv_id: str
    loaded_start: float
    arrival: float
```

### `FTask`

```python
@dataclass
class FTask:
    task_id: str
    op_id: str
    island_id: str
    agv_id: str
    start: float
    arrival: float
    return_end: float
```

## 6.2 F 任务插入

F 任务固定从 WH 出发并返回 WH。

若选择 `(o,m,f)`：

```text
out = tau_FO[f,m]
ret = tau_FR[f,m]
duration = out + ret
```

在 `f_timeline[f]` 中寻找最早可插入 gap。

候选 gap `[prev.return_end, next.start]` 满足：

```text
start >= prev.return_end
start + duration <= next.start
```

选择最早 `start`。

返回：

```text
F_arrival = start + out
F_return_end = start + duration
```

F 允许提前配送，所以不设置 release time。

## 6.3 W 任务插入

若产品第一次工序：

```text
pickup = WH
release = 0
```

否则：

```text
pickup = island(previous actual product op)
release = completion(previous actual product op)
```

若 `pickup == target_island`：

```text
w_task = None
W_ready = release
```

否则选择 W-AGV `w`。

在 `w_timeline[w]` 中寻找插入位置。

若插入到两个现有任务 `q_prev -> q_next` 中间：

```text
empty_to_new = tau_WE[w, q_prev.delivery, pickup]
start_new >= max(release, q_prev.arrival + empty_to_new)
arrival_new = start_new + tau_WL[w,pickup,target]
empty_new_to_next = tau_WE[w,target,q_next.pickup]
arrival_new + empty_new_to_next <= q_next.loaded_start
```

若是车辆第一任务：

```text
q_prev.delivery = WH
q_prev.arrival = 0
```

若是最后任务，无后继约束。

**必须测试插入行为是否正确处理空驶。**

## 6.4 装配岛工序插入

当 `(o,m,w,f)` 已确定，先计算：

```text
product_ready = completion(previous actual product op) or 0
W_ready       = W arrival or product_ready if same island
F_ready       = material arrival
base_ready    = max(product_ready, W_ready, F_ready)
processing    = p[o,m]
config        = c_o
```

在 `machine_timeline[m]` 的每个 gap 尝试插入。

### 插入到 `prev -> next` 之间

前置重构：

```text
prev_config = c_m^0 if prev is None else prev.config
setup_before = rho[m,prev_config,c_o]
```

后置兼容：

```text
setup_after = 0 if next is None else rho[m,c_o,next.config]
```

最早开始：

```text
candidate_start = max(base_ready,
                      0 if prev is None else prev.end) + setup_before
```

若 `prev is None`：

```text
candidate_start = max(base_ready, rho[m,c_m0,c_o])
```

若存在 `next`，必须满足：

```text
candidate_start + processing + setup_after <= next.start
```

选择最小 feasible `candidate_start` 的 gap。

### 关键要求

插入新工序会把原先 `prev -> next` 的直接配置转换替换为：

```text
prev -> new -> next
```

所以 decoder 计算成本时必须：

```text
incremental_reconfig_cost
= kappa(prev,new)
+ kappa(new,next)
- kappa(prev,next)
```

边界 first/last 类似处理。

这一步是很多调度代码会写错的地方，必须单元测试。

---

# 7. 动态 Capability–Logistics Coupled Graph (CLCG)

## 7.1 节点类型

使用五类节点：

$$
V=V^O\cup V^J\cup V^M\cup V^W\cup V^F.
$$

即：

1. operation；
2. product；
3. assembly island；
4. W-AGV；
5. F-AGV。

**配置 C 不单独作为普通图节点。**

配置采用 learnable embedding，并注入 operation / island / O-M edge feature。这样避免额外 C 节点制造大量弱信息跳转，同时突出“配置是资源状态”。

## 7.2 Edge relations

至少定义：

```text
O --precedence--> O
O --precedence_rev--> O
J --contains--> O
O --belongs_to--> J
O --eligible_on--> M
M --can_process--> O
M --spatial--> M
W --reachable_to--> M
M --reachable_by--> W
F --deliver_to--> M
M --served_by--> F
```

已调度后动态增加：

```text
O --actual_product_prev--> O
O --machine_prev--> O
```

不要使用一个统一 adjacency 后简单 GAT；必须保留 relation type。

---

# 8. 节点与边特征

所有连续特征必须用实例尺度归一化；不要直接输入原始 1000 级时间。

## 8.1 Operation node

静态特征：

```text
min_processing_time
mean_processing_time
max_processing_time
num_eligible_islands
DAG_in_degree
DAG_out_degree
transitive_pred_ratio
transitive_succ_ratio
required_configuration_embedding
```

动态特征：

```text
is_scheduled
is_topological_ready
remaining_unscheduled_successors_ratio
product_progress_ratio
actual_prev_completion / horizon
actual_prev_island_embedding or flag
scheduled_start / horizon
scheduled_end / horizon
```

## 8.2 Product node

```text
num_operations
scheduled_ratio
remaining_workload_estimate
last_actual_completion
last_actual_island_embedding
current_workpiece_location
```

## 8.3 Island node

```text
initial_configuration_embedding
last/tail configuration embedding
current_total_processing_load
current_total_reconfiguration_time
last_completion_time
utilization_estimate
num_scheduled_operations
num_supported_configurations
num_free_gaps
```

如果使用 insertion decoder，`current config` 不是一个唯一时间点状态，因此 island node 的 `tail config` 只描述末尾状态；候选操作真正的前/后配置差异由 O-M edge 动态特征表达。

## 8.4 W node

```text
num_tasks
total_loaded_time
total_empty_time
last_delivery_node embedding
tail_completion_time
utilization
largest_free_gap
```

## 8.5 F node

```text
num_tasks
total_busy_time
tail_return_time
utilization
largest_free_gap
```

## 8.6 O-M eligible edge

这是最重要的边。

对每个 ready or unscheduled `o` 与候选岛 `m`，动态计算：

```text
processing_time[o,m]
earliest_machine_insertion_start
predicted_operation_completion
setup_before_at_best_gap
setup_after_at_best_gap
incremental_reconfiguration_cost
workpiece_loaded_distance_from_actual_prev
best_W_arrival_lower_bound
best_F_arrival_lower_bound
predicted_sync_wait
```

其中 `best_W/F` 只是用于表示学习的 lower-bound feature，不替代后续 W/F policy 决策。

## 8.7 M-M spatial edge

```text
distance[m,n]
normalized_distance
```

可以限制到 k-nearest islands，避免大规模实例形成完全图。

## 8.8 W-M / F-M edges

W-M：

```text
empty_travel_from_tail_location
effective_loaded_time_to_m
```

F-M：

```text
outbound_time
round_trip_time
```

---

# 9. RT-HGT Encoder

实现一个 relation-aware temporal heterogeneous transformer。

不要直接调用默认 HGTConv 后结束；至少加入两个问题特定 bias。

## 9.1 Relation-specific attention

对 relation `r: s -> t`：

$$
q_i=W^Q_t h_i,
\qquad
k_j=W^K_{r,s} h_j,
\qquad
v_j=W^V_{r,s} h_j.
$$

attention logit：

$$
e_{ij}^{(r)}
=
\frac{q_i^T k_j}{\sqrt d}
+b_r
+b^{time}_{ij}
+b^{cap}_{ij}.
$$

其中：

- `b_time`：由 earliest-time / transport-time / current delay feature 经 MLP 产生；
- `b_cap`：仅 O-M 关系使用，由 `required config` 与候选 gap 的前后配置 transition feature 产生。

## 9.2 Residual + normalization

每层：

```text
relation attention
-> gated relation fusion
-> residual
-> LayerNorm
-> FFN
-> residual
-> LayerNorm
```

## 9.3 Recommended default

```yaml
hidden_dim: 128
num_layers: 4
num_heads: 8
dropout: 0.1
relation_hidden: 64
```

首先保证 correctness，再做超参数实验。

---

# 10. Preference-conditioned Masked Autoregressive Policy

## 10.1 多目标 preference

若保留论文双目标，训练时采样：

$$
\lambda=(\lambda_1,\lambda_2),
\qquad \lambda_i\ge0,
\quad \lambda_1+\lambda_2=1.
$$

建议：

```python
lambda1 ~ Uniform(0.05, 0.95)
lambda2 = 1-lambda1
```

把 preference embedding 注入所有 policy head 与 critic。

不要单纯使用线性加权和作为最终优化标量。

使用 normalized augmented Tchebycheff：

$$
g_\lambda(f)
=
\max_k\lambda_k\hat f_k
+\epsilon\sum_k\lambda_k\hat f_k,
$$

其中 `epsilon=0.05`。

`hat f` 使用训练集统计 ideal/nadir 或每实例启发式归一化尺度。

如果后续论文改为单目标，设置 `lambda=(1,0)` 即可复用代码。

## 10.2 Operation head

候选：topological ready unscheduled operations。

score：

```text
MLP([h_o, h_product(o), global_pool, pref_embed])
```

mask 非 ready operation 为 `-inf`。

## 10.3 Island head

给定选中 `o` 后，只在 `eligible_islands[o]` 中选择。

score：

```text
MLP([h_o, h_m, h_edge(o,m), global_pool, pref_embed])
```

必须包含动态 O-M edge feature。

## 10.4 W head

确定 `o,m` 后：

```text
if first product operation or last_product_island != m:
    choose W AGV
else:
    action = NONE
```

对每辆 W 车实时调用 lightweight `probe_w_insertion()`，得到：

```text
earliest_arrival
incremental_empty_distance
incremental_loaded_distance
new_tail_time
```

W head 输入：

```text
[h_o, h_m, h_w, probe_features, pref_embed]
```

## 10.5 F head

对每辆 F 车调用 `probe_f_insertion(o,m,f)`。

输入：

```text
[h_o, h_m, h_f, probe_features, pref_embed]
```

## 10.6 为什么自回归而不是组合动作

如果直接枚举

```text
(o,m,w,f)
```

动作数近似：

$$
|O^{ready}|\times|M_o|\times|W|\times|F|,
$$

在规模迁移时迅速膨胀。

自回归策略保持条件依赖并共享参数，且符合问题因果结构：

```text
operation determines eligible islands
operation + island determines whether W is needed
operation + island determines W/F travel context
```

---

# 11. Reward 设计

避免只在 episode 末给终局奖励。

## 11.1 Potential-based dense reward

每次插入一个工序后计算当前部分调度：

```text
partial_makespan
accumulated_reconfiguration_cost
accumulated_W_cost
accumulated_F_cost
```

构造 normalized scalar potential：

$$
\Phi_t=g_\lambda(\hat C_{\max,t},\hat C_{cost,t}).
$$

reward：

$$
r_t=\Phi_{t-1}-\Phi_t.
$$

由于每一步通常使 objective 增大，reward 可能为负；可乘常数 `reward_scale`。

最后加入 terminal bonus：

```text
- final scalarized objective
```

但避免 double count；推荐直接让 telescoping potential 完成终局等价。

## 11.2 Synchronization auxiliary signal

不要直接把辅助信号加成主 reward 权重堆叠。

额外训练一个 auxiliary head 预测：

```text
machine-caused waiting
W-caused waiting
F-caused waiting
reconfiguration-caused waiting
```

真实标签来自 decoder。

auxiliary loss：

$$
L_{sync}=\mathrm{Huber}(\hat d,d).
$$

这会迫使 encoder 学到同步瓶颈，而不改变原优化目标。

---

# 12. PPO 训练

使用 clipped PPO，但论文创新不能写成“we propose PPO”。

默认：

```yaml
gamma: 1.0              # finite deterministic construction episode
gae_lambda: 0.95
clip_ratio: 0.2
entropy_coef: 0.01
value_coef: 0.5
sync_aux_coef: 0.1
learning_rate: 3.0e-4
max_grad_norm: 0.5
ppo_epochs: 4
minibatch_size: 512
```

因为 episode 长度 = number of operations，`gamma=1.0` 更自然；如数值不稳定再测试 0.99。

每个 action 的 log-prob：

```text
logpi = logpi_o + logpi_m + logpi_w(if used) + logpi_f
```

PPO ratio 使用联合 log-prob。

---

# 13. 强示范预训练：只作为训练增强

## 13.1 Demonstration 来源

优先顺序：

1. 小实例 Gurobi / CP 最优或近最优；
2. 大实例强启发式 / memetic algorithm；
3. 不使用简单 dispatching rules 作为“expert”。

## 13.2 行为克隆

把专家完整 schedule 转换回自回归 action sequence：

```text
operation topological order
island assignment
W assignment
F assignment
```

训练：

$$
L_{BC}
=-\sum_t
(\log\pi_o+\log\pi_m+\log\pi_w+\log\pi_f).
$$

推荐先预训练 10–30 epochs，再 PPO。

不要复制 CIE 2025 的“GA-guided imitation + REINFORCE”叙事；本文的核心仍然是 CLCG representation + constrained autoregressive policy + neural improvement。

---

# 14. Curriculum

训练实例不能只用固定 10x6。

至少同时随机化：

```text
num_products
num_operations per product
num_islands
num_W
num_F
DAG density
capability diversity
reconfiguration intensity
transport intensity
```

定义两个难度参数：

```text
reconfiguration_intensity = mean(rho_nonzero) / mean(processing_time)
transport_intensity = mean(W_loaded_time) / mean(processing_time)
```

curriculum：

```text
Phase 1: small + low coupling
Phase 2: medium + mixed coupling
Phase 3: full scale + wide coupling distribution
```

不要按固定 epoch 硬切；使用 rolling validation score 达阈值后提升难度。

---

# 15. Critical Synchronization Graph (CSG)

初始调度完成后，建立一个**实现调度图**，这与输入 CLCG 不同。

## 15.1 节点

```text
operation execution nodes
W transport task nodes
F delivery task nodes
reconfiguration transition nodes (optional explicit nodes)
```

## 15.2 弧

```text
actual_product_sequence
machine_sequence
W_vehicle_sequence
F_vehicle_sequence
W_delivery_to_operation
F_delivery_to_operation
reconfiguration_to_operation
```

所有弧携带时间长度。

## 15.3 Critical path

通过 DAG longest-path 计算 makespan critical path。

如果因为 schedule graph 存在显式资源顺序，它应为 DAG；若检测出 cycle，说明 decoder 有 bug，立即 raise，不得静默修复。

## 15.4 Synchronization criticality

对工序 `o` 记录：

```text
machine_ready
product_ready
W_ready
F_ready
actual_start
```

定义每个来源 slack：

```text
slack_source = actual_start - ready_source
```

主瓶颈来源是 slack 最小的 readiness source。

定义 coupling criticality：

```text
critical_path_flag
number_of_near_tight_sync_sources
waiting_time_caused_to_successors
reconfiguration_share
transport_share
```

这些特征输入 destroy selector。

---

# 16. Critical-Synchronization Graph Neural Improvement (CSG-NI)

这是 CLGRI 区别于单次构造 DRL 的核心模块。

## 16.1 Destroy size

每轮破坏：

```text
k = clamp(round(ratio * N_ops), k_min, k_max)
ratio in {0.05, 0.10, 0.15}
```

训练时随机采样；推理时多尺度尝试。

## 16.2 Neural destroy selector

每个 operation 得到 score：

```text
score_o = DestroyMLP([
    encoder_embedding_o,
    critical_sync_features_o,
    preference_embedding,
    global_schedule_embedding
])
```

采用无放回采样选择 k 个 operation。

训练早期先采用 heuristic mixture：

```text
50% critical-path neighborhood
25% high-reconfiguration neighborhood
25% high-logistics-wait neighborhood
```

后期逐步提高 neural selector 比例。

## 16.3 Coupled closure destroy

不能只删除选中的 operation 本身。

若删除操作 `o`，必须同时解除：

```text
its island assignment
its W task
its F task
its product actual predecessor/successor links
its machine predecessor/successor links
```

但不删除原 DAG。

若删除后留下产品拓扑 sequence 断点，repair 时重新线性化被 destroy 子集与边界操作的关系。

## 16.4 Repair

冻结未 destroy 部分的时间与资源顺序，允许 destroy operation 在可行 gap 中重新插入。

repair order 仍由 PC-MAP policy 决定，但 ready mask 必须考虑：

```text
fixed DAG predecessors
already-repaired destroyed predecessors
frozen boundary predecessors
```

所有动作通过同一 `probe_*` 与 insertion decoder。

## 16.5 Acceptance

多目标场景：

- 若新解 Pareto-dominates 当前解：接受；
- 若 non-dominated：加入 archive，并根据当前 preference `lambda` 比较 augmented Tchebycheff 值；
- 若 worse：训练阶段可按小概率接受以提供探索，推理阶段默认拒绝。

维护 bounded Pareto archive，例如最大 64 个解，用 crowding distance / hypervolume contribution 截断。

---

# 17. Neural destroy selector 的训练

第一版不要一上来使用复杂 differentiable top-k。

推荐两阶段：

### Stage A：监督 warm-up

收集多种 heuristic destroy 的 improvement 数据：

```text
(state, destroy_set, improvement)
```

把 improvement top-quantile 的操作作为正样本，训练 operation score ranking。

### Stage B：policy-gradient fine-tuning

destroy set 的 log-prob 为无放回采样的联合概率。

reward：

$$
r^{NI}=g_\lambda(S_{before})-g_\lambda(S_{after}).
$$

仅当完整 repair 后计算。

使用 baseline 降低方差。

---

# 18. Self-improvement / self-imitation

在 PPO 稳定后启用。

流程：

```text
policy constructs S0
CSG-NI produces S1...SK
best/Pareto-improved trajectories stored in improvement buffer
periodically distill improved decisions back into construction policy
```

只存储明显改进样本：

```text
relative scalar improvement >= 0.5%
或 Pareto-dominates parent
```

self-imitation loss 与 PPO loss 分开采样，不要在同一 mini-batch 混入未经标注的旧 policy logprob。

---

# 19. 总损失

推荐：

$$
L
=
L_{PPO}
+\lambda_V L_V
+\lambda_{sync}L_{sync}
+\lambda_{BC}L_{BC/imitation}.
$$

其中 `BC/imitation` 在对应训练阶段开启，不是永久固定大权重。

destroy selector 单独优化：

$$
L_D=L_{PG}^{destroy}+\lambda_{rank}L_{rank}.
$$

---

# 20. 训练阶段顺序

Codex 严格按以下顺序实现，不能全部同时打开。

## Phase 0：正确性

1. loader；
2. instance validator；
3. insertion decoder；
4. objective calculator；
5. random masked policy 可生成 100% feasible solution。

**只有 1000 个随机实例无 infeasible 后才能进入 Phase 1。**

## Phase 1：Graph + supervised sanity

1. graph builder；
2. RT-HGT forward；
3. action masks；
4. 用 heuristic/expert action 做 cross-entropy，确认 loss 可下降。

## Phase 2：Constructive PPO

关闭 neural improvement，只训练 PC-MAP。

达到：

```text
valid schedule rate = 100%
优于主要 dispatching rules
跨 seed 收敛稳定
```

后进入 Phase 3。

## Phase 3：CSG-NI

先 heuristic destroy + policy repair，证明 LNS 本身有效。

再训练 neural destroy selector。

## Phase 4：Preference-conditioned multiobjective

如果此前单目标调通，再引入 sampled preference。

不要在 decoder 尚未稳定时同时调 Pareto + neural LNS。

## Phase 5：generalization curriculum

扩规模、扩 coupling intensity、跨生成器训练测试。

---

# 21. 关键代码接口

## 21.1 Decoder probe 必须无副作用

```python
def probe_machine_insertion(schedule, op_id, island_id, base_ready) -> MachineProbe:
    ...

def probe_w_insertion(schedule, w_id, pickup, delivery, release) -> WProbe:
    ...

def probe_f_insertion(schedule, f_id, island_id) -> FProbe:
    ...
```

probe 不得修改 schedule。

真正写入：

```python
def commit_action(schedule, action, probes) -> TransitionResult:
    ...
```

## 21.2 Environment

```python
class RCIASConstructionEnv:
    def reset(self, instance, preference): ...
    def get_graph_state(self): ...
    def get_operation_mask(self): ...
    def get_island_mask(self, op_id): ...
    def get_w_mask_and_features(self, op_id, island_id): ...
    def get_f_mask_and_features(self, op_id, island_id): ...
    def step(self, action): ...
    def objective(self): ...
    def validate_schedule(self): ...
```

## 21.3 Model

```python
class CLGRIModel(nn.Module):
    def encode(self, hetero_graph, preference): ...
    def operation_policy(self, embeddings, mask, preference): ...
    def island_policy(self, op_id, embeddings, edge_features, mask, preference): ...
    def w_policy(self, op_id, island_id, embeddings, probe_features, mask, preference): ...
    def f_policy(self, op_id, island_id, embeddings, probe_features, mask, preference): ...
    def value(self, embeddings, preference): ...
    def destroy_scores(self, schedule_graph, preference): ...
```

---

# 22. 单元测试要求

## 22.1 DAG

必须测试：

```text
precedence acyclic
ready set correct
policy-generated product order is a topological order
same-product operations never overlap
```

## 22.2 Reconfiguration

构造岛序列：

```text
C1 -> C1 -> C3 -> C2
```

断言：

```text
C1->C1 setup = 0
other setup exactly equals matrix
machine insertion updates both predecessor and successor transition
```

## 22.3 W

至少测试：

1. WH -> M1 first operation；
2. M1 -> M1 no W task；
3. M1 -> M2 creates W task；
4. W empty reposition between two tasks；
5. insertion into W timeline middle gap；
6. two tasks cannot overlap。

## 22.4 F

测试：

```text
material arrival before return
operation may start after arrival but before F return
same F cannot overlap round-trip tasks
F task may be delivered early
```

## 22.5 Objective

手工 2-product 小实例逐项核对：

```text
makespan
reconfiguration cost
W loaded cost
W empty cost
F outbound/return cost
```

---

# 23. Baseline 设计

论文不能只与简单启发式比较。

至少包括：

### Classical / optimization

1. Gurobi MILP：small instances；
2. CP-SAT 或 CP Optimizer：small/medium；
3. NSGA-II / MOEA/D：双目标 evolutionary baseline；
4. problem-specific memetic algorithm：强元启发式。

### DRL

5. hand-crafted feature PPO；
6. homogeneous GAT + PPO；
7. standard HGT + flat tuple policy；
8. RT-HGT + autoregressive policy without neural improvement；
9. full CLGRI。

如果可复现，加入近年 FJSPT HGNN/Graph Transformer 类方法的适配版本，但必须保证相同 decoder 与 time budget 公平比较。

---

# 24. 消融实验

至少：

```text
A0 Full CLGRI
A1 - capability-aware bias
A2 - logistics relation features
A3 HGT -> homogeneous GAT
A4 autoregressive -> flat composite action
A5 - synchronization auxiliary loss
A6 - CSG-NI
A7 heuristic destroy instead of neural destroy
A8 - preference conditioning (train separate weights)
A9 - imitation warm-up
```

核心论文消融建议只展示最关键 5–6 个，其余放 supplementary。

---

# 25. 泛化实验

这是冲击 ToC/TNNLS 类期刊必须重点强化的部分。

## 25.1 Scale generalization

训练：

```text
small/medium mixed scale
```

测试：

```text
seen scale
1.5x operations
2x operations
more islands
fewer W/F vehicles
```

## 25.2 Structural generalization

分别在两套生成器间做：

```text
train FJSP-expanded -> test automotive semantic
train automotive -> test FJSP-expanded
mixed train -> both test
```

这能证明网络不是只记忆某种 DAG 模板。

## 25.3 Coupling generalization

测试：

```text
low/high reconfiguration intensity
low/high transport intensity
W-scarce
F-scarce
balanced logistics
```

并分析 attention / bottleneck predictions 是否随系统瓶颈迁移。

---

# 26. 双目标指标

若保留双目标：

```text
Hypervolume (HV)
IGD+
epsilon indicator
nondominated set size
runtime
```

同时报告代表 preference 下：

```text
makespan
cost
```

统计检验：

```text
Wilcoxon signed-rank
Holm correction
Vargha–Delaney A12 or Cliff's delta
```

对每个 stochastic algorithm 至少 20 independent runs；深度模型应区分：

```text
training seed
inference stochastic seed
instance seed
```

---

# 27. 论文算法贡献建议写法

最终算法贡献不要写成：

> We apply a heterogeneous graph neural network and PPO...

应围绕以下三点：

### Contribution A — structure representation

> A capability–logistics coupled heterogeneous graph is developed to preserve the endogenous dependencies among DAG linearization, capability transitions, workpiece routing, and component-kit synchronization. Unlike conventional FJSP graphs, the graph explicitly represents reconfiguration-aware operation–island compatibility and the two distinct logistics layers.

### Contribution B — decision architecture

> A preference-conditioned masked autoregressive policy is designed according to the causal decision structure of the problem. It decomposes the high-dimensional composite action into precedence-feasible operation selection, reconfiguration-aware island assignment, conditional workpiece-AGV allocation, and kit-AGV allocation while preserving end-to-end learning.

### Contribution C — neural improvement

> A critical-synchronization graph neural improvement mechanism is introduced to refine constructed schedules. It identifies bottleneck subgraphs induced by production–reconfiguration–logistics synchronization and performs learned destroy–repair search, enabling the policy to go beyond one-pass constructive scheduling.

这三点才是投稿高水平 cybernetics / learning journal 时应突出的方法创新。

---

# 28. 论文中不要做的过度宣称

禁止：

```text
first GNN for scheduling
first DRL for AGV scheduling
first heterogeneous graph for FJSP
```

这些已经不成立。

可以更精确地声称：

```text
first/one of the first structure-aware learning frameworks specifically coupling
reconfigurable capability states, assembly DAG linearization, and two synchronized
logistics layers
```

但投稿前仍需再次完整检索确认 “first” 是否安全；如果不能确认，使用：

```text
to the best of our knowledge, existing studies have not jointly modeled ...
```

并用文献表格支持。

---

# 29. 性能工程要求

1. graph static topology 尽量缓存；每 step 只更新 dynamic features；
2. DAG transitive data 在 loader 预处理；
3. insertion probe 使用有序 timeline + binary search；
4. 不允许在每个 action candidate 上 deepcopy 整个 schedule；
5. probe 返回增量结构，commit 时局部更新；
6. GPU 只做 neural forward/backward；decoder 保持轻量 CPU 或批量 vectorized；
7. 支持 batched environments；
8. `torch.no_grad()` inference；
9. 所有 random seed 集中管理；
10. logging 记录 decoder time / graph build time / model forward time 分解。

---

# 30. 训练日志

每个 epoch 至少记录：

```text
mean scalarized objective
mean makespan
mean cost
HV on validation subset
policy entropy per action head
value loss
sync auxiliary loss
invalid action count (must be 0 after mask)
decoder feasibility failures (must be 0)
mean reconfiguration time share
mean W waiting share
mean F waiting share
CSG-NI improvement rate
neural destroy success rate
runtime components
```

---

# 31. Checkpoint 规则

不要只按 training reward 保存。

至少保存：

```text
best seen-scale validation HV
best unseen-scale validation HV
best mean normalized objective
latest
```

模型文件必须同时保存：

```text
network state
optimizer state
normalization statistics
configuration catalogue embedding mapping
training schema version
code git hash if available
```

---

# 32. Reproducibility

最终仓库应提供：

```text
requirements.txt / environment.yml
fixed seeds
generator commands
training config files
evaluation config files
pretrained checkpoints
raw result CSV
statistical-analysis script
figure-generation script
```

结果文件每一行至少包含：

```text
instance_id
instance_seed
method
training_seed
run_seed
lambda1
makespan
cost
runtime
HV contribution
```

---

# 33. Codex 的实施优先级

严格执行：

```text
P0 loader + validator
P1 deterministic insertion decoder
P2 random feasible constructive solver
P3 objective audit against hand cases
P4 CLCG builder
P5 RT-HGT encoder
P6 autoregressive masked policy
P7 behavior cloning sanity
P8 PPO
P9 CSG extraction
P10 heuristic critical destroy + learned repair
P11 neural destroy selector
P12 preference-conditioned multiobjective training
P13 large-scale/generalization experiments
```

如果 P0–P3 任何一项失败，不允许继续写神经网络。

---

# 34. 最小验收标准

代码第一阶段完成时必须满足：

```text
[ ] 两套 RCIAS-2.0 generator 的 JSON 均可加载
[ ] 1000 个随机实例 decoder 生成可行解率 100%
[ ] 所有产品执行顺序均为输入 DAG 的拓扑序
[ ] 同产品工序无重叠
[ ] 装配岛工序无重叠且配置转换时间正确
[ ] W 运输仅在产品第一工序或相邻工序跨岛时产生
[ ] W AGV 空驶位置连续
[ ] F AGV 往返任务不重叠
[ ] operation.start >= W arrival（若存在）
[ ] operation.start >= F arrival
[ ] objective 与手工实例逐项一致
```

学习算法阶段：

```text
[ ] action mask 永不产生非法动作
[ ] behavior cloning loss 显著下降
[ ] PPO 优于随机与主要 dispatching heuristics
[ ] CSG-NI 相对纯构造策略有稳定正改进率
[ ] unseen scale 上性能退化可控
[ ] two-generator cross-domain test 有明显泛化能力
```

---

# 35. 推荐的最终研究叙事

不要把论文讲成“在可重构问题上增加两个 AGV”。

主叙事应是：

$$
\boxed{
\text{capability reconfiguration}
+
\text{assembly DAG routing}
+
\text{dual-flow synchronization}
}
$$

共同导致**调度关系本身内生变化**：

1. 拓扑线性化决定产品实际相邻工序；
2. 实际相邻工序与岛分配共同决定 W 运输链；
3. 岛分配同时决定配置转换与 F 配送目的地；
4. 配置、主工件和物料到达共同决定工序可开始时刻；
5. 一个局部决策会跨越工艺、生产资源和两类物流形成长程影响。

因此普通手工状态向量和单一 disjunctive graph 难以完整表达该耦合结构；CLGRI 的图表示、自回归决策和同步关键子图改进机制都应围绕这一核心展开。

这才是达到 IEEE Transactions on Cybernetics 等高水平算法期刊要求时，最值得强调的“cybernetics / intelligent optimization”部分。
