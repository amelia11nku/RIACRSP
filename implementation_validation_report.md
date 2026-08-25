# RCIAS 第一阶段实现与验证报告

## 1. 验收结论

本轮已闭环实现：

```text
RCIAS-2.0 instance
-> strict loader / validator
-> deterministic insertion decoder
-> complete schedule records
-> independent feasibility checker
-> tiny exhaustive exact validation
```

两套生成器、两个 demo、四个小规模验证实例已统一到 `RCIAS-2.0`。三种非学习构造式启发式共享同一个 decoder，在全部验证实例上均得到可行调度。两个 tiny 实例完成穷举分支定界，H2 均达到最优 makespan。1,000 个不同 seed 的双生成器压力测试可行率为 100%。

## 2. 新增与修改文件

### 修改

- `generate_fjsp_reconfigurable.py`：从旧模块实例模型重写为固定能力配置的 RCIAS-2.0 生成器。
- `generate_automotive_semantic.py`：删除模块类型/实例/配置覆盖，改为语义任务到唯一固定配置的映射。
- `fjsp_reconfigurable_demo.json`、`automotive_semantic_demo.json`：重新生成 RCIAS-2.0 demo。

### 新增

- `rcias_clgri/data/`：dataclass、loader、strict validator、生成公共函数。
- `rcias_clgri/env/`：schedule records、状态视图、W/F/岛插入探测、decoder、environment、objective、feasibility checker。
- `rcias_clgri/exact/tiny_exact_solver.py`：有规模保护的穷举活动调度分支定界。
- `rcias_clgri/heuristic/dispatching.py`：H1/H2/H3 三类构造式基线。
- `rcias_clgri/graph/`：五类节点的动态异构图、特征、typed relations 与 action masks。
- `instances/tiny/`：两套 tiny 与两套稍大实例。
- `tests/`：28 项单元与交叉验证测试。
- `scripts/validate_instances.py`：批量严格加载。
- `scripts/run_small_validation.py`：基线、checker、graph 与 exact 验证。
- `scripts/stress_random_validation.py`：随机实例压力测试。
- `pytest.ini`、`notes.txt`、`validation_results.json`、`stress_validation_results.json`。

## 3. 数学模型到代码的映射

| 数学对象/约束 | 实现位置 | 实现方式 |
|---|---|---|
| 产品、工序、DAG | `data/instance.py`, `data/loader.py` | `ProductData`、预计算 `pred/succ/transitive closure` |
| 唯一固定配置 `c_o` | `OperationData.required_config` | loader/validator 禁止配置选择或 coverage |
| 候选岛与 `p_om` | `OperationData`, tuple-key lookup | 加工时间键必须与候选岛集合严格相等 |
| 初始配置与 `rho/kappa` | `IslandData`, instance transition maps | 支持配置上的完整矩阵，严格零对角 |
| 产品实际拓扑线性化 | `Schedule.product_sequences` | ready mask 基于 DAG，实际 predecessor/successor 显式保存 |
| 岛序列与配置重构 | `env/timelines.py` | 前后相邻 setup 均检查；插入时更新增量重构成本 |
| W 主工件物流 | `probe_w_insertion` | 第一工序 WH 出发；跨岛生成；同岛强制 `NONE`；含空驶连续性 |
| F 物料包物流 | `probe_f_insertion` | 完整往返占车；到岛时间与返库可用时间分离 |
| 多源同步 | `OperationSchedule` | 保存 product/island/config/W/F ready、start/end、binding resource |
| 最大完工时间与成本 | `env/objective.py` | 从最终原始记录独立重算各成本分项 |
| 独立可行性 | `env/feasibility.py` | 不调用 decoder，返回结构化 category/resource/IDs/interval violations |

decoder 对工序开始时间采用：

```text
start = max(product_ready, W_ready, F_ready, previous_island_end + setup_before)
```

因此能力重构可以在物料或主工件尚未到达时提前完成，这与数学模型中的独立下界约束一致。重构占用岛，不能和前后加工重叠。

## 4. 实例

| 文件 | 产品 | 工序 | 岛 | W/F | 结构用途 |
|---|---:|---:|---:|---:|---|
| `instances/tiny/fjsp_tiny.json` | 2 | 5 | 2 | 1/1 | FJSP 加工域、非线性 DAG、exact |
| `instances/tiny/automotive_tiny.json` | 2 | 6 | 3 | 1/1 | 分支/汇合 DAG、同配置、exact |
| `instances/tiny/fjsp_small.json` | 3 | 9 | 3 | 2/2 | 多车、多岛稍大验证 |
| `instances/tiny/automotive_small.json` | 3 | 12 | 3 | 2/2 | 三类语义 DAG 稍大验证 |

受控 automotive tiny 决策覆盖：非线性 DAG、不可比工序、多候选岛、配置切换、同配置连续加工、跨岛 W、同岛无 W、W 空驶与单 F 资源竞争。

## 5. 测试结果

最终命令：

```text
python -m compileall .
pytest -q
```

结果：`28 passed`。

| 验收项 | 对应测试 | 结果 |
|---|---|---|
| A DAG 违规拒绝 | `test_dag.py` | 通过 |
| B 产品不可分割 | `test_decoder.py`, `test_feasibility.py` | 通过 |
| C 配置变化插入重构 | `test_reconfiguration.py` | 通过 |
| D 同配置重构为 0 | `test_reconfiguration.py` | 通过 |
| E 同岛无 W | `test_w_agv.py` | 通过 |
| F 跨岛生成 W | `test_w_agv.py` | 通过 |
| G W 空驶与中间 gap 插入 | `test_w_agv.py` | 通过 |
| H F 到达/返库分离与同步 | `test_f_agv.py` | 通过 |
| I 岛 processing/reconfiguration 不冲突 | `test_reconfiguration.py` | 通过 |
| J objective 从原始记录重算一致 | `test_feasibility.py` | 通过 |
| 图节点/关系/action masks | `test_graph.py` | 通过 |
| tiny exact-vs-decoder | `test_exact_small.py` | 通过 |

压力测试：

```text
requested=1000
completed=1000
feasible=1000
feasible_rate=1.0
failures=[]
```

## 6. 精确小算例结果

当前环境检测结果：`gurobipy=False`，`ortools=False`。因此按任务要求使用 `exhaustive-active-schedule-bnb`，枚举 precedence-feasible operation order、island、W 与 F 决策，并对 makespan 做分支定界。该后备求解器只允许最多 8 道工序，避免误用于大实例。

| 实例 | 状态 | 最优 makespan | 搜索节点 | H2 makespan | checker |
|---|---|---:|---:|---:|---|
| fjsp-tiny | OPTIMAL | 57 | 273 | 57 | feasible |
| automotive-tiny | OPTIMAL | 157 | 8,942 | 157 | feasible |

该“最优”指穷举的活动调度决策空间；对于 regular makespan 目标，存在最优活动调度。最终解仍由独立 feasibility checker 逐项验证，而不是只信任 decoder。

## 7. 三类基线

| 实例 | H1 makespan/cost | H2 makespan/cost | H3 makespan/cost |
|---|---:|---:|---:|
| fjsp-tiny | 58 / 39.373 | 57 / 35.033 | 149 / 98.466 |
| automotive-tiny | 157 / 79.917 | 157 / 79.551 | 237 / 129.816 |
| fjsp-small | 174 / 233.584 | 176 / 224.070 | 330 / 217.074 |
| automotive-small | 241 / 269.252 | 224 / 231.754 | 421 / 268.120 |

H3 的目标是验证 configuration-aware 规则和环境稳定性，不宣称其质量优于 H1/H2；结果也说明“只优先同配置”会因物流和时间同步付出明显代价，后续策略必须联合建模。

## 8. 发现并修复的不一致

1. **现有生成器/demo 是 RAIS-1.0，而权威模型是 RCIAS-2.0。** 原文件含 module type、module instance、module allocation/transfer、compatible configurations。已全部删除并重生成数据。
2. **旧 ID 为 P/I，当前定义要求 J/o/M。** 已统一为 `J*`、`o*`、`M*`、`C*`、`W*`、`F*`；可能歧义的长编号使用下划线。
3. **旧工序可在多个 compatible configurations 中选择。** 已改为唯一 `required_configuration`，候选岛必须直接支持该配置。
4. **旧 FJSP 生成器保留线性 job chain。** 已改为 sparse assembly DAG；原 FJSP 只保留机器候选域与加工时间。
5. **旧岛没有明确 initial_configuration。** 已显式生成并验证。
6. **旧物流成本只有单一车辆费率。** 已按模型拆分 W loaded/empty 与 F outbound/return 成本。
7. **算法说明中的插入伪代码写成 `max(base_ready, prev.end) + setup`，但 MILP 将物流就绪与重构就绪作为独立下界。** 实现服从数学模型，使用 `max(base_ready, prev.end + setup)`，允许提前重构。
8. **旧 JSON 路径使用 `agv_w/agv_f`。** 已统一为模型说明中的 `logistics.W` 与 `logistics.F`。

## 9. 当前未实现的 CLGRI 模块

本阶段按要求没有创建占位神经网络。尚未实现：

- RT-HGT encoder；
- preference-conditioned autoregressive neural policy；
- behavior cloning 与 PPO；
- Critical Synchronization Graph；
- neural destroy selector 与 graph-guided repair；
- self-improvement/self-imitation；
- 多目标 Pareto archive 与训练课程。

已为下一阶段提供可直接使用的五类节点动态图、typed relations、动态 O-M 特征、`operation -> island -> W/NONE -> F` hard masks，以及完整同步瓶颈标签。

## 10. 下一阶段建议顺序

1. 固定本阶段 schema 与 decoder 测试作为回归门禁。
2. 实现批处理友好的 graph tensor adapter，并对 RT-HGT forward 做 shape/mask 测试。
3. 用 H1/H2/H3 与 tiny exact action sequence 做 behavior-cloning sanity check。
4. 只训练单目标 constructive PPO，保持 checker 可行率 100%。
5. 构建 Critical Synchronization Graph，先验证 heuristic destroy + repair。
6. 再引入 neural destroy、preference conditioning、Pareto archive 与泛化实验。

机器可读完整结果见 `validation_results.json` 与 `stress_validation_results.json`。
