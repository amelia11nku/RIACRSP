# Phase 3 Constructive PPO 实现与验证报告

## 1. 阶段结论

本阶段已实现一套以 makespan 为唯一学习目标的
`RT-HGT + O→M→W→F autoregressive PPO` 构造式求解器。训练仅使用独立
synthetic 分布，固定 synthetic validation seed 用于 checkpoint 选择；130 个
canonical RCIAS-2.0 实例在 checkpoint 与配置冻结后只执行一次正式测试。

最终验收标志：

```text
PPO_IMPLEMENTED = TRUE
PPO_NUMERICALLY_STABLE = TRUE
PPO_FEASIBILITY_100 = TRUE
PPO_REPRODUCIBLE = TRUE
PPO_GENERALIZATION_VALIDATED = TRUE
REPOSITORY_CLEAN = TRUE
READY_FOR_CSG = TRUE
```

## 2. 数据边界与无泄漏设计

| 数据角色 | 来源 | 种子/数量 | 是否产生梯度 | 用途 |
|---|---|---:|---|---|
| BC/PPO training | `TrainingInstanceFactory` 新生成 DAG、资格、时间、能力、布局与物流 | episode seeds 从 1,000,000 起 | 是 | BC expert replay 与 PPO rollout |
| checkpoint validation | 同一生成规则、固定且互斥的 held-out seeds | S: 41001–41005；M: 42001–42003；L: 43001–43002 | 否 | curriculum gate 与 best checkpoint |
| structural generalization | 独立 M-level 场景 seeds 44001–44003 | reconfiguration、fleet scarcity、travel 各 3 例 | 否 | 结构外推 |
| public test | `instances/canonical/RCIAS-2.0/` | 130 例 | 否 | 冻结后的正式跨族测试 |
| Tiny sanity | `tiny_01`、`tiny_03` | 2 例 | 仅 sanity overfit | exact 目标与 PPO 数值闭环 |

Level L 的范围由 canonical manifest 的 P10/P50/P90 统计约束，但实例工厂不读取或
复制 canonical operation-level 记录。正式训练和 validation seed 集合零重叠，三个
PPO run 的 `canonical_gradient_instances` 均为 0。

## 3. 实现范围

| 模块 | 实现 |
|---|---|
| Synthetic distribution | deterministic `seed → Instance`；S/M/L；新 DAG、eligible islands、processing time、configuration transitions、layout 和 W/F logistics |
| Curriculum | validation performance gate；当前 level 75%，旧 level replay 25%；可达到 S→M→L |
| RT-HGT | 数值稳定的 `scatter_reduce_ + index_add_` vectorized segment softmax |
| Graph batching | 五类节点及全部关系的 disjoint union；保存 graph ptr、node batch index 和 candidate-to-graph |
| Policy | 保留 BC API；新增 stochastic/greedy sample、joint log probability、stage/joint/normalized entropy 和 action reevaluation |
| Reward | `-(C_partial(t+1)-C_partial(t))/H_I`，严格 telescope 到 `-Cmax/H_I` |
| Learning | detached CPU rollout buffer、GAE(γ=1, λ=0.95)、advantage normalization、clipped PPO、Huber value loss、entropy bonus、gradient clipping、KL early stop |
| Logging | performance、四阶段 entropy、KL、clip fraction、explained variance、grad norm、时间分解、action/resource statistics 和 memory |

本阶段没有实现 CSG、destroy/repair、多目标 preference policy、Pareto archive 或
self-imitation；也没有改变 decoder、timeline、checker、objective 或 canonical 定义。

## 4. Tiny 与数值 sanity

`run_ppo_sanity.py` 对两个 Tiny 实例各运行 4 次 PPO update：

| 实例 | exact | BC | PPO final | rollout feasibility | finite update | 参数变化 |
|---|---:|---:|---:|---:|---|---|
| tiny_01 | 157 | 157 | 157 | 100% | 是 | 是 |
| tiny_03 | 36 | 36 | 36 | 100% | 是 | 是 |

单元测试覆盖 vectorized softmax 的 reference/sum/gradient/no-NaN、single-vs-batch
encoder、hard mask、deterministic singleton stage、joint log probability、ratio=1、
reward telescope、GAE、PPO loss、buffer detach、curriculum 和 checkpoint replay。

## 5. BC warm start

BC 从 12 个独立 S/M synthetic 实例生成 best-of-H1/H2/H3 expert，共 515 个决策
state，训练 20 epochs。训练 seed 为 20260825，best epoch 为 4；固定 S validation
的 normalized makespan 从随机初始化的 0.293877 降至 best 0.236553，最终权重为
0.237476，validation feasibility 为 100%。运行时间 507.05 s。

## 6. 三随机种子 PPO 训练

硬件为 NVIDIA GeForce RTX 4060 Laptop GPU，PyTorch 2.12.1+cu130，CUDA 13.0。
三个 run 均从冻结 BC best checkpoint 初始化；best checkpoint criterion 是固定 S/M
synthetic validation 的 mean normalized makespan，canonical 结果从未用于选择。

| seed | updates | env steps | episodes | curriculum updates | init val | best val | 改善 | best update | wall time |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 31001 | 6 | 1261 | 47 | S,S,S,M,M,M | 0.205586 | 0.198271 | 3.56% | 2 | 1068.85 s |
| 31002 | 6 | 1239 | 41 | S,S,S,M,M,M | 0.205586 | 0.202372 | 1.56% | 3 | 1009.35 s |
| 31003 | 6 | 1306 | 43 | S,S,S,M,M,M | 0.205586 | 0.199826 | 2.80% | 4 | 1057.06 s |

总计 3806 environment steps、131 episodes、3135.25 s。三种子 best validation
均值 0.200156，population std 0.001691；三个 run 均改善相同初始化。最大 observed
approx KL 为 0.01777，低于 target 0.03；最小 active-stage normalized entropy 为
0.867，未发生严重 collapse；训练 rollout feasibility 为 100%，且无 NaN/Inf。

## 7. Held-out synthetic 与结构泛化

正式冻结评估包含 S 5 例、M 3 例、L 2 例和三个结构场景各 3 例，共 19 个实例、
133 method-runs。gap 定义为同一实例上 H1/H2/H3/BC/三个 PPO seeds 中最佳
makespan 的百分比差距。

| 组 | H1 gap | H2 gap | H3 gap | BC gap | PPO gap | PPO 相对 BC |
|---|---:|---:|---:|---:|---:|---:|
| Level S | 6.34% | 8.59% | 61.08% | 1.65% | 3.37% | -1.72 pp |
| Level M | 1.13% | 6.70% | 135.87% | 33.66% | 17.00% | +16.66 pp |
| Level L | 0.00% | 4.81% | 102.43% | 63.83% | 53.25% | +10.58 pp |
| fleet scarcity | 7.31% | 17.69% | 17.27% | 6.13% | 10.20% | -4.07 pp |
| high reconfiguration | 0.70% | 9.52% | 62.03% | 38.58% | 37.76% | +0.82 pp |
| high travel | 1.50% | 18.84% | 79.28% | 21.97% | 18.31% | +3.65 pp |
| Overall | 3.35% | 11.10% | 73.35% | 23.00% | 19.64% | +3.36 pp |

PPO 因此在总体、M/L、high-reconfiguration 和 high-travel 上改善 BC，但在 Level S
和 fleet-scarcity 上回退；同时整体仍未超过 H1。这一结果支持“存在独立 RL 学习信号
和跨规模改善”，不支持“PPO 已成为最强构造启发式”的结论。

## 8. Frozen canonical public test

冻结记录在正式评估前写入四个 checkpoint 的 SHA256 和 metadata。随后用 CPU 侧
6 个单线程 worker 完成 130 个实例、910 个 method-runs；总 wall time 为
2941.72 s（49.03 min），全部通过独立 feasibility checker。

| family | H1 gap | H2 gap | H3 gap | BC gap | PPO gap | PPO 相对 BC |
|---|---:|---:|---:|---:|---:|---:|
| Brandimarte (10) | 0.04% | 12.39% | 81.88% | 23.52% | 18.73% | +4.79 pp |
| Hurink E (40) | 1.15% | 7.28% | 77.07% | 33.87% | 28.13% | +5.75 pp |
| Hurink R (40) | 1.37% | 6.21% | 83.30% | 38.74% | 32.82% | +5.92 pp |
| Hurink V (40) | 0.32% | 13.41% | 106.27% | 78.22% | 69.39% | +8.83 pp |
| Overall (130) | 0.87% | 9.23% | 88.34% | 48.22% | 41.54% | +6.67 pp |

三个 PPO seed 的 overall mean gap 分别为 38.89%、42.84%、42.91%；跨 seed
均值 41.54%、population std 1.88%、best 38.89%。PPO 对 BC 的改善在四个 family
方向一致，并在 flexibility 从 E/R 到 V 增大时仍保持改善，但对 H1 没有竞争力。
PPO pooled mean runtime 为 26.78 s/instance，其中 network inference 为 7.41 s；
BC 为 27.78/7.56 s，H1/H2/H3 分别为 3.06/0.81/19.35 s。

## 9. 性能与内存 profiling

三次正式训练 history 的 3806 steps 时间分解为：graph construction 10.96%、
RT-HGT forward 25.87%、policy scoring 4.41%、decoder 0.05%、PPO update 58.70%。
这表明当前第一优化目标应是 PPO minibatch reevaluation/update，而不是 decoder。

独立 CUDA profiling 连续执行 300 个 detached S-level episodes、6200 environment
steps，rollout wall time 为 785.17 s，吞吐为 7.90 steps/s、0.382 episodes/s。随后
一个 M-level collection 为 49.92 s，一个 4-epoch PPO update 为 63.99 s。

该独立 profile 的测量占比为 RT-HGT forward 68.64%、graph construction 12.53%、
policy scoring 11.13%、PPO update 7.58%、decoder 0.12%。两种时间分解的 workload
不同：三次训练 history 含每 update 的大量 minibatch reevaluation，故 update 占比更高；
300-episode profile 以 rollout 为主，故 forward 占比更高。

从 episode 1 到 300，GPU allocated 恒为 9.24 MB，reserved 恒为 23.07 MB；tail
reserved growth 为 0。CPU RSS tail growth 为 1.62 MB，约占 1.256 GB 的 0.13%，
且样本存在回落。脚本判定 `cpu_memory_stable=true`、`gpu_memory_stable=true`、
`memory_stable=true`。

## 10. 回归、复现与仓库审计

最终清理后的回归结果：

| 检查 | 结果 |
|---|---|
| `python -m compileall .` | PASS |
| `pytest -q` | 69 passed in 21.63 s |
| canonical verify-only | 130/130 checksum 与 byte regeneration PASS |
| small validation | tiny_01/02/03 exact makespan 157/57/36，全部 checker feasible |
| native Tiny exact | Gurobi=CP-SAT=exhaustive=36，bound=36，gap=0，replay feasible |
| graph profile | `GRAPH_PROFILE_VALID = TRUE`，probe bound 全部精确吻合 |
| Phase 2 BC regression | loss 0.117917，joint accuracy 1.0，feasible，makespan 157 |
| repository audit | cache 已清除；duplicate/legacy/redundant/obsolete 均为空；clean=true |

正式机器可读产物包括 checkpoint SHA256 freeze record、逐实例 raw CSV、分组与
Overall summary、完整训练 history、memory samples 和 final gate JSON。所有报告数值
可回溯到 `outputs/phase3/`，图表同时提供 PNG 与 PDF。

## 11. 验收判断依据

`PPO_GENERALIZATION_VALIDATED=TRUE` 的含义是：三种子都在未见的 checkpoint
validation 上改善初始化；冻结 PPO 在 synthetic overall 和 canonical overall 上都
改善 BC，且 canonical 四个 family 改善方向一致。它不表示 PPO 超过 H1，报告已显式
保留这一限制。

`READY_FOR_CSG=TRUE` 的依据是：冻结接口和所有历史回归未变；PPO 数值、ratio、
reward、GAE、hard mask、checkpoint tests 通过；Tiny、训练、synthetic 和 canonical
rollout 全部可行；三种子均出现 held-out improvement；KL 受控、entropy 未 collapse、
300-episode memory 稳定；至少一个完整 M-level curriculum run 完成；canonical
compatibility 和 clean audit 均通过。因此下一阶段可以把当前结果作为单一正式
constructive baseline，量化 `PPO → PPO + CSG improvement`，而无需保留并行旧实现。
