# 可重构岛式汽车总装系统集成调度：问题描述、符号体系与数学模型

> **研究对象**：Reconfigurable Capability Island Assembly System with Dual-Flow Logistics（简称 RCIAS）  
> **模型版本**：RCIAS-2.0  
> **模型类型**：静态、确定性、连续时间双目标 MILP  
> **核心边界**：DAG 工艺顺序柔性 + 装配岛能力配置重构 + W 类主工件物流 + F 类物料包物流  
> **删除内容**：模块类型、物理模块实例、模块保持/迁移、A 类模块 AGV  
> **数据接口**：与 `generate_fjsp_reconfigurable.py` 和 `generate_automotive_semantic.py` 完全一致

---

## 0. 本版相对上一版的结构性变化

本版本不再对“能力配置是由哪些物理辅助模块形成的”进行内部展开，而将每个配置视为装配岛可以进入的一个**离散能力状态（capability configuration/state）**。这一建模粒度与近年来可重构制造单元调度文献的主流抽象更加一致，也避免有限模块实例与模块物流把论文主线从“可重构装配—双层物流—智能调度”转移到更细粒度的设备资源配置问题。

本版做出以下确定性调整：

1. 每道工序具有唯一固定能力配置 `c_o`，不存在 operation-to-configuration 选择。
2. 每个装配岛 `m` 具有一个可支持配置集合 `C_m` 和一个已知初始配置 `c_m^0`。
3. 同一装配岛相邻工序若配置不同，则产生 sequence-dependent reconfiguration time 和 cost；若配置相同，则重构时间与成本均为 0。
4. 删除模块类型、模块实例、模块保留、模块迁移以及 A 类 AGV。
5. 保留产品 DAG，不预先给定唯一线性工艺路线；由于主工件不可分割，调度需内生确定 DAG 的一个拓扑线性化路径。
6. 保留两层有限物流：W 类主工件 AGV 与 F 类物料包 AGV。
7. W 物流任务不是预先给定，而由“产品实际拓扑相邻关系 + 两工序的装配岛分配”共同内生生成。
8. F 物流对每道工序均存在一次仓库到目标装配岛的配送任务，并占用有限 F-AGV 完成“出库—配送—返库”全过程。

由此，论文的核心耦合关系变为：

\[
\boxed{
\text{DAG linearization}
\rightarrow
\text{actual operation adjacency}
\rightarrow
\text{workpiece route}
}
\]

以及

\[
\boxed{
\text{operation-island assignment}
\rightarrow
\text{capability transition}
}
\]

最终每道工序的开始由**工艺就绪、主工件就绪、物料包就绪、能力配置就绪与装配岛就绪**共同决定。

---

# 1. 问题描述

本文研究一种面向汽车总装场景的**可重构岛式装配系统**。系统由若干相互独立的装配岛组成。与具有固定加工能力的传统装配单元不同，每个装配岛可在若干预定义能力配置之间切换，从而在不同时间承担不同类型的装配任务。装配岛的可重构性在本文中抽象为离散能力状态转换，而不进一步追踪能力状态内部的具体模块组成。

每个装配产品由若干装配工序组成，工艺约束使用有向无环图（DAG）描述。DAG 仅规定必要的偏序关系，不预先规定唯一线性工艺路线。由于同一产品的主工件不可分割，其工序在实际执行时不能并行，因此所有工序必须内生形成一条满足原 DAG 的拓扑线性化路径。该路径决定每个工序的实际直接前序，也进一步决定主工件在装配岛之间的实际运输链。

每道装配工序 `o` 具有唯一固定的能力配置需求 `c_o`。工序只能分配至其候选装配岛集合 `M_o` 中的一个装配岛，且候选集合已保证目标岛支持配置 `c_o`。若某装配岛连续执行两道配置不同的工序，则必须在两工序之间完成相应能力重构；重构时间与成本依赖于装配岛以及前后配置状态。若相邻工序所需配置相同，则不发生能力重构。

系统进一步包含两类有限物流资源：

- **W 类主工件 AGV**：负责产品主工件从仓库/起始区运输至第一实际装配岛，以及产品实际相邻工序所在装配岛之间的运输。若实际相邻工序分配到同一装配岛，则不产生 W 类运输任务。W-AGV 完成一次运输后停留在交付节点，下一任务开始前需从当前位置空驶至下一提货节点。
- **F 类物料包 AGV**：负责将每道工序所需的零部件/物料包从仓库配送到该工序的目标装配岛。F-AGV 完成配送后返回仓库，整个“出库配送—返库”周期占用该车辆；但工序在物料包到达目标岛时即可开始，无需等待 AGV 返回仓库。

因此，某道工序只有在以下条件均满足后才能开始：

1. 其在实际产品拓扑线性化中的前一道工序已经完成；
2. 原 DAG 中的所有必要前驱关系均得到满足；
3. 若需要 W 类运输，则主工件已抵达目标装配岛；
4. 该工序对应的 F 类物料包已抵达目标装配岛；
5. 目标装配岛空闲，且已从上一工序对应配置完成必要的能力切换。

本文考虑两个优化目标：

\[
\min f_1=C_{\max},
\]

即最小化最大完工时间；以及

\[
\min f_2=C^{\mathrm{total}},
\]

即最小化配置重构与 W/F 两层物流产生的综合运行成本。

---

# 2. 核心概念定义

## 2.1 装配岛与能力配置

装配岛集合记为 `M`。装配岛 `m` 可进入的能力配置集合为

\[
\mathcal C_m\subseteq\mathcal C.
\]

其中 `C` 为全局能力配置集合。每个配置 `c` 表示一种预定义装配能力状态，例如“车身定位与检测能力”“机器人安装与拧紧能力”“轮胎安装与定位能力”等。

本文不进一步展开配置内部由哪些物理模块构成。换言之，配置在模型中是**不可再分的能力状态标签**。

每个装配岛在调度开始时具有已知初始配置

\[
c_m^0\in\mathcal C_m.
\]

若岛 `m` 从配置 `c` 切换至 `c'`，所需时间与成本分别为

\[
\rho_{mcc'},\qquad \kappa_{mcc'}.
\]

满足

\[
\rho_{mcc}=0,\qquad \kappa_{mcc}=0.
\]

## 2.2 工序固定配置

每道工序 `o` 对应唯一固定配置

\[
c_o\in\mathcal C.
\]

因此工序并不在多个配置之间选择。若 `o` 被分配至装配岛 `m`，则 `m` 在执行 `o` 时必须处于 `c_o`。

候选装配岛集合在数据生成阶段已经完成技术兼容性预处理：

\[
\mathcal M_o
=\{m:\;m\text{ 可以执行 }o\text{ 且 }c_o\in\mathcal C_m\}.
\]

## 2.3 DAG 与实际拓扑线性化

对产品 `j`，其工序集合记为 `O_j`，DAG 直接优先弧集合为

\[
\mathcal A_j\subseteq\mathcal O_j\times\mathcal O_j.
\]

DAG 中不存在路径关系的两个工序在工艺层面可交换顺序，但由于主工件不可分割，实际调度仍需对所有工序给出一个满足 DAG 的全序。因此模型需要同时决定：

- 哪些不可比工序谁先谁后；
- 产品实际相邻工序是谁；
- 相邻工序的主工件是否需要跨岛运输。

## 2.4 W 类主工件物流

若 `v` 是产品实际线性化中的第一道工序，则其主工件从仓库节点 `0` 运输到 `v` 所在装配岛。

若 `u` 是 `v` 的实际直接前序：

- 若 `u` 与 `v` 在同一装配岛，则主工件直接留在原岛，不生成 W 任务；
- 若两者在不同装配岛，则形成一项从 `u` 所在岛到 `v` 所在岛的 W 运输任务。

因此 W 运输任务集合是**调度内生生成的**。

## 2.5 F 类物料包物流

每道工序都产生一项物料包配送需求。若工序 `o` 分配至岛 `m`，则形成

\[
0\rightarrow m\rightarrow 0
\]

的 F-AGV 任务。物料到达 `m` 后即可被工序使用，而 F-AGV 仍需继续完成返库过程后才可执行下一任务。

---

# 3. 基本假设

1. 所有产品、工序、DAG 工艺约束、候选装配岛、加工时间、能力配置、配置切换参数与物流参数在调度开始前已知，系统为静态确定性环境。
2. 每道工序属于且仅属于一个装配产品。
3. 同一产品的主工件不可分割，因此任意两道该产品工序均不得并行；实际工序顺序必须形成原 DAG 的一个拓扑线性化。
4. 每道工序仅对应一个固定能力配置 `c_o`。
5. 每个装配岛具有有限支持配置集合 `C_m`，其初始配置 `c_m^0` 已知。
6. 每个装配岛任一时刻最多执行一道工序，工序不可抢占。
7. 若装配岛连续执行配置不同的工序，则两工序之间需完成 sequence-dependent capability reconfiguration；相同配置之间重构时间与成本均为 0。
8. W 与 F 两类 AGV 数量有限，每台 AGV 任一时刻最多执行一个运输任务。
9. W-AGV 初始位于仓库。完成运输后停留在交付节点，执行下一任务前需要空驶至下一提货节点。
10. F-AGV 初始位于仓库。每次任务均从仓库出发、将物料包送至目标岛并返回仓库；整个往返过程占用该车辆。
11. 产品第一道实际工序的主工件必须由仓库运输至目标岛；产品最后一道工序完成后不再计成品运输。
12. 若产品实际相邻工序位于同一装配岛，则不产生 W 运输任务。
13. 不考虑 AGV 路径冲突、交通拥堵、道路容量与充电约束；节点间运输时间已知且确定。
14. 不考虑动态订单到达、设备故障、随机加工时间和随机运输时间。
15. 时间参数为非负整数，以便同一实例既可用于连续时间 MILP，也可直接用于 CP-SAT；成本可按统一倍率缩放为整数。

---

# 4. 符号体系

## 4.1 集合与索引

| 符号 | 定义 |
|---|---|
| `\mathcal J` | 装配产品集合，索引 `j`；数据中为 J1,J2,... |
| `\mathcal O_j` | 产品 `j` 的工序集合 |
| `\mathcal O=\cup_j\mathcal O_j` | 全部工序集合，索引 `o,u,v`；数据中如 o11,o12,o21 |
| `\mathcal A_j` | 产品 `j` 的 DAG 直接优先弧集合 |
| `\mathcal M` | 装配岛集合，索引 `m,n`；数据中为 M1,M2,... |
| `\mathcal M_o` | 工序 `o` 的候选装配岛集合 |
| `\mathcal C` | 能力配置集合，索引 `c,c'`；数据中为 C1,C2,... |
| `\mathcal C_m` | 装配岛 `m` 支持的配置集合 |
| `\mathcal W` | W 类 AGV 集合，索引 `a` |
| `\mathcal F` | F 类 AGV 集合，索引 `f` |
| `\mathcal N=\{0\}\cup\mathcal M` | 物流节点集合，`0` 对应数据中的 WH |
| `s_j,t_j` | 产品拓扑路径的虚拟源与虚拟汇 |
| `s_m,t_m` | 装配岛工序路径的虚拟源与虚拟汇 |
| `s_a^W,t_a^W` | W-AGV 任务路径虚拟源/汇 |
| `s_f^F,t_f^F` | F-AGV 任务路径虚拟源/汇 |

## 4.2 加工与重构参数

| 符号 | 定义 |
|---|---|
| `c_o` | 工序 `o` 的唯一固定能力配置 |
| `c_m^0` | 装配岛 `m` 的已知初始配置 |
| `p_{om}` | 工序 `o` 在候选岛 `m` 上的加工时间 |
| `\rho_{mcc'}` | 岛 `m` 从配置 `c` 切换至 `c'` 的时间 |
| `\kappa_{mcc'}` | 岛 `m` 从配置 `c` 切换至 `c'` 的成本 |
| `H` | 足够大的连续时间上界/Big-M 常数 |

## 4.3 物流参数

| 符号 | 定义 |
|---|---|
| `d_{np}` | 节点 `n,p` 之间的距离 |
| `\tau^{WL}_{a,np}` | W-AGV `a` 从 `n` 到 `p` 的载主工件运输时间 |
| `\tau^{WE}_{a,np}` | W-AGV `a` 从 `n` 到 `p` 的空驶时间 |
| `c^{WL}_a,c^{WE}_a` | W-AGV 的单位距离载货/空驶成本 |
| `\tau^{FO}_{f,m}` | F-AGV `f` 从仓库到岛 `m` 的配送时间 |
| `\tau^{FR}_{f,m}` | F-AGV `f` 从岛 `m` 返回仓库时间 |
| `c^{FO}_f,c^{FR}_f` | F-AGV 的单位距离出库/返库成本 |

---

# 5. 决策变量

## 5.1 工序与装配岛变量

- `x_{om}\in\{0,1\}`：工序 `o` 是否分配至岛 `m`。
- `S_o,C_o\ge0`：工序开始时间与完成时间。
- `C_{\max}\ge0`：最大完工时间。

## 5.2 产品实际拓扑路径变量

- `\pi^j_{uv}\in\{0,1\}`：在产品 `j` 的实际拓扑线性化路径中，`v` 是否是 `u` 的直接后继。
- 同时定义 `\pi^j_{s_jv}` 与 `\pi^j_{ut_j}`。

## 5.3 装配岛工序路径变量

- `y^m_{uv}\in\{0,1\}`：在装配岛 `m` 的实际工序序列中，`v` 是否直接位于 `u` 之后。
- 同时定义 `y^m_{s_mv}` 与 `y^m_{ut_m}`。
- `z_m\in\{0,1\}`：装配岛 `m` 是否被使用。

## 5.4 W 类运输任务变量

为避免让一个“工序级运输变量”同时承担未知提货点与交付点，本文预定义所有可能的 W 运输任务候选。

定义候选任务集合 `Theta^W`。候选任务 `theta` 具有固定提货节点 `P_theta`、固定交付节点 `D_theta` 和目标工序 `v(theta)`：

1. 第一工序候选任务 `theta=(s_j,v,0,n)`；
2. 相邻工序跨岛候选任务 `theta=(u,v,m,n)`，其中 `m != n`。

变量：

- `e^W_\theta\in\{0,1\}`：候选 W 任务 `theta` 是否被实际激活。
- `\alpha^W_{\theta a}\in\{0,1\}`：任务 `theta` 是否由 W-AGV `a` 执行。
- `U^W_\theta,A^W_\theta\ge0`：W 任务开始时间与载主工件到达时间。
- `\eta^W_{\theta\theta'a}\in\{0,1\}`：在 W-AGV `a` 的任务序列中，任务 `theta'` 是否直接位于任务 `theta` 之后。
- `\eta^W_{s_a\theta a},\eta^W_{\theta t_a a}`：车辆任务路径源/汇弧。
- `z^W_a\in\{0,1\}`：W-AGV `a` 是否被使用。

## 5.5 F 类物流变量

- `\beta^F_{omf}\in\{0,1\}`：工序 `o` 分配至岛 `m` 且其物料包由 F-AGV `f` 配送。
- `\alpha^F_{of}=\sum_m\beta^F_{omf}`：工序 `o` 是否由车辆 `f` 执行配送。
- `U^F_o,A^F_o,R^F_o\ge0`：F 任务开始、物料包到达、车辆返库完成时间。
- `\eta^F_{ovf}\in\{0,1\}`：在 F-AGV `f` 上，配送任务 `v` 是否直接位于任务 `o` 后。
- `\eta^F_{s_f o f},\eta^F_{o t_f f}`：F 车辆任务路径源/汇弧。
- `z^F_f\in\{0,1\}`：F-AGV `f` 是否被使用。

---

# 6. 双目标 MILP

## 6.1 工序分配与加工时间

每道工序必须且只能选择一个候选装配岛：

\[
\sum_{m\in\mathcal M_o}x_{om}=1,
\qquad \forall o\in\mathcal O.
\tag{1}
\]

完成时间：

\[
C_o=S_o+\sum_{m\in\mathcal M_o}p_{om}x_{om},
\qquad \forall o\in\mathcal O.
\tag{2}
\]

最大完工时间：

\[
C_{\max}\ge C_o,
\qquad \forall o\in\mathcal O.
\tag{3}
\]

---

## 6.2 DAG 工艺约束与实际拓扑线性化

### 6.2.1 原 DAG 优先关系

\[
S_v\ge C_u,
\qquad \forall j\in\mathcal J,\;(u,v)\in\mathcal A_j.
\tag{4}
\]

### 6.2.2 产品实际 Hamilton 路径

对任意 `v in O_j`：

\[
\pi^j_{s_jv}
+\sum_{u\in\mathcal O_j\setminus\{v\}}\pi^j_{uv}=1.
\tag{5}
\]

对任意 `u in O_j`：

\[
\pi^j_{ut_j}
+\sum_{v\in\mathcal O_j\setminus\{u\}}\pi^j_{uv}=1.
\tag{6}
\]

源汇流：

\[
\sum_{v\in\mathcal O_j}\pi^j_{s_jv}=1,
\qquad
\sum_{u\in\mathcal O_j}\pi^j_{ut_j}=1.
\tag{7}
\]

若 `u` 在实际产品路径中直接位于 `v` 前：

\[
S_v\ge C_u-H(1-\pi^j_{uv}),
\qquad u\neq v.
\tag{8}
\]

由于 `p_om>0`，式 (8) 会排除任何产品路径子环；与式 (4) 联合后，该 Hamilton 路径只能对应原 DAG 的合法拓扑线性化，因此无需额外 MTZ/rank 变量。

---

## 6.3 装配岛工序排序与能力配置转换

对任意 `o` 和可加工该工序的岛 `m`：

\[
y^m_{s_mo}
+\sum_{u\in\mathcal O\setminus\{o\}}y^m_{uo}
=x_{om},
\tag{9}
\]

\[
y^m_{ot_m}
+\sum_{v\in\mathcal O\setminus\{o\}}y^m_{ov}
=x_{om}.
\tag{10}
\]

岛路径源汇：

\[
\sum_o y^m_{s_mo}=z_m,
\qquad
\sum_o y^m_{ot_m}=z_m.
\tag{11}
\]

若 `v` 是 `u` 在岛 `m` 上的直接后继：

\[
S_v\ge
C_u+\rho_{m,c_u,c_v}
-H(1-y^m_{uv}).
\tag{12}
\]

若 `v` 是岛 `m` 上第一道工序，则需要从初始配置切换至 `c_v`：

\[
S_v\ge
\rho_{m,c_m^0,c_v}
-H(1-y^m_{s_mv}).
\tag{13}
\]

式 (12) 同时承担三项功能：

1. 保证装配岛工序不重叠；
2. 内生确定装配岛上的实际工序序列；
3. 根据相邻工序固定配置自动触发能力重构。

因此无需再增加经典 pairwise machine-disjunctive 变量。

---

# 7. W 类主工件物流建模

## 7.1 候选 W 运输任务的预定义

### 7.1.1 产品第一工序任务

对 `n in M_v` 定义候选任务

\[
\theta=(s_j,v,0,n).
\]

其激活条件为：`v` 是产品第一实际工序且分配到 `n`。

使用标准 AND 线性化：

\[
e^W_{s_jv0n}\le \pi^j_{s_jv},
\tag{14a}
\]

\[
e^W_{s_jv0n}\le x_{vn},
\tag{14b}
\]

\[
e^W_{s_jv0n}\ge \pi^j_{s_jv}+x_{vn}-1.
\tag{14c}
\]

### 7.1.2 产品相邻工序跨岛任务

对 `m in M_u,n in M_v,m != n` 定义

\[
\theta=(u,v,m,n).
\]

其激活条件为：`u` 是 `v` 的实际直接前序、`u` 在 `m`、`v` 在 `n`。

\[
e^W_{uvmn}\le \pi^j_{uv},
\tag{15a}
\]

\[
e^W_{uvmn}\le x_{um},
\tag{15b}
\]

\[
e^W_{uvmn}\le x_{vn},
\tag{15c}
\]

\[
e^W_{uvmn}\ge \pi^j_{uv}+x_{um}+x_{vn}-2.
\tag{15d}
\]

当 `m=n` 时不定义 W 候选任务，因此模型天然表达“同岛无需主工件运输”。

## 7.2 W-AGV 分配

每个激活任务必须选择一辆 W-AGV：

\[
\sum_{a\in\mathcal W}\alpha^W_{\theta a}=e^W_\theta,
\qquad \forall \theta\in\Theta^W.
\tag{16}
\]

并用

\[
0\le U^W_\theta\le H e^W_\theta,
\qquad
0\le A^W_\theta\le H e^W_\theta
\tag{17}
\]

关闭未激活任务。

## 7.3 W 任务释放与载货运输

若 `theta=(u,v,m,n)`：

\[
U^W_\theta\ge C_u-H(1-e^W_\theta).
\tag{18}
\]

第一工序从仓库出发，无产品前序释放时间，只需满足

\[
U^W_\theta\ge0.
\tag{19}
\]

任务到达时间：

\[
A^W_\theta
=
U^W_\theta
+
\sum_{a\in\mathcal W}
\tau^{WL}_{a,P_\theta,D_\theta}\alpha^W_{\theta a}.
\tag{20}
\]

工序必须等待其实际 W 任务到达：

\[
S_{v(\theta)}
\ge
A^W_\theta-H(1-e^W_\theta),
\qquad \forall\theta\in\Theta^W.
\tag{21}
\]

若相邻两工序同岛，则没有激活 W 任务，但式 (8) 仍保证后工序必须在前工序完成后执行。

---

# 8. W-AGV 任务序列与空驶

定义车辆是否被使用：

\[
\eta^W_{s_a\theta a}
+\sum_{\theta'\neq\theta}\eta^W_{\theta'\theta a}
=\alpha^W_{\theta a},
\tag{22}
\]

\[
\eta^W_{\theta t_a a}
+\sum_{\theta'\neq\theta}\eta^W_{\theta\theta'a}
=\alpha^W_{\theta a}.
\tag{23}
\]

\[
\sum_{\theta}\eta^W_{s_a\theta a}=z^W_a,
\qquad
\sum_{\theta}\eta^W_{\theta t_a a}=z^W_a.
\tag{24}
\]

若 `theta` 是车辆 `a` 的第一项任务，则车辆从仓库空驶到任务提货节点：

\[
U^W_\theta
\ge
\tau^{WE}_{a,0,P_\theta}
-H(1-\eta^W_{s_a\theta a}).
\tag{25}
\]

若任务 `theta'` 直接位于 `theta` 后：

\[
U^W_{\theta'}
\ge
A^W_\theta
+\tau^{WE}_{a,D_\theta,P_{\theta'}}
-H(1-\eta^W_{\theta\theta'a}).
\tag{26}
\]

由于所有载货运输时间均为正，式 (26) 会排除单辆 W-AGV 任务序列中的有向子环，无需额外 MTZ 变量。

---

# 9. F 类物料包物流

## 9.1 岛—F-AGV 联合分配

每道工序选择一个岛和一辆 F-AGV：

\[
\sum_{m\in\mathcal M_o}
\sum_{f\in\mathcal F}
\beta^F_{omf}=1,
\qquad \forall o.
\tag{27}
\]

联合变量必须与加工岛分配一致：

\[
\sum_{f\in\mathcal F}\beta^F_{omf}=x_{om},
\qquad \forall o,m\in\mathcal M_o.
\tag{28}
\]

定义

\[
\alpha^F_{of}=\sum_{m\in\mathcal M_o}\beta^F_{omf}.
\tag{29}
\]

## 9.2 F 配送时刻

物料到达时间：

\[
A^F_o
=
U^F_o
+
\sum_{m\in\mathcal M_o}
\sum_{f\in\mathcal F}
\tau^{FO}_{f,m}\beta^F_{omf}.
\tag{30}
\]

车辆返库完成时间：

\[
R^F_o
=
U^F_o
+
\sum_{m\in\mathcal M_o}
\sum_{f\in\mathcal F}
\left(\tau^{FO}_{f,m}+\tau^{FR}_{f,m}\right)\beta^F_{omf}.
\tag{31}
\]

工序必须等待物料包到达：

\[
S_o\ge A^F_o,
\qquad \forall o.
\tag{32}
\]

F 任务允许提前配送，因此没有约束要求 `U_o^F` 必须晚于工序前驱完成。该设计体现现实中的提前齐套。

---

# 10. F-AGV 任务序列

对每辆 F-AGV：

\[
\eta^F_{s_f o f}
+
\sum_{v\neq o}\eta^F_{vof}
=
\alpha^F_{of},
\tag{33}
\]

\[
\eta^F_{o t_f f}
+
\sum_{v\neq o}\eta^F_{ovf}
=
\alpha^F_{of}.
\tag{34}
\]

\[
\sum_o\eta^F_{s_f o f}=z^F_f,
\qquad
\sum_o\eta^F_{o t_f f}=z^F_f.
\tag{35}
\]

若任务 `v` 紧接任务 `o`：

\[
U^F_v
\ge
R^F_o-H(1-\eta^F_{ovf}).
\tag{36}
\]

由于每个 F 任务均以仓库为起点和终点，不需要额外空驶位置变量。

---

# 11. 综合成本目标

## 11.1 配置重构成本

首工序配置成本：

\[
C^{R,0}
=
\sum_{m\in\mathcal M}
\sum_{v\in\mathcal O}
\kappa_{m,c_m^0,c_v}y^m_{s_mv}.
\tag{37}
\]

岛内相邻工序配置转换成本：

\[
C^{R}
=
\sum_{m\in\mathcal M}
\sum_{u\neq v}
\kappa_{m,c_u,c_v}y^m_{uv}.
\tag{38}
\]

## 11.2 W 类物流成本

载货成本：

\[
C^{WL}
=
\sum_{\theta\in\Theta^W}
\sum_{a\in\mathcal W}
 c^{WL}_a d_{P_\theta,D_\theta}\alpha^W_{\theta a}.
\tag{39}
\]

车辆首次任务前空驶：

\[
C^{WE,0}
=
\sum_a\sum_\theta
c^{WE}_a d_{0,P_\theta}\eta^W_{s_a\theta a}.
\tag{40}
\]

任务间空驶：

\[
C^{WE}
=
\sum_a
\sum_{\theta\neq\theta'}
 c^{WE}_a d_{D_\theta,P_{\theta'}}
 \eta^W_{\theta\theta'a}.
\tag{41}
\]

## 11.3 F 类物流成本

\[
C^F
=
\sum_o\sum_{m\in\mathcal M_o}\sum_{f\in\mathcal F}
\left(c^{FO}_f d_{0m}+c^{FR}_f d_{m0}\right)
\beta^F_{omf}.
\tag{42}
\]

## 11.4 总成本

\[
C^{\mathrm{total}}
=
C^{R,0}+C^R+C^{WL}+C^{WE,0}+C^{WE}+C^F.
\tag{43}
\]

双目标模型为：

\[
\min\left(C_{\max},C^{\mathrm{total}}\right).
\tag{44}
\]

对于 Gurobi 精确求解，推荐：

- 小规模 Pareto 前沿：epsilon-constraint；
- 单次折中解：归一化加权 Tchebycheff；
- 仅用于验证可行性/下界：可固定成本上界后最小化 makespan。

---

# 12. Big-M 与线性化说明

模型中的 AND 关系均可使用标准线性化。

若

\[
z=x_1\land x_2,
\]

则

\[
z\le x_1,
\quad z\le x_2,
\quad z\ge x_1+x_2-1.
\]

若

\[
z=x_1\land x_2\land x_3,
\]

则

\[
z\le x_k,\;k=1,2,3,
\qquad
z\ge x_1+x_2+x_3-2.
\]

因此整个模型只包含二元变量、连续变量及线性等式/不等式，是纯 MILP，不包含双线性项或非凸二次项。

Big-M 推荐采用实例相关时间上界，而不是任意巨大常数。可使用保守上界：

\[
H=
\sum_{o\in\mathcal O}\max_{m\in\mathcal M_o}p_{om}
+
(|\mathcal O|+|\mathcal M|)\rho^{\max}
+
N_W\tau_W^{\max}
+
N_F\tau_F^{\max},
\]

其中 `N_W,N_F` 为 W/F 潜在任务数量的安全上界。

---

# 13. 冗余约束审查

本模型刻意不再增加以下常见但实质重复的约束。

## 13.1 不增加经典机器两两 disjunctive 变量

装配岛的 Hamilton 路径 `y` 与式 (12) 已同时保证工序排序、非重叠与 sequence-dependent reconfiguration。再加入 pairwise `u before v / v before u` 会重复描述同一资源互斥关系。

## 13.2 不增加“同一产品任意两工序不能重叠”约束

产品实际 Hamilton 路径 `pi` 与式 (8) 已把同产品所有工序串成一条实际拓扑路径，因此额外 pairwise serial constraints 冗余。

## 13.3 不增加产品路径、岛路径、AGV 路径的 MTZ 子环消除

所有真实活动都具有严格正持续时间；若路径产生有向子环，相应时间推进约束会导致

\[
S>S+\text{positive duration},
\]

从而自动不可行。MTZ/rank 可作为数值强化约束测试，但不是模型正确性的必要条件。

## 13.4 不增加 operation-to-configuration 变量

`c_o` 是参数，不是决策。配置状态变化完全由岛序列中相邻工序的 `c_u -> c_v` 确定。

## 13.5 不增加 W 运输的固定工序链

W 任务必须由实际产品拓扑相邻关系与岛分配共同生成，不能提前按工序编号预设运输链。

---

# 14. Gurobi 可解性

该模型可以直接用 Gurobi 建模求解，原因如下：

1. 所有离散决策均为 binary variables；
2. 所有时间变量为 continuous nonnegative variables；
3. 加工时间、能力切换、AGV travel time 均以参数形式进入；
4. 所有逻辑 AND 均可以线性化；
5. 不含二次乘积、max/min 非线性或非凸约束。

主要规模瓶颈并非连续时间，而是：

- 产品实际拓扑路径变量 `pi`，约为 `O(sum_j |O_j|^2)`；
- 岛序列变量 `y`，约为 `O(|M||O|^2)` 的稀疏子集；
- 候选 W 任务及车辆序列变量，取决于产品相邻候选与候选岛组合；
- F 车辆顺序变量，约为 `O(|F||O|^2)`。

因此精确 MILP 主要用于小规模算例验证与最优性基准，大规模实例应使用学习/启发式算法。

---

# 15. CP-SAT / CP Optimizer 转换

本问题也可以自然转化为 CP 模型。

## 15.1 工序与装配岛

对每个 `(o,m)` 创建 optional interval：

\[
I_{om},\qquad \mathrm{length}(I_{om})=p_{om}.
\]

并使用

\[
\mathrm{alternative}(I_o,\{I_{om}\}_{m\in M_o})
\]

实现装配岛选择。

对每个岛建立 sequence variable，并通过 transition matrix 表达

\[
\rho_{m,c,c'}.
\]

这比 MILP 中的 `y` 变量更加原生。

## 15.2 产品 DAG 与实际线性化

DAG 原始优先关系使用 `endBeforeStart`。同产品主工件不可分割可使用 NoOverlap；若还需要显式恢复实际直接前序以生成 W 任务，则可：

- 在 CP Optimizer 中使用 sequence variable 获取产品工序顺序；
- 在 CP-SAT 中使用 circuit/path binary variables 与 interval constraints 组合。

## 15.3 W 与 F AGV

W 运输任务使用 optional interval；任务是否存在由产品相邻关系与两工序岛分配共同决定。每辆 W-AGV 使用 NoOverlap/sequence，并设置任务间 empty-travel transition。

F 任务对每道工序必然存在，车辆选择用 alternative，车辆 sequence/no-overlap 控制容量；任务持续时间为 outbound + return，而物料到达时间是 interval 内的一个中间时间点。

因此 RCIAS-2.0 不依赖 MILP 独有结构，可以同时作为 Gurobi 和 CP 的统一问题定义。

---

# 16. 两套算例与模型字段的一一对应

两套生成器均输出 `RCIAS-2.0` JSON。

## 16.1 集合符号

| MILP | JSON |
|---|---|
| `J` | `sets.products`: J1,J2,... |
| `O` | `sets.operations`: o11,o12,... |
| `M` | `sets.islands`: M1,M2,... |
| `C` | `sets.configurations`: C1,C2,... |
| `W` | `sets.agvs_w`: W1,W2,... |
| `F` | `sets.agvs_f`: F1,F2,... |
| `N` | `sets.nodes`: WH,M1,M2,... |

## 16.2 工序与 DAG

`products[Jk].operations` 对应 `O_j`。

`products[Jk].precedence` 对应 `A_j`，并明确是 DAG 弧集合，不是默认线性链。

`operations[o].required_configuration` 对应 `c_o`。

`operations[o].eligible_islands` 对应 `M_o`。

`operations[o].processing_time[Mk]` 对应 `p_om`。

## 16.3 装配岛与重构

`islands[Mk].supported_configurations` 对应 `C_m`。

`islands[Mk].initial_configuration` 对应 `c_m^0`。

`reconfiguration.time[Mk][C][C2]` 对应 `rho_mcc'`。

`reconfiguration.cost[Mk][C][C2]` 对应 `kappa_mcc'`。

## 16.4 W 物流

`logistics.W.loaded_time[Wk][n][p]` 对应 `tau^WL_a,np`。

`logistics.W.empty_time[Wk][n][p]` 对应 `tau^WE_a,np`。

车辆成本字段对应 `c^WL_a,c^WE_a`。

## 16.5 F 物流

`logistics.F.outbound_time[Fk][Mm]` 对应 `tau^FO_f,m`。

`logistics.F.return_time[Fk][Mm]` 对应 `tau^FR_f,m`。

车辆成本字段对应 `c^FO_f,c^FR_f`。

---

# 17. 两套算例生成逻辑

## 17.1 Generator A：公开 FJSP 扩展

`generate_fjsp_reconfigurable.py` 采用以下原则：

1. FJSP job 映射为 `J1,J2,...`；
2. FJSP operation 映射为 `o11,o12,...`；
3. FJSP machine 映射为 `M1,M2,...`；
4. **原候选机器集合与加工时间严格保留**；
5. 原 job-shop 线性 precedence 不保留；
6. 每个产品重新生成具有不可比节点的 sparse assembly DAG；
7. 每道工序随机赋予一个固定能力配置；
8. 原本可加工该工序的每个岛都会被赋予该配置的支持能力，从而不破坏公开基准的机器可行域；
9. 生成 sequence-dependent reconfiguration matrix；
10. 根据二维布局生成 W/F 物流时间与成本。

因此公开基准承担的是**加工可行域与加工时间基准**，而不是继承原始线性 job-shop 工艺结构。

## 17.2 Generator B：汽车总装语义实例

`generate_automotive_semantic.py` 从汽车总装任务模板出发，为不同产品变型生成：

- 车身定位；
- 高压电池安装；
- 线束安装；
- 仪表台/座舱安装；
- 座椅安装；
- 车门安装；
- 车轮安装；
- 加注；
- ADAS 标定；
- EOL 检测等。

其中工艺约束直接按照装配逻辑生成多分支 DAG，而不是线性工序链。不同语义任务绑定固定能力配置；装配岛随机生成能力支持集合并进行可行性修复，保证每道工序至少具有给定数量的候选岛。

---

# 18. 推荐论文问题表述

本文问题可以表述为：

> **Reconfigurable Capability Island Assembly Scheduling Problem with Dual-Flow Logistics (RCIASP-DL)**。

其核心不是细化可重构设备内部模块，而是研究**能力状态转换如何与装配 DAG 路由和双层物流同步共同影响系统调度**。

与经典 RMC/FJSP 可重构调度相比，本问题增加了两个关键结构：

1. 产品工艺不是固定线性链，而是 DAG，实际拓扑路径会内生改变主工件运输链；
2. 每道工序除等待主工件外，还必须等待独立有限 F-AGV 物料包配送，因此生产与两类物流形成双重同步。

这使得一个 operation-island 决策不仅改变加工时间，还会同时改变：

- 装配岛后续能力切换；
- 产品主工件的实际空间路由；
- W-AGV 运输任务；
- F-AGV 配送目标；
- 多资源同步等待关系。

这也是后续图学习算法应重点利用的结构来源。

---

# 19. 最终建模边界总结

本模型最终固定为：

\[
\boxed{
\begin{aligned}
&\text{DAG topological sequencing}\\
&+\text{assembly-island assignment}\\
&+\text{capability reconfiguration}\\
&+\text{W-AGV workpiece transportation}\\
&+\text{F-AGV component-kit delivery}.
\end{aligned}}
\]

明确不包含：

- module type；
- physical module instance；
- module allocation；
- module retention；
- module relocation；
- A-class module AGV。

这使问题结构与现有可重构制造单元调度文献保持可比，同时将论文的主要新颖性集中在**DAG 装配顺序柔性、能力配置重构与双层物流同步的联合决策**，并为后续结构感知图学习算法提供清晰、可解释的异构关系基础。
