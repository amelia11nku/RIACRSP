# CSG-NI 算法项目交接文档：Phase 6G

生成日期：2026-09-01（Asia/Shanghai）  
仓库路径：`/home/liulei/ayx/RI-ACRSP`  
主要 Python 环境：`/home/liulei/miniconda3/envs/gnn311/bin/python`  
原始执行说明：`/home/liulei/下载/Phase6G_Codex_Execution_Instructions.md`

## 1. 当前结论

Phase 6G 在当前环境中可执行的任务已经完成。正式 Gate 为：

```text
status = PERFORMANCE_PASS_SAFETY_HOLD
PHASE6H_RECOMMENDATION = REVISE_CALIBRATION
```

算法性能、最终解可行性、配对统计、墙钟预算、解码效率和 tiny 精确验证均达到预期；但 live 概率校准明显失真，且 Phase 6C TRAIN 到 Phase 6G live 的状态分布漂移被分类为 `HIGH`。因此当前不能直接执行最终 CB1-Core 实验，也不能将 Phase 6G 判定为无条件通过。

最终 Gate：

- `NI_DISABLED_EQUALS_ALNS = TRUE`
- `LIVE_CSGNI_IMPLEMENTED = TRUE`
- `LIVE_CSGNI_FEASIBILITY_RATE = 1.000000`
- `FROZEN_INTERVENTION_RATE = R100`
- `LIVE_INTERVENTION_COVERAGE = 0.295117`
- `LIVE_FALLBACK_RATE = 0.704883`
- `LIVE_CALIBRATION_STABLE = FALSE`
- `LIVE_STATE_DISTRIBUTION_DRIFT = HIGH`
- `CSGNI_MEAN_IMPROVEMENT_VS_H1 = 0.106102`
- `CSGNI_MEAN_IMPROVEMENT_VS_ALNS = 0.011328`
- `CSGNI_MEAN_IMPROVEMENT_VS_GA = 0.033352`
- `CSGNI_WILCOXON_VS_ALNS_P = 0.01367188`
- `CSGNI_BEATS_ALNS_ON_DEV_HOLDOUT = TRUE`
- `CSGNI_BEATS_GA_ON_DEV_HOLDOUT = TRUE`
- `GUROBI_VALIDATION_EXECUTED = PARTIAL`
- `LIVE_OVERHEAD_ACCEPTABLE = TRUE`
- `PHASE6H_RECOMMENDATION = REVISE_CALIBRATION`

权威结论文件：

- `outputs/phase6g/audit/phase6g_gate.json`
- `docs/reports/phase6g_live_csgni_integration_report.md`

## 2. 严格边界

后续对话必须遵守以下边界：

1. 不得自动运行 CB1-Core、CB1-Sensitivity、Legacy-130 或 PPO/在线 RL。
2. 不得修改或覆盖 Phase 6F checkpoint、H1、公共 decoder、独立可行性检查、基线 ALNS、基线 GA、CSG-1.0 或冻结的 `transport_aware` repair 语义。
3. 不得利用已经打开的 DEV-HOLDOUT 主结果重新调参后，再把同一 DEV-HOLDOUT 当作无偏验证集。
4. 不得把 Gurobi incumbent 称为 optimum；只有 `optimality_proven=true` 时才能报告最优间隙。
5. 不得绕过 Gurobi 许可证限制。
6. Phase 6G 主实验的 279 份结果已经冻结，后续校准实验必须输出到新目录，不得覆盖这些结果。
7. 漂移补采是独立诊断任务，不属于 279 份主性能统计。

## 3. 冻结模型和配置

Phase 6F 部署模型：

```text
seed = 660301
checkpoint = outputs/phase6f/training/final_seeds/seed_660301/checkpoint_best.pt
SHA-256 = f1ccceb607b0e453dfb74e7aa7a946616001db8ec08c12dc9900a66c6f165fc7
architecture = C1_H128_L2
```

冻结配置：`configs/phase6g_live_solver.json`

关键约束：

- 初始解：H1
- destroy fraction：0.15
- NI repair：`transport_aware`
- candidate trials：8
- 搜索预算：`2 × N_operations` 秒
- DEV-TUNE seeds：670201、670202、670203
- DEV-HOLDOUT seeds：670301–670310
- tiny seeds：670401、670402、670403
- DEV-TUNE：R01，共 9 个实例
- DEV-HOLDOUT：R02，共 9 个实例
- 候选频率：R20、R50、R100
- 最终冻结频率：R100

环境/拆分记录：

- `outputs/phase6g/environment/phase6g_environment_freeze.json`
- `outputs/phase6g/environment/dev_split.csv`
- `outputs/phase6g/environment/concurrency_protocol_amendment.json`
- `outputs/phase6g/environment/dev_holdout_concurrency_protocol.json`
- `outputs/phase6g/frequency_study/selected_rate_freeze.json`

## 4. 已完成的 Phase 6G 实现

### 4.1 Live CSG-NI

新增的主要实现：

- `rcias_clgri/ni/proposal_bank.py`
- `rcias_clgri/ni/live_policy.py`
- `rcias_clgri/ni/live_inference.py`
- `rcias_clgri/search/csgni.py`
- `rcias_clgri/analysis/phase6g.py`

关键语义：

- CSG-NI 是独立 solver wrapper，没有修改基线 ALNS。
- R0 直接委托冻结 ALNS，从而保证零干预完全一致。
- NI 与 fallback 使用相同 SA acceptance。
- NI 迭代不向 ALNS destroy/repair 权重错误记功。
- proposal、NI repair、acceptance 和 diagnostics 使用隔离 RNG namespace。
- 模型在一个 worker 内常驻 GPU，不在每次干预时重载。
- 只有模型选中的 target set 被 repair 和 decode。
- frozen calibration gate 拒绝时执行真正的普通 ALNS fallback。

### 4.2 通用 Gurobi

新增：

- `rcias_clgri/exact/general_gurobi.py`
- `rcias_clgri/exact/__init__.py` 中的导出
- `tests/test_general_gurobi.py`

该 MILP 支持一般小实例语义，而不是只支持 tiny_03 profile：

- 任意产品 DAG；
- operation-island assignment；
- island sequence 和 sequence-dependent reconfiguration；
- F-AGV warehouse round trip；
- W-AGV pickup/delivery routing；
- same-island 产品相邻操作不产生 W 任务；
- 求解结果通过公共 action replay 和独立 checker。

### 4.3 实验和分析脚本

- `scripts/freeze_phase6g_environment.py`
- `scripts/run_phase6g_integration_regression.py`
- `scripts/run_phase6g_smoke.py`
- `scripts/tune_phase6g_intervention_rate.py`
- `scripts/run_phase6g_frequency_parallel.sh`
- `scripts/run_phase6g_dev_holdout.py`
- `scripts/run_phase6g_dev_holdout_parallel.sh`
- `scripts/analyze_phase6g_live_results.py`
- `scripts/run_phase6g_drift_audit.py`
- `scripts/run_phase6g_drift_audit.sh`
- `scripts/analyze_phase6g_drift.py`
- `scripts/run_phase6g_exact_validation.py`
- `scripts/run_phase6g_small_gurobi.sh`
- `scripts/run_phase6g_tiny_comparison.py`
- `scripts/finalize_phase6g.py`

测试：

- `tests/test_phase6g_live_solver.py`
- `tests/test_general_gurobi.py`

## 5. 零干预与 smoke 结果

零干预回归：

```text
NI_DISABLED_EQUALS_ALNS = TRUE
```

证据：`outputs/phase6g/integration_regression/zero_intervention_regression.json`

已验证一致项包括：

- 初始 makespan；
- destroy/repair 序列；
- candidate makespan；
- accept/reject 序列；
- global-best trajectory；
- 最终 best candidate 和 makespan；
- decoder evaluation count。

真实 GPU smoke：`outputs/phase6g/integration_regression/live_smoke.json`

## 6. DEV-TUNE

DEV-TUNE 已完成 108/108 个有效运行：

```text
ALNS = 27
R20 = 27
R50 = 27
R100 = 27
```

频率汇总：`outputs/phase6g/frequency_study/frequency_study_summary.csv`

主要结果：

| Method | Mean final makespan | Improvement vs paired ALNS | Mean decoder evals |
| --- | ---: | ---: | ---: |
| R100 | 2963.6667 | 0.4391% | 15514.63 |
| R20 | 2978.4444 | 0.3024% | 26291.89 |
| R50 | 2978.9630 | 0.8153% | 20638.33 |

R100 按预注册主规则“最低平均最终 makespan”被选中。R50 的平均相对改善值更高不构成选择规则冲突，因为主指标是绝对平均 makespan，且相对改善是逐配对比例的均值。

### 6.1 资源污染事件

DEV-TUNE 加速期间曾发现一个遗留孤儿进程，造成 15 个 rate 结果存在资源重叠风险。这 15 个结果已移动到隔离目录并从正式统计中排除：

`outputs/phase6g/audit/quarantine_frequency_resource_overlap_20260831/`

随后从零重新运行受影响任务，最终 108 个正式结果均为干净运行。不得把 quarantine 中的 JSON 或 Parquet 合并回正式结果。

## 7. DEV-HOLDOUT

任务总数：279/279 完成。

```text
H1 = 9（每实例一次）
ALNS = 90（9 × 10 seeds）
GA = 90（9 × 10 seeds）
CSGNI-R100 = 90（9 × 10 seeds）
```

所有结果状态均为 `COMPLETE`，所有 schedule 均通过独立可行性检查。CSG-NI 具有 90 份逐迭代 Parquet，共 190,985 行。

完整性审计：`outputs/phase6g/audit/dev_holdout_integrity.json`

方法汇总：

| Method | Mean of instance means | Mean of instance best | Mean runtime (s) | Mean decoder evals |
| --- | ---: | ---: | ---: | ---: |
| H1 | 3406.2222 | 3406.2222 | 0.2060 | 1.00 |
| ALNS | 3063.7556 | 2962.7778 | 247.1162 | 34621.60 |
| GA | 3211.3000 | 3031.8889 | 247.1162 | 33856.02 |
| CSGNI | 3027.6778 | 2930.5556 | 247.1280 | 16972.12 |

配对统计使用每个实例的 10-seed 平均 makespan 作为统计单元：

| Baseline | Mean improvement | 95% bootstrap CI | W/T/L | One-sided Wilcoxon p |
| --- | ---: | --- | --- | ---: |
| H1 | 10.6102% | [6.8856%, 14.2558%] | 9/0/0 | 0.001953125 |
| ALNS | 1.1328% | [0.4127%, 1.8861%] | 7/0/2 | 0.013671875 |
| GA | 3.3352% | [-0.2985%, 7.0115%] | 6/0/3 | 0.048828125 |

对 ALNS 的三个规模组和三个 CF 组平均改善均为正：

- S：0.3024%
- M：2.4280%
- L：0.6678%
- CF1：1.3096%
- CF2：1.4118%
- CF3：0.6769%

两个 CSG-NI vs ALNS 负向实例：

- `CB1_DEV_S_CF3_R02`：-0.6516%
- `CB1_DEV_L_CF3_R02`：-0.0447%

主要结果：

- `outputs/phase6g/dev_holdout/dev_holdout_run_results.csv`
- `outputs/phase6g/dev_holdout/dev_holdout_instance_summary.csv`
- `outputs/phase6g/dev_holdout/dev_holdout_method_summary.csv`
- `outputs/phase6g/statistics/pairwise_statistics.csv`
- `outputs/phase6g/statistics/subgroup_improvement.csv`
- `outputs/phase6g/statistics/paired_instance_means.csv`

## 8. Live intervention 与 fallback

整体数据：

```text
eligible iterations = 190985
interventions = 56363
fallbacks = 134622
intervention coverage = 29.5117%
fallback rate = 70.4883%
```

NI 相比 fallback 的行为：

- 即时改善率：5.4823% vs 2.5947%
- acceptance rate：8.8480% vs 5.5986%
- global-best rate：0.7878% vs 0.2711%
- mean immediate relative utility：-6.9152% vs -9.0587%

平均即时 utility 为负不代表 acceptance 实现错误；Phase 6G 明确保留了允许恶化解以 SA 概率被接受的语义。

证据：`outputs/phase6g/statistics/live_intervention_summary.csv`

## 9. 校准诊断

校准结果：

```text
evaluated interventions = 56363
mean predicted probability = 0.390614
realized positive fraction = 0.054823
probability ECE = 0.335791
mean predicted utility = -0.030419
mean realized utility = -0.069152
utility MAE = 0.047802
utility Spearman r = 0.321423
utility Spearman p ≈ 0
```

解释：

- 模型排序仍有信息量，utility rank association 为显著正向；
- 但概率头在 live 状态上严重过度乐观；
- frozen threshold 很低，最终仍有约 29.5% 的 intervention coverage；
- 当前不能把 frozen offline calibration 视为 live-stable。

结论：`LIVE_CALIBRATION_STABLE = FALSE`

证据：

- `outputs/phase6g/audit/live_calibration_audit.json`
- `outputs/phase6g/statistics/live_calibration_summary.csv`

## 10. 状态漂移补采

主 DEV-HOLDOUT 日志最初没有保存六类 graph-state feature，无法从既有主日志严谨计算 drift。为避免伪造结论，新增了一个与主统计隔离的 drift audit：

```text
instances = 9 DEV-HOLDOUT cells
seed = 670301
rate = R100
budget = full 2N seconds
live states = 20836
Phase6C TRAIN reference states = 60000
```

该补采不会进入 279 份主性能统计。

比较特征：

- mean slack ratio；
- mean W delay ratio；
- mean F delay ratio；
- mean island relative load；
- mean local reconfiguration ratio；
- search progress。

整体分类：`HIGH`

- search progress：LOW；
- 其余五类特征：HIGH。

参考统计采用 Phase 6C TRAIN 的 `target_membership.parquet`，先对 `(state_id, operation_id)` 去重，再聚合到 state-level mean。时间型特征除以 current makespan，漂移指标包括 standardized mean shift、quantile shift 和 PSI。

证据：

- `outputs/phase6g/drift_audit/progress.json`
- `outputs/phase6g/audit/live_state_drift_audit.json`
- `outputs/phase6g/statistics/live_drift_summary.csv`
- `outputs/phase6g/drift_audit/phase6c_training_state_reference.parquet`
- `outputs/phase6g/live_logs/drift_audit/`

## 11. Runtime 与 decoder efficiency

主要结论：

```text
CSG-NI decoder evaluation reduction vs ALNS = 50.9782%
neural decision overhead fraction = 40.8224% of CSG-NI wall time
mean GPU forward = 22.529 ms
mean total neural decision overhead = 47.540 ms
p95 total neural decision overhead = 87.792 ms
```

尽管 neural overhead 占比不低，CSG-NI 在相同墙钟预算内减少了约一半 decoder evaluations，并取得更好的最终 makespan，因此 Gate 中 `LIVE_OVERHEAD_ACCEPTABLE = TRUE`。后续仍应优化 CSG build 和 GPU forward。

证据：`outputs/phase6g/profiling/runtime_profile.csv`

## 12. Exact/Gurobi 验证

冻结协议：`outputs/phase6g/exact_validation/exact_protocol_freeze.json`

### 12.1 tiny

| Instance | Gurobi status | Optimum | H1 | ALNS 3 seeds | GA 3 seeds | CSG-NI 3 seeds |
| --- | --- | ---: | ---: | --- | --- | --- |
| tiny_01 | OPTIMAL | 157 | 157 | 全部 157 | 全部 157 | 全部 157 |
| tiny_03 | OPTIMAL | 36 | 39 | 全部 36 | 全部 36 | 全部 36 |

Gurobi、公共语义 replay 和独立 checker 一致。tiny_01 还由 exact active-schedule BnB 证明为 157。

### 12.2 DEV-Small

预注册的两个最小 DEV-HOLDOUT Small 实例：

- `CB1_DEV_S_CF2_R02`，56 operations；
- `CB1_DEV_S_CF3_R02`，61 operations。

两者均在建模后被当前 Gurobi license 拒绝：

```text
GurobiError: Model too large for size-limited license
```

这属于环境限制，不属于算法或 MILP 语义失败。不得把它们称为已求解，也不能报告 optimum/incumbent/bound。

在 unrestricted Gurobi 环境中的精确命令：

```bash
cd /home/liulei/ayx/RI-ACRSP
/home/liulei/miniconda3/envs/gnn311/bin/python scripts/run_phase6g_exact_validation.py --stage small
```

如果换环境重跑，现有两个错误 JSON 会使 runner 判断任务已存在。应先在获得用户明确授权后，把这两个错误 JSON 移到新的 quarantine/archive 目录，再运行；不要直接删除，不要覆盖 tiny 结果。

证据：

- `outputs/phase6g/gurobi/gurobi_results.csv`
- `outputs/phase6g/exact_validation/exact_solver_comparison.csv`
- `outputs/phase6g/gurobi/runs/tiny/`
- `outputs/phase6g/gurobi/runs/dev_small/`

## 13. 回归验证

最终已执行：

```bash
/home/liulei/miniconda3/envs/gnn311/bin/python -m compileall -q rcias_clgri scripts tests
/home/liulei/miniconda3/envs/gnn311/bin/python -m pytest -q
/home/liulei/miniconda3/envs/gnn311/bin/python scripts/generate_canonical_benchmarks.py --verify-only
/home/liulei/miniconda3/envs/gnn311/bin/python scripts/run_small_validation.py
/home/liulei/miniconda3/envs/gnn311/bin/python scripts/run_native_tiny_validation.py
```

结果：

```text
pytest = 182 passed
canonical benchmark = 130/130 byte-level regeneration verified
small validation = feasible
tiny_03 Gurobi/CP-SAT/exhaustive = 36/36/36
frozen Phase6F checkpoint SHA-256 = MATCH
```

审计：`outputs/phase6g/audit/regression_summary.json`

## 14. 必需产物完整性

Phase 6G 说明书要求的 15 类核心输出均已存在：

- zero intervention regression；
- frequency summary；
- 三份 DEV-HOLDOUT 表；
- intervention/calibration/drift/trajectory；
- exact/Gurobi；
- pairwise statistics；
- runtime profile；
- Gate；
- 最终报告。

10 类图均生成 PNG 和 PDF，共 20 个文件，位于：

`outputs/phase6g/figures/`

## 15. 工作区状态

本阶段大量实现尚未提交。交接时 `git status --short` 的核心状态为：

```text
M  rcias_clgri/exact/__init__.py
?? configs/phase6g_live_solver.json
?? docs/reports/phase6g_live_csgni_integration_report.md
?? rcias_clgri/analysis/phase6g.py
?? rcias_clgri/exact/general_gurobi.py
?? rcias_clgri/ni/live_inference.py
?? rcias_clgri/ni/live_policy.py
?? rcias_clgri/ni/proposal_bank.py
?? rcias_clgri/search/csgni.py
?? scripts/analyze_phase6g_drift.py
?? scripts/analyze_phase6g_live_results.py
?? scripts/finalize_phase6g.py
?? scripts/freeze_phase6g_environment.py
?? scripts/run_phase6g_dev_holdout.py
?? scripts/run_phase6g_dev_holdout_parallel.sh
?? scripts/run_phase6g_drift_audit.py
?? scripts/run_phase6g_drift_audit.sh
?? scripts/run_phase6g_exact_validation.py
?? scripts/run_phase6g_frequency_parallel.sh
?? scripts/run_phase6g_integration_regression.py
?? scripts/run_phase6g_small_gurobi.sh
?? scripts/run_phase6g_smoke.py
?? scripts/run_phase6g_tiny_comparison.py
?? scripts/tune_phase6g_intervention_rate.py
?? tests/test_general_gurobi.py
?? tests/test_phase6g_live_solver.py
```

注意：工作区可能还包含用户先前的其他修改。新对话不得用 `git reset --hard`、`git checkout --` 或类似命令清理工作区。应先重新执行 `git status --short` 和 `git diff --stat`，区分现有用户改动与 Phase 6G 改动。

大部分 `outputs/` 受 `.gitignore` 管理，所以不一定显示在 `git status`，但它们是实验权威证据，不能因未被 Git 跟踪而删除。

## 16. 当前进程状态

交接时 DEV-TUNE、DEV-HOLDOUT、drift audit、tiny exact 和 DEV-Small Gurobi 任务均已返回，没有需要继续等待的长任务。

新对话开始时仍应执行只读检查：

```bash
ps -eo pid,ppid,etime,stat,cmd | rg 'phase6g|gurobi|csgni' | rg -v 'rg ' || true
```

## 17. 下一阶段建议

### 17.1 首要目标

不要直接进入最终 Core。先设计并冻结一个 live calibration revision protocol，目标是：

1. 保留模型权重、CSG-1.0、solver、R100 和主搜索语义；
2. 只修订概率/utility calibration 或 intervention gate；
3. 不使用最终 CB1-Core 标签；
4. 不把已打开的 DEV-HOLDOUT 再当作无偏验证；
5. 建立新的、隔离的 live calibration 和 validation 边界；
6. 在任何新结果出现前冻结数据来源、阈值选择规则、种子、预算和 Gate；
7. 校准修订后重新证明 R0 与 ALNS 完全一致；
8. 重新评估 coverage、fallback、ECE、utility ranking、drift、墙钟开销和最终 makespan；
9. 只有校准/漂移安全 Gate 通过后，才讨论最终 Core。

### 17.2 数据泄漏风险

DEV-HOLDOUT 已经被打开并用于正式 Gate，因此不能在其上选择新阈值后继续声称其结果无偏。可考虑的方案必须先书面冻结，例如：

- 使用 DEV-TUNE 的 live logs 做 calibration，并按 seed/instance 建立内部 calibration/validation 切分；
- 生成或预注册新的非 Core shadow-development replicate，作为校准后的新验证集；
- 使用 cross-fitting 仅作开发诊断，同时明确它不替代新的未见验证集。

具体选择会改变统计有效性，下一对话应先制定方案，不应静默选择。

### 17.3 推荐的校准研究内容

- 检查 frozen probability threshold、predicted-utility threshold、decision-margin threshold 各自造成的 fallback；
- 在 DEV-TUNE live 数据上分别评估 temperature scaling、isotonic 或现有 FrozenCalibrator 同族重拟合，避免扩张到新模型训练；
- 比较 calibration-only 与保守 coverage cap；
- 用可靠性图、ECE/Brier、utility calibration、coverage-risk curve 选择方案；
- 保持 target-bank、repair、acceptance 和 RNG semantics 不变；
- 预注册新的推荐门槛，例如 live ECE、最小 fallback、性能非劣和 feasibility 100%。

### 17.4 不建议立即做的事情

- 不要直接重训 Phase 6F 模型；
- 不要引入 PPO/在线 RL；
- 不要在看完 DEV-HOLDOUT 后手工调 threshold 并重报同一 holdout；
- 不要运行最终 45 Core 或 45 Sensitivity；
- 不要以 tiny 最优恢复替代 live calibration 安全性判断。

## 18. 新对话建议首条提示词

可以直接将以下内容发给新对话：

```text
请先完整阅读：
1. docs/reports/phase6g_project_handoff.md
2. docs/reports/phase6g_live_csgni_integration_report.md
3. outputs/phase6g/audit/phase6g_gate.json
4. /home/liulei/下载/Phase6G_Codex_Execution_Instructions.md

当前 Phase6G Gate 为 PERFORMANCE_PASS_SAFETY_HOLD，推荐 REVISE_CALIBRATION。
不要运行 CB1-Core/CB1-Sensitivity/Legacy/PPO，不要覆盖 279 份 DEV-HOLDOUT 主结果，也不要把已打开的 DEV-HOLDOUT 用于重新选阈值后再次声称无偏。

请先只读检查工作区、权威输出和当前进程，然后以资深算法工程师角度制定一个无数据泄漏的 live calibration revision protocol：明确 calibration/validation 数据边界、冻结项、候选校准方法、coverage-risk 指标、种子、预算、统计单元和通过 Gate。方案确认/冻结后再逐步实现和验证。
```

## 19. 快速索引

### 最终结论

- `outputs/phase6g/audit/phase6g_gate.json`
- `docs/reports/phase6g_live_csgni_integration_report.md`

### DEV-TUNE

- `outputs/phase6g/frequency_study/progress.json`
- `outputs/phase6g/frequency_study/frequency_study_summary.csv`
- `outputs/phase6g/frequency_study/selected_rate_freeze.json`

### DEV-HOLDOUT

- `outputs/phase6g/dev_holdout/progress.json`
- `outputs/phase6g/dev_holdout/dev_holdout_run_results.csv`
- `outputs/phase6g/dev_holdout/dev_holdout_instance_summary.csv`
- `outputs/phase6g/dev_holdout/dev_holdout_method_summary.csv`
- `outputs/phase6g/live_logs/dev_holdout/`

### 诊断

- `outputs/phase6g/statistics/live_intervention_summary.csv`
- `outputs/phase6g/statistics/live_calibration_summary.csv`
- `outputs/phase6g/statistics/live_drift_summary.csv`
- `outputs/phase6g/statistics/trajectory_summary.csv`
- `outputs/phase6g/profiling/runtime_profile.csv`
- `outputs/phase6g/audit/live_calibration_audit.json`
- `outputs/phase6g/audit/live_state_drift_audit.json`

### Exact/Gurobi

- `outputs/phase6g/exact_validation/exact_protocol_freeze.json`
- `outputs/phase6g/exact_validation/exact_solver_comparison.csv`
- `outputs/phase6g/gurobi/gurobi_results.csv`

### 回归与图

- `outputs/phase6g/audit/regression_summary.json`
- `outputs/phase6g/figures/`

## 20. 一句话交接

CSG-NI live solver 已被证明能在 DEV-HOLDOUT 上以 100% 可行率、约一半 decoder evaluations 和统计显著的 1.13% makespan 改善击败 ALNS，并在两个 tiny 算例上稳定恢复精确最优；但 live calibration 严重失真且状态漂移为 HIGH，因此下一阶段必须先完成无泄漏的校准修订和新验证边界设计，不能直接进入最终 Core。
