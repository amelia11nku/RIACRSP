# LG_HGA-RIACRSP source-fidelity audit

Status: implementation gate passed for deterministic tests and smoke testing. The offline knowledge base and frozen DTR artifacts must be generated before any formal LG_HGA run.

The source is Hu et al., “A learning-guided hybrid genetic algorithm and multi-neighborhood search for the integrated process planning and scheduling problem with reconfigurable manufacturing cells,” *Robotics and Computer-Integrated Manufacturing* 93 (2025) 102919.

| Source mechanism | Source location | Paper-stated behavior | RIACRSP implementation | Classification | Reason | Unit test |
|---|---|---|---|---|---|---|
| Objective/global replacement | Secs. 3, 4.2, 4.3 | NSGA-II for makespan and total weighted tardiness | Makespan-only elitist generational replacement over parents, genetic offspring, and local offspring | Adapted | RIACRSP has no due dates or tardiness objective; no synthetic objective is introduced | local-size/population test |
| Representation | Sec. 4.1 | `Lp` process plan, `Lm` cell, `Lj` job sequence | Product-DAG topology, island assignment, global operation priority, plus neutral W/F layers | Adapted | RIACRSP has fixed operations and alternative topological orders, not alternative routes | N3/topology tests |
| Initialization | Sec. 4.2 Step 1 | Random population | Forty fully decoded random candidates | Exact mechanism | Same population construction intent | defaults test |
| Global crossover | Sec. 4.3.2 | UX on `Lm`, POX on `Lj`, no `Lp` crossover | UX on islands and W/F; POX on global operation priority | Adapted | W/F do not exist in the source | shared POX/UX tests |
| Global mutation | Sec. 4.3.2 | Rearrange several jobs' plans; switch a few cells; swap a few sequence positions | One topology, island, sequence, and neutral W/F mutation within a `Pm` event | Source-gap assumption | Paper does not state the selection-count distributions or per-layer probability linkage | topology and domain tests |
| Critical path | Sec. 4.4.1 | Randomly choose one when several exist | Seeded choice on the shared generalized event DAG | Adapted | W/F and reconfiguration events affect RIACRSP makespan | shared critical-path tests |
| N1 | Sec. 4.4.1(1) | EDD critical job, move one `Lj` occurrence earlier | `N1_CTU`: critical-tail urgency product and earlier priority insertion | Adapted | RIACRSP benchmark has no due dates | N1/N2 test |
| N2 | Sec. 4.4.1(2) | Earliest-release critical job, move one `Lj` occurrence earlier | `N2_EST`: earliest realized `OperationSchedule.start_time` and earlier priority insertion | Adapted | Static RIACRSP has no nondegenerate release-date field | N1/N2 test |
| N3 | Sec. 4.4.1(3) | Rearrange a random critical job's process plan | Generate and inject a different valid topology of a random critical product | Adapted | Fixed product DAG replaces source route selection | N3 test |
| N4 | Sec. 4.4.1(4) | Reassign several critical stages to the eligible cell with minimum operation count | Reassign one eligible critical operation per proposal to a minimum-count island | Adapted, frozen choice | “Several” has no reported count distribution; one operation is the minimum neighborhood move allowed by the reproduction protocol | N4 test |
| Knowledge target | Eqs. (10)–(11) | `U=0.6T+0.4Cmax`; `R=|BS|/|P|`; twenty runs | `U=Cmax`; strict makespan improvement percentage on 0–100 scale; twenty frozen seeds | Adapted | Tardiness is absent; improvement-rate concept is preserved | improvement-rate and KB-budget tests |
| DTR models | Sec. 4.4.2; Table 6 | One DTR trajectory per neighborhood, predicted by generation | Four deterministic `DecisionTreeRegressor` models using normalized generation only | Exact model family, source-minimal feature assumption | Paper plots generation-dependent trajectories but reports no larger feature vector or hyperparameters | DTR hash round-trip test |
| Online gate | Sec. 4.4.2 Steps 3–4 | Select `argmax Ri`; search only if `Rn > Tls` | Deterministic argmax and strict `>50` gate; no online refit | Exact | Directly reported behavior and tuned threshold | strict-gate tests |
| Local mechanics | Table 7 and Sec. 4.4 | `lsize=5`, `MaxIterNum=5`, `nsize=20` | Best five seeds; twenty round-robin proposals per local iteration; elitist five; repeat five times | Source-gap assumption | The paper defines the parameters but omits their connecting replacement pseudocode | exact local decoder-count test |
| Termination | Secs. 4.2 and 5.3 | `MAXGEN=100` | Maximum 100 generations, stopping earlier at matched `2N` wall-clock budget | Experiment adaptation | Repository formal fairness protocol | runner/config audit |

## Leakage control

Knowledge generation is preregistered on nine balanced `RCIAS-CB1-TRAIN` R01 instances (`3 scales × 3 CF`, RI2/TI2), with twenty seeds. This split is training-only and disjoint from the 45-instance CB1 Core formal suite. Training refuses to proceed unless all 180 generation runs exist and the instance-ID/hash overlap audit is empty. Formal evaluation only loads hash-verified models and never fits them.

## Gate result

The implementation includes the random population, reported parameters, source-analog genetic operations, all four critical-path neighborhoods, offline knowledge generation, four DTRs, percentage `R`, strict threshold, and explicit single-objective/source-gap adaptations. Once the frozen model manifest exists, the online method is eligible for the name `LG_HGA-RIACRSP`.
