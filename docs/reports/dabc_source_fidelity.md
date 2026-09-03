# DABC-RIACRSP source-fidelity audit

Status: implementation gate passed for deterministic tests and smoke testing. No formal benchmark result is reported here.

The source is Li et al., “A Dual-Space Artificial Bee Colony Algorithm Integrating Configuration-Coupled Heterogeneous Disjunctive Graph for Scheduling Problem in Reconfigurable Manufacturing Systems,” *IEEE TSMC: Systems* 55(12), 2025. The RIACRSP implementation keeps the source makespan objective while adding the repository's fixed product-DAG and W/F logistics decisions.

| Source mechanism | Source location | Paper-stated behavior | RIACRSP implementation | Classification | Reason | Unit test |
|---|---|---|---|---|---|---|
| Objective | Abstract; Sec. III | Minimize makespan | Common decoder makespan only | Exact objective, adapted problem | RIACRSP adds logistics constraints but no extra objective | `test_dabc_decoder_count_trace_and_restart_are_exact` |
| OS/CS representation | Sec. V-B | Operation order plus machine/configuration selection | Operation priority plus island, W, and F assignment layers; required configuration is fixed by the instance | Adapted | RIACRSP has a general product DAG and explicit logistics genes | shared operator tests |
| CHDG | Sec. IV-A | Operation, machine/configuration, and reconfiguration graph | Generalized realized event DAG including reconfiguration, operations, W/F travel, and resource order | Adapted | Logistics events are necessary for exact RIACRSP makespan | `test_generalized_event_dag_is_exact_and_acyclic` |
| F/R recurrences | Eqs. (1)–(3) | Head/tail recurrences with adjacent reconfiguration time | Exact recurrence on an explicit operation/island RMSSP projection | Exact formula on projection | W/F events are excluded only from the theorem projection | `test_source_projection_f_and_r_recurrences_match_paper` |
| Theorems 1–4 | Sec. IV-B | Sufficient insertion-feasibility inequalities | Exact strict inequalities on the RMSSP projection; independent full product/resource reachability is authoritative | Exact formula, adapted authority | Source conditions are sufficient and omit RIACRSP logistics/product-DAG structure | `test_source_theorems_are_exact_shadow_audits` |
| Theorems 5–10 | Sec. IV-C | Reconfiguration-time inequalities identify nonimproving same-block insertions | All six formulas implemented and logged in `shadow` mode | Exact formula, adapted transfer | A source “nonimproving” proof does not include W/F rescheduling effects | `test_source_theorems_are_exact_shadow_audits` |
| Hierarchical crossover | Sec. V-C | Mate from first-level population; POX on OS and UX on CS; retain best of current and two children | Stable top 20%, POX, neutral UX on all assignment layers, best-of-three | Adapted | Top-20% is the frozen source setting; W/F UX is neutral extension | DABC default and shared POX/UX tests |
| Self-exploration | Sec. V-C | Rate 0.1; equal OS/CS split; insertion or one feasible assignment change; direct replacement | Same; CS analog selects one island/W/F layer and changes at most one gene | Adapted | Three assignment layers replace source CS | self-exploration tests |
| CNS1 | Sec. V-D | Random critical-block operation, enumerate feasible positions in the block, return best | Same over generalized critical island blocks; decode only representable projections | Adapted | Encoding-to-schedule mapping is not one-to-one under the ready-operation decoder | DABC decoder-count and round-trip tests |
| CNS2 | Sec. V-D | Random alternate capable machine and exhaustive feasible insertion | Random alternate eligible island and exhaustive target-timeline insertion | Adapted | Island is the RIACRSP machine analog | DABC decoder-count tests |
| Onlooker/restart parameters | Algorithm 2; Sec. V-A; Sec. VI-B | `ns=2`, `mn=40`, ten self-explorations on restart; `ps=120` | Same defaults | Exact | Directly reported settings | `test_dabc_source_defaults_are_frozen` |
| Termination | Source experiments | Source experimental termination protocol | Matched wall-clock budget `2N`, with optional iteration cap only for tests | Experiment adaptation | Repository comparison protocol requires equal time | runner/config audit |

## Transfer limitation

The source clipping proofs compare processing and reconfiguration paths in RMSSP. Moving an operation in RIACRSP can also change workpiece transport, kit delivery, and their vehicle queues. Therefore `source_clipping_mode="shadow"` is mandatory: theorem hits are counted, every otherwise reachable proposal is still decoded, and any source-clipped proposal that improves the RIACRSP objective is recorded as a shadow false clip. Calling these formulas unavailable or using them as hard pruning would both be inaccurate.

## Gate result

The implementation includes the top-20% hierarchy, 0.1 self-exploration, POX/UX, best-of-three replacement, equal self-exploration split, CNS1/CNS2, `ns=2`, `mn=40`, ten-step restart, Theorems 1–10, and explicit shadow transfer. It is eligible for the name `DABC-RIACRSP`.
