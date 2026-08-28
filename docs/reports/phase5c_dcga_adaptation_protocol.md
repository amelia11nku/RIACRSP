# Phase 5C DCGA adaptation protocol

## Fidelity decision

`DCGA_FIDELITY = FAITHFUL_ADAPTATION`

Primary source: X. Han et al., “A dual population collaborative genetic algorithm for solving flexible job shop scheduling problem with AGV,” *Swarm and Evolutionary Computation* 86 (2024), 101538, DOI 10.1016/j.swevo.2024.101538. The complete 15-page publisher PDF supplied by the user was inspected before revising the implementation.

| Original DCGA component | Original paper definition | RIACRSP implementation | Reason adaptation is necessary |
|---|---|---|---|
| Representation | Integer OS and MS strings, each of length `No`; OS is occurrence-based job order and MS indexes an optional machine. No AGV string. | Operation-priority permutation and aligned eligible-island assignment. W/F assignments are ignored by DCGA evaluation. | Products have DAGs rather than chains; ready-set filtering converts priority into a feasible topological order. Islands replace machines. |
| Population structure | Two independent sub-populations; each uses a different decoder. | Two populations of 500, evaluated by Decoding1 and Decoding2 respectively. | Preserved. |
| Initialization | Random chromosomes using the stated encoding/decoding. | Random operation priorities and feasible islands; AGVs are derived during decoding. | Direct structural mapping. |
| Selection | Binary tournament plus two-parent elites copied directly. | Binary tournament and exactly two elites per sub-population. | Preserved. |
| OS crossover | POX: randomly partition jobs, retain group positions from one parent, fill with the other. | Products are partitioned; product-operation positions are retained and filled exactly by POX. | Product is the RIACRSP analogue of a job. |
| MS crossover | Uniform crossover with one shared binary mask producing reciprocal offspring. | Uniform island-assignment crossover with reciprocal offspring. | Islands replace machines. |
| OS mutation | Randomly choose Swap, Insert, or Inversion. | Same three equiprobable mutations. | Ready-set decoding guarantees DAG feasibility. |
| MS mutation | Reassign one gene to another optional machine. | Reassign one operation to an eligible island. | Direct mapping. |
| Decoding1 | Choose the AGV delivering earliest; if tied, use a fixed AGV. | W and F fleets are independently minimized by arrival time; deterministic fleet order breaks ties. | RCIAS has two synchronized fleets rather than one. |
| Decoding2 | Choose the AGV delivering earliest; if tied, choose minimum accumulated idle-plus-loaded transport time. | W/F earliest arrivals are primary; accumulated W empty+loaded or F outbound+return time breaks ties. | F round trips are the mathematical analogue of empty+loaded occupation. |
| Diversity check | Every `Nt=300` generations, if makespans match and MS similarity is at least 80%, regenerate one individual. | Same interval, equality test, island-string similarity, and random regeneration. | Direct mapping. |
| Collaboration | Repeat `Np` times each generation: binary-tournament one parent per population, cross-population POX, replace each selected parent only if its corresponding child is better. | Same `Np` trials, cross-population POX, decoder-specific child evaluation, strict improving replacement. | Direct mapping. |
| Parameters | Orthogonal study selects `Nt=300`, `Np=500`, `Pc=0.9`, `Pm=0.15`. | Exact values frozen in `configs/phase5c_dcga.json`. | Preserved without retuning on Phase 5C outcomes. |
| Stopping condition | Wall clock `2No` seconds in experiments. | Wall clock `2 × N_operations` seconds. | Exact match. |
| Local improvement | None stated. | None. | Preserved. |

The adaptation changes only the additional RCIAS structure: DAG precedence, reconfigurable islands, and separate W/F logistics. All final schedules still pass through the same frozen `RCIASConstructionEnv` and independent checker used by GA and ALNS.
