# Phase 6D — CSG-1.0 Validation Report

## 1. Executive conclusion

**CSG-1.0 passes the Phase 6D structural acceptance gate.** The framework-neutral builder was validated on 10,000 balanced Phase 6C states with 10,000 passes and zero failures. It also projected 236,254 real target sets containing 4,297,000 target-operation memberships without an unmappable operation or same-state graph change. All temporal gaps were consistent and non-negative, every causal subgraph was acyclic, identifier permutation equivalence passed, and S/M/L construction remained practical.

The final recommendation is:

```text
PHASE6E_RECOMMENDATION = PROCEED_TO_SUPERVISED_NI_MODEL
```

This recommends only the next supervised set-level NI experiment. It is not evidence that a particular GNN architecture will improve the solver, and no neural tensorizer or model was implemented in Phase 6D.

## 2. Frozen boundary

The Phase 6C dataset was verified complete before CSG implementation:

- state count: 100,000;
- target-set count: 2,362,722;
- schema/label generation: `phase6c-v1`;
- dataset freeze hash: `695307ac6193ecbbeb0f73e81a94ea20672ffd47aa9419bb37610be9d3161437`;
- Phase 6C completion and counterfactual-integrity gates: passed.

No H1, ALNS, decoder, feasibility checker, benchmark, Phase 3 graph, Phase 6A/6B/6C artifact, or historical policy implementation was changed. CSG is a new package under `rcias_clgri/csg/` and reuses `rcias_clgri.data.phase6c.reconstruct_state`.

## 3. Implementation overview

The implementation contains:

- machine-readable schema: `configs/csg_v1_schema.json`;
- typed immutable records and schema validation: `schema.py`;
- leakage-safe current-state features and per-instance normalization: `features.py`;
- unclipped temporal features: `temporal.py`;
- deterministic complete-schedule construction: `builder.py`;
- action-independent target-set projection: `actions.py`;
- canonical serialization, SHA-256 hashing, and table export: `serialize.py`;
- exact relation, timing, event, DAG, and leakage validation: `validate.py`;
- structural summaries and Graphviz neighborhood export: `diagnostics.py`.

CSG-1.0 defines 8 node types, 20 forward relation types, and 4 instantiated edge classes. Reverse relations are deliberately absent.

## 4. Dataset/state sampling

The deterministic seed `667006004` selected 10,000 states by round-robin sampling over the joint cells of split, Scale, CF, RI, TI, and search stage. Only the frozen `TRAIN_VALIDATION` and `TRAIN_INTERNAL_HOLDOUT` splits were used.

| Factor | Counts |
|---|---|
| Split | validation 4,995; internal holdout 5,005 |
| Scale | S 3,338; M 3,326; L 3,336 |
| CF | CF1 3,328; CF2 3,330; CF3 3,342 |
| RI | RI1 3,334; RI2 3,343; RI3 3,323 |
| TI | TI1 3,341; TI2 3,334; TI3 3,325 |
| Search stage | 2,000; 1,997; 1,998; 2,002; 2,003 across the five consecutive stages |

The selected manifest and resumable per-state progress log are stored under `outputs/phase6d/validation/`.

## 5. Determinism

Every sampled state was reconstructed once and built twice from the same authoritative reconstructed object. Canonical JSON and graph SHA-256 were identical for all 10,000 pairs. Typed nodes and edges are sorted by canonical keys, feature dictionaries are sorted during serialization, and non-finite numeric values are rejected.

The deterministic-build gate passed for 10,000/10,000 states.

## 6. Information preservation

The formal audit maps all 13 frozen Phase 6C requirements to explicit CSG locations and validation tests. Five additional correctness-critical items—current makespan, assignment, configuration context, W-release causality, and transition context—are also explicit. All 18 mappings passed.

No predictive feature name matched a counterfactual, improvement, rank, regret, future-best, repair-outcome, or numeric raw-ID field. Node keys remain identifiers for joins and action mapping only. The audit is in `outputs/phase6d/information_audit/information_preservation_audit.csv`.

## 7. Temporal consistency

Temporal validation checked source time, target time, raw gap, normalized gap, and binding indicator for every temporal edge. Across the 10,000-state sample:

- negative temporal gaps: **0**;
- all normalized gaps equaled raw gap divided by `max(current_makespan, 1)`;
- all binding indicators equaled `abs(gap) <= 1e-9`;
- no raw gap was clipped.

The largest observed gap was 11,058 time units on a technological precedence edge. Zero-gap resource-chain counts matched decoder serialization: island, W, and F consecutive-resource edges were all binding under the current no-idle insertion semantics, while enablement relations retained informative positive waits.

## 8. Resource-chain completeness

Exact set equality was checked independently for every product, island, W resource, and F resource chain. Aggregate realized relation counts were:

| Relation | Edges |
|---|---:|
| `PRODUCT_NEXT` | 1,053,817 |
| `ISLAND_NEXT` | 1,100,968 |
| `W_NEXT` | 990,657 |
| `F_NEXT` | 1,183,803 |

For W and F, the number of next edges equaled total events minus nonempty resource chains in every graph. No exception was observed.

## 9. Synchronization completeness

The validator compared graph relations with decoder records, not with inferred temporal overlap. Aggregate event and enablement counts were:

| Item | Count |
|---|---:|
| W events / W enablement edges | 1,010,657 |
| F events / F enablement edges | 1,203,803 |
| Positive reconfiguration events / enablement edges | 657,725 |
| W predecessor release edges | 860,671 |
| Prior-operation reconfiguration trigger edges | 605,252 |

Every W/F event had exactly one correct resource owner and enabled operation. Every positive reconfiguration had the correct island, destination operation, source configuration, and target configuration. First-on-island reconfigurations correctly omitted an artificial prior-operation trigger.

## 10. Causal-subgraph validity

For each graph, Kahn topological validation was applied to the union of `REALIZED_RESOURCE_ORDER`, `TEMPORAL_CAUSAL`, and `SYNCHRONIZATION` edges. `STATIC_CONTEXT` was excluded. All 10,000 causal subgraphs were DAGs. Causal depth is retained only as a deterministic diagnostic and is not a learned node label.

## 11. Permutation / identifier invariance

The identifier test renamed and permuted operation, product, island, configuration, W-resource, and F-resource identifiers in `tiny_01`, rebuilt the instance and schedule, and constructed both graphs independently. After applying the explicit typed-node bijection—including event keys—the two graphs had equal numeric features and typed relations. This checks semantic equivariance rather than incorrectly requiring a raw hash to remain unchanged after key renaming.

Canonical repeated-build stability and identifier permutation equivalence both passed.

## 12. Same-state action invariance

All 236,254 real Phase 6C target sets in the sample were projected onto their shared state graphs. In total, 4,297,000 targeted operation memberships were mapped. Every projected view retained the original graph hash; no target-specific feature or edge was injected. Duplicate operations, missing operations, and forbidden label metadata are rejected by the action interface.

## 13. Human-audited examples

Seven independently validated examples were exported:

1. one Small state;
2. one Medium state;
3. one Large state;
4. one reconfiguration-pressure state;
5. one W-logistics state;
6. one F-logistics state;
7. one cross-resource-synchronization state.

Each directory contains typed node/edge CSVs, canonical graph JSON, a timeline reference, relation summary, graph diagnostics, and a focused Graphviz neighborhood. All seven passed exact validation. The example manifest is `outputs/phase6d/examples/example_manifest.csv`.

## 14. Complexity and memory

Thirty validation-split states (10 per scale) were profiled with traced Python allocations.

| Scale | Mean operations | Mean nodes | Mean edges | Mean build | Mean validation | Mean serialized size | Maximum traced peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| S | 61 | 203 | 908 | 23.2 ms | 3.7 ms | 234 KB | 2.70 MB |
| M | 130 | 450 | 2,136 | 55.3 ms | 8.7 ms | 533 KB | 6.21 MB |
| L | 167 | 608 | 3,046 | 76.9 ms | 13.0 ms | 727 KB | 7.70 MB |

The implementation pre-indexes product/island/event positions and configuration counts. Construction traverses operations, precedence, eligibility, capabilities, realized events, and chains without an all-operation-pairs scan. The empirical linear explanatory model obtained \(R^2=0.951\); its individual coefficients are diagnostic only because structural predictors are correlated.

The final stricter 10,000-state run completed at an effective 63.3 states/s with eight workers. Its canonical per-graph payloads would total 5.29 GB if stored individually; Phase 6D deliberately stores validation summaries and selected examples rather than duplicating the frozen dataset as a second graph corpus.

## 15. Redundancy analysis

The audit retained operation-local W/F/reconfiguration descriptors alongside explicit events because Phase 6C directly requires the local signals while events preserve causal topology. It retained static precedence and realized product order because they differ on branching products. Assignment plus chain relations and event ownership plus next relations are retained mainly for exact validation and unambiguous downstream message passing.

Numeric raw-ID encodings and mechanically derived reverse relations were removed. Reverse edges may be introduced only in a later tensorizer as explicitly classified derived relations, without changing the CSG-1.0 canonical graph.

## 16. Known limitations

- CSG-1.0 represents complete feasible schedules; it is not a replacement for the Phase 3 partial constructive graph.
- The graph is structurally validated but has not yet been tested through a learned encoder. Predictive utility remains a Phase 6E question.
- Current resource timelines often have zero-gap consecutive tasks because of frozen decoder insertion semantics; this is represented faithfully rather than artificially perturbed.
- Raw identifiers remain in audit tables and graph keys for exact reconstruction. They must remain lookup-only in any neural tensorizer.
- Benchmark factors are diagnostic metadata, not universally available deployment features.
- No reverse-relation expansion, batching, PyG/HGT tensorization, or memory-optimized graph store is part of this phase.

## 17. Phase 6E recommendation

All structural acceptance conditions are satisfied, so Phase 6E should proceed to the first supervised NI model:

```text
CSG state encoding
+ target-set action encoding
→ target-set improvement classification/ranking
```

The frozen experiment boundary should retain `transport_aware` repair and destroy size 0.15. Model selection must evaluate genuine held-out predictive and solver value; the Phase 6D structural pass is not a substitute for that evidence.

## 18. Reproducibility checklist

- [x] CSG-1.0 schema is machine-readable.
- [x] Phase 6C `reconstruct_state` is reused.
- [x] 10,000-state balanced manifest is stored.
- [x] Validation is resumable through `validation_progress.jsonl` and `--resume`.
- [x] Exact node, relation, event, chain, temporal, synchronization, and DAG checks are stored.
- [x] Real target-set operation mapping and same-state graph invariance are checked.
- [x] Identifier permutation is tested.
- [x] Seven human-audited examples are exported.
- [x] S/M/L timing, serialized size, and traced memory are profiled.
- [x] Six explanatory figures are generated as PNG and PDF.
- [x] Full test suite passes: **149 passed**.
- [x] Canonical, tiny, CB1, CB1-TRAIN, Phase 6A, and Phase 6C regressions pass.
- [x] Historical graph sources are unchanged.
- [x] No neural framework is imported by the CSG package.

Primary artifacts:

- `outputs/phase6d/validation/csg_validation_summary.json`;
- `outputs/phase6d/information_audit/information_audit_summary.json`;
- `outputs/phase6d/profiling/profiling_summary.csv`;
- `outputs/phase6d/examples/example_manifest.csv`;
- `outputs/phase6d/audit/regression_summary.json`;
- `outputs/phase6d/audit/repository_audit.json`.
