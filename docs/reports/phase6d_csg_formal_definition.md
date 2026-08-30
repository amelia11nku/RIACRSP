# Phase 6D — Formal Definition of the Critical Synchronization Graph (CSG-1.0)

## 1. Motivation

Phase 6C established that target-set improvement is learnable at the set level, while a direct scalar supervision target for individual operations is not ready. It also showed that useful current-state information spans technological precedence, realized production and resource orders, logistics delays, reconfiguration, and cross-resource synchronization. A flat operation table cannot express all of these relations without either losing topology or manufacturing outcome-derived operation labels.

The Critical Synchronization Graph (CSG) is therefore a deterministic, framework-neutral representation of one complete, feasible, pre-action schedule. It preserves the two distinct information layers required by Neural Improvement (NI): feasible alternatives and the realized causal/synchronization structure of the current schedule. CSG-1.0 contains no neural model, tensorizer, learned importance score, counterfactual outcome, or after-state information.

The authoritative machine-readable contract is [`configs/csg_v1_schema.json`](../../configs/csg_v1_schema.json). The authoritative input contract remains [`phase6c_ni_dataset_contract.md`](phase6c_ni_dataset_contract.md).

## 2. Difference from the constructive-policy graph

The historical Phase 3 graph and CSG have different state units and purposes.

| Dimension | Phase 3 constructive-policy graph | CSG-1.0 |
|---|---|---|
| Purpose | Select the next constructive operation/resource action | Encode a complete current solution for target-set NI ranking |
| Input | Partial constructive schedule | Frozen complete feasible pre-action schedule |
| Node types | `O/J/M/W/F` | `OP/ISLAND/CONFIG/W_AGV/F_AGV/W_EVENT/F_EVENT/RECONF_EVENT` |
| Relations | Constructive feasibility and context | Static alternatives plus realized order, causality, and synchronization |
| Resource chains | Partial decision context | Explicit product, island, W, and F chains |
| Reconfiguration | Implicit readiness/context | Explicit positive-duration transition events with from/to configuration |
| Logistics | Resource availability/action context | Explicit W/F events, event chains, and operation enablement |
| Action interface | One constructive compound action | One shared state graph plus a set of target `OP` nodes |

CSG is implemented in a separate package. No historical graph implementation or policy behavior is changed.

## 3. Formal graph definition

For a reconstructed Phase 6C state (s), define

\[
\mathcal{G}_s = (\mathcal{V}_s, \mathcal{E}_s, X_s, A_s, g_s, m_s),
\]

where:

- \(\mathcal{V}_s = \biguplus_{\tau \in \mathcal{T}_V}\mathcal{V}^{\tau}_s\) is a disjoint union of typed node sets;
- \(\mathcal{E}_s = \biguplus_{r \in \mathcal{T}_E}\mathcal{E}^{r}_s\) is a directed typed multirelation;
- \(X_s\) contains finite numeric node features;
- \(A_s\) contains finite numeric edge features;
- \(g_s\) is the graph-level current-state record;
- \(m_s\) is non-predictive diagnostic metadata.

The node-type set is

\[
\mathcal{T}_V = \{\mathrm{OP},\mathrm{ISLAND},\mathrm{CONFIG},\mathrm{W\_AGV},
\mathrm{F\_AGV},\mathrm{W\_EVENT},\mathrm{F\_EVENT},\mathrm{RECONF\_EVENT}\}.
\]

CSG has two information layers:

1. **Feasible-alternative structure**: technological precedence, eligibility, required configuration, and island capability.
2. **Realized-synchronization structure**: current assignment, realized product/island/resource chains, events, temporal causality, and synchronization.

Typed node keys are identifiers used only for exact lookup, deterministic serialization, and action mapping. Their numeric or lexical values are not predictive features.

## 4. Node types

### 4.1 `OP`

One node exists for every operation. It is the only action-target node type. Features cover current processing and timing, non-negative current slack, frozen Phase 6C criticality proxies, W/F delays, synchronization wait, local reconfiguration contribution, island load, normalized positions in realized chains, event-presence flags, and normalized static degrees.

The field `criticality_proxy` is exactly the current-state Phase 6C descriptor

\[
c_o = \frac{1}{1+\operatorname{slack}(o)},
\]

not an outcome-derived importance label.

### 4.2 `ISLAND`

One node exists for every assembly island. Features describe assigned processing load, relative load, scheduled-operation count, capability coverage, positive-duration reconfiguration count/burden, busy/idle time, and last completion.

### 4.3 `CONFIG`

One node exists for every discrete configuration. Its identity remains categorical. Features are the fractions of islands supporting it and operations requiring it.

### 4.4 `W_AGV` and `F_AGV`

One node exists for each resource. Features summarize realized task count, busy/travel burden, relative load, and last completion. `W_AGV` additionally stores pickup waiting burden.

### 4.5 `W_EVENT`

One node exists for every realized non-same-island workpiece transport. It stores event start/end, normalized duration, empty/loaded travel, pickup wait, chain position, and warehouse/first-transport indicators.

### 4.6 `F_EVENT`

One node exists for every realized component-kit round trip. Under the frozen decoder semantics every operation has one F event. It stores departure, island arrival, return, travel components, duration, and resource-chain position.

### 4.7 `RECONF_EVENT`

One node exists for each strictly positive-duration realized island configuration transition. Zero-duration transitions do not create an event. Features contain start/end/duration, normalized duration, island-chain position, and whether the transition starts from the island's initial configuration.

## 5. Static relations

The canonical static/context relations are:

| Relation | Meaning |
|---|---|
| `OP__PRECEDES__OP` | Technological precedence; it also carries exact schedule temporal features |
| `OP__ELIGIBLE_ON__ISLAND` | Feasible operation-island alternative with processing-time edge features |
| `OP__ASSIGNED_TO__ISLAND` | Current realized assignment |
| `OP__REQUIRES__CONFIG` | Required operation configuration |
| `ISLAND__SUPPORTS__CONFIG` | Island capability |
| `ISLAND__CURRENT_CONFIG__CONFIG` | Configuration after the island's last scheduled operation, or its initial configuration when idle |

Eligibility and assignment remain distinct. The former describes alternatives; the latter describes the current realized schedule.

## 6. Realized schedule relations

The four realized order relations are:

\[
\begin{aligned}
&\mathrm{OP}\xrightarrow{\mathrm{PRODUCT\_NEXT}}\mathrm{OP},\\
&\mathrm{OP}\xrightarrow{\mathrm{ISLAND\_NEXT}}\mathrm{OP},\\
&\mathrm{W\_EVENT}\xrightarrow{\mathrm{W\_NEXT}}\mathrm{W\_EVENT},\\
&\mathrm{F\_EVENT}\xrightarrow{\mathrm{F\_NEXT}}\mathrm{F\_EVENT}.
\end{aligned}
\]

They are exact consecutive pairs in the decoder's product, island, W-resource, and F-resource timelines. For any resource family, the expected number of chain edges is

\[
|E_{\text{next}}| = |V_{\text{event}}| - |\{q:\text{resource chain }q\text{ is nonempty}\}|.
\]

No dependency is inferred merely from temporal overlap.

## 7. Synchronization relations

Every realized logistics or reconfiguration event enables exactly the operation governed by the frozen decoder semantics:

\[
\mathrm{W\_EVENT}\xrightarrow{\mathrm{ENABLES}}\mathrm{OP},\quad
\mathrm{F\_EVENT}\xrightarrow{\mathrm{ENABLES}}\mathrm{OP},\quad
\mathrm{RECONF\_EVENT}\xrightarrow{\mathrm{ENABLES}}\mathrm{OP}.
\]

Resource ownership is represented by `W_EVENT__EXECUTED_BY__W_AGV` and `F_EVENT__EXECUTED_BY__F_AGV`. Reconfiguration location is represented by `RECONF_EVENT__OCCURS_ON__ISLAND`.

Reconfiguration causal context uses:

- `OP__TRIGGERS_RECONF__RECONF_EVENT` when a previous operation exists on the island;
- `RECONF_EVENT__FROM_CONFIG__CONFIG`;
- `RECONF_EVENT__TO_CONFIG__CONFIG`.

Workpiece release causality uses `OP__RELEASES_WORKPIECE_TO__W_EVENT` when a W task has a predecessor operation. A warehouse-origin first transport has no artificial predecessor-operation edge.

## 8. Temporal edge semantics

Every temporal relation stores the same five fields. For an edge \(e=(u,r,v)\), let \(t_u^{\mathrm{end}}\) be the relation-specific source availability time and \(t_v^{\mathrm{start}}\) the relation-specific target demand time. Then

\[
\Delta_e = t_v^{\mathrm{start}} - t_u^{\mathrm{end}}, \qquad
\widehat{\Delta}_e = \frac{\Delta_e}{\max(C_{\max}(s),1)},
\]

and

\[
b_e = \mathbb{1}\{|\Delta_e|\le 10^{-9}\}.
\]

The stored fields are `source_end_time`, `target_start_time`, `temporal_gap`, `normalized_temporal_gap`, and `binding_indicator`. Gaps are never clipped. A feasible decoded schedule must produce \(\Delta_e\ge -10^{-9}\) for every realized causal edge; the 10,000-state validation observed no negative gap.

Relation-specific endpoints include completion-to-start for precedence/product order, completion-to-next-reconfiguration-start for island order, event completion-to-next-event start for W/F chains, event availability-to-operation start for enablement, previous-operation completion-to-reconfiguration start, and predecessor completion-to-W loaded departure for workpiece release.

## 9. Feature definitions

All features are deterministic functions of the current reconstructed instance and schedule. The complete names, semantic types, and per-type dimensions are frozen in the JSON schema. CSG-1.0 contains 31 `OP`, 12 `ISLAND`, 2 `CONFIG`, 7 `W_AGV`, 6 `F_AGV`, 9 `W_EVENT`, 7 `F_EVENT`, and 6 `RECONF_EVENT` features.

The graph-level numeric state is:

\[
g_s^{\mathrm{num}}=(C_{\max}(s),\operatorname{search\_progress}(s)).
\]

The graph-level categorical state contains `search_stage` and the current-state `bottleneck_proxy`. `scale`, `CF_level`, `RI_level`, and `TI_level` are retained only as benchmark diagnostic metadata by the builder and are not node features.

## 10. Normalization

Normalization statistics are computed per current instance/schedule without test-set aggregates or future outcomes:

- absolute current times and temporal gaps: divide by \(\max(C_{\max},1)\);
- processing durations: divide by mean positive eligible processing time;
- W/F travel durations: divide by mean positive current-instance travel time;
- reconfiguration durations: divide by mean positive current-instance reconfiguration time, or 1 when absent;
- counts: divide by the relevant positive operation/resource/configuration count;
- chain positions: \(i/\max(n-1,1)\).

Raw time fields are retained where the schema explicitly requires auditability. Every numeric value must be finite.

## 11. Action projection

For state graph \(\mathcal{G}_s\) and target set \(D\subseteq\mathcal{V}^{\mathrm{OP}}_s\), the action view is

\[
a(s,D)=(\mathcal{G}_s, I_D, M_D, z_D),
\]

where \(I_D\) is the sorted tuple of canonical `OP` node indices, \(M_D\) is a Boolean node mask, and \(z_D\) contains only permitted origin-rule metadata. The base graph and its hash are identical for every target set belonging to the same state. Duplicate, absent, or label-bearing target specifications are rejected.

Counterfactual makespan, improvement, rank, regret, repair seed, and repair outcome are forbidden action metadata.

## 12. Causal subgraph

Every edge belongs to one of `STATIC_CONTEXT`, `REALIZED_RESOURCE_ORDER`, `TEMPORAL_CAUSAL`, `SYNCHRONIZATION`, or reserved `DERIVED_REVERSE`. CSG-1.0 defines no reverse relations.

The causal subgraph is

\[
\mathcal{G}^{\mathrm{causal}}_s=(\mathcal{V}_s,
\{e\in\mathcal{E}_s:\operatorname{class}(e)\in
\{\mathrm{REALIZED\_RESOURCE\_ORDER},\mathrm{TEMPORAL\_CAUSAL},\mathrm{SYNCHRONIZATION}\}\}).
\]

It must be a directed acyclic graph. Static alternative/context relations are excluded from the acyclicity claim. Kahn topological validation is applied to every sampled graph and also returns a deterministic causal-depth diagnostic.

## 13. Complexity

Let \(O\), \(P\), \(L\), \(K\), \(E_W\), \(E_F\), and \(E_R\) denote operations, precedence edges, eligibility edges, capability edges, W events, F events, and positive reconfiguration events. Construction is

\[
O(|O|+|P|+|L|+|K|+|E_W|+|E_F|+|E_R|),
\]

plus linear passes over product/island/resource chains. Product, island, and event positions and configuration counts are pre-indexed; there is no all-operation-pairs scan. Canonical sorting adds the usual \(O(n\log n)\) serialization overhead within typed collections.

On 30 representative states, mean construction time was 23.2 ms (S), 55.3 ms (M), and 76.9 ms (L); mean graph sizes were 203/908, 450/2,136, and 608/3,046 nodes/edges respectively.

## 14. Information-leakage boundary

CSG construction may read only the frozen Phase 6C current state and its deterministic reconstruction. It never reads a later trajectory row, after-state, counterfactual schedule, repair result, outcome aggregate, rank, regret, or future best. Raw operation, island, configuration, and resource identifiers are node keys only. No numeric ID encoding is exposed as a feature.

Graph diagnostics such as binding-edge count, causal depth, fan-in, or zero-gap ratio are current-state structural descriptors. They are validation outputs, not supervision labels and not automatically promoted into the feature schema.

## 15. Relationship to the Phase 6C NI dataset contract

The builder calls `rcias_clgri.data.phase6c.reconstruct_state`; it does not define a competing reconstruction path. Each selected dataset record follows:

```text
Phase6C state record
→ reconstruct_state
→ deterministic feasible schedule
→ build CSG-1.0
→ validate exact structure/timing
→ project one or more Phase6C target sets onto OP nodes
```

The frozen Phase 6C dataset hash is `695307ac6193ecbbeb0f73e81a94ea20672ffd47aa9419bb37610be9d3161437`. CSG-1.0 preserves all 13 evidence-required signals plus current assignment, configuration context, current makespan, W-release causality, and explicit reconfiguration-transition context. The formal mapping and redundancy classification are stored in `outputs/phase6d/information_audit/`.
