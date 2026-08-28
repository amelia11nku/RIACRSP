# Phase 6A Diagnostic Bottleneck Attribution

This attribution is a deterministic diagnostic proxy, not ground-truth causal attribution.

For each pre-action schedule, find every operation whose completion equals the makespan. Read its decoder-maintained `binding_resource` values and map them as follows:

| Decoder binding | Proxy category |
|---|---|
| `PRODUCT` | `PRECEDENCE_SEQUENCE` |
| `ISLAND` | `ISLAND_PROCESSING_LOAD` |
| `ISLAND_CONFIG` | `RECONFIGURATION` |
| `W_AGV` | `W_LOGISTICS` |
| `F_AGV` | `F_LOGISTICS` |

If terminal operations expose bindings mapping to multiple categories, classify the state as `CROSS_RESOURCE_SYNCHRONIZATION`. If no binding is exposed or a binding is unknown, classify it as `MIXED_OR_UNCERTAIN`. `ISLAND_PROCESSING_LOAD` is retained in the schema even when the current decoder represents a terminal island delay as `ISLAND_CONFIG`.

The processing-critical-path proxy starts from makespan-terminal operations and walks backward over zero-gap product and island-processing arcs. The resource-critical-chain proxy marks the terminal operation on every island timeline. W/F chain positions and decoder bindings are logged separately; Phase 6A does not construct a CSG or claim that these proxies prove causality.
