# Phase 2D — RT-HGT and Autoregressive Policy Foundation

## Tensor and encoder structure

The tensorizer preserves five node types (`O`, `J`, `M`, `W`, `F`) and 13
directed relation types covering precedence, membership, eligibility, spatial
relations, W/F service relations, and realized product/machine order. Feature
schemas are checked at tensorization time.

The RT-HGT encoder uses:

- type-specific input, query, key, value, and output projections;
- relation-specific key and message transformations;
- edge-feature attention bias and edge-feature message gates;
- multi-head incoming-edge attention;
- residual connections, layer normalization, and feed-forward blocks.

The central default is 128 dimensions, four heads, and two layers.

## Policy and value heads

The policy is factored exactly as
`P(o|s) P(m|o,s) P(w|o,m,s) P(f|o,m,w,s)`. Each head receives graph
embeddings and the corresponding candidate-level numeric features. It scores
only candidates admitted by the current hard mask; no flattened
`O x M x W x F` head exists. Same-island workpiece movement exposes only the
learned `NONE` W candidate. A graph-pooled scalar value head is included for
future work but is not trained in this phase.

## Verification

- Typed tensor construction, encoder forward pass, imitation losses, and full
  backward propagation are tested.
- Operation/island/W/F hard masks and same-island `NONE` behavior are tested.
- The repository audit found no duplicate or unreferenced implementation.
- Full regression after audit: **42 passed**.

This phase implements no PPO, multiobjective preference policy, critical
synchronization graph, or neural destroy/repair component.
