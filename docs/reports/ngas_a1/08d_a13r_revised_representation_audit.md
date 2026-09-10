# A1.3R revised representation audit

Status: **PASS**.

C1 preserves compact relation-aware mean aggregation while using the revised
typed CPM feature schema. R1 implements type-specific input and Q/K/V
transformations, multi-head relation-specific attention, edge contributions,
residual connections, feed-forward blocks, and per-type normalization. Both
models share type-balanced mean/max/critical pooling and the batched target
mean/max, critical-overlap, graph-boundary, size and repair representation.

The largest cached state (722 nodes,
90 joint actions) required one encoder call for
all actions. CUDA FP32 outputs and gradients were finite and repeated inference
was bitwise equal. C1 has 261,762 parameters; R1 has
1,457,730. AMP, TF32 and candidate-wise graph forward
passes are absent.
