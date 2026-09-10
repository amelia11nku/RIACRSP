# A1.3R OOF representation comparison

All variants use the same 72 frozen R12 states, 6,465 joint actions, three
training seeds and three held-instance folds. C1 and R1 use identical fixed 60
epochs, joint loss, revised features, pooling and action heads. C0 is immutable
historical output evaluated against the same labels and folds.

| Variant | Spearman | Pair accuracy | NDCG@1 | NDCG@3 | NDCG@5 | Selected advantage | Top-1 regret | Uniform regret | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.357974 | 0.812006 | 0.494255 | 0.523178 | 0.545011 | 0.018811 | 0.022403 | 0.040481 | 0.084660 |
| C1 | 0.350615 | 0.808400 | 0.459704 | 0.498847 | 0.520820 | 0.016320 | 0.024894 | 0.040481 | 0.085548 |
| R1 | 0.360814 | 0.812919 | 0.478086 | 0.521825 | 0.553702 | 0.017100 | 0.024114 | 0.040481 | 0.087145 |

C0, C1 and R1 each pass all three seed gates. Relative to C1, R1 changes
Spearman by +0.010199, material-pair accuracy by +0.004520, selected advantage
by +0.000781, and top-1 regret by -0.000781. These changes are below the
predeclared material thresholds of 0.020, 0.010, 0.002 and 0.002.

Full seed/state metrics, S/M/L and CF1/CF2/CF3 subgroups, selected
beats-fallback probabilities/frequencies, parameter counts, memory and
throughput are retained in `representation_comparison.json`. The independent
completion audit recalculated all headline metrics directly from the frozen
replicate labels and original OOF predictions and reproduced the saved values.
