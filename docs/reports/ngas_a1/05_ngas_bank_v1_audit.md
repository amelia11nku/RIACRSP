# NGAS bank v1 audit

PASS: all 18 canonical R12 instances yield distinct sizes at fractions .08/.15/.22.
Use floor(n*f + .5), lower bound min(2,n), upper bound n. Tiny-instance collisions
are allowed and explicit; joint IDs retain size semantics. Historical effective
15% counts and all new cardinalities are recorded in the audit JSON.

The full bank generates exactly 24 rules per size (72 before cross-size action
expansion), then deduplicates exact operation sets within size. It is never a top-k
shortlist. Inherited diversity retains operator, related-variant, random-control,
local-perturbation and structured-neighbor families. The legacy critical proposal is
replaced with `csg_critical_sync`; `near_low_slack` is explicitly renamed
`near_legacy_completion_tail` because it retains the old completion-tail proxy.

All original rule/family/operator memberships are sorted and retained; online features
use their multi-hot union and count, with no first-origin field and no outcome fields.
Stable IDs depend on version, state, size and canonical operation set. Repair is part
of a separate stable joint action ID. All five inherited repair operators remain.
NGAS passes a sorted tuple to the shared repair primitive, eliminating set-order
dependence without changing any frozen comparator code.

There are 60 comparisons with Jaccard overlap <= .5. Exact sets,
critical reasons, duplicates and padding are in `outputs/ngas_a1/audit/bank_validation.json`.
