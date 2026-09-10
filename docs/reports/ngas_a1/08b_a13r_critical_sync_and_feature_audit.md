# A1.3R critical synchronization and feature audit

Status: **PASS**.

The revised analysis uses CPM over the complete realized generalized CHDG and
represents the union of all zero-slack chains that reach the makespan sink.
Across 72 frozen R12 development states, 8,453 critical
event nodes independently satisfied the longest-path identity within the frozen
tolerance. 144 activities without a causal
path to the sink retained undefined slack and were never encoded as zero slack.

Every nonboundary event was mapped deterministically to OP, W_EVENT, F_EVENT,
or RECONF_EVENT, with documented aggregation/projection for W empty/loaded, F
outbound/return, zero-duration reconfiguration, and realized idle. The feature
schema explicitly separates validity, zero slack, critical degree, all seven
relation categories, active margin, participation, and unreachable fraction.
The heuristic operation score is not used as the criticality definition.

The cache rebuild ran no continuation rollout. All 72 states, 6,465 actions and
58,185 replicate-label vectors equal the historical compact cache exactly.
R13/R14 remained locked; Gurobi was not invoked.
