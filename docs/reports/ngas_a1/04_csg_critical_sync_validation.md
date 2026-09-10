# CSG critical synchronization validation

PASS on three tiny schedules and 18 canonical R12 DEVELOPMENT H1 schedules.
The unchanged generalized event DAG contains technological and realized product
precedence, island order, reconfiguration readiness, W empty/loaded travel,
workpiece release, F outbound/return travel, and operation start synchronization.
Realized insertion idle is explicitly represented; it is a schedule-specific constraint,
not a claim that the idle time is unavoidable under another schedule.

NGAS computes earliest and latest activity times on ALL chains to the makespan sink.
Final F return activities without a path to that sink have unconstrained latest times
(JSON null), and are not made critical by an artificial makespan bound.
Tolerance is max(1e-8 schedule time units, 1e-10 * max(1, makespan)).
An edge is critical only when both endpoints have zero slack and its realized
temporal margin is within tolerance. Critical W/F/reconfiguration nodes project to
their associated operations. Score = 4 * critical activity count + incident critical
edge count + distinct critical resource count; ties break by operation ID.

Only operations with explicit critical reasons are called critical. If fewer than k
exist, remaining slots are explicitly marked noncritical padding and ordered by
operation slack then ID. Every selected critical member has stored node/edge reasons.

`outputs/ngas_a1/audit/bank_validation.json` contains full small-schedule records,
per-node earliest/latest/slack, per-edge margins and reasons, and 63 size/state overlap
comparisons against the unchanged legacy completion-descending rule.
The analytic regression fixture separately checks hand-computed OP/RECONFIG/W/F
slack, a branch with positive slack, and an irrelevant final F return.
