#!/usr/bin/env python3
"""Validate full-bank semantics on tiny schedules and canonical R12 H1 states."""
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rcias_clgri.data.loader import load_instance
from rcias_clgri.heuristic.dispatching import solve_dispatching
from rcias_clgri.search.alns import _destroy
from rcias_clgri.search.common import candidate_from_actions, decode_candidate
from rcias_ngas.actions.destroy_size import SIZE_FRACTIONS, destroy_count
from rcias_ngas.bank.ngas_bank_v1 import build_bank
from rcias_ngas.csg.critical_sync import critical_sync
from rcias_ngas.evaluation.bks import write_immutable
from rcias_ngas.rng import RNGStreams
from scripts.audit_ngas_starting_state import digest


def main():
    manifest = json.loads((ROOT / 'outputs/frozen_2o_baselines/instance_manifest.json').read_text())
    specs = [{'path': f'instances/tiny/tiny_{i:02d}.json'} for i in range(1, 4)]
    specs += [{**row, 'path': 'instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14/' + row['relative_path']} for row in manifest['instances']]
    rows, examples = [], []
    for spec in specs:
        instance = load_instance(ROOT / spec['path'])
        if 'sha256' in spec:
            assert digest(spec['path']) == spec['sha256']
        h1 = solve_dispatching(instance, 'H1')
        current = decode_candidate(instance, candidate_from_actions(instance, h1.actions))
        analysis = critical_sync(instance, current)
        for size in SIZE_FRACTIONS:
            bank = build_bank(instance, current, 'H1:' + instance.instance_id, size,
                              RNGStreams(instance.instance_id, 746101), analysis)
            count = destroy_count(instance.num_operations, size)
            old = _destroy(instance, current, 'critical', count, random.Random(0))
            new = next(p.destroyed_operations for p in bank.proposals if p.origin_rule == 'csg_critical_sync')
            assert bank.requested_count == 24
            assert all(len(t.operations) == count for t in bank.targets)
            assert all(analysis.operation_reasons[op] for op in bank.critical_operations)
            rows.append({
                'instance_id': instance.instance_id, 'num_operations': instance.num_operations,
                'size': size, 'count': count,
                'legacy_effective_count': min(instance.num_operations, max(2, round(.15 * instance.num_operations))),
                'requested_rules': bank.requested_count, 'unique_targets': len(bank.targets),
                'duplicate_rules': bank.duplicate_count, 'critical_members': list(bank.critical_operations),
                'noncritical_padding': list(bank.noncritical_padding),
                'legacy_critical': sorted(old), 'csg_critical_sync': list(new),
                'jaccard': len(old & set(new)) / len(old | set(new)),
            })
        if spec['path'].startswith('instances/tiny'):
            examples.append({'instance_id': instance.instance_id, 'makespan': current.makespan,
                             'schedule': current.schedule.to_dict(), 'analysis': asdict(analysis)})
    canonical = [r for r in rows if r['instance_id'].startswith('CB1')]
    distinct = all(len({r['count'] for r in canonical if r['instance_id'] == spec['instance_id']}) == 3 for spec in manifest['instances'])
    material = [r for r in rows if r['jaccard'] <= .5]
    assert distinct and material
    output = {'status': 'PASS', 'rows': rows, 'small_schedule_examples': examples,
              'distinct_cardinalities_all_18': distinct, 'material_divergence_cases': len(material),
              'critical_score': '4 * zero_slack_activity_count + incident_critical_edge_count + distinct_resource_count',
              'size_fractions': SIZE_FRACTIONS, 'rounding': 'floor(n*fraction + 0.5), clamp [min(2,n), n]'}
    write_immutable(ROOT / 'outputs/ngas_a1/audit/bank_validation.json', output)
    report = ROOT / 'docs/reports/ngas_a1'
    report.joinpath('04_csg_critical_sync_validation.md').write_text('''# CSG critical synchronization validation

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
''')
    report.joinpath('05_ngas_bank_v1_audit.md').write_text(f'''# NGAS bank v1 audit

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

There are {len(material)} comparisons with Jaccard overlap <= .5. Exact sets,
critical reasons, duplicates and padding are in `outputs/ngas_a1/audit/bank_validation.json`.
''')
    print(json.dumps({'status': 'PASS', 'states': len(specs), 'material_divergence_cases': len(material)}))


if __name__ == '__main__':
    main()
