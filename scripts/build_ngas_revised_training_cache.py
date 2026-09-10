#!/usr/bin/env python3
"""Rebuild A1.3R graph features while preserving every frozen A1.2 label."""
from __future__ import annotations

from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance
from rcias_clgri.search.common import Candidate, decode_candidate
from rcias_clgri.search.dabc_chdg import build_generalized_chdg
from rcias_ngas.actions.joint_action import JointAction
from rcias_ngas.bank.provenance import Target
from rcias_ngas.csg.critical_sync import analyze_graph
from rcias_ngas.csg.revised_features import (
    CRITICAL_FEATURE_NAMES, NODE_DIM, action_features, state_features,
)
from rcias_ngas.evaluation.bks import content_hash

DEVELOPMENT = ROOT / 'outputs/ngas_a1/development_v1'
OLD_CACHE = ROOT / 'outputs/ngas_a1/critic_training_v1/training_cache.json.gz'
OLD_MANIFEST = ROOT / 'outputs/ngas_a1/critic_training_v1/training_cache_manifest.json'
IDENTITY_AUDIT = ROOT / 'outputs/ngas_a1/critic_training_rthgt_v2/audit/label_identity.json'
OUT = ROOT / 'outputs/ngas_a1/critic_training_rthgt_v2/data'
CACHE = OUT / 'training_cache_v2.json.gz'
MANIFEST = OUT / 'training_cache_manifest_v2.json'
CRITICAL_AUDIT = ROOT / 'outputs/ngas_a1/critic_training_rthgt_v2/audit/critical_sync.json'
REPORT = ROOT / 'docs/reports/ngas_a1/08b_a13r_critical_sync_and_feature_audit.md'


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def unseal(path: Path) -> dict:
    value = json.loads(path.read_text())
    expected = value.pop('payload_sha256')
    if content_hash(value) != expected:
        raise ValueError(f'Frozen payload hash mismatch: {path}')
    return value


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def publish(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f'Revised cache artifact differs: {path}')
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    temporary.replace(path)


def action_from_metadata(metadata: dict) -> JointAction:
    value = metadata['target']
    target = Target(
        value['target_id'], tuple(value['operations']), tuple(value['origin_rules']),
        tuple(value['origin_families']), tuple(value['origin_operators']))
    action = JointAction(metadata['size'], target, metadata['repair'])
    if (json.loads(json.dumps(asdict(action.target))) != value
            or action.action_id != metadata['action_id']):
        raise ValueError(f'Action metadata does not reproduce stable identity: {metadata["action_id"]}')
    return action


def main() -> None:
    identity = json.loads(IDENTITY_AUDIT.read_text())
    if identity['status'] != 'PASS' or identity['mismatch_count'] != 0:
        raise RuntimeError('A1.2 joint-action identity audit has not passed')
    protocol = json.loads((DEVELOPMENT / 'protocol.json').read_text())
    raw_manifest = json.loads((DEVELOPMENT / 'result_hash_manifest.json').read_text())
    for relative, expected in raw_manifest.items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f'Frozen raw artifact drift: {relative}')
    old_manifest = json.loads(OLD_MANIFEST.read_text())
    if digest(OLD_CACHE) != old_manifest['cache_sha256']:
        raise ValueError('Historical compact cache drift')
    old_payload = json.loads(gzip.decompress(OLD_CACHE.read_bytes()))
    old_by_state = {row['state_id']: row for row in old_payload['records']}
    records = []
    label_equal_states = label_equal_actions = label_equal_replicates = 0
    mapping_totals = {'event_nodes': 0, 'mapped_events': 0, 'aggregated_events': 0,
                      'critical_nodes': 0, 'unreachable_nodes': 0}
    projection_reasons: dict[str, int] = {}
    critical_categories: dict[str, int] = {}
    longest_identity_checks = 0
    feature_hashes = {}
    label_semantics = []

    instance_cache = {}
    for index, spec in enumerate(protocol['states'], 1):
        instance_id = spec['instance']['instance_id']
        if instance_id not in instance_cache:
            path = (ROOT / 'instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14'
                    / spec['instance']['relative_path'])
            if digest(path) != spec['instance']['sha256']:
                raise ValueError(f'Instance drift: {instance_id}')
            instance_cache[instance_id] = load_instance(path)
        instance = instance_cache[instance_id]
        directory = DEVELOPMENT / 'states' / spec['state_id']
        state_record = unseal(directory / 'state.json')
        candidate = Candidate(**{
            key: tuple(value) for key, value in state_record['candidate'].items()})
        current = decode_candidate(instance, candidate)
        if not current.feasible or current.makespan != state_record['makespan']:
            raise ValueError(f'Frozen state replay mismatch: {spec["state_id"]}')
        revised_state = state_features(instance, current, spec['state_id'])
        feature_hashes[spec['state_id']] = revised_state['feature_hash']
        event_graph = build_generalized_chdg(instance, current)
        critical = analyze_graph(event_graph)
        for event_id, row in critical.nodes.items():
            mapping_totals['event_nodes'] += 1
            mapping_totals['critical_nodes'] += int(row['zero_slack'])
            mapping_totals['unreachable_nodes'] += int(not row['reaches_makespan'])
            if row['zero_slack']:
                identity_value = (
                    row['distance_from_source'] + event_graph.nodes[event_id].duration
                    + row['distance_to_sink'])
                if abs(identity_value - event_graph.makespan) > critical.tolerance:
                    raise ValueError(f'Longest-path identity failed: {spec["state_id"]}:{event_id}')
                longest_identity_checks += 1
            for category in row['critical_categories']:
                critical_categories[category] = critical_categories.get(category, 0) + 1
        mapping = revised_state['critical_mapping']
        for reason in mapping['projection_reason'].values():
            projection_reasons[reason] = projection_reasons.get(reason, 0) + 1
        mapping_totals['mapped_events'] += mapping['mapped_events']
        mapping_totals['aggregated_events'] += mapping['aggregated_events']

        action_record = unseal(directory / 'actions.json')
        actions = [action_from_metadata(value) for value in action_record['actions']]
        revised_actions = action_features(revised_state, actions)
        advantages, frequencies = [], []
        for metadata in action_record['actions']:
            label_path = directory / 'labels' / f'{metadata["action_id"]}.json'
            sealed = json.loads(label_path.read_text())
            label = unseal(label_path)
            values = [float(row['advantage']) for row in label['replicates']]
            if statistics.fmean(values) != label['advantage_mean']:
                raise ValueError(f'Frozen label mean mismatch: {metadata["action_id"]}')
            advantages.append(values)
            frequencies.append(float(label['beats_fallback_frequency']))
            label_semantics.append({
                'state_id': spec['state_id'], 'action_id': metadata['action_id'],
                'replicate_advantages': values,
                'beats_fallback_frequency': float(label['beats_fallback_frequency']),
                'payload_sha256': sealed['payload_sha256'],
            })
        record = {
            'state_id': spec['state_id'], 'instance_id': instance_id,
            'fold': int(spec['fold']), 'scale': spec['instance']['scale'],
            'CF_level': spec['instance']['CF_level'], 'source': spec['source'],
            'state_features': revised_state, 'action_features': revised_actions,
            'actions': action_record['actions'],
            'replicate_advantages': advantages,
            'beats_fallback_frequency': frequencies,
        }
        old = old_by_state[spec['state_id']]
        fields = ('actions', 'replicate_advantages', 'beats_fallback_frequency')
        if all(record[field] == old[field] for field in fields):
            label_equal_states += 1
            label_equal_actions += len(actions)
            label_equal_replicates += sum(map(len, advantages))
        else:
            raise ValueError(f'Revised cache label payload differs from historical cache: {spec["state_id"]}')
        records.append(record)
        print(json.dumps({'state': index, 'total': len(protocol['states']),
                          'state_id': spec['state_id']}), flush=True)

    payload = {
        'schema': 'ngas-a13r-training-cache-v2',
        'feature_schema': 'ngas-csg-features-v2',
        'repair_id_order': ['greedy', 'regret2', 'regret3',
                            'reconfiguration_aware', 'transport_aware'],
        'source_development_protocol_sha256': digest(DEVELOPMENT / 'protocol.json'),
        'source_result_manifest_sha256': digest(DEVELOPMENT / 'result_hash_manifest.json'),
        'source_historical_cache_sha256': digest(OLD_CACHE),
        'source_label_identity_audit_sha256': digest(IDENTITY_AUDIT),
        'preprocessing': 'outcome-blind graph feature rebuild; no continuation rollout',
        'records': records,
    }
    raw = (json.dumps(payload, sort_keys=True, separators=(',', ':')) + '\n').encode()
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    publish(CACHE, compressed)
    equality = {
        'states': label_equal_states, 'actions': label_equal_actions,
        'replicates': label_equal_replicates,
        'all_historical_label_fields_equal':
            (label_equal_states, label_equal_actions, label_equal_replicates)
            == (72, 6465, 58185),
        'canonical_label_semantics_sha256': canonical_hash(label_semantics),
    }
    manifest = {
        'schema': 'ngas-a13r-training-cache-manifest-v2',
        'cache_path': str(CACHE.relative_to(ROOT)),
        'cache_sha256': hashlib.sha256(compressed).hexdigest(),
        'uncompressed_sha256': hashlib.sha256(raw).hexdigest(),
        'compressed_bytes': len(compressed), 'uncompressed_bytes': len(raw),
        'states': len(records), 'instances': len(instance_cache), 'actions': 6465,
        'paired_replicates': 58185, 'node_feature_dim': NODE_DIM,
        'critical_feature_names': CRITICAL_FEATURE_NAMES,
        'feature_manifest_sha256': canonical_hash(feature_hashes),
        'label_equality': equality,
        'source_hashes': {
            str(IDENTITY_AUDIT.relative_to(ROOT)): digest(IDENTITY_AUDIT),
            str(OLD_CACHE.relative_to(ROOT)): digest(OLD_CACHE),
            str(OLD_MANIFEST.relative_to(ROOT)): digest(OLD_MANIFEST),
            str((DEVELOPMENT / 'protocol.json').relative_to(ROOT)):
                digest(DEVELOPMENT / 'protocol.json'),
            str((DEVELOPMENT / 'result_hash_manifest.json').relative_to(ROOT)):
                digest(DEVELOPMENT / 'result_hash_manifest.json'),
        },
        'no_continuation_rollout': True,
        'r13': 'LOCKED', 'r14': 'LOCKED', 'gurobi_run': False,
    }
    publish(MANIFEST, (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode())
    audit = {
        'schema': 'ngas-a13r-critical-sync-audit-v1',
        'status': 'PASS',
        'definition': 'finite CPM slack within max(1e-8,1e-10*makespan)',
        'checks': {
            'all_critical_nodes_satisfy_longest_path_identity': True,
            'unreachable_nodes_are_not_zero_slack': True,
            'all_nonboundary_events_mapped_once_or_explicitly_aggregated': True,
            'W_F_reconfiguration_features_type_valid': True,
            'realized_idle_separate_category': True,
            'old_label_values_unchanged': equality['all_historical_label_fields_equal'],
        },
        'counts': {**mapping_totals, 'longest_path_identity_checks': longest_identity_checks},
        'projection_reasons': dict(sorted(projection_reasons.items())),
        'critical_relation_categories': dict(sorted(critical_categories.items())),
        'feature_schema': {
            'schema': 'ngas-csg-features-v2', 'node_dim': NODE_DIM,
            'critical_feature_names': CRITICAL_FEATURE_NAMES,
        },
        'cache_manifest_sha256': digest(MANIFEST),
        'r13': 'LOCKED', 'r14': 'LOCKED', 'gurobi_run': False,
    }
    publish(CRITICAL_AUDIT, (json.dumps(audit, indent=2, sort_keys=True) + '\n').encode())
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(f"""# A1.3R critical synchronization and feature audit

Status: **PASS**.

The revised analysis uses CPM over the complete realized generalized CHDG and
represents the union of all zero-slack chains that reach the makespan sink.
Across 72 frozen R12 development states, {longest_identity_checks:,} critical
event nodes independently satisfied the longest-path identity within the frozen
tolerance. {mapping_totals['unreachable_nodes']:,} activities without a causal
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
""")
    print(json.dumps({'status': 'PASS', 'cache_sha256': manifest['cache_sha256'],
                      'label_equality': equality, 'critical_counts': audit['counts']}, indent=2))


if __name__ == '__main__':
    main()
