#!/usr/bin/env python3
"""Evaluate the frozen full-bank critic audit outside formal search budgets."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from scipy.stats import spearmanr
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance  # noqa: E402
from rcias_clgri.env.feasibility import check_schedule  # noqa: E402
from rcias_clgri.search.common import Candidate, decode_candidate  # noqa: E402
from rcias_ngas.actions.repair import construct_neighbor  # noqa: E402
from rcias_ngas.critic.inference import FrozenJointCritic  # noqa: E402
from rcias_ngas.evaluation.a16_io import write_new_json  # noqa: E402
from rcias_ngas.evaluation.a16_integrity import digest, load_json  # noqa: E402
from rcias_ngas.execution import ExperimentSessionLock  # noqa: E402
from rcias_ngas.rng import RNGStreams  # noqa: E402
from rcias_ngas.runtime import ProductionRefreshRuntime  # noqa: E402
from scripts.freeze_ngas_a16r_offline_audit import canonical_hash  # noqa: E402
from scripts.run_ngas_a16r_solver_comparison import (  # noqa: E402
    CONFIG, FORMAL_OWNER_ID, OUT,
)


OFFLINE_PROTOCOL = OUT / 'diagnostics/offline_protocol.json'
RAW = OUT / 'diagnostics/offline_raw'
SUMMARY = OUT / 'diagnostics/offline_critic_regret.json'


def candidate_from_payload(payload: dict) -> Candidate:
    return Candidate(*(tuple(payload[name]) for name in (
        'operation_order', 'island_assignment', 'w_assignment', 'f_assignment')))


def safe_spearman(left, right) -> dict:
    if len(set(left)) < 2 or len(set(right)) < 2:
        return {'rho': None, 'p_value': None, 'valid': False}
    result = spearmanr(left, right)
    return {'rho': float(result.statistic), 'p_value': float(result.pvalue),
            'valid': True}


def ndcg(predicted_order: list[int], gains: list[float]) -> float | None:
    ideal = sorted(gains, reverse=True)
    if not ideal or ideal[0] <= 0:
        return None
    discounts = [math.log2(index + 2.) for index in range(len(gains))]
    actual = math.fsum(gains[index] / discounts[rank]
                       for rank, index in enumerate(predicted_order))
    maximum = math.fsum(value / discounts[rank] for rank, value in enumerate(ideal))
    return actual / maximum


def raw_name(state_key: str) -> str:
    return hashlib.sha256(state_key.encode()).hexdigest()[:24] + '.json'


def validate_protocol() -> tuple[dict, dict, str]:
    protocol = load_json(OFFLINE_PROTOCOL)
    config = load_json(CONFIG)
    if protocol.get('status') != 'FROZEN_BEFORE_COUNTERFACTUAL_RESULTS':
        raise RuntimeError('offline A1.6R protocol is not frozen')
    for relative, expected in protocol['source_hashes'].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f'offline source changed: {relative}')
    if digest(CONFIG) != protocol['config_sha256']:
        raise RuntimeError('A1.6R config changed before offline audit')
    return protocol, config, digest(OFFLINE_PROTOCOL)


def execute_state(state: dict, protocol_sha256: str, config: dict,
                  runtime: ProductionRefreshRuntime) -> dict:
    formal = load_json(ROOT / state['formal_raw_path'])
    if digest(ROOT / state['formal_raw_path']) != state['formal_raw_sha256']:
        raise RuntimeError('formal raw changed before offline audit')
    snapshot = next(
        row for row in formal['search_diagnostics']['a16r_replayable_states']
        if row['capture_fraction'] == state['capture_fraction'])
    if canonical_hash(snapshot) != state['snapshot_sha256']:
        raise RuntimeError('frozen replayable state changed')
    instance_path = ROOT / config['scope']['instance_root'] / state['instance_relative_path']
    if digest(instance_path) != state['instance_sha256']:
        raise RuntimeError('offline instance changed')
    instance = load_instance(instance_path)
    current = decode_candidate(instance, candidate_from_payload(snapshot['current_candidate']))
    feasibility = check_schedule(instance, current.schedule)
    if not current.feasible or not feasibility['feasible'] \
            or current.makespan != snapshot['current_makespan']:
        raise RuntimeError('offline state replay failed')
    streams = RNGStreams(instance.instance_id, state['seed'])
    refresh = runtime.refresh(
        instance, current, snapshot['state_id'], streams,
        sample_seed=streams.seed(
            'neural_prior', snapshot['state_id'], snapshot['iteration']))
    if ([action.action_id for action in refresh.actions] != snapshot['action_ids']
            or max(abs(a - b) for a, b in zip(refresh.prior, snapshot['prior'])) > 1e-7
            or max(abs(a - b) for a, b in zip(
                refresh.advantage, snapshot['advantage'])) > 1e-7):
        raise RuntimeError('offline full bank does not reproduce frozen state')
    rows = []
    common_seed = streams.seed('neighbor', snapshot['state_id'], 0)
    for index, action in enumerate(refresh.actions):
        started = time.perf_counter()
        neighbor = construct_neighbor(instance, current, action, random.Random(common_seed))
        candidate = decode_candidate(instance, neighbor)
        elapsed = time.perf_counter() - started
        if not candidate.feasible:
            raise RuntimeError('offline action produced infeasible candidate')
        rows.append({
            'action_index': index, 'action_id': action.action_id,
            'size': action.size, 'repair': action.repair,
            'advantage': refresh.advantage[index],
            'prior_probability': refresh.prior[index],
            'fallback_probability': refresh.beats_fallback_probability[index],
            'candidate_makespan': candidate.makespan,
            'realized_improvement': max(0., current.makespan - candidate.makespan),
            'decode_repair_seconds': elapsed,
        })
    critic_order = sorted(
        range(len(rows)), key=lambda i: (-rows[i]['advantage'], rows[i]['action_id']))
    realized_order = sorted(
        range(len(rows)),
        key=lambda i: (-rows[i]['realized_improvement'], rows[i]['action_id']))
    realized_rank = {index: rank + 1 for rank, index in enumerate(realized_order)}
    best_gain = rows[realized_order[0]]['realized_improvement']
    best_set = {index for index, row in enumerate(rows)
                if row['realized_improvement'] == best_gain}
    top1 = critic_order[0]
    metrics = {
        'critic_top1_action_id': rows[top1]['action_id'],
        'critic_top1_realized_rank': realized_rank[top1],
        'critic_top1_realized_gain': rows[top1]['realized_improvement'],
        'best_audited_realized_gain': best_gain,
        'critic_top1_regret': best_gain - rows[top1]['realized_improvement'],
        'best_action_recall_at_1': len(best_set & set(critic_order[:1])) / len(best_set),
        'best_action_recall_at_5': len(best_set & set(critic_order[:5])) / len(best_set),
        'best_action_recall_at_10': len(best_set & set(critic_order[:10])) / len(best_set),
        'ndcg_full': ndcg(critic_order, [row['realized_improvement'] for row in rows]),
        'spearman_advantage_realized_gain': safe_spearman(
            [row['advantage'] for row in rows],
            [row['realized_improvement'] for row in rows]),
    }
    return {
        'schema': 'ngas-a16r-offline-state-result-v1', 'status': 'COMPLETE',
        'protocol_sha256': protocol_sha256, **state,
        'evaluated_at_utc': datetime.now(timezone.utc).isoformat(),
        'current_makespan': current.makespan,
        'full_unique_bank_actions': len(rows),
        'matched_neighbor_seed': common_seed,
        'metrics': metrics, 'actions': rows,
        'formal_budget_inclusion': False,
    }


def mean(values) -> float | None:
    values = [float(value) for value in values if value is not None]
    return math.fsum(values) / len(values) if values else None


def summarize(payloads: list[dict], protocol_sha256: str) -> dict:
    groups = defaultdict(list)
    for row in payloads:
        groups[('scale', row['scale'])].append(row)
        stage = ('early' if row['capture_fraction'] < .3 else
                 'mid' if row['capture_fraction'] < .7 else 'late')
        groups[('stage', stage)].append(row)
    breakdowns = []
    for (field, value), rows in sorted(groups.items()):
        breakdowns.append({
            'field': field, 'value': value, 'states': len(rows),
            'mean_top1_realized_rank': mean(
                row['metrics']['critic_top1_realized_rank'] for row in rows),
            'mean_top1_regret': mean(row['metrics']['critic_top1_regret'] for row in rows),
            'mean_recall_at_5': mean(
                row['metrics']['best_action_recall_at_5'] for row in rows),
            'mean_ndcg_full': mean(row['metrics']['ndcg_full'] for row in rows),
        })
    action_groups = defaultdict(list)
    for state in payloads:
        for action in state['actions']:
            action_groups[('size', action['size'])].append(action)
            action_groups[('repair', action['repair'])].append(action)
    action_breakdowns = [
        {'field': field, 'value': value, 'actions': len(rows),
         'mean_realized_improvement': mean(row['realized_improvement'] for row in rows),
         'positive_improvement_rate': mean(row['realized_improvement'] > 0 for row in rows)}
        for (field, value), rows in sorted(action_groups.items())
    ]
    checks = {
        'exact_18_states': len(payloads) == 18,
        'all_complete': all(row['status'] == 'COMPLETE' for row in payloads),
        'all_full_bank_larger_than_8': all(
            row['full_unique_bank_actions'] > 8 for row in payloads),
        'outside_formal_budget': all(not row['formal_budget_inclusion'] for row in payloads),
    }
    return {
        'schema': 'ngas-a16r-offline-critic-regret-summary-v1',
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': protocol_sha256, 'checks': checks,
        'state_count': len(payloads),
        'overall': {
            'mean_top1_realized_rank': mean(
                row['metrics']['critic_top1_realized_rank'] for row in payloads),
            'mean_top1_regret': mean(
                row['metrics']['critic_top1_regret'] for row in payloads),
            'mean_recall_at_1': mean(
                row['metrics']['best_action_recall_at_1'] for row in payloads),
            'mean_recall_at_5': mean(
                row['metrics']['best_action_recall_at_5'] for row in payloads),
            'mean_recall_at_10': mean(
                row['metrics']['best_action_recall_at_10'] for row in payloads),
            'mean_ndcg_full': mean(row['metrics']['ndcg_full'] for row in payloads),
        },
        'scale_and_stage': breakdowns,
        'size_and_repair': action_breakdowns,
        'interpretation': 'frozen offline counterfactual audit; excluded from formal 2|O| results',
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('A1.6R offline audit requires CUDA')
    if (OUT / 'integrity/formal.lock').exists():
        raise RuntimeError('formal A1.6R worker is still active')
    protocol, config, protocol_sha256 = validate_protocol()
    if SUMMARY.exists():
        existing = load_json(SUMMARY)
        if existing.get('protocol_sha256') != protocol_sha256 \
                or existing.get('status') != 'PASS':
            raise RuntimeError('existing offline summary is invalid')
        print(json.dumps({'status': 'VALID_EXISTING_RESULT',
                          'path': str(SUMMARY.relative_to(ROOT))}))
        return
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    critic = FrozenJointCritic(
        ROOT / config['production_solver']['checkpoint_path'], args.device,
        config['production_solver']['checkpoint_sha256'],
        config['production_solver']['variant'])
    runtime = ProductionRefreshRuntime(
        critic,
        prior_advantage_scale=config['production_solver']['search']['prior_advantage_scale'],
        prior_uniform_mix=config['production_solver']['search']['prior_uniform_mix'])
    lock = ExperimentSessionLock(
        OUT / 'integrity/offline.lock', OUT / 'integrity/offline_sessions',
        experiment='NGAS_A1_6R_OFFLINE', stage='A1.6R_OFFLINE',
        implementation_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        command=[sys.executable, *sys.argv], formal_owner_id=FORMAL_OWNER_ID,
        heartbeat_interval_seconds=30.)
    with lock as session:
        payloads = []
        for state in protocol['states']:
            path = RAW / raw_name(state['state_key'])
            if path.exists():
                row = load_json(path)
                if row.get('protocol_sha256') != protocol_sha256 \
                        or row.get('state_key') != state['state_key']:
                    raise RuntimeError('existing offline raw is invalid')
            else:
                session.record('offline_state_started', state_key=state['state_key'])
                row = execute_state(state, protocol_sha256, config, runtime)
                write_new_json(path, row)
                session.record(
                    'offline_state_completed', state_key=state['state_key'],
                    raw_path=str(path.relative_to(ROOT)), raw_sha256=digest(path))
            payloads.append(row)
        summary = summarize(payloads, protocol_sha256)
        write_new_json(SUMMARY, summary)
        session.record('offline_audit_complete', states=len(payloads))
    print(json.dumps({'status': summary['status'], 'states': len(payloads),
                      'path': str(SUMMARY.relative_to(ROOT))}, indent=2))


if __name__ == '__main__':
    main()
