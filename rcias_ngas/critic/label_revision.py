"""A1.2-v2: matched trial-count diagnostic with independent CRN replication.

The v1 transition and continuation semantics remain frozen. V2 adds replication
and an explicitly accounted shared fallback cache; it never changes old labels.
"""
import statistics

from rcias_ngas.rng import RNGStreams
from .dataset import continuation, fallback_action

REVISION = 'ngas-a12-label-v2'


def replicate_seeds(instance_id, state_id, canonical_seeds, repeats):
    """Repeat-major ordering keeps each three-replicate prefix root balanced."""
    return tuple(RNGStreams(instance_id, root).seed(
        'continuation_crn', REVISION + ':' + state_id, repeat)
        for repeat in range(repeats) for root in canonical_seeds)


def paired_streams(instance_id, state_id, replicate_seed):
    base = RNGStreams(instance_id, replicate_seed)
    return RNGStreams(instance_id, base.seed('continuation_crn', state_id))


def evaluate_fallback(instance, current, state_id, seed, steps, trials):
    rngs = paired_streams(instance.instance_id, state_id, seed)
    action = fallback_action(instance, current, rngs, f'{state_id}:continuation:0')
    return continuation(instance, current, action, rngs, state_id, steps, trials)


def evaluate_replicate(instance, current, action, state_id, seed, steps, trials, fallback):
    candidate = continuation(instance, current, action,
                             paired_streams(instance.instance_id, state_id, seed),
                             state_id, steps, trials)
    first = candidate['steps'][0]
    if first['repair'] != action.repair or first['action_id'] != action.action_id:
        raise ValueError('Joint action differs from executed label action')
    if len(fallback['steps']) != steps + 1 or any(s['decoder_evals'] != trials for s in fallback['steps']):
        raise ValueError('Fallback policy accounting mismatch')
    for c, f in zip(candidate['steps'], fallback['steps']):
        if c['neighbor_seed'] != f['neighbor_seed'] or c['acceptance_seed'] != f['acceptance_seed']:
            raise ValueError('Fallback CRN mismatch')
    return {
        'crn_seed': seed, 'candidate': candidate, 'fallback': fallback,
        'advantage': (fallback['best_makespan'] - candidate['best_makespan']) / current.makespan,
        'immediate_normalized_improvement': (current.makespan - first['proposal_makespan']) / current.makespan,
    }


def aggregate(action_metadata, initial_makespan, replicates, trials, steps):
    if len(replicates) < 2 or len({r['crn_seed'] for r in replicates}) != len(replicates):
        raise ValueError('Need at least two independent CRN replicates')
    advantages = [r['advantage'] for r in replicates]
    return {
        'action': action_metadata, 'initial_makespan': initial_makespan,
        'revision': REVISION, 'transition_policy': 'ngas-native-uniform-crn-v1',
        'repair_trials': trials, 'continuation_steps': steps,
        'advantage_mean': statistics.fmean(advantages),
        'advantage_variance': statistics.variance(advantages),
        'beats_fallback_frequency': statistics.fmean(float(a > 0) for a in advantages),
        'immediate_normalized_improvement_mean': statistics.fmean(r['immediate_normalized_improvement'] for r in replicates),
        'feasible': all(r['candidate']['feasible'] and r['fallback']['feasible'] for r in replicates),
        'repair_failure_count': 0, 'replicates': replicates,
    }
