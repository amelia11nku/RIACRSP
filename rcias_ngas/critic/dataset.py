"""Balanced joint-action sampling and paired, namespace-isolated continuation."""
import math
import statistics

from rcias_clgri.search.alns import REPAIR, _destroy
from rcias_ngas.actions.destroy_size import SIZE_FRACTIONS, destroy_count
from rcias_ngas.actions.joint_action import JointAction
from rcias_ngas.actions.repair import execute_action
from rcias_ngas.bank.ngas_bank_v1 import build_bank
from rcias_ngas.bank.provenance import Target
from rcias_ngas.csg.critical_sync import critical_sync
from rcias_ngas.evaluation.bks import content_hash
from rcias_ngas.rng import RNGStreams

POLICY_VERSION = 'ngas-native-uniform-crn-v1'
FALLBACK_OPERATORS = ('random', 'related', 'overloaded_island', 'high_reconfiguration', 'w_bottleneck', 'f_bottleneck')
SAMPLING_RULES = ('csg_critical_sync', 'related_variant_1', 'matched_random_1',
                  'related_replace_25', 'near_same_island_chain')


def balanced_actions(instance, current, state_id, rngs):
    analysis = critical_sync(instance, current)
    selected = {}
    banks = []
    for size in SIZE_FRACTIONS:
        bank = build_bank(instance, current, state_id, size, rngs, analysis)
        banks.append(bank)
        for rule in SAMPLING_RULES:
            target = next(t for t in bank.targets if rule in t.origin_rules)
            for repair in REPAIR:
                action = JointAction(size, target, repair)
                selected[action.action_id] = action
    return tuple(sorted(selected.values(), key=lambda a: a.action_id)), banks


def fallback_action(instance, current, rngs, key):
    size = rngs.stream('destroy_size', key).choice(tuple(SIZE_FRACTIONS))
    repair = rngs.stream('repair', key).choice(REPAIR)
    operator = rngs.stream('fallback', key).choice(FALLBACK_OPERATORS)
    operations = tuple(sorted(_destroy(instance, current, operator,
                                       destroy_count(instance.num_operations, size),
                                       rngs.stream('target', key))))
    target = Target('native_' + content_hash([POLICY_VERSION, size, operations])[:24],
                    operations, ('native_' + operator,), ('NATIVE',), (operator,))
    return JointAction(size, target, repair)


def transition(instance, current, action, rngs, key, trials, temperature):
    candidate, evaluations = execute_action(instance, current, action, rngs, key, trials)
    delta = candidate.makespan - current.makespan
    accepted = delta <= 0 or rngs.stream('acceptance', key).random() < math.exp(-delta / max(temperature, 1e-12))
    return (candidate if accepted else current), candidate, evaluations, accepted


def continuation(instance, current, first_action, rngs, state_id, steps=2, trials=2):
    initial_makespan = current.makespan
    best = initial_makespan
    records = []
    for step in range(steps + 1):
        key = f'{state_id}:continuation:{step}'
        action = first_action if step == 0 else fallback_action(instance, current, rngs, key)
        current, proposed, evaluations, accepted = transition(
            instance, current, action, rngs, key, trials,
            .05 * initial_makespan * (.995 ** step))
        best = min(best, proposed.makespan)
        records.append({'action_id': action.action_id, 'size': action.size, 'repair': action.repair,
                        'destroyed_operations': action.target.operations, 'accepted': accepted,
                        'proposal_makespan': proposed.makespan, 'current_makespan': current.makespan,
                        'best_makespan': best, 'feasible': proposed.feasible,
                        'decoder_evals': evaluations,
                        'neighbor_seed': rngs.seed('neighbor', key),
                        'acceptance_seed': rngs.seed('acceptance', key)})
    return {'best_makespan': best, 'steps': records, 'feasible': all(r['feasible'] for r in records)}


def label_action(instance, current, action, state_id, crn_seeds, steps=2, trials=2):
    rows = []
    for seed in crn_seeds:
        base = RNGStreams(instance.instance_id, seed)
        rngs = RNGStreams(instance.instance_id, base.seed('continuation_crn', state_id))
        key = f'{state_id}:continuation:0'
        fallback = fallback_action(instance, current, rngs, key)
        candidate_run = continuation(instance, current, action, rngs, state_id, steps, trials)
        fallback_run = continuation(instance, current, fallback, rngs, state_id, steps, trials)
        if candidate_run['steps'][0]['repair'] != action.repair or candidate_run['steps'][0]['action_id'] != action.action_id:
            raise ValueError('Action/label mismatch')
        rows.append({
            'crn_seed': seed, 'candidate': candidate_run, 'fallback': fallback_run,
            'advantage': (fallback_run['best_makespan'] - candidate_run['best_makespan']) / current.makespan,
            'immediate_normalized_improvement': (current.makespan - candidate_run['steps'][0]['proposal_makespan']) / current.makespan,
        })
    advantage = [r['advantage'] for r in rows]
    return {
        'action': action.metadata(), 'continuation_policy': POLICY_VERSION,
        'continuation_steps': steps, 'repair_trials': trials,
        'initial_makespan': current.makespan,
        'advantage_mean': statistics.fmean(advantage),
        'advantage_variance': statistics.variance(advantage) if len(advantage) > 1 else 0.,
        'beats_fallback_frequency': statistics.fmean(float(a > 0) for a in advantage),
        'immediate_normalized_improvement_mean': statistics.fmean(r['immediate_normalized_improvement'] for r in rows),
        'feasible': all(r['candidate']['feasible'] and r['fallback']['feasible'] for r in rows),
        'repair_failure_count': 0, 'replicates': rows,
    }


def separated_pair(left, right, noise_multiplier=2., minimum_gap=.001):
    """Paired-CRN differences; tiny gaps are never hard ranking labels."""
    if [r['crn_seed'] for r in left['replicates']] != [r['crn_seed'] for r in right['replicates']]:
        raise ValueError('Pair must use matched CRN replicates')
    differences = [a['advantage'] - b['advantage'] for a, b in zip(left['replicates'], right['replicates'])]
    stderr = statistics.stdev(differences) / math.sqrt(len(differences)) if len(differences) > 1 else math.inf
    return abs(statistics.fmean(differences)) > max(minimum_gap, noise_multiplier * stderr)
