"""Outcome-blind source plan, grouped folds and balanced full-bank sampling."""
from dataclasses import asdict

from rcias_clgri.search.alns import REPAIR
from rcias_ngas.actions.destroy_size import SIZE_FRACTIONS
from rcias_ngas.actions.joint_action import JointAction
from rcias_ngas.bank.ngas_bank_v1 import build_bank
from rcias_ngas.csg.features import RULE_GROUPS
from rcias_ngas.csg.critical_sync import critical_sync
from rcias_ngas.critic.dataset import fallback_action, transition
from rcias_ngas.rng import RNGStreams

DEVELOPMENT_VERSION = 'ngas-a13-development-v1'


def instance_fold(spec):
    # Both cell replicates stay together. Each fold has all three scales and CFs.
    return (('S', 'M', 'L').index(spec['scale']) + ('CF1', 'CF2', 'CF3').index(spec['CF_level'])) % 3


def state_plan(instances, roots):
    rows = []
    for spec in sorted(instances, key=lambda s: s['instance_id']):
        for source, seed, steps in [('H1', roots[0], 0), *((f'NATIVE16_{seed}', seed, 16) for seed in roots)]:
            rows.append({'state_id': f'{DEVELOPMENT_VERSION}:{spec["instance_id"]}:{source}',
                         'instance': spec, 'fold': instance_fold(spec), 'source': source,
                         'source_seed': seed, 'source_steps': steps, 'ordinal': len(rows)})
    return rows


def source_state(instance, initial, spec):
    current = initial
    rngs = RNGStreams(instance.instance_id, spec['source_seed'])
    trace = []
    for step in range(spec['source_steps']):
        key = f'{DEVELOPMENT_VERSION}:source:{step}'
        action = fallback_action(instance, current, rngs, key)
        current, proposal, count, accepted = transition(instance, current, action, rngs, key, 8,
                                                       .05 * initial.makespan * .995**step)
        trace.append({'action': action.metadata(), 'candidate': asdict(current.candidate),
                      'current_makespan': current.makespan, 'proposal_makespan': proposal.makespan,
                      'accepted': accepted, 'decoder_evals': count})
    return current, trace


def selected_rules(ordinal, size_index):
    return ('csg_critical_sync', *(group[(ordinal + size_index) % len(group)] for group in RULE_GROUPS))


def sample_actions(instance, current, spec):
    analysis = critical_sync(instance, current)
    rngs = RNGStreams(instance.instance_id, spec['source_seed'])
    actions, banks = {}, []
    for index, size in enumerate(SIZE_FRACTIONS):
        bank = build_bank(instance, current, spec['state_id'], size, rngs, analysis)
        banks.append(bank)
        for rule in selected_rules(spec['ordinal'], index):
            target = next(t for t in bank.targets if rule in t.origin_rules)
            for repair in REPAIR:
                action = JointAction(size, target, repair)
                actions[action.action_id] = action
    return tuple(sorted(actions.values(), key=lambda a: a.action_id)), banks
