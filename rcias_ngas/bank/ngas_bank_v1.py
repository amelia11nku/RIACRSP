"""Full 24-rule design per size, before any scoring or shortlist."""
from collections import Counter
from dataclasses import dataclass, replace
import random

from rcias_clgri.analysis.phase6a import schedule_features
from rcias_clgri.search.alns import DESTROY, _destroy
from rcias_clgri.search.counterfactual import stable_seed
from rcias_clgri.search.phase6c import (
    ArmGenerationResult, ArmProposal, Phase6CTargetArm,
    _random_swap, _ranked_replacement, generate_revised_target_arms,
    target_set_id,
)
from rcias_ngas.actions.destroy_size import destroy_count
from rcias_ngas.csg.critical_sync import critical_sync
from .provenance import deduplicate


@dataclass(frozen=True)
class Bank:
    version: str
    state_id: str
    size: str
    destroy_count: int
    targets: tuple
    proposals: tuple
    critical_operations: tuple[str, ...]
    noncritical_padding: tuple[str, ...]

    @property
    def requested_count(self):
        return len(self.proposals)

    @property
    def duplicate_count(self):
        return len(self.proposals) - len(self.targets)


def _related_destroy(instance, current, count, rng, positions):
    operations = list(instance.operations)
    pivot = rng.choice(operations)
    product = instance.product_of[pivot]
    pivot_position = positions[pivot]
    ranked = sorted(operations, key=lambda operation: (
        instance.product_of[operation] != product,
        abs(positions[operation] - pivot_position),
    ))
    return set(ranked[:count])


def _cached_target_arms(instance, current, state_id, count, seed_namespace,
                        features):
    """Exact Phase 6C design using one shared per-refresh feature mapping."""
    proposals = []
    operator_targets = {}
    positions = {operation: index for index, operation in enumerate(instance.operations)}
    for operator in DESTROY:
        rng = random.Random(stable_seed(
            state_id, 'operator', operator, namespace=seed_namespace))
        destroyed = (_related_destroy(instance, current, count, rng, positions)
                     if operator == 'related'
                     else _destroy(instance, current, operator, count, rng))
        target = tuple(sorted(destroyed))
        operator_targets[operator] = target
        proposals.append(ArmProposal(
            f'operator_{operator}', 'ORIGINAL_OPERATOR', operator, target))
    for index in range(1, 5):
        rng = random.Random(stable_seed(
            state_id, 'related_variant', index, namespace=seed_namespace))
        target = tuple(sorted(_related_destroy(
            instance, current, count, rng, positions)))
        proposals.append(ArmProposal(
            f'related_variant_{index}', 'RELATED_VARIANT', 'related', target))
    for index in range(1, 4):
        rng = random.Random(stable_seed(
            state_id, 'matched_random', index, namespace=seed_namespace))
        target = tuple(sorted(_destroy(
            instance, current, 'random', count, rng)))
        proposals.append(ArmProposal(
            f'matched_random_{index}', 'MATCHED_RANDOM', 'random', target))

    reference = operator_targets['related']
    local_rules = (
        ('one_operation_swap', 1), ('two_operation_swap', 2),
        ('related_replace_25', max(1, round(count * .25))),
        ('related_replace_50', max(1, round(count * .50))),
    )
    for rule, replace_count in local_rules:
        target, removed, added = _random_swap(
            reference, instance.operations, replace_count,
            stable_seed(state_id, rule, namespace=seed_namespace))
        proposals.append(ArmProposal(
            rule, 'LOCAL_PERTURBATION', 'related', target,
            reference, removed, added))

    product_counts = Counter(instance.product_of[operation] for operation in reference)
    island_counts = Counter(features[operation]['assigned_island'] for operation in reference)
    direct = {
        neighbor for operation in reference
        for neighbor in (*instance.predecessors[operation], *instance.successors[operation])}
    transitive = {
        neighbor for operation in reference
        for neighbor in (*instance.transitive_predecessors[operation],
                          *instance.transitive_successors[operation])}
    scores = {
        'same_product': {
            operation: float(product_counts[instance.product_of[operation]])
            for operation in instance.operations},
        'precedence_neighbor': {
            operation: float(2 if operation in direct else 1 if operation in transitive else 0)
            for operation in instance.operations},
        'same_island_chain': {
            operation: float(island_counts[features[operation]['assigned_island']])
            for operation in instance.operations},
        'high_W_delay': {
            operation: float(features[operation]['W_waiting_or_delay_contribution'])
            for operation in instance.operations},
        'high_F_delay': {
            operation: float(features[operation]['F_waiting_or_delay_contribution'])
            for operation in instance.operations},
        'low_slack': {
            operation: -float(features[operation]['operation_slack'])
            for operation in instance.operations},
    }
    replace_count = max(1, round(count * .25))
    for rule, score in scores.items():
        target, removed, added = _ranked_replacement(
            reference, instance.operations, replace_count, score,
            state_id, rule, seed_namespace)
        proposals.append(ArmProposal(
            f'near_{rule}', 'STRUCTURED_NEAR_NEIGHBOR', 'related', target,
            reference, removed, added))
    by_target = {}
    for proposal in proposals:
        by_target.setdefault(proposal.destroyed_operations, []).append(proposal)
    arms = []
    for operations, origins in by_target.items():
        first = origins[0]
        arms.append(Phase6CTargetArm(
            target_set_id(state_id, operations), first.arm_family,
            first.origin_destroy_operator,
            tuple(origin.origin_rule for origin in origins),
            tuple(dict.fromkeys(origin.arm_family for origin in origins)), operations))
    return ArmGenerationResult(
        tuple(arms), tuple(proposals), reference, len(proposals), len(arms),
        len(proposals) - len(arms))


def build_bank(instance, current, state_id, size, rngs, analysis=None,
               schedule_feature_cache=None):
    count = destroy_count(instance.num_operations, size)
    analysis = analysis or critical_sync(instance, current)
    seed_namespace = rngs.seed('target', state_id + ':' + size)
    inherited = (_cached_target_arms(
        instance, current, state_id, count, seed_namespace,
        schedule_feature_cache) if schedule_feature_cache is not None
        else generate_revised_target_arms(
            instance, current, state_id, count, seed_namespace))
    critical = analysis.ranked_operations[:count]
    # Padding is explicitly NOT described as a critical operation. Its nearest
    # finite operation slack and stable ID determine the supplemental ordering.
    def operation_slack(operation):
        if hasattr(analysis, 'operation_slack'):
            return analysis.operation_slack(operation)
        return analysis.nodes['OP:' + operation]['slack']

    remaining = sorted(set(instance.operations) - set(critical), key=lambda op: (
        operation_slack(op) if operation_slack(op) is not None else float('inf'), op))
    padding = tuple(remaining[:count - len(critical)])
    proposals = []
    for p in inherited.proposals:
        if p.origin_rule == 'operator_critical':
            p = replace(p, origin_rule='csg_critical_sync', origin_destroy_operator='csg_critical_sync',
                        destroyed_operations=tuple(sorted(critical + padding)))
        elif p.origin_rule == 'near_low_slack':
            # Preserve the useful historical completion-tail perturbation without
            # mislabelling its proxy as true temporal slack.
            p = replace(p, origin_rule='near_legacy_completion_tail')
        proposals.append(p)
    targets = deduplicate(proposals, state_id, size)
    return Bank('ngas-bank-v1', state_id, size, count, targets, tuple(proposals), critical, padding)


def build_all_banks(instance, current, state_id, rngs, analysis,
                    schedule_feature_cache=None):
    """Build all frozen sizes while sharing schedule-derived operation features."""
    shared = schedule_feature_cache or schedule_features(instance, current.schedule)
    return tuple(build_bank(
        instance, current, state_id, size, rngs, analysis=analysis,
        schedule_feature_cache=shared,
    ) for size in ('small', 'medium', 'large'))
