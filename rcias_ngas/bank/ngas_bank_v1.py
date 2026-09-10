"""Full 24-rule design per size, before any scoring or shortlist."""
from dataclasses import dataclass, replace

from rcias_clgri.search.phase6c import generate_revised_target_arms
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


def build_bank(instance, current, state_id, size, rngs, analysis=None):
    count = destroy_count(instance.num_operations, size)
    analysis = analysis or critical_sync(instance, current)
    inherited = generate_revised_target_arms(
        instance, current, state_id, count, rngs.seed('target', state_id + ':' + size))
    critical = analysis.ranked_operations[:count]
    # Padding is explicitly NOT described as a critical operation. Its nearest
    # finite operation slack and stable ID determine the supplemental ordering.
    remaining = sorted(set(instance.operations) - set(critical), key=lambda op: (
        analysis.nodes['OP:' + op]['slack'] if analysis.nodes['OP:' + op]['slack'] is not None else float('inf'), op))
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
