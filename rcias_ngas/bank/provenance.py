from dataclasses import dataclass

from rcias_ngas.evaluation.bks import content_hash


@dataclass(frozen=True)
class Target:
    target_id: str
    operations: tuple[str, ...]
    origin_rules: tuple[str, ...]
    origin_families: tuple[str, ...]
    origin_operators: tuple[str, ...]

    def provenance_features(self, rules, families, operators):
        """Multi-hot union over ALL origins; independent of insertion order."""
        return tuple(float(name in present) for vocab, present in (
            (rules, self.origin_rules), (families, self.origin_families),
            (operators, self.origin_operators)) for name in vocab) + (float(len(self.origin_rules)),)


def deduplicate(proposals, state_id, size):
    groups = {}
    for p in proposals:
        groups.setdefault(tuple(sorted(set(p.destroyed_operations))), []).append(p)
    return tuple(sorted((Target(
        'ngas_target_' + content_hash(['ngas-bank-v1', state_id, size, operations])[:24],
        operations, tuple(sorted({p.origin_rule for p in origins})),
        tuple(sorted({p.arm_family for p in origins})),
        tuple(sorted({p.origin_destroy_operator for p in origins})),
    ) for operations, origins in groups.items()), key=lambda t: t.target_id))
