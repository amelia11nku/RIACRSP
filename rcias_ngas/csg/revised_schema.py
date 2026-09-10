"""Frozen A1.3R feature vocabularies independent of historical ALNS globals."""

RULE_GROUPS = (
    ('operator_random', 'operator_overloaded_island', 'operator_high_reconfiguration',
     'operator_w_bottleneck', 'operator_f_bottleneck', 'operator_related'),
    tuple(f'related_variant_{index}' for index in range(1, 5)),
    tuple(f'matched_random_{index}' for index in range(1, 4)),
    ('one_operation_swap', 'two_operation_swap', 'related_replace_25', 'related_replace_50'),
    ('near_same_product', 'near_precedence_neighbor', 'near_same_island_chain',
     'near_high_W_delay', 'near_high_F_delay', 'near_legacy_completion_tail'),
)
RULES = tuple(sorted(('csg_critical_sync', *(
    rule for group in RULE_GROUPS for rule in group))))
FAMILIES = (
    'ORIGINAL_OPERATOR', 'RELATED_VARIANT', 'MATCHED_RANDOM',
    'LOCAL_PERTURBATION', 'STRUCTURED_NEAR_NEIGHBOR',
)
OPERATORS = (
    'random', 'csg_critical_sync', 'overloaded_island',
    'high_reconfiguration', 'w_bottleneck', 'f_bottleneck', 'related',
)
PROVENANCE_DIM = len(RULES) + len(FAMILIES) + len(OPERATORS) + 1
