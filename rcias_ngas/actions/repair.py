"""Frozen NGAS repair IDs mapped separately to reused legacy callables."""
from rcias_clgri.search.alns import REPAIR as LEGACY_REPAIR_IDS, _neighbor
from rcias_clgri.search.common import decode_candidate


NGAS_REPAIR_IDS = (
    'greedy', 'regret2', 'regret3', 'reconfiguration_aware', 'transport_aware',
)

# The current implementation is deliberately reused, while the embedding/index
# contract above remains owned by NGAS and cannot drift with the legacy tuple.
if set(NGAS_REPAIR_IDS) != set(LEGACY_REPAIR_IDS):
    raise RuntimeError('Legacy repair implementations are incompatible with frozen NGAS IDs')

def _legacy_implementation(repair_id):
    def apply(instance, base, removed, rng):
        return _neighbor(instance, base, removed, repair_id, rng)
    return apply


REPAIR_CALLABLES = {
    repair_id: _legacy_implementation(repair_id) for repair_id in NGAS_REPAIR_IDS}


def repair_index(repair_id: str) -> int:
    try:
        return NGAS_REPAIR_IDS.index(repair_id)
    except ValueError as error:
        raise ValueError(f'Unknown NGAS repair ID: {repair_id}') from error


def construct_neighbor(instance, current, action, rng):
    if action.repair not in REPAIR_CALLABLES or not set(action.target.operations) <= set(instance.operations):
        raise ValueError('Invalid joint repair')
    # The shared primitive only iterates and tests membership on removed. A
    # canonical tuple fixes both tied sorting and assignment-draw ordering.
    return REPAIR_CALLABLES[action.repair](
        instance, current.candidate, tuple(sorted(action.target.operations)), rng)


def execute_action(instance, current, action, rngs, state_id, trials=8):
    if trials < 1:
        raise ValueError('At least one repair trial required')
    candidates = [decode_candidate(instance, construct_neighbor(
        instance, current, action, rngs.stream('neighbor', state_id, trial))) for trial in range(trials)]
    if not all(c.feasible for c in candidates):
        raise ValueError('Repair produced an infeasible schedule')
    return min(candidates, key=lambda c: c.makespan), trials
