"""Inherited repair semantics with canonical iteration order in NGAS only."""
from rcias_clgri.search.alns import REPAIR, _neighbor
from rcias_clgri.search.common import decode_candidate


def construct_neighbor(instance, current, action, rng):
    if action.repair not in REPAIR or not set(action.target.operations) <= set(instance.operations):
        raise ValueError('Invalid joint repair')
    # The shared primitive only iterates and tests membership on removed. A
    # canonical tuple fixes both tied sorting and assignment-draw ordering.
    return _neighbor(instance, current.candidate, tuple(sorted(action.target.operations)), action.repair, rng)


def execute_action(instance, current, action, rngs, state_id, trials=8):
    if trials < 1:
        raise ValueError('At least one repair trial required')
    candidates = [decode_candidate(instance, construct_neighbor(
        instance, current, action, rngs.stream('neighbor', state_id, trial))) for trial in range(trials)]
    if not all(c.feasible for c in candidates):
        raise ValueError('Repair produced an infeasible schedule')
    return min(candidates, key=lambda c: c.makespan), trials
