"""A1.4 telemetry with causal intermediate checkpoints and a true final point."""
from __future__ import annotations

from dataclasses import asdict
import math

from rcias_ngas.search.telemetry import RunState


class A14Telemetry:
    def __init__(self, budget_seconds: float, fractions=(.10, .25, .50, .75)) -> None:
        if not math.isfinite(budget_seconds) or budget_seconds <= 0:
            raise ValueError('Budget must be finite and positive')
        self.budget = budget_seconds
        self.fractions = tuple(fractions)
        self.elapsed = 0.
        self.state = RunState()
        self.checkpoints = []
        self.best_events = []

    def observe(self, elapsed: float, state: RunState) -> None:
        if elapsed < self.elapsed:
            raise ValueError('Event clock regressed')
        old = self.state
        for name in ('decoder_evals', 'neural_calls', 'iteration', 'accepted_moves',
                     'improving_moves', 'new_best_moves'):
            if getattr(state, name) < getattr(old, name):
                raise ValueError(f'Cumulative counter regressed: {name}')
        if old.best_makespan is not None and (
                state.best_makespan is None or state.best_makespan > old.best_makespan):
            raise ValueError('Best makespan regressed')
        for fraction in self.fractions[len(self.checkpoints):]:
            deadline = fraction * self.budget
            if deadline > elapsed:
                break
            snapshot = state if deadline == elapsed else old
            self.checkpoints.append({
                'budget_fraction': fraction,
                'elapsed_time_sec': deadline,
                **asdict(snapshot),
            })
        if state.best_makespan is not None and (
                old.best_makespan is None or state.best_makespan < old.best_makespan):
            self.best_events.append({
                'best_event_time_sec': elapsed,
                'best_event_decoder_evals': state.decoder_evals,
                'best_event_iteration': state.iteration,
                'best_makespan': state.best_makespan,
            })
        self.elapsed, self.state = elapsed, state

    def finish(self, elapsed: float) -> dict:
        self.observe(elapsed, self.state)
        final = {'budget_fraction': 1., 'elapsed_time_sec': elapsed, **asdict(self.state)}
        checkpoints = [*self.checkpoints, final]
        last = self.best_events[-1] if self.best_events else {}
        return {
            'budget_checkpoints': checkpoints,
            'best_events': self.best_events,
            'termination': {'elapsed_time_sec': elapsed, **asdict(self.state)},
            'last_best_time_sec': last.get('best_event_time_sec'),
            'last_best_decoder_evals': last.get('best_event_decoder_evals'),
            'last_best_iteration': last.get('best_event_iteration'),
            'final_best_makespan': self.state.best_makespan,
        }
