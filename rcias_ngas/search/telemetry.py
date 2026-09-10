"""Separate completed-work events, budget checkpoints, and best events.

Call observe after EVERY completed decoder/critic call and state transition.
Checkpoints crossed during an atomic call retain the pre-call state; no future
work or incumbent is attributed backwards to an earlier wall-clock checkpoint.
"""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class RunState:
    decoder_evals: int = 0
    neural_calls: int = 0
    current_makespan: float | None = None
    best_makespan: float | None = None
    iteration: int = 0
    accepted_moves: int = 0
    improving_moves: int = 0
    new_best_moves: int = 0


class Telemetry:
    def __init__(self, budget_seconds, fractions=(.10, .25, .50, .75, 1.0)):
        if not math.isfinite(budget_seconds) or budget_seconds <= 0:
            raise ValueError("Budget must be finite and positive")
        if tuple(sorted(set(fractions))) != tuple(fractions) or any(not 0 < f <= 1 for f in fractions):
            raise ValueError("Checkpoint fractions must be unique and increasing")
        self.budget = budget_seconds
        self.fractions = tuple(fractions)
        self.state = RunState()
        self.elapsed = 0.0
        self.checkpoints = []
        self.best_events = []

    def _checkpoint(self, fraction, state):
        self.checkpoints.append({
            "budget_fraction": fraction, "elapsed_time_sec": fraction * self.budget,
            **asdict(state),
        })

    def observe(self, elapsed, state):
        if not math.isfinite(elapsed) or elapsed < self.elapsed:
            raise ValueError("Event clock regressed")
        for name in ("decoder_evals", "neural_calls", "iteration", "accepted_moves", "improving_moves", "new_best_moves"):
            if getattr(state, name) < getattr(self.state, name):
                raise ValueError(f"Cumulative counter regressed: {name}")
        if self.state.best_makespan is not None and (state.best_makespan is None or state.best_makespan > self.state.best_makespan):
            raise ValueError("Best makespan regressed")
        for fraction in self.fractions[len(self.checkpoints):]:
            deadline = fraction * self.budget
            if deadline > elapsed:
                break
            self._checkpoint(fraction, state if deadline == elapsed else self.state)
        if state.best_makespan is not None and (self.state.best_makespan is None or state.best_makespan < self.state.best_makespan):
            self.best_events.append({
                "best_event_time_sec": elapsed,
                "best_event_decoder_evals": state.decoder_evals,
                "best_event_iteration": state.iteration,
                "best_makespan": state.best_makespan,
            })
        self.elapsed, self.state = elapsed, state

    def finish(self, elapsed):
        self.observe(elapsed, self.state)
        last = self.best_events[-1] if self.best_events else {}
        return {
            "budget_checkpoints": self.checkpoints, "best_events": self.best_events,
            "termination": {"elapsed_time_sec": elapsed, **asdict(self.state)},
            "last_best_time_sec": last.get("best_event_time_sec"),
            "last_best_decoder_evals": last.get("best_event_decoder_evals"),
            "last_best_iteration": last.get("best_event_iteration"),
            "final_best_makespan": self.state.best_makespan,
        }
