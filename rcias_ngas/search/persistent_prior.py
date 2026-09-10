"""Persistent frozen-neural priors and explicit refresh decisions."""
from __future__ import annotations

from dataclasses import dataclass
import math


def normalized_prior(advantages: list[float], scale: float,
                     uniform_mix: float) -> tuple[float, ...]:
    if not advantages or scale <= 0 or not 0 <= uniform_mix < 1:
        raise ValueError('Invalid persistent-prior configuration')
    if not all(math.isfinite(value) for value in advantages):
        raise FloatingPointError('Non-finite advantage')
    logits = [max(-40., min(40., value / scale)) for value in advantages]
    ceiling = max(logits)
    weights = [math.exp(value - ceiling) for value in logits]
    total = sum(weights)
    neural = [value / total for value in weights]
    uniform = 1. / len(neural)
    return tuple((1. - uniform_mix) * value + uniform_mix * uniform for value in neural)


def distribution_summary(actions: tuple, probabilities: tuple[float, ...]) -> dict:
    order = sorted(range(len(actions)), key=lambda index: (-probabilities[index], actions[index].action_id))
    entropy = -sum(value * math.log(max(value, 1e-300)) for value in probabilities)
    uniform = 1. / len(probabilities)
    return {
        'action_count': len(actions),
        'entropy': entropy,
        'effective_action_count': math.exp(entropy),
        'total_variation_from_uniform': .5 * sum(abs(value - uniform) for value in probabilities),
        'top_actions': [
            {'action_id': actions[index].action_id, 'probability': probabilities[index],
             'size': actions[index].size, 'repair': actions[index].repair}
            for index in order[:5]
        ],
    }


@dataclass(frozen=True)
class PriorCache:
    state_id: str
    refresh_iteration: int
    actions: tuple
    probabilities: tuple[float, ...]
    advantages: tuple[float, ...]
    beats_fallback_probability: tuple[float, ...]
    refresh_reasons: tuple[str, ...]
    state_feature_hash: str
    graph_hash: str
    critical_signature: str
    dominant_bottleneck: str | None


@dataclass(frozen=True)
class RefreshConfig:
    fixed_horizon: int = 20
    maximum_age: int = 30
    minimum_spacing: int = 5
    stagnation_horizon: int = 15
    repeated_high_prior_failures: int = 3
    meaningful_improvement_fraction: float = .001


def refresh_reasons(*, mode: str, iteration: int, cache: PriorCache | None,
                    new_global_best: bool, meaningful_current_improvement: bool,
                    critical_structure_changed: bool, bottleneck_changed: bool,
                    stagnation_age: int, high_prior_failures: int,
                    stage_changed: bool, config: RefreshConfig) -> tuple[str, ...]:
    if cache is None:
        return ('initial_state',)
    age = iteration - cache.refresh_iteration
    if mode == 'fixed':
        return ('fixed_horizon',) if age >= config.fixed_horizon else ()
    if mode != 'event':
        return ()
    if age < config.minimum_spacing:
        return ()
    reasons = []
    if new_global_best:
        reasons.append('new_global_best')
    if meaningful_current_improvement:
        reasons.append('meaningful_current_improvement')
    if critical_structure_changed:
        reasons.append('critical_structure_change')
    if bottleneck_changed:
        reasons.append('bottleneck_dominance_change')
    if stagnation_age >= config.stagnation_horizon:
        reasons.append('stagnation')
    if high_prior_failures >= config.repeated_high_prior_failures:
        reasons.append('repeated_high_prior_failure')
    if age >= config.maximum_age:
        reasons.append('maximum_age')
    if stage_changed:
        reasons.append('search_stage_change')
    return tuple(reasons)
