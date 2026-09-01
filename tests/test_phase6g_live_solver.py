from pathlib import Path

from rcias_clgri.data.loader import load_instance
from rcias_clgri.ni.live_policy import InterventionDecision
from rcias_clgri.ni.proposal_bank import build_live_proposal_bank
from rcias_clgri.search.alns import ALNSConfig, solve_alns
from rcias_clgri.search.csgni import CSGNIConfig, intervention_eligible, solve_csgni


ROOT = Path(__file__).resolve().parents[1]


def _event_signature(event):
    return {
        "iteration": event["iteration"],
        "decoder_evaluations": event["decoder_evaluations"],
        "current_before": event["current_before"].makespan,
        "best_before": event["best_before"].makespan,
        "candidate": event["candidate"].makespan,
        "candidate_encoding": event["candidate"].candidate,
        "current_after": event["current_after"].makespan,
        "best_after": event["best_after"].makespan,
        "destroy_operator": event["destroy_operator"],
        "repair_operator": event["repair_operator"],
        "destroyed_operation_ids": event["destroyed_operation_ids"],
        "accepted": event["accepted"],
        "new_global_best": event["new_global_best"],
        "operator_weights_before": event["operator_weights_before"],
        "operator_weights_after": event["operator_weights_after"],
        "repair_decoder_evaluations": event["repair_decoder_evaluations"],
        "candidate_trials_completed": event["candidate_trials_completed"],
        "temperature_before": event["temperature_before"],
    }


def test_zero_intervention_exactly_replays_frozen_alns():
    instance = load_instance(ROOT / "instances/tiny/tiny_03.json")
    config = ALNSConfig(candidate_trials=3, iteration_limit=30)
    baseline_events, wrapped_events = [], []
    baseline = solve_alns(instance, 60.0, 670001, config, baseline_events.append)
    wrapped = solve_csgni(
        instance,
        60.0,
        670001,
        None,
        alns_config=config,
        csgni_config=CSGNIConfig(intervention_rate=0),
        observer=wrapped_events.append,
    )
    assert [_event_signature(row) for row in wrapped_events] == [
        _event_signature(row) for row in baseline_events
    ]
    assert wrapped.best.makespan == baseline.best.makespan
    assert wrapped.best.candidate == baseline.best.candidate
    assert wrapped.decoder_evaluations == baseline.decoder_evaluations
    assert wrapped.iterations == baseline.iterations
    assert [
        (point.decoder_evaluations, point.current_best_makespan)
        for point in wrapped.convergence_trace
    ] == [
        (point.decoder_evaluations, point.current_best_makespan)
        for point in baseline.convergence_trace
    ]
    assert wrapped.diagnostics["zero_intervention_delegated_to_frozen_alns"] is True


def test_intervention_schedules_have_frozen_rates():
    assert sum(intervention_eligible(i, 20) for i in range(100)) == 20
    assert sum(intervention_eligible(i, 50) for i in range(100)) == 50
    assert sum(intervention_eligible(i, 100) for i in range(100)) == 100
    assert not any(intervention_eligible(i, 0) for i in range(100))


def test_live_proposal_bank_is_deterministic_and_outcome_blind():
    instance = load_instance(ROOT / "instances/tiny/tiny_03.json")
    events = []
    solve_alns(
        instance,
        60.0,
        1,
        ALNSConfig(candidate_trials=1, iteration_limit=1),
        events.append,
    )
    current = events[0]["current_before"]
    first, records = build_live_proposal_bank(
        instance, current, state_id="live-test", destroy_count=2, seed_namespace=670102
    )
    replay, replay_records = build_live_proposal_bank(
        instance, current, state_id="live-test", destroy_count=2, seed_namespace=670102
    )
    assert first == replay
    assert records == replay_records
    assert first.requested_arm_count == 24
    assert first.unique_arm_count == len(records)
    assert all(row["mean_relative_improvement"] == 0.0 for row in records)


class _AlwaysIntervene:
    def decide(self, instance, current, *, state_id, destroy_count, search_progress, search_stage):
        removed = tuple(instance.operations[:destroy_count])
        return InterventionDecision(
            True, state_id, "test-target", removed, .9, .1, .5, None, 20, 24, 4
        )


def test_neural_iterations_do_not_receive_alns_weight_credit():
    instance = load_instance(ROOT / "instances/tiny/tiny_03.json")
    events = []
    result = solve_csgni(
        instance,
        60.0,
        670002,
        _AlwaysIntervene(),
        alns_config=ALNSConfig(candidate_trials=2, iteration_limit=5),
        csgni_config=CSGNIConfig(intervention_rate=100),
        observer=events.append,
    )
    assert result.diagnostics["ni_interventions"] == 5
    assert all(row["operator_weights_before"] == row["operator_weights_after"] for row in events)
    assert all(row["alns_weight_credit"] is False for row in events)
    assert all(row["repair_operator"] == "transport_aware" for row in events)
