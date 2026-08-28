from rcias_clgri.analysis.phase6a import Phase6AObserver
from rcias_clgri.data.loader import load_instance
from rcias_clgri.search.alns import ALNSConfig, solve_alns


def test_phase6a_observer_is_search_passive():
    root = Path(__file__).resolve().parents[1]
    instance = load_instance(root / "instances/tiny/tiny_03.json")
    config = ALNSConfig(candidate_trials=2, iteration_limit=25)
    compact = []
    solve_alns(instance, 60.0, 610001, config, lambda event: compact.append((
        event["destroy_operator"], event["repair_operator"], event["accepted"],
        event["new_global_best"], event["candidate"].makespan,
    )))
    detailed = Phase6AObserver(instance, {
        "run_id": "test", "instance_id": instance.instance_id, "suite": "TINY",
        "scale": "TINY", "CF_level": "NA", "seed": 610001,
    })
    result = solve_alns(instance, 60.0, 610001, config, detailed)
    expanded = [(
        row["destroy_operator"], row["repair_operator"], row["accepted"],
        row["new_global_best"], row["candidate_makespan"],
    ) for row in detailed.transitions]
    assert expanded == compact
    assert result.iterations == 25
    assert len(detailed.targets) == sum(row["destroy_count"] for row in detailed.transitions)
from pathlib import Path
