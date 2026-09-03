import json

from scripts.run_paper_phase6h_core import (
    ALGORITHM,
    CONFIG_PATH,
    PROTOCOL_PATH,
    build_tasks,
    partition_tasks,
    result_path,
)


def test_provisional_phase6h_core_protocol_has_225_matched_tasks():
    config = json.loads(CONFIG_PATH.read_text())
    protocol = json.loads(PROTOCOL_PATH.read_text())
    tasks = build_tasks(config)
    assert ALGORITHM == "CSG_NI_PROVISIONAL_PHASE6H"
    assert config["experiment_status"] == "PROVISIONAL"
    assert config["seeds"] == list(range(530101, 530106))
    assert protocol["primary_seeds"] == config["seeds"]
    assert protocol["supplementary_seeds"] == []
    assert len(tasks) == 225
    assert len({(task["instance_id"], task["seed"]) for task in tasks}) == 225
    assert {task["scale"] for task in tasks} == {"S", "M", "L"}


def test_provisional_phase6h_core_paths_retain_exact_algorithm_identity():
    config = json.loads(CONFIG_PATH.read_text())
    task = build_tasks(config)[0]
    path = result_path(task)
    assert "CSG_NI_PROVISIONAL_PHASE6H" in str(path)
    assert path.name == "seed_530101.json"


def test_two_shards_are_disjoint_and_cover_all_core_tasks():
    config = json.loads(CONFIG_PATH.read_text())
    tasks = build_tasks(config)
    shards = [partition_tasks(tasks, 2, index) for index in range(2)]
    shard_keys = [
        {(task["instance_id"], task["seed"]) for task in shard}
        for shard in shards
    ]
    assert [len(shard) for shard in shards] == [113, 112]
    assert len(set().union(*shard_keys)) == 225
    assert all(
        shard_keys[left].isdisjoint(shard_keys[right])
        for left in range(2)
        for right in range(left + 1, 2)
    )
