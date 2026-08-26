from __future__ import annotations

from scripts.train_bc_pretrain import _configure_capacity_study, _demonstration_plan


def test_phase4_demonstration_plan_is_level_stratified_and_seed_unique():
    plan = _demonstration_plan(
        {"demonstration_instances": {"S": 2, "M": 1, "L": 2}}, 1000
    )
    assert plan == [
        ("S", 1000), ("S", 1001), ("M", 1002), ("L", 1003), ("L", 1004)
    ]


def test_phase3_demonstration_plan_remains_round_robin_compatible():
    plan = _demonstration_plan(
        {"demonstration_episodes": 5, "levels": ["S", "M"]}, 2000
    )
    assert plan == [
        ("S", 2000), ("M", 2001), ("S", 2002), ("M", 2003), ("S", 2004)
    ]


def test_capacity_study_overrides_only_model_size_and_pilot_bc_budget():
    config = {
        "model": {"embedding_dim": 32, "heads": 4, "layers": 2},
        "bc_warm_start": {"demonstration_instances": {"S": 100, "M": 100, "L": 50}, "epochs": 20},
        "model_capacity_study": [
            {"name": "D64-L3", "embedding_dim": 64, "layers": 3}
        ],
        "capacity_study_settings": {
            "demonstration_instances": {"S": 20, "M": 5, "L": 1},
            "epochs": 1,
        },
    }
    model, bc = _configure_capacity_study(config, "D64-L3")
    assert model == {"embedding_dim": 64, "heads": 4, "layers": 3}
    assert bc["demonstration_instances"] == {"S": 20, "M": 5, "L": 1}
    assert bc["epochs"] == 1
