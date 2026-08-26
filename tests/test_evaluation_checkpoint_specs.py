from pathlib import Path

from scripts.evaluate_ppo import _checkpoint_specs


def test_phase4_checkpoint_specs_use_large_bc_directory():
    config = {
        "bc_warm_start": {"demonstration_instances": {"S": 1}},
        "training_seed_policy": {"independent_training_seeds": [11, 12, 13]},
    }
    specs = _checkpoint_specs(config, Path("outputs/phase4"))
    assert specs[0][2] == Path("outputs/phase4/bc_large/best.pt")
    assert [item[2].name for item in specs[1:]] == ["best.pt"] * 3


def test_phase3_checkpoint_specs_remain_compatible():
    config = {
        "bc_warm_start": {"demonstration_episodes": 1},
        "training_seed_policy": {"independent_training_seeds": [1]},
    }
    specs = _checkpoint_specs(config, Path("outputs/phase3"))
    assert specs[0][2] == Path("outputs/phase3/bc_pretrain/best.pt")
