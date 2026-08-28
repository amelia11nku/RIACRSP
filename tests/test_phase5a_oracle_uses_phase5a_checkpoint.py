from pathlib import Path

from rcias_clgri.learning.experiment import load_phase3_config


def test_phase5b_oracle_uses_all_three_phase5a_best_checkpoints():
    config = load_phase3_config(Path("configs/phase5b_training.json"))
    assert config["phase5a_oracle_checkpoints"] == [
        "outputs/phase5a/seed_1/best_mean.pt",
        "outputs/phase5a/seed_2/best_mean.pt",
        "outputs/phase5a/seed_3/best_mean.pt",
    ]
    assert all(Path(path).is_file() for path in config["phase5a_oracle_checkpoints"])

