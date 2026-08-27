import pytest

from scripts.build_phase5a_development import _validate_seed_boundaries


def _config():
    return {
        "development_seeds": {
            "S": list(range(10)), "M": list(range(10, 20)), "L": list(range(20, 30))
        },
        "historical_validation_seeds": {"S": [100], "M": [101], "L": [102]},
        "training_seed_policy": {"independent_training_seeds": [200, 201, 202]},
    }


def test_phase5a_development_seeds_are_disjoint():
    _validate_seed_boundaries(_config())


def test_phase5a_development_rejects_overlap():
    config = _config()
    config["historical_validation_seeds"]["S"] = [0]
    with pytest.raises(ValueError, match="overlap"):
        _validate_seed_boundaries(config)
