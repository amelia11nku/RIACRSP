import copy

import pytest

from scripts.build_phase5b_holdout import validate_seed_boundaries


def _config():
    return {
        "phase5b_holdout_seeds": {
            "S": list(range(100, 120)),
            "M": list(range(120, 140)),
            "L": list(range(140, 160)),
        },
        "phase5b_structural_scenarios": {"seeds": list(range(200, 210))},
        "development_seeds": {"S": [1], "M": [2], "L": [3]},
        "historical_validation_seeds": {"S": [4], "M": [5], "L": [6]},
        "structural_scenarios": {"seeds": [7, 8, 9]},
        "training_seed_policy": {"independent_training_seeds": [300, 301, 302]},
    }


def test_phase5b_holdout_and_structural_seeds_are_disjoint():
    validate_seed_boundaries(_config())


def test_phase5b_holdout_rejects_prior_seed_overlap():
    config = copy.deepcopy(_config())
    config["historical_validation_seeds"]["S"] = [100]
    with pytest.raises(ValueError, match="overlap"):
        validate_seed_boundaries(config)

