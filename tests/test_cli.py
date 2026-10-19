import json

import pytest

from wiping.cli import main


def test_cli_evaluates_scripted_baseline_without_checkpoint(tmp_path):
    config = tmp_path / "short.json"
    config.write_text(json.dumps({"profile": "smoke", "seed": 11, "demo_episodes": 8,
                                  "total_steps": 8192, "env": {"max_steps": 1}}))
    output = tmp_path / "eval"
    assert main(["evaluate", "--method", "scripted", "--config", str(config),
                 "--split", "validation", "--suite", "id", "--output", str(output)]) == 0
    summary = json.loads((output / "summary.json").read_text())
    group = summary["groups"][0]
    assert group["episodes"] == 20 and group["n_training_seeds"] == 0
    assert group["profile"] == "smoke"


def test_cli_requires_checkpoint_for_learned_method(tmp_path):
    with pytest.raises(ValueError, match="checkpoint"):
        main(["evaluate", "--method", "ppo", "--suite", "id", "--output", str(tmp_path)])
