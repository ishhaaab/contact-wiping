"""Release orchestration must preserve artifacts and never skip partial work."""

import json
import subprocess
import sys

import numpy as np
import pytest

from wiping.config import EnvConfig, RunConfig
from wiping.env import WipingEnv
from wiping.learning import ValidationCallback, make_ppo
from wiping.release import choose_output, run_logged, training_complete


def test_output_selection_preserves_partial_directory(tmp_path):
    base = tmp_path / "run"
    base.mkdir()
    (base / "partial.txt").write_text("retain")
    chosen = choose_output(base, "attempt-1")
    assert chosen != base and not chosen.exists()
    assert (base / "partial.txt").read_text() == "retain"
    assert choose_output(tmp_path / "new", "attempt-1") == tmp_path / "new"


def test_incomplete_training_is_not_skipped(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"status": "running"}))
    assert not training_complete(tmp_path, "ppo", 11)
    (tmp_path / "run.json").write_text(json.dumps({"status": "complete", "profile": "smoke"}))
    assert not training_complete(tmp_path, "ppo", 11)


def test_failed_command_stops_and_retains_output(tmp_path):
    log = tmp_path / "failure.log"
    with pytest.raises(subprocess.CalledProcessError):
        run_logged([sys.executable, "-c", "print('diagnostic',flush=True); raise SystemExit(3)"],
                   log, tmp_path)
    assert "diagnostic" in log.read_text()
    with pytest.raises(FileExistsError):
        run_logged([sys.executable, "-c", "pass"], log, tmp_path)


def test_passive_ppo_start_logging_preserves_training(tmp_path):
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    config = RunConfig(env=EnvConfig(max_steps=8))
    vec = DummyVecEnv([lambda: Monitor(WipingEnv(config.env)) for _ in range(4)])
    model = make_ppo(vec, 11, 4096)
    model.set_logger(configure(str(tmp_path), ["csv"]))
    callback = ValidationCallback(config, tmp_path)
    model.learn(4096, callback=callback)
    record = json.loads((tmp_path / "ppo_start.json").read_text())
    assert record["initial_transitions"] == 0
    assert record["initial_policy_log_std"] == [0.0, 0.0, 0.0]
    assert record["first_rollout_transitions"] == 4096
    assert np.isfinite(record["pre_update_value_mse"])
    assert np.isfinite(record["first_optimizer_value_loss"])
    learned = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
    model.get_env().close()
    vec = DummyVecEnv([lambda: Monitor(WipingEnv(config.env)) for _ in range(4)])
    plain = make_ppo(vec, 11, 4096)
    plain.set_logger(configure(str(tmp_path / "plain"), ["csv"]))
    plain.learn(4096)
    import torch

    assert all(torch.equal(v, plain.policy.state_dict()[k]) for k, v in learned.items())
    plain.get_env().close()
