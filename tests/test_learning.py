"""Numerical learning and BC-to-PPO transfer regression tests."""

from pathlib import Path

import numpy as np
import pytest
import torch

from wiping.config import EnvConfig
from wiping.dataset import Episode, save_episodes
from wiping.env import WipingEnv
from wiping.learning import actor_mean, fit_bc, make_ppo, transfer_bc, validation_rank


def learnable_dataset(path: Path):
    rng = np.random.default_rng(42)
    episodes = []
    for seed, split in [(3000, "train"), (3001, "validation")]:
        obs = np.zeros((257, 273), dtype=np.float32)
        obs[:, :3] = rng.uniform(-1, 1, (257, 3))
        actions = 0.6 * obs[:-1, :3]
        trunc = np.zeros(256, dtype=bool)
        trunc[-1] = True
        episodes.append(Episode(
            obs=obs, actions=actions, rewards=np.zeros(256, dtype=np.float32),
            terminated=np.zeros(256, dtype=bool), truncated=trunc,
            metadata={"coverage": np.full(256, 0.6), "failure_reason": np.full(256, ""),
                      "success": np.zeros(256, dtype=bool)},
            seed=seed, split=split, normal_offset_m=0,
        ))
    save_episodes(episodes, path, {"profile": "test_fixture", "source": "synthetic_test_fixture"})
    return episodes


def test_bc_halves_mse_without_training_critic_and_transfers_exactly(tmp_path):
    episodes = learnable_dataset(tmp_path / "tiny.h5")
    model = make_ppo(WipingEnv(EnvConfig()), seed=11, total_steps=8192)
    obs = torch.as_tensor(episodes[1].obs[:-1])
    actions = torch.as_tensor(episodes[1].actions)
    critic = {k: v.clone() for k, v in model.policy.state_dict().items() if "value" in k}
    with torch.no_grad():
        before = torch.mean((actor_mean(model, obs) - actions) ** 2).item()
    checkpoint = fit_bc(model, tmp_path / "tiny.h5", tmp_path / "bc")
    with torch.no_grad():
        after = torch.mean((actor_mean(model, obs) - actions) ** 2).item()
    assert after < 0.5 * before
    for name, value in critic.items():
        assert torch.equal(model.policy.state_dict()[name], value)
    fresh = make_ppo(WipingEnv(EnvConfig()), seed=11, total_steps=8192)
    transfer_bc(fresh, checkpoint)
    np.testing.assert_allclose(model.predict(obs.numpy(), deterministic=True)[0],
                               fresh.predict(obs.numpy(), deterministic=True)[0], atol=1e-6)
    assert not fresh.policy.optimizer.state
    model.get_env().close()
    fresh.get_env().close()


def test_rl_update_changes_actor_and_leaves_finite_critic():
    model = make_ppo(WipingEnv(EnvConfig(max_steps=8)), seed=22, total_steps=4096)
    before = model.policy.action_net.weight.detach().clone()
    model.learn(total_timesteps=4096)
    assert model.num_timesteps == 4096 and model.n_envs == 4
    assert not torch.equal(before, model.policy.action_net.weight)
    assert all(torch.isfinite(p).all() for p in model.policy.parameters())
    model.get_env().close()


def test_checkpoint_selection_is_lexicographic():
    def rows(success, coverage, unsafe):
        return [{"success": success, "coverage": coverage, "unsafe_fraction": unsafe}]

    assert validation_rank(rows(True, 0.91, 0.04)) > validation_rank(rows(False, 0.99, 0))
    assert validation_rank(rows(False, 0.8, 0.04)) > validation_rank(rows(False, 0.7, 0))
    assert validation_rank(rows(False, 0.8, 0)) > validation_rank(rows(False, 0.8, 0.04))


def test_bc_rejects_nonfinite_demonstrations(tmp_path):
    import h5py

    path = tmp_path / "bad.h5"
    learnable_dataset(path)
    with h5py.File(path, "a") as file:
        file["episodes/3000/obs"][0, 0] = np.nan
    model = make_ppo(WipingEnv(EnvConfig()), seed=11, total_steps=8192)
    with pytest.raises(RuntimeError, match="Invalid physics"):
        fit_bc(model, path, tmp_path / "out")
    model.get_env().close()
