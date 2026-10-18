"""Environment contracts, including safety state across control boundaries."""

import numpy as np
import pytest

from wiping.config import EnvConfig
from wiping.coverage import CoverageGrid
from wiping.env import WipingEnv, step_reward


def test_observation_order_shape_dtype_and_reset():
    env = WipingEnv(EnvConfig())
    obs, info = env.reset(seed=11)
    assert obs.shape == (273,) and obs.dtype == np.float32
    np.testing.assert_allclose(obs[:3], env.data.qpos / 0.1)
    np.testing.assert_array_equal(obs[3:6], [0, 0, 0])
    np.testing.assert_allclose(obs[6:9], obs[:3])
    np.testing.assert_array_equal(obs[9:13], [0, 0, 0, 0])
    assert np.linalg.norm(obs[14:17]) == pytest.approx(1)
    assert np.count_nonzero(obs[17:]) == 0
    assert info["coverage"] == 0 and info["elapsed_s"] == 0
    env.coverage.update(env.coverage.centres[0], 1)
    env.step([1, 0, 0])
    obs, _ = env.reset(seed=11)
    assert not obs[17:].any() and not obs[10:13].any()
    assert env.unsafe_substeps == 0 and env.excessive_count == 0
    env.close()


def test_same_seed_and_actions_reproduce():
    a, b = WipingEnv(EnvConfig()), WipingEnv(EnvConfig())
    oa, ia = a.reset(seed=22)
    ob, ib = b.reset(seed=22)
    np.testing.assert_array_equal(oa, ob)
    assert ia == ib
    rng = np.random.default_rng(17)
    for action in rng.uniform(-1, 1, (30, 3)):
        xa, xb = a.step(action), b.step(action)
        np.testing.assert_allclose(xa[0], xb[0], atol=1e-6)
        assert xa[1:] == xb[1:]
        if xa[2] or xa[3]:
            break


def test_coverage_requires_force_and_is_monotonic():
    grid = CoverageGrid(0.26)
    point = grid.centres[0].copy()
    assert grid.update(point, 0) == 0
    assert grid.update(point, 0.49) == 0
    assert grid.update(point, 8.01) == 0
    assert grid.update(point + [0, 0, 0.02], 1) == 0
    assert grid.update(point, 0.5) == 1 / 256
    assert grid.update(point, 8) == 0
    assert grid.fraction == 1 / 256


def test_hover_does_not_mark_grid():
    env = WipingEnv(EnvConfig())
    env.reset(seed=11)
    for _ in range(10):
        _, _, terminated, _, info = env.step([0, 0, 0])
        assert not terminated
        assert info["coverage"] == 0
        assert info["force_mean_n"] < 1e-4


def test_action_clipping_and_incremental_target():
    env = WipingEnv(EnvConfig())
    env.reset(seed=11)
    initial = env.target.copy()
    obs, _, _, _, info = env.step([2, -3, 0.5])
    np.testing.assert_allclose(env.target - initial, [0.004, -0.004, 0.001])
    np.testing.assert_allclose(obs[10:13], [1, -1, 0.5])
    assert info["action_clipped"] is True
    assert info["executed_action"] == [1.0, -1.0, 0.5]
    env.step([0, 0, 0])
    np.testing.assert_allclose(env.target - initial, [0.004, -0.004, 0.001])


@pytest.mark.parametrize("action", [[1, 2], [[1, 2, 3]], [np.nan, 0, 0], [0, np.inf, 0]])
def test_invalid_actions_rejected(action):
    env = WipingEnv(EnvConfig())
    env.reset(seed=11)
    with pytest.raises(ValueError):
        env.step(action)


def test_exact_reward_from_synthetic_measurements():
    assert step_reward(0.1, np.array([0, 4, 12, 16]),
                       np.array([1, 0, 0]), np.zeros(3)) == pytest.approx(0.96925)


def test_excessive_force_counter_crosses_control_boundary(monkeypatch):
    forces = iter([0.0] * 23 + [16.0] * 3)
    monkeypatch.setattr("wiping.env.read_normal_force", lambda *_: next(forces))
    env = WipingEnv(EnvConfig())
    env.reset(seed=11)
    first = env.step([0, 0, 0])
    assert not first[2] and not first[3]
    _, reward, terminated, truncated, info = env.step([0, 0, 0])
    assert terminated and not truncated and reward < -2
    assert info["failure_reason"] == "excessive_force"
    assert info["elapsed_s"] == pytest.approx(0.052)
    assert info["unsafe_fraction"] == pytest.approx(3 / 26)


def test_timeout_is_not_success_or_physical_failure():
    env = WipingEnv(EnvConfig(max_steps=1))
    env.reset(seed=11)
    _, _, terminated, truncated, info = env.step([0, 0, 0])
    assert not terminated and truncated
    assert info["success"] is False and info["failure_reason"] is None
    with pytest.raises(RuntimeError):
        env.step([0, 0, 0])


def test_success_bonus_once_and_unsafe_condition(monkeypatch):
    env = WipingEnv(EnvConfig())
    env.reset(seed=11)
    env.coverage.covered[:231] = True
    _, reward, terminated, truncated, info = env.step([0, 0, 0])
    assert reward == pytest.approx(1.998)
    assert terminated and not truncated and info["success"]
    env.reset(seed=11)
    env.coverage.covered[:231] = True
    monkeypatch.setattr("wiping.env.read_normal_force", lambda *_: 9.0)
    assert env.step([0, 0, 0])[4]["success"] is False


@pytest.mark.parametrize("position,reason", [([0.151, 0, 0], "out_of_bounds"),
                                            ([np.nan, 0, 0], "nonfinite_physics")])
def test_invalid_state_is_failure_before_mujoco_can_reset_it(position, reason):
    env = WipingEnv(EnvConfig())
    env.reset(seed=11)
    env.data.qpos[:] = position
    _, reward, terminated, truncated, info = env.step([0, 0, 0])
    assert terminated and not truncated and reward == pytest.approx(-2.002)
    assert info["failure_reason"] == reason


def test_sb3_environment_checker():
    from stable_baselines3.common.env_checker import check_env

    check_env(WipingEnv(EnvConfig()), warn=True)


def test_config_profiles_and_unknown_fields(tmp_path):
    from wiping.config import load_config

    full = load_config("configs/full.json")
    assert full.demo_episodes == 200 and full.total_steps == 2_000_000
    assert full.env.max_steps == 400
    bad = tmp_path / "bad.json"
    bad.write_text('{"profile":"smoke","reward_scale":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(bad)


def test_training_seed_validation_excludes_evaluation():
    from wiping.config import RunConfig

    for seed in [1000, 2000, 3000]:
        with pytest.raises(ValueError, match="seed"):
            RunConfig(seed=seed)
