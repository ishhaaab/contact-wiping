"""Catch split leakage, lost terminal observations, and fabricated quality gates."""

import hashlib
import json

import h5py
import numpy as np
import pytest

from wiping.config import EnvConfig, RunConfig
from wiping.dataset import (
    DemonstrationGateError,
    Episode,
    assert_demonstration_gate,
    collect_demonstrations,
    load_episodes,
    save_episodes,
)
from wiping.env import WipingEnv
from wiping.scripted import RasterPolicy


def fixture_episode(seed, length, coverage=0.6, reason="", split="train"):
    obs = np.zeros((length + 1, 273), dtype=np.float32)
    obs[:, 0] = np.arange(length + 1)
    actions = np.zeros((length, 3), dtype=np.float32)
    actions[:, 0] = np.arange(length) / max(1, length)
    terminated = np.zeros(length, dtype=bool)
    truncated = np.zeros(length, dtype=bool)
    terminated[-1] = bool(reason)
    truncated[-1] = not bool(reason)
    return Episode(
        obs=obs, actions=actions, rewards=np.zeros(length, dtype=np.float32),
        terminated=terminated, truncated=truncated,
        metadata={"coverage": np.full(length, coverage),
                  "failure_reason": np.array([""] * (length - 1) + [reason]),
                  "success": np.zeros(length, dtype=bool)},
        seed=seed, split=split, normal_offset_m=0.0,
    )


def test_unequal_lengths_alignment_and_episode_split(tmp_path):
    path = tmp_path / "fixture.h5"
    episodes = [fixture_episode(3000, 2), fixture_episode(3001, 5, split="validation")]
    save_episodes(episodes, path, {"profile": "test_fixture"})
    train, val = load_episodes(path, "train"), load_episodes(path, "validation")
    assert {e.seed for e in train}.isdisjoint(e.seed for e in val)
    assert train[0].obs.shape == (3, 273) and val[0].obs.shape == (6, 273)
    np.testing.assert_array_equal(val[0].obs[:-1, 0], np.arange(5))
    np.testing.assert_allclose(val[0].actions[:, 0], [0, 0.2, 0.4, 0.6, 0.8])
    assert val[0].obs[-1, 0] == 5 and val[0].truncated[-1]
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    assert manifest["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with h5py.File(path) as file:
        assert file.attrs["schema_version"] == 1


def test_misaligned_transition_arrays_rejected(tmp_path):
    episode = fixture_episode(3000, 2)
    episode.obs = episode.obs[:-1]
    with pytest.raises(ValueError, match="observations"):
        save_episodes([episode], tmp_path / "bad.h5", {})


def test_gate_includes_failed_episodes(tmp_path):
    episodes = [fixture_episode(3000, 2, coverage=0.6)]
    episodes += [fixture_episode(3000 + i, 2, coverage=0.7, reason="excessive_force")
                 for i in range(1, 5)]
    path = tmp_path / "twenty_percent.h5"
    save_episodes(episodes, path, {})
    assert assert_demonstration_gate(path)["qualifying_fraction"] == 0.2
    episodes.append(fixture_episode(3005, 2, coverage=0.1))
    path = tmp_path / "below_twenty.h5"
    save_episodes(episodes, path, {})
    with pytest.raises(DemonstrationGateError):
        assert_demonstration_gate(path)
    assert len(load_episodes(path, "all")) == 6


def test_scripted_actions_are_seeded_clipped_and_surface_following():
    env = WipingEnv(EnvConfig())
    env.reset(seed=3000)
    a, b = RasterPolicy(seed=3000), RasterPolicy(seed=3000)
    np.testing.assert_array_equal(a.predict(env), b.predict(env))
    assert -0.003 <= a.normal_offset_m <= 0.003
    for _ in range(20):
        action = a.predict(env)
        assert action.shape == (3,) and action.dtype == np.float32
        assert np.max(np.abs(action)) <= 1
        result = env.step(action)
        if result[2] or result[3]:
            break


def test_collection_retains_all_failed_episodes_before_gate_error(tmp_path):
    config = RunConfig(env=EnvConfig(max_steps=2))
    path = tmp_path / "failed.h5"
    with pytest.raises(DemonstrationGateError):
        collect_demonstrations(config, path)
    train, val = load_episodes(path, "train"), load_episodes(path, "validation")
    assert [e.seed for e in train] == list(range(3000, 3006))
    assert [e.seed for e in val] == [3006, 3007]
    assert all(e.truncated[-1] for e in train + val)


def test_imperfection_does_not_reuse_surface_random_stream():
    env = WipingEnv(EnvConfig())
    radii, offsets = [], []
    for seed in range(3000, 3200):
        _, info = env.reset(seed=seed)
        radii.append(info["radius_m"])
        offsets.append(RasterPolicy(seed=seed).normal_offset_m)
    env.close()
    assert abs(np.corrcoef(radii, offsets)[0, 1]) < 0.25
