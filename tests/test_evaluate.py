"""Evaluation preserves failures and passes only observations to learned policies."""

import numpy as np

from wiping.config import EnvConfig
from wiping.evaluate import evaluate


class HoverPolicy:
    def predict(self, obs, deterministic=True):
        assert isinstance(obs, np.ndarray) and obs.shape == (273,)
        return np.zeros(3, dtype=np.float32), None


def test_evaluation_retains_timeouts_and_only_exposes_observations():
    rows = evaluate(HoverPolicy(), "id", [1000, 1001], config=EnvConfig(max_steps=2))
    assert len(rows) == 2 and [r["seed"] for r in rows] == [1000, 1001]
    assert all(r["truncated"] and not r["terminated"] and not r["success"] for r in rows)
    assert all(r["coverage"] == 0 and r["elapsed_s"] == 0.1 for r in rows)
