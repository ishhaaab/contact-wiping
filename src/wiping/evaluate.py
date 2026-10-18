"""Paired seeded evaluation. Final-test cases never select checkpoints."""

import numpy as np

from wiping.config import EnvConfig
from wiping.env import WipingEnv
from wiping.scripted import RasterPolicy


class EvaluationPhysicsError(RuntimeError):
    def __init__(self, rows):
        self.rows = rows
        super().__init__(f"Evaluation stopped on invalid physics: {rows[-1]}")


def evaluate(policy, suite: str, seeds: list[int], *, config: EnvConfig | None = None):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation requires nonempty unique seeds")
    env = WipingEnv(config or EnvConfig())
    rows = []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=int(seed), options={"suite": suite})
            if isinstance(policy, RasterPolicy):
                policy.reset(int(seed))
            total_reward = 0.0
            contact_seen = contact_lost = recovered = False
            while True:
                if isinstance(policy, RasterPolicy):
                    action = policy.predict(env)
                else:
                    action, _ = policy.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                in_contact = info["force_mean_n"] >= 0.5
                if contact_seen and not in_contact:
                    contact_lost = True
                if contact_lost and in_contact:
                    recovered = True
                contact_seen |= in_contact
                if terminated or truncated:
                    break
            row = {
                **info, "seed": int(seed), "suite": suite, "return": total_reward,
                "steps": env.steps, "terminated": terminated, "truncated": truncated,
                "force_mean_n": info["episode_force_mean_n"],
                "force_max_n": info["episode_force_max_n"],
                "completion_time_s": info["elapsed_s"] if info["success"] else None,
                "contact_loss": contact_lost, "contact_recovery": recovered,
            }
            row.pop("executed_action", None)
            rows.append(row)
            if (info["failure_reason"] in ("nonfinite_physics", "mujoco_warning", "out_of_bounds")
                    or not np.isfinite(obs).all()):
                raise EvaluationPhysicsError(rows)
    finally:
        env.close()
    return rows
