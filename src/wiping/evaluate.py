"""Paired seeded evaluation. Final-test cases never select checkpoints."""

from collections import defaultdict

import numpy as np

from wiping.config import TEST_SEEDS, TRAINING_SEEDS, EnvConfig
from wiping.env import WipingEnv
from wiping.scripted import RasterPolicy

REQUIRED_METRICS = {
    "profile", "method", "training_seed", "suite", "seed", "coverage", "force_mean_n",
    "force_max_n", "unsafe_fraction", "elapsed_s", "success", "terminated", "truncated",
    "failure_reason", "completion_time_s",
}


def _boolean(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value in ("True", "true", "1"):
        return True
    if value in ("False", "false", "0"):
        return False
    raise ValueError(f"Invalid boolean metric: {value!r}")


def normalize_rows(rows: list[dict]) -> list[dict]:
    if not rows:
        raise ValueError("No evaluation episodes")
    result, identities = [], set()
    for raw in rows:
        missing = REQUIRED_METRICS - raw.keys()
        if missing:
            raise ValueError(f"Missing required metrics: {sorted(missing)}")
        row = dict(raw)
        if row["method"] not in ("scripted", "bc", "ppo", "bc_ppo"):
            raise ValueError("Unknown evaluation method")
        if row["suite"] not in ("id", "curvature", "friction"):
            raise ValueError("Unknown evaluation suite")
        if row["profile"] not in ("smoke", "pilot", "full"):
            raise ValueError("Unknown evaluation profile")
        row["training_seed"] = (None if row["training_seed"] in (None, "")
                                else int(row["training_seed"]))
        if row["method"] == "scripted" and row["training_seed"] is not None:
            raise ValueError("The scripted baseline has no training seed")
        if row["method"] != "scripted" and row["training_seed"] not in TRAINING_SEEDS:
            raise ValueError("Learned policies require an approved training seed")
        row["seed"] = int(row["seed"])
        for key in ("coverage", "force_mean_n", "force_max_n", "unsafe_fraction", "elapsed_s"):
            row[key] = float(row[key])
            if not np.isfinite(row[key]) or row[key] < 0:
                raise ValueError(f"Invalid metric: {key}")
        if row["coverage"] > 1 or row["unsafe_fraction"] > 1:
            raise ValueError("Coverage and unsafe fraction must lie in [0,1]")
        for key in ("success", "terminated", "truncated"):
            row[key] = _boolean(row[key])
        row["failure_reason"] = row["failure_reason"] or None
        if row["terminated"] == row["truncated"]:
            raise ValueError("A completed episode must terminate or truncate, never both")
        if row["success"] and (row["truncated"] or row["failure_reason"]
                               or row["coverage"] < 0.90 or row["unsafe_fraction"] > 0.05):
            raise ValueError("Inconsistent success metrics")
        completion = row["completion_time_s"]
        row["completion_time_s"] = None if completion in (None, "") else float(completion)
        if row["success"]:
            if (row["completion_time_s"] is None
                    or not np.isclose(row["completion_time_s"], row["elapsed_s"])):
                raise ValueError("Successful completion time must match elapsed time")
        elif row["completion_time_s"] is not None:
            raise ValueError("Unsuccessful episodes have no completion time")
        identity = tuple(row[k] for k in ("profile", "method", "training_seed", "suite", "seed"))
        if identity in identities:
            raise ValueError(f"Duplicate evaluation case: {identity}")
        identities.add(identity)
        result.append(row)
    return result


def aggregate(rows: list[dict]) -> dict:
    rows = normalize_rows(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["profile"], row["suite"], row["method"])].append(row)
    groups = []
    for (profile, suite, method), cases in sorted(grouped.items()):
        per_model = []
        seeds = sorted({r["training_seed"] for r in cases}, key=lambda x: -1 if x is None else x)
        for seed in seeds:
            model_rows = [r for r in cases if r["training_seed"] == seed]
            successes = [r for r in model_rows if r["success"]]
            per_model.append({
                "training_seed": seed, "episodes": len(model_rows),
                "evaluation_seeds": sorted(r["seed"] for r in model_rows),
                "coverage": float(np.mean([r["coverage"] for r in model_rows])),
                "force_mean_n": float(np.mean([r["force_mean_n"] for r in model_rows])),
                "force_max_n": float(max(r["force_max_n"] for r in model_rows)),
                "unsafe_fraction": float(np.mean([r["unsafe_fraction"] for r in model_rows])),
                "elapsed_s": float(np.mean([r["elapsed_s"] for r in model_rows])),
                "success_rate": len(successes) / len(model_rows),
                "completion_time_s": float(np.mean([r["elapsed_s"] for r in successes]))
                if successes else None,
            })
        metrics = {}
        for key in ("coverage", "force_mean_n", "force_max_n", "unsafe_fraction",
                    "elapsed_s", "success_rate", "completion_time_s"):
            values = [r[key] for r in per_model if r[key] is not None]
            metrics[key] = {
                "mean": float(np.mean(values)) if values else None,
                "std_across_seeds": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "contributing_models": len(values),
            }
        groups.append({
            "profile": profile, "suite": suite, "method": method,
            "episodes": len(cases), "paired_case_count": len({r["seed"] for r in cases}),
            "n_training_seeds": 0 if method == "scripted" else len(seeds),
            "training_seeds": [] if method == "scripted" else seeds,
            "per_training_seed": per_model, "metrics": metrics,
        })
    return {"episodes": len(rows), "groups": groups,
            "notes": ["All episodes, including failures, contribute to coverage and success.",
                      "Force means include non-contact substeps.",
                      "Completion time is conditional on success; null means no successes.",
                      "Across-seed SD is sample SD; null for fewer than two models."]}


def validate_full_protocol(rows):
    rows = normalize_rows(rows)
    expected = {(method, training_seed, suite, seed)
                for method in ("scripted", "bc", "ppo", "bc_ppo")
                for training_seed in ([None] if method == "scripted" else TRAINING_SEEDS)
                for suite in ("id", "curvature", "friction") for seed in TEST_SEEDS}
    actual = {(r["method"], r["training_seed"], r["suite"], r["seed"]) for r in rows}
    if actual != expected or any(r["profile"] != "full" for r in rows):
        raise ValueError("Incomplete full protocol: require all 3000 paired final-test episodes")


class EvaluationPhysicsError(RuntimeError):
    def __init__(self, rows):
        self.rows = rows
        super().__init__(f"Evaluation stopped on invalid physics: {rows[-1]}")


def rollout_episode(env: WipingEnv, policy, suite: str, seed: int, on_frame=None) -> dict:
    obs, info = env.reset(seed=int(seed), options={"suite": suite})
    if isinstance(policy, RasterPolicy):
        policy.reset(int(seed))
    if on_frame:
        on_frame(env, info)
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
        if on_frame:
            on_frame(env, info)
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
    return row


def evaluate(policy, suite: str, seeds: list[int], *, config: EnvConfig | None = None):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation requires nonempty unique seeds")
    env = WipingEnv(config or EnvConfig())
    rows = []
    try:
        for seed in seeds:
            row = rollout_episode(env, policy, suite, seed)
            rows.append(row)
            if row["failure_reason"] in ("nonfinite_physics", "mujoco_warning", "out_of_bounds"):
                raise EvaluationPhysicsError(rows)
    finally:
        env.close()
    return rows
