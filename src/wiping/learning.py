"""Behaviour cloning and comparable PPO policies."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import h5py
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from wiping.artifacts import run_provenance, write_csv, write_json
from wiping.config import TRAINING_SEEDS, VALIDATION_SEEDS, RunConfig
from wiping.dataset import assert_demonstration_gate, load_episodes
from wiping.env import WipingEnv
from wiping.evaluate import EvaluationPhysicsError, evaluate

PPO_SETTINGS = {
    "learning_rate": 3e-4, "n_steps": 1024, "batch_size": 256, "n_epochs": 10,
    "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0,
    "vf_coef": 0.5, "max_grad_norm": 0.5,
}


def make_ppo(env, seed: int, total_steps: int) -> PPO:
    if seed not in TRAINING_SEEDS or total_steps <= 0:
        raise ValueError("Use an approved training seed and positive transition budget")
    torch.set_num_threads(1)
    if isinstance(env, WipingEnv):
        config = env.config
        env = DummyVecEnv([lambda env=env: env] + [lambda: WipingEnv(config) for _ in range(3)])
    if not isinstance(env, VecEnv) or env.num_envs != 4:
        raise ValueError("PPO requires exactly four environments")
    return PPO(
        "MlpPolicy", env, seed=seed, device="cpu", verbose=0, **PPO_SETTINGS,
        policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]},
                       "activation_fn": torch.nn.Tanh},
    )


def actor_mean(model: PPO, obs: torch.Tensor) -> torch.Tensor:
    return model.policy.get_distribution(obs.to(model.device)).distribution.mean


def fit_bc(model: PPO, dataset: Path, output: Path) -> Path:
    assert_demonstration_gate(dataset)
    train_episodes = load_episodes(dataset, "train")
    val_episodes = load_episodes(dataset, "validation")
    if not train_episodes or not val_episodes:
        raise ValueError("BC requires nonempty episode-level train and validation splits")
    if {e.seed for e in train_episodes} & {e.seed for e in val_episodes}:
        raise ValueError("BC splits overlap")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "best_bc.zip"
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    x = torch.as_tensor(np.concatenate([e.obs[:-1] for e in train_episodes]))
    y = torch.as_tensor(np.concatenate([e.actions for e in train_episodes]))
    vx = torch.as_tensor(np.concatenate([e.obs[:-1] for e in val_episodes]))
    vy = torch.as_tensor(np.concatenate([e.actions for e in val_episodes]))
    parameters = list(model.policy.pi_features_extractor.parameters())
    parameters += list(model.policy.mlp_extractor.policy_net.parameters())
    parameters += list(model.policy.action_net.parameters())
    parameters = list({id(p): p for p in parameters}.values())
    optimizer = torch.optim.Adam(parameters, lr=3e-4)
    generator = torch.Generator().manual_seed(model.seed)
    best, stale, history = float("inf"), 0, []
    started = perf_counter()
    model.policy.set_training_mode(True)
    for epoch in range(1, 101):
        order = torch.randperm(len(x), generator=generator)
        for indices in order.split(256):
            optimizer.zero_grad()
            loss = torch.mean((actor_mean(model, x[indices]) - y[indices]) ** 2)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite BC loss; training stopped")
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            train_mse = torch.mean((actor_mean(model, x) - y) ** 2).item()
            val_mse = torch.mean((actor_mean(model, vx) - vy) ** 2).item()
        history.append({"epoch": epoch, "train_mse": train_mse, "validation_mse": val_mse})
        if not np.isfinite([train_mse, val_mse]).all():
            raise RuntimeError("Nonfinite BC metrics; training stopped")
        if val_mse < best:
            best, stale = val_mse, 0
            model.save(checkpoint)
        else:
            stale += 1
        if stale >= 10:
            break
    transfer_bc(model, checkpoint)
    write_csv(output / "bc_history.csv", history)
    write_json(output / "bc_metrics.json", {
        "best_validation_mse": best, "epochs": len(history),
        "wall_seconds": perf_counter() - started,
        "train_episodes": len(train_episodes), "validation_episodes": len(val_episodes),
        "train_transitions": len(x), "validation_transitions": len(vx),
        "seed": model.seed,
        "dataset_sha256": hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
        "checkpoint": str(checkpoint), "critic_fitted": False,
    })
    return checkpoint


def transfer_bc(model: PPO, checkpoint: Path):
    saved = PPO.load(checkpoint, device="cpu")
    model.policy.load_state_dict(saved.policy.state_dict(), strict=True)
    # Deliberately do not load saved.policy.optimizer, timesteps, or rollout buffer.
    if any(not torch.isfinite(p).all() for p in model.policy.parameters()):
        raise RuntimeError("Checkpoint contains nonfinite parameters")


def validation_rank(rows: list[dict]) -> tuple[float, float, float]:
    return (float(np.mean([r["success"] for r in rows])),
            float(np.mean([r["coverage"] for r in rows])),
            -float(np.mean([r["unsafe_fraction"] for r in rows])))


class FailFastEnv(gym.Wrapper):
    """Prevent vector auto-reset from concealing simulator failures."""

    def __init__(self, env, diagnostic: Path):
        super().__init__(env)
        self.diagnostic = diagnostic

    def step(self, action):
        transition = self.env.step(action)
        obs, reward, _, _, info = transition
        if (info["failure_reason"] in ("nonfinite_physics", "mujoco_warning", "out_of_bounds")
                or not np.isfinite(obs).all() or not np.isfinite(reward)):
            write_json(self.diagnostic, {"info": info, "raw_observation": repr(obs),
                                         "action": np.asarray(action).tolist()})
            raise RuntimeError(f"Invalid physics; retained diagnostics at {self.diagnostic}")
        return transition


class ValidationCallback(BaseCallback):
    def __init__(self, config: RunConfig, output: Path):
        super().__init__()
        self.config, self.output = config, output
        self.next_eval = 50_000
        self.best_rank = None
        self.best_steps = 0
        self.validation_seconds = 0.0
        self.validation_rows, self.episode_rows = [], []
        self.start_diagnostics = {}

    def _on_training_start(self):
        self.start_diagnostics = {
            "initial_transitions": self.model.num_timesteps,
            "initial_policy_log_std": self.model.policy.log_std.detach().cpu().tolist(),
            "value_loss_definition": (
                "pre_update_value_mse is first rollout mean squared return-minus-value before "
                "optimization; first_optimizer_value_loss is SB3 train/value_loss averaged "
                "over the first PPO optimization pass. No targets exist at transition zero."
            ),
        }
        write_json(self.output / "ppo_start.json", self.start_diagnostics)

    def _capture_first_optimizer_loss(self):
        if (self.start_diagnostics and "first_optimizer_value_loss" not in self.start_diagnostics
                and "train/value_loss" in self.model.logger.name_to_value):
            self.start_diagnostics["first_optimizer_value_loss"] = float(
                self.model.logger.name_to_value["train/value_loss"]
            )
            self.start_diagnostics["first_optimizer_policy_log_std"] = (
                self.model.policy.log_std.detach().cpu().tolist()
            )
            write_json(self.output / "ppo_start.json", self.start_diagnostics)

    def _on_rollout_start(self):
        self._capture_first_optimizer_loss()

    def _on_training_end(self):
        self._capture_first_optimizer_loss()

    def validate(self):
        started = perf_counter()
        try:
            rows = evaluate(self.model, "id", list(VALIDATION_SEEDS), config=self.config.env)
        except EvaluationPhysicsError as error:
            write_json(self.output / "validation_failed.json", error.rows)
            raise
        self.validation_seconds += perf_counter() - started
        for row in rows:
            row.update(transitions=self.model.num_timesteps, training_seed=self.config.seed)
        self.validation_rows.extend(rows)
        rank = validation_rank(rows)
        if self.best_rank is None or rank > self.best_rank:
            self.best_rank, self.best_steps = rank, self.model.num_timesteps
            self.model.save(self.output / "best.zip")
        write_csv(self.output / "validation_episodes.csv", self.validation_rows)

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if done:
                self.episode_rows.append({
                    "transitions": self.num_timesteps, "coverage": info["coverage"],
                    "force_mean_n": info["episode_force_mean_n"],
                    "unsafe_fraction": info["unsafe_fraction"], "success": info["success"],
                    "failure_reason": info["failure_reason"], "elapsed_s": info["elapsed_s"],
                    "return": info["episode"]["r"], "steps": info["episode"]["l"],
                    "truncated": bool(info.get("TimeLimit.truncated", False)),
                })
        if self.num_timesteps >= self.next_eval:
            self.validate()
            self.next_eval += 50_000
        return True

    def _on_rollout_end(self):
        if "pre_update_value_mse" not in self.start_diagnostics:
            buffer = self.model.rollout_buffer
            self.start_diagnostics.update(
                first_rollout_transitions=self.model.num_timesteps,
                pre_update_value_mse=float(np.mean((buffer.returns - buffer.values) ** 2)),
            )
            write_json(self.output / "ppo_start.json", self.start_diagnostics)
        if self.episode_rows:
            write_csv(self.output / "training_episodes.csv", self.episode_rows)
        if any(not torch.isfinite(p).all() for p in self.model.policy.parameters()):
            raise RuntimeError("Nonfinite PPO parameters; training stopped")


def _verify_dataset_for_run(path: Path, config: RunConfig):
    summary = assert_demonstration_gate(path)
    if summary["seeds"] != list(range(3000, 3000 + config.demo_episodes)):
        raise ValueError("Dataset episode seeds/count do not match the selected profile")
    cutoff = int(config.demo_episodes * 0.8)
    expected_validation = list(range(3000 + cutoff, 3000 + config.demo_episodes))
    if (summary["train_seeds"] != list(range(3000, 3000 + cutoff))
            or summary["validation_seeds"] != expected_validation):
        raise ValueError("Dataset split does not match the experiment contract")
    with h5py.File(path) as file:
        generation = json.loads(file.attrs["generation_config"])
    if (generation.get("profile") != config.profile or generation.get("env") != asdict(config.env)
            or not generation.get("complete")):
        raise ValueError("Dataset generation configuration does not match this run")
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if manifest["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Dataset checksum mismatch")


def train(method: str, config: RunConfig, output: Path) -> Path:
    if method not in ("bc", "ppo", "bc_ppo"):
        raise ValueError(f"Unknown learning method: {method}")
    dataset = Path(config.dataset) if config.dataset else None
    if method != "ppo":
        if dataset is None:
            raise ValueError("BC methods require a dataset")
        _verify_dataset_for_run(dataset, config)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    provenance = run_provenance(output)
    settings = {**config.to_dict(), "method": method, "ppo": PPO_SETTINGS,
                "network": {"actor": [256, 256], "critic": [256, 256], "activation": "Tanh"},
                "bc": {"learning_rate": 3e-4, "batch_size": 256, "max_epochs": 100,
                       "patience": 10}, "n_envs": 4, "observation_normalization": False}
    write_json(output / "config.json", settings)
    record = {**provenance, "method": method, "profile": config.profile, "seed": config.seed,
              "status": "running", "actual_transitions": 0,
              "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest()
              if dataset is not None and method != "ppo" else None}
    write_json(output / "run.json", record)
    started = perf_counter()
    def build_env(rank):
        return Monitor(FailFastEnv(WipingEnv(config.env), output / f"fatal-env-{rank}.json"),
                       str(output / f"env-{rank}"))
    vec = DummyVecEnv([lambda rank=rank: build_env(rank) for rank in range(4)])
    model = None
    try:
        model = make_ppo(vec, config.seed, config.total_steps)
        if method in ("bc", "bc_ppo"):
            bc_checkpoint = fit_bc(model, dataset, output / "bc_init")
            record["bc_checkpoint"] = str(bc_checkpoint)
            if method == "bc_ppo":
                model = make_ppo(vec, config.seed, config.total_steps)
                transfer_bc(model, bc_checkpoint)
        callback = ValidationCallback(config, output)
        callback.init_callback(model)
        rl_seconds = 0.0
        if method != "bc":
            model.set_logger(configure(str(output), ["csv"]))
            rl_start = perf_counter()
            model.learn(total_timesteps=config.total_steps, callback=callback)
            rl_seconds = perf_counter() - rl_start - callback.validation_seconds
            model.save(output / "last.zip")
        callback.validate()
        record.update(
            status="complete", actual_transitions=model.num_timesteps,
            requested_transitions=config.total_steps if method != "bc" else 0,
            rl_seconds_excluding_validation=rl_seconds,
            transitions_per_second=model.num_timesteps / rl_seconds if rl_seconds else None,
            validation_seconds=callback.validation_seconds,
            validation_episodes=len(callback.validation_rows),
            selected_checkpoint_transitions=callback.best_steps,
            checkpoint=str(output / "best.zip"), wall_seconds=perf_counter() - started,
        )
        write_json(output / "run.json", record)
        return output / "best.zip"
    except Exception as error:
        record.update(status="failed", error=repr(error), wall_seconds=perf_counter() - started,
                      actual_transitions=model.num_timesteps if model else 0)
        write_json(output / "run.json", record)
        raise
    finally:
        vec.close()


if __name__ == "__main__":
    import argparse
    from dataclasses import replace

    from wiping.config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["bc", "ppo", "bc_ppo"], required=True)
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(train(args.method, replace(load_config(args.config), seed=args.seed), args.output))
