"""Episode-preserving HDF5 datasets and demonstration quality gates."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from wiping.config import RunConfig, load_config
from wiping.env import WipingEnv
from wiping.scripted import RasterPolicy


class DemonstrationGateError(RuntimeError):
    pass


@dataclass
class Episode:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    metadata: dict
    seed: int
    split: str
    normal_offset_m: float


def _validate_episode(episode: Episode):
    length = len(episode.actions)
    if length < 1 or episode.actions.shape != (length, 3):
        raise ValueError("Episode must have at least one 3D action")
    if episode.obs.shape != (length + 1, 273):
        raise ValueError("Episode observations must retain T+1 states including terminal")
    for field in (episode.rewards, episode.terminated, episode.truncated):
        if field.shape != (length,):
            raise ValueError("Transition arrays must have matching length")
    done = episode.terminated | episode.truncated
    if done[:-1].any() or not done[-1] or (episode.terminated & episode.truncated).any():
        raise ValueError("Episode must end exactly once, with termination or truncation")
    if episode.split not in ("train", "validation"):
        raise ValueError("Unknown episode split")
    if any(len(value) != length for value in episode.metadata.values()):
        raise ValueError("Metadata arrays must align with transitions")


def dataset_summary(episodes: list[Episode]) -> dict:
    if not episodes:
        raise ValueError("Dataset is empty")
    qualifying = [float(e.metadata["coverage"][-1]) >= 0.5
                  and e.metadata["failure_reason"][-1] != "excessive_force"
                  for e in episodes]
    return {
        "episodes": len(episodes),
        "transitions": sum(len(e.actions) for e in episodes),
        "seeds": [e.seed for e in episodes],
        "train_seeds": [e.seed for e in episodes if e.split == "train"],
        "validation_seeds": [e.seed for e in episodes if e.split == "validation"],
        "qualifying_fraction": float(np.mean(qualifying)),
        "mean_coverage": float(np.mean([e.metadata["coverage"][-1] for e in episodes])),
        "success_fraction": float(np.mean([e.metadata["success"][-1] for e in episodes])),
        "physical_failure_fraction": float(np.mean([
            bool(e.metadata["failure_reason"][-1]) for e in episodes
        ])),
        "unsuccessful_fraction": float(np.mean([
            not e.metadata["success"][-1] for e in episodes
        ])),
        "episode_results": [{
            "seed": e.seed, "split": e.split, "steps": len(e.actions),
            "coverage": float(e.metadata["coverage"][-1]),
            "success": bool(e.metadata["success"][-1]),
            "failure_reason": str(e.metadata["failure_reason"][-1]) or None,
            "terminated": bool(e.terminated[-1]), "truncated": bool(e.truncated[-1]),
            "normal_offset_m": e.normal_offset_m,
        } for e in episodes],
    }


def save_episodes(episodes: list[Episode], path: Path, generation_config: dict) -> Path:
    path = Path(path)
    for episode in episodes:
        _validate_episode(episode)
    if len({e.seed for e in episodes}) != len(episodes):
        raise ValueError("Episode seeds must be unique across splits")
    summary = dataset_summary(episodes)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "x") as file:
        file.attrs["schema_version"] = 1
        file.attrs["generation_config"] = json.dumps(generation_config, sort_keys=True)
        file.attrs["source"] = generation_config.get("source", "unspecified")
        root = file.create_group("episodes")
        for episode in episodes:
            group = root.create_group(str(episode.seed))
            group.attrs.update(seed=episode.seed, split=episode.split,
                               normal_offset_m=episode.normal_offset_m)
            for name in ("obs", "actions", "rewards", "terminated", "truncated"):
                group.create_dataset(name, data=getattr(episode, name), compression="gzip")
            metadata = group.create_group("metadata")
            for name, values in episode.metadata.items():
                values = np.asarray(values)
                if values.dtype.kind in "UO":
                    values = values.astype(h5py.string_dtype("utf-8"))
                metadata.create_dataset(name, data=values, compression="gzip")
    summary.update(schema_version=1, generation_config=generation_config,
                   sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path.with_suffix(".manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def load_episodes(path: Path, split: str) -> list[Episode]:
    if split not in ("train", "validation", "all"):
        raise ValueError(f"Unknown split: {split}")
    episodes = []
    with h5py.File(path, "r") as file:
        if file.attrs.get("schema_version") != 1:
            raise ValueError("Unsupported dataset schema")
        for seed in sorted(file["episodes"], key=int):
            group = file["episodes"][seed]
            if split != "all" and group.attrs["split"] != split:
                continue
            metadata = {}
            for name, dataset in group["metadata"].items():
                metadata[name] = dataset.asstr()[:] if dataset.dtype.kind == "O" else dataset[:]
            episode = Episode(
                **{name: group[name][:] for name in
                   ("obs", "actions", "rewards", "terminated", "truncated")},
                metadata=metadata, seed=int(group.attrs["seed"]), split=group.attrs["split"],
                normal_offset_m=float(group.attrs["normal_offset_m"]),
            )
            _validate_episode(episode)
            episodes.append(episode)
    return episodes


def assert_demonstration_gate(path: Path) -> dict:
    episodes = load_episodes(path, "all")
    for episode in episodes:
        if (not all(np.isfinite(x).all() for x in
                    (episode.obs, episode.actions, episode.rewards))
                or episode.metadata["failure_reason"][-1] in
                ("nonfinite_physics", "mujoco_warning", "out_of_bounds")):
            raise DemonstrationGateError(f"Invalid physics in retained episode {episode.seed}")
    summary = dataset_summary(episodes)
    if summary["qualifying_fraction"] < 0.2:
        raise DemonstrationGateError(
            f"Demonstration gate failed: {summary['qualifying_fraction']:.1%} qualify; "
            f"require at least 20%. All {len(episodes)} episodes retained at {path}. "
            "Inspect physics/controller and obtain approval before changing the experiment."
        )
    return summary


def record_scripted_episode(env: WipingEnv, seed: int, split: str) -> Episode:
    obs, _ = env.reset(seed=seed)
    policy = RasterPolicy(seed=seed)
    observations, actions, rewards, terms, truncs, infos = [obs], [], [], [], [], []
    while True:
        obs, reward, terminated, truncated, info = env.step(policy.predict(env))
        observations.append(obs)
        actions.append(info["executed_action"])
        rewards.append(reward)
        terms.append(terminated)
        truncs.append(truncated)
        infos.append(info)
        if terminated or truncated:
            break
    metadata = {name: np.asarray([i[name] if i[name] is not None else "" for i in infos])
                for name in infos[0]}
    return Episode(
        obs=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        terminated=np.asarray(terms), truncated=np.asarray(truncs), metadata=metadata,
        seed=seed, split=split, normal_offset_m=policy.normal_offset_m,
    )


def collect_demonstrations(config: RunConfig, output: Path) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite dataset {output}")
    started = perf_counter()
    episodes = []
    env = WipingEnv(config.env)
    try:
        for index in range(config.demo_episodes):
            split = "train" if index < int(config.demo_episodes * 0.8) else "validation"
            episode = record_scripted_episode(env, 3000 + index, split)
            episodes.append(episode)
            if episode.metadata["failure_reason"][-1] in (
                "nonfinite_physics", "mujoco_warning", "out_of_bounds"
            ):
                break
    finally:
        env.close()
    generation = {**config.to_dict(), "source": "imperfect_scripted_raster",
                  "wall_seconds": perf_counter() - started,
                  "complete": len(episodes) == config.demo_episodes}
    save_episodes(episodes, output, generation)
    assert_demonstration_gate(output)
    if not generation["complete"]:
        raise DemonstrationGateError(f"Partial dataset retained at {output}; training blocked")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--output", type=Path, default=Path("data/smoke/demos.h5"))
    args = parser.parse_args()
    path = collect_demonstrations(load_config(args.config), args.output)
    print(json.dumps(assert_demonstration_gate(path), indent=2))
