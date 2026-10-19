"""Measured summaries, plots, labelled replay videos, and compute estimates."""

import csv
import json
import math
from pathlib import Path
from time import perf_counter

import imageio.v2 as iio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wiping.artifacts import write_csv, write_json  # noqa: E402
from wiping.config import EnvConfig  # noqa: E402
from wiping.env import WipingEnv  # noqa: E402
from wiping.evaluate import (  # noqa: E402
    aggregate,
    normalize_rows,
    rollout_episode,
    validate_full_protocol,
)


def render_episode(policy, suite: str, seed: int, output: Path, *, method: str,
                   profile: str, config: EnvConfig | None = None) -> dict:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    env = WipingEnv(config or EnvConfig(), render_mode="rgb_array")
    trace = []
    font = ImageFont.load_default(size=17)
    started = perf_counter()
    try:
        with iio.get_writer(output, fps=20, codec="libx264", quality=7) as writer:
            def frame(env, info):
                image = Image.fromarray(env.render())
                draw = ImageDraw.Draw(image)
                if info["success"]:
                    outcome = "success"
                elif info["failure_reason"]:
                    outcome = "failure: " + info["failure_reason"]
                elif env.steps >= env.config.max_steps:
                    outcome = "timeout"
                else:
                    outcome = "running"
                label = (f"SIMULATION / {profile} / {method} / {suite} / seed {seed}\n"
                         f"t={info['elapsed_s']:.2f}s  coverage={info['coverage']:.1%}  "
                         f"force={info['force_mean_n']:.2f}N\n{outcome}")
                draw.rectangle((0, 0, 640, 75), fill=(10, 15, 20))
                draw.multiline_text((10, 6), label, fill=(255, 255, 255), font=font)
                writer.append_data(np.asarray(image))
                trace.append({
                    "step": env.steps, "elapsed_s": info["elapsed_s"],
                    "coverage": info["coverage"], "force_mean_n": info["force_mean_n"],
                    "force_max_n": info["force_max_n"], "unsafe_fraction": info["unsafe_fraction"],
                    "x_m": float(env.data.qpos[0]), "y_m": float(env.data.qpos[1]),
                    "z_m": float(env.data.qpos[2]),
                })
            metrics = rollout_episode(env, policy, suite, seed, on_frame=frame)
    finally:
        env.close()
    write_csv(output.with_suffix(".trace.csv"), trace)
    result = {"path": str(output), "profile": profile, "method": method,
              "suite": suite, "seed": seed, "frames": len(trace), "metrics": metrics,
              "wall_seconds": perf_counter() - started}
    write_json(output.with_suffix(".json"), result)
    figure, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    time = [r["elapsed_s"] for r in trace]
    axes[0].plot(time, [r["coverage"] for r in trace])
    axes[0].set(ylabel="Coverage fraction", ylim=(0, 1))
    axes[1].plot(time, [r["force_mean_n"] for r in trace], label="Control-step mean")
    axes[1].plot(time, [r["force_max_n"] for r in trace], label="Substep maximum", alpha=0.6)
    axes[1].axhline(8, color="red", linestyle="--", label="Unsafe threshold")
    axes[1].set(xlabel="Simulated time, s", ylabel="Force, N")
    axes[1].legend()
    figure.suptitle(f"{profile}: {method}, {suite}, seed {seed}")
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=140)
    plt.close(figure)
    return result


def training_plots(training_root: Path, output: Path) -> list[str]:
    plots = []
    for run_path in sorted(Path(training_root).rglob("run.json")):
        record = json.loads(run_path.read_text(encoding="utf-8"))
        history_path = run_path.parent / "bc_init/bc_history.csv"
        progress_path = run_path.parent / "progress.csv"
        figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        if history_path.exists():
            with history_path.open(newline="") as file:
                rows = list(csv.DictReader(file))
            axes[0].plot([int(r["epoch"]) for r in rows],
                         [float(r["validation_mse"]) for r in rows], label="BC validation")
            axes[0].plot([int(r["epoch"]) for r in rows],
                         [float(r["train_mse"]) for r in rows], label="BC train")
            axes[0].legend()
        axes[0].set(xlabel="BC epoch", ylabel="Action MSE")
        if progress_path.exists():
            with progress_path.open(newline="") as file:
                rows = list(csv.DictReader(file))
            valid = [r for r in rows if r.get("rollout/ep_rew_mean")]
            axes[1].plot([int(r["time/total_timesteps"]) for r in valid],
                         [float(r["rollout/ep_rew_mean"]) for r in valid])
        axes[1].set(xlabel="RL transitions", ylabel="Rolling episode return")
        label = f"{record['profile']}-{record['method']}-{record['seed']}"
        figure.suptitle(label)
        figure.tight_layout()
        destination = output / f"training-{label}.png"
        figure.savefig(destination, dpi=140)
        plt.close(figure)
        plots.append(str(destination))
    return plots


def estimate_full_compute(training_root: Path, dataset: Path | None = None) -> dict:
    records, bc_times = [], []
    for path in Path(training_root).rglob("run.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["status"] != "complete":
            continue
        records.append(record)
        bc = path.parent / "bc_init/bc_metrics.json"
        if bc.exists():
            metrics = json.loads(bc.read_text(encoding="utf-8"))
            bc_times.append(metrics["wall_seconds"] * 64_000 / metrics["train_transitions"])
    transitions = math.ceil(2_000_000 / 4096) * 4096
    estimates = []
    for method in ("ppo", "bc_ppo"):
        matching = [r for r in records if r["method"] == method and r["transitions_per_second"]]
        if not matching:
            raise ValueError("Compute estimate requires completed PPO and BC+PPO measurements")
        throughput = float(np.median([r["transitions_per_second"] for r in matching]))
        validation_round = float(np.median([
            r["validation_seconds"] * 20 / r["validation_episodes"] for r in matching
        ]))
        estimates.append({
            "method": method, "measured_transitions_per_second": throughput,
            "training_seeds": 3, "actual_transitions_per_seed": transitions,
            "validation_rounds_per_seed": 41,
            "estimated_seconds_three_seeds": 3 * (transitions / throughput + 41 * validation_round),
        })
    full_rl = sum(r["estimated_seconds_three_seeds"] for r in estimates)
    seconds_per_episode = max(r["validation_seconds"] / r["validation_episodes"] for r in records)
    final_evaluation = 3000 * seconds_per_episode
    bc_seconds = 6 * float(np.median(bc_times)) if bc_times else 0
    collection_seconds = estimated_dataset_bytes = None
    if dataset:
        manifest_path = Path(dataset).with_suffix(".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        collection_seconds = (
            manifest["generation_config"]["wall_seconds"] * 200 / manifest["episodes"]
        )
        estimated_dataset_bytes = Path(dataset).stat().st_size * 200 / manifest["episodes"]
    total = full_rl + final_evaluation + bc_seconds + (collection_seconds or 0)
    return {
        "basis": "Measured smoke/pilot throughput, linear extrapolation; not a runtime promise",
        "rl_methods": estimates, "estimated_rl_and_validation_hours": full_rl / 3600,
        "estimated_bc_fitting_hours": bc_seconds / 3600,
        "estimated_final_evaluation_hours": final_evaluation / 3600,
        "estimated_collection_seconds": collection_seconds,
        "estimated_dataset_bytes": estimated_dataset_bytes,
        "estimated_total_hours_excluding_rendering": total / 3600,
        "full_final_evaluation_episodes": 3000,
        "hardware": records[0]["hardware"],
        "limitations": ["Laptop power and thermal limits affect throughput.",
                        "Episode lengths and BC early stopping can change runtime.",
                        "Extrapolation uses only the available completed timing measurements."],
    }


def write_report(inputs: list[Path], output: Path, *, training_root: Path | None = None,
                 require_full: bool = False) -> Path:
    files = []
    for path in map(Path, inputs):
        files.extend(sorted(path.rglob("episodes.csv")) if path.is_dir() else [path])
    rows = []
    for path in files:
        with path.open(newline="", encoding="utf-8") as file:
            rows.extend(csv.DictReader(file))
    rows = normalize_rows(rows)
    if require_full:
        validate_full_protocol(rows)
    summary = aggregate(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "episodes.csv", rows)
    write_json(output / "summary.json", summary)
    lines = ["# Measured evaluation results", "",
             "All attempted episodes are included. Smoke results only verify execution.", "",
             "| Profile | Method | Suite | Train seeds | Episodes | Coverage | Success | Unsafe |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for group in summary["groups"]:
        m = group["metrics"]
        lines.append(f"| {group['profile']} | {group['method']} | {group['suite']} | "
                     f"{group['n_training_seeds']} | {group['episodes']} | "
                     f"{m['coverage']['mean']:.4f} | {m['success_rate']['mean']:.4f} | "
                     f"{m['unsafe_fraction']['mean']:.4f} |")
    lines.extend(["", "## Per-training-seed results", "",
                  "| Method | Suite | Training seed | Episodes | Coverage | Success |",
                  "| --- | --- | --- | ---: | ---: | ---: |"])
    for group in summary["groups"]:
        for model in group["per_training_seed"]:
            seed = model["training_seed"] if model["training_seed"] is not None else "baseline"
            lines.append(f"| {group['method']} | {group['suite']} | {seed} | {model['episodes']} | "
                         f"{model['coverage']:.4f} | {model['success_rate']:.4f} |")
    lines.extend(["", *summary["notes"], "",
                  "`summary.json` includes force, elapsed time, successful completion time, "
                  "and sample standard deviation across training seeds. Null SD means it cannot "
                  "be estimated from the available models."])
    (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    groups = summary["groups"]
    labels = [f"{g['method']}\n{g['suite']}" for g in groups]
    figure, axes = plt.subplots(2, 1, figsize=(max(8, len(groups)), 6), sharex=True)
    for axis, metric, ylabel in zip(axes, ["coverage", "force_mean_n"],
                                    ["Coverage fraction", "Mean contact force, N"], strict=True):
        axis.bar(np.arange(len(groups)), [g["metrics"][metric]["mean"] for g in groups],
                 yerr=[g["metrics"][metric]["std_across_seeds"] or 0 for g in groups])
        axis.set_ylabel(ylabel)
    axes[0].set_ylim(0, 1)
    axes[1].set_xticks(np.arange(len(groups)), labels, fontsize=8)
    figure.suptitle("Measured episodes; bars show across-training-seed sample SD when available")
    figure.tight_layout()
    figure.savefig(output / "coverage-force.png", dpi=140)
    plt.close(figure)
    if training_root:
        training_plots(training_root, output)
    return output
