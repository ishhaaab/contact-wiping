"""Full-release statistical tables and diagnostics, without changing the experiment."""

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wiping.artifacts import write_csv, write_json  # noqa: E402
from wiping.evaluate import aggregate, normalize_rows, validate_full_protocol  # noqa: E402

METHODS = ("scripted", "bc", "ppo", "bc_ppo")
SUITES = {"id": "ID", "curvature": "Curvature OOD", "friction": "Friction OOD"}


def _number(value):
    return "NA" if value is None else f"{value:.4f}"


def _spread(metric):
    return f"{_number(metric['mean'])} +/- {_number(metric['std_across_seeds'])}"


def suite_tables(rows: list[dict]) -> tuple[str, dict]:
    rows = normalize_rows(rows)
    summary = aggregate(rows)
    text = ["# Full experimental release", "",
            "Fractions use [0,1]. Every final-test episode, including failures, contributes. "
            "Completion time is conditional on success. NA means undefined, not zero.", "",
            "The aggregate row is mean +/- sample SD across the trained models, not an "
            "episode-level confidence interval. The scripted policy has no training seed "
            "and is counted once. With no successes, completion time and its SD are undefined."]
    for suite, label in SUITES.items():
        text.extend(["", f"## {label}", "",
                     "| Method | Training seed | Successes/cases | Success rate | Coverage | "
                     "Unsafe fraction | Completion seconds, successful cases only |",
                     "| --- | --- | ---: | ---: | ---: | ---: | --- |"])
        for method in METHODS:
            group = next((g for g in summary["groups"]
                          if g["suite"] == suite and g["method"] == method), None)
            if group is None:
                continue
            total_success = 0
            for model in group["per_training_seed"]:
                matching = [r for r in rows if r["suite"] == suite and r["method"] == method
                            and r["training_seed"] == model["training_seed"]]
                successes = sum(r["success"] for r in matching)
                total_success += successes
                completion = (f"{_number(model['completion_time_s'])}, {successes} successes"
                              if successes else "NA, 0 successes")
                seed = model["training_seed"] if method != "scripted" else "baseline once"
                text.append(f"| {method} | {seed} | {successes}/{model['episodes']} | "
                            f"{model['success_rate']:.4f} | {model['coverage']:.4f} | "
                            f"{model['unsafe_fraction']:.4f} | {completion} |")
            if method != "scripted":
                m = group["metrics"]
                contributors = m["completion_time_s"]["contributing_models"]
                completion = (f"{_spread(m['completion_time_s'])}, {total_success} successes, "
                              f"{contributors} models with success")
                count = f"{total_success}/{group['episodes']}"
                text.append(f"| {method} | mean +/- sample SD | {count} | "
                            f"{_spread(m['success_rate'])} | {_spread(m['coverage'])} | "
                            f"{_spread(m['unsafe_fraction'])} | {completion} |")
    return "\n".join(text) + "\n", summary


def bc_diagnostics(path: Path) -> dict:
    path = Path(path) / "bc_init"
    metrics = json.loads((path / "bc_metrics.json").read_text(encoding="utf-8"))
    with (path / "bc_history.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    history = [{"epoch": int(r["epoch"]), "train_mse": float(r["train_mse"]),
                "validation_mse": float(r["validation_mse"])} for r in rows]
    if len(history) != metrics["epochs"] or not history:
        raise ValueError(f"BC history/epoch count mismatch at {path}")
    best = min(history, key=lambda row: row["validation_mse"])
    if not np.isclose(best["validation_mse"], metrics["best_validation_mse"], atol=1e-12):
        raise ValueError(f"BC best-validation mismatch at {path}")
    return {
        "epochs_trained": len(history), "best_epoch": best["epoch"],
        "best_validation_mse": best["validation_mse"],
        "early_stopping_epoch": len(history) if len(history) < 100 else None,
        "epochs_since_improvement": len(history) - best["epoch"],
        "validation_mse_history": history,
        "stop_reason": "patience_10" if len(history) < 100 else "maximum_100_epochs",
    }


def _training_curve(run: Path, destination: Path) -> dict:
    with (run / "training_episodes.csv").open(newline="", encoding="utf-8") as file:
        raw = list(csv.DictReader(file))
    bins = {}
    for row in raw:
        # Non-overlapping 50,000-transition windows, include every finished episode.
        key = ((int(row["transitions"]) - 1) // 50_000 + 1) * 50_000
        bins.setdefault(key, []).append(row)
    points = []
    for key, episodes in sorted(bins.items()):
        points.append({
            "window_end_transitions": key, "completed_episodes": len(episodes),
            "success_rate": float(np.mean([r["success"] == "True" for r in episodes])),
            "coverage": float(np.mean([float(r["coverage"]) for r in episodes])),
            "unsafe_fraction": float(np.mean([float(r["unsafe_fraction"]) for r in episodes])),
        })
    if not points:
        raise ValueError(f"No training episodes at {run}")
    write_csv(destination.with_suffix(".csv"), points)
    figure, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for axis, metric in zip(axes, ("success_rate", "coverage", "unsafe_fraction"), strict=True):
        axis.plot([r["window_end_transitions"] for r in points], [r[metric] for r in points])
        axis.set(ylabel=metric.replace("_", " "), ylim=(0, 1))
    axes[-1].set_xlabel("Environment transitions, nonoverlapping 50,000-transition bins")
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=150)
    plt.close(figure)
    return {"completed_episodes": len(raw), "bins": len(points),
            "plot": str(destination.with_suffix(".png"))}


def _interpretation(summary: dict) -> str:
    lines = ["## What these results show", "",
             "These comparisons are descriptive. Three training seeds do not support strong "
             "significance claims. Episode counts do not replace independent training seeds."]
    for suite, label in SUITES.items():
        groups = {g["method"]: g for g in summary["groups"] if g["suite"] == suite}
        lines.extend(["", f"### {label}"])
        for other in ("ppo", "bc", "scripted"):
            a, b = groups["bc_ppo"]["metrics"], groups[other]["metrics"]
            success_delta = a["success_rate"]["mean"] - b["success_rate"]["mean"]
            coverage_delta = a["coverage"]["mean"] - b["coverage"]["mean"]
            relation = ("higher" if coverage_delta > 0 else
                        "lower" if coverage_delta < 0 else "equal")
            success_relation = ("higher" if success_delta > 0 else
                                "lower" if success_delta < 0 else "equal")
            band = max(a["coverage"]["std_across_seeds"] or 0,
                       b["coverage"]["std_across_seeds"] or 0)
            uncertainty = ("The coverage difference is within the larger observed "
                           "seed-to-seed sample SD." if abs(coverage_delta) <= band else
                           "The coverage difference exceeds the available across-seed sample SDs; "
                           "this is not a statistical significance test.")
            lines.append(f"- BC+PPO versus {other}: {success_relation} mean success rate, "
                         f"difference {success_delta:+.4f}; {relation} mean coverage, "
                         f"difference {coverage_delta:+.4f}. {uncertainty}")
        for method in METHODS:
            m = groups[method]["metrics"]
            lines.append(f"- {method}: success {_number(m['success_rate']['mean'])}, "
                         f"coverage {_number(m['coverage']['mean'])}, unsafe fraction "
                         f"{_number(m['unsafe_fraction']['mean'])}.")
    lines.extend(["", "Higher coverage alone does not mean task success or safer forces. "
                  "No causal explanation of any difference has been tested in this release. "
                  "Explanations involving exploration, imitation error, or critic initialization "
                  "would be a hypothesis, not an established finding."])
    return "\n".join(lines)


def build_release_analysis(control: Path, output: Path) -> Path:
    control, output = Path(control), Path(output)
    state = json.loads((control / "state.json").read_text(encoding="utf-8"))
    if state["status"] != "complete":
        raise ValueError("Full comparison has not completed; do not report partial results as full")
    report = Path(state["report"])
    with (report / "episodes.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    validate_full_protocol(rows)
    text, summary = suite_tables(rows)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", summary)
    inventory, diagnostics, curve_inventory = [], [], []
    text += "\n## Actual training and validation-selected checkpoints\n\n"
    text += ("| Method | Seed | Actual RL transitions | Training wall seconds | "
             "Selected transition | Checkpoint |\n| --- | ---: | ---: | ---: | ---: | --- |\n")
    for entry in state["training"]:
        path = Path(entry["path"])
        record = json.loads((path / "run.json").read_text(encoding="utf-8"))
        identity = {"method": entry["method"], "seed": entry["seed"]}
        inventory.append({**identity, "path": str(path), "run": record})
        text += (f"| {entry['method']} | {entry['seed']} | {record['actual_transitions']} | "
                 f"{record['wall_seconds']:.2f} | {record['selected_checkpoint_transitions']} | "
                 f"`{path / 'best.zip'}` |\n")
        if entry["method"] in ("bc", "bc_ppo"):
            bc = bc_diagnostics(path)
            diagnostics.append({**identity, "kind": "bc", **bc})
            shutil.copy2(path / "bc_init/bc_history.csv",
                         output / f"bc-history-{entry['method']}-{entry['seed']}.csv")
        if entry["method"] in ("ppo", "bc_ppo"):
            start = json.loads((path / "ppo_start.json").read_text(encoding="utf-8"))
            required = (
                "initial_policy_log_std", "pre_update_value_mse", "first_optimizer_value_loss"
            )
            for key in required:
                if key not in start:
                    raise ValueError(f"Missing required diagnostic {key}: {path}")
            diagnostics.append({**identity, "kind": "ppo_start", **start})
            curve = _training_curve(path, output / f"training-{entry['method']}-{entry['seed']}")
            curve_inventory.append({**identity, **curve})
    write_json(output / "run-inventory.json", inventory)
    write_json(output / "diagnostics.json", diagnostics)
    write_json(output / "curves.json", curve_inventory)
    text += "\n## BC fitting and early stopping\n\n"
    text += ("| Method | Seed | Epochs trained | Best epoch | Validation MSE | Stop |\n"
             "| --- | ---: | ---: | ---: | ---: | --- |\n")
    for d in diagnostics:
        if d["kind"] == "bc":
            text += (f"| {d['method']} | {d['seed']} | {d['epochs_trained']} | {d['best_epoch']} | "
                     f"{d['best_validation_mse']:.8f} | {d['stop_reason']} |\n")
    text += "\nPer-epoch train and validation MSE histories are preserved alongside this report.\n"
    text += "\n## PPO-start diagnostics\n\n"
    text += ("No value-loss target exists at transition zero. The pre-update MSE below uses the "
             "first rollout's returns and predictions before optimization. The first optimizer "
             "loss is SB3's mean value loss over the first PPO optimizer pass. "
             "These are logging-only measurements.\n\n"
             "| Method | Seed | Initial policy log-std xyz | Pre-update value MSE | "
             "First optimizer value loss |\n| --- | ---: | --- | ---: | ---: |\n")
    for d in diagnostics:
        if d["kind"] == "ppo_start":
            text += (f"| {d['method']} | {d['seed']} | {d['initial_policy_log_std']} | "
                     f"{d['pre_update_value_mse']:.8f} | {d['first_optimizer_value_loss']:.8f} |\n")
    text += ("\n## Timing\n\n"
             f"This orchestration attempt took {state['elapsed_seconds'] / 3600:.4f} wall hours. "
             "Per-command elapsed times, including final evaluations, are in `orchestration.json`. "
             "Previous attempts must also be included in total consumed compute after a resume.\n")
    text += "\n" + _interpretation(summary) + "\n"
    text += ("\n## Limitations and deviations\n\n"
             "Rigid spherical contact and geometric coverage are end-effector abstractions, not "
             "cleaning chemistry or articulated-arm performance. Local geometry is ideal and "
             "observations have no sensor noise. Native Windows was used instead of the Linux "
             "reference platform. Only three training seeds are available. The baseline is "
             "scripted "
             "and deliberately imperfect, not human data. Final-test cases were used only for "
             "evaluation; checkpoints were selected on validation seeds 1000..1019. No reward, "
             "physics, network, hyperparameter, budget, or split was changed. The launcher used "
             "idle-sleep inhibition, not a guarantee against explicit sleep or power loss. "
             "There is no physical-robot validation, sim-to-real claim, or safety claim.\n")
    write_json(output / "orchestration.json", state)
    (output / "results.md").write_text(text, encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build_release_analysis(args.control, args.output))
