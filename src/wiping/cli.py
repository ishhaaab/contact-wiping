"""Command-line entry points for the experiment."""

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from stable_baselines3 import PPO

from wiping.artifacts import run_provenance, write_csv, write_json
from wiping.config import TEST_SEEDS, VALIDATION_SEEDS, EnvConfig, load_config
from wiping.dataset import collect_demonstrations
from wiping.evaluate import EvaluationPhysicsError, aggregate, evaluate
from wiping.feasibility import run_physics_gate
from wiping.learning import train
from wiping.reporting import estimate_full_compute, render_episode, write_report
from wiping.scripted import RasterPolicy


def _empty_output(path: Path):
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _label(rows, *, method, profile, training_seed, split):
    return [{**row, "method": method, "profile": profile, "training_seed": training_seed,
             "evaluation_split": split} for row in rows]


def _evaluate_command(args):
    torch.set_num_threads(1)
    if args.method == "scripted":
        if args.checkpoint:
            raise ValueError("The scripted baseline does not load a checkpoint")
        config = load_config(args.config or "configs/full.json")
        policy, method, seed = RasterPolicy(), "scripted", None
        profile, env_config = config.profile, config.env
        checkpoint_hash = None
    else:
        if not args.checkpoint:
            raise ValueError("A learned method requires --checkpoint")
        checkpoint = Path(args.checkpoint)
        record_path = checkpoint.parent / "run.json"
        if not record_path.exists():
            raise ValueError("Checkpoint must have its training run.json beside it")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["status"] != "complete":
            raise ValueError("Training run did not complete")
        method, profile, seed = record["method"], record["profile"], record["seed"]
        if args.method and args.method != method:
            raise ValueError("Requested method does not match checkpoint provenance")
        resolved = json.loads((checkpoint.parent / "config.json").read_text(encoding="utf-8"))
        env_config = EnvConfig(**resolved["env"])
        if args.config:
            config = load_config(args.config)
            if config.profile != profile or config.env != env_config:
                raise ValueError("Evaluation config does not match checkpoint profile/environment")
        policy = PPO.load(checkpoint, device="cpu")
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output = Path(args.output)
    _empty_output(output)
    provenance = run_provenance(output)
    seeds = list(TEST_SEEDS if args.split == "test" else VALIDATION_SEEDS)
    suites = ("id", "curvature", "friction") if args.suite == "all" else (args.suite,)
    rows, start = [], perf_counter()
    for suite in suites:
        try:
            cases = evaluate(policy, suite, seeds, config=env_config)
        except EvaluationPhysicsError as error:
            rows.extend(_label(error.rows, method=method, profile=profile,
                               training_seed=seed, split=args.split))
            write_csv(output / "episodes.partial.csv", rows)
            write_json(output / "evaluation.json", {**provenance, "status": "failed",
                                                      "error": str(error)})
            raise
        rows.extend(_label(cases, method=method, profile=profile,
                           training_seed=seed, split=args.split))
    write_csv(output / "episodes.csv", rows)
    write_json(output / "summary.json", aggregate(rows))
    write_json(output / "evaluation.json", {
        **provenance, "status": "complete", "method": method, "training_seed": seed,
        "profile": profile, "split": args.split, "seeds": seeds,
        "checkpoint_sha256": checkpoint_hash, "wall_seconds": perf_counter() - start,
        "episodes": len(rows),
    })
    print(output)


def smoke(config, output: Path) -> Path:
    if config.profile != "smoke" or config.env.max_steps != 400:
        raise ValueError("smoke requires the smoke profile with 400-action episodes")
    output = Path(output)
    _empty_output(output)
    started = perf_counter()
    provenance = run_provenance(output)
    write_json(output / "pipeline.json", {**provenance, "status": "running", "profile": "smoke"})
    try:
        run_physics_gate(output / "physics")
        dataset = collect_demonstrations(config, output / "data/demos.h5")
        config = replace(config, dataset=str(dataset))
        write_json(output / "config.json", config.to_dict())
        policies = {"scripted": RasterPolicy()}
        checkpoints = {}
        for method in ("bc", "ppo", "bc_ppo"):
            checkpoint = train(method, config, output / "runs" / method / str(config.seed))
            checkpoints[method] = str(checkpoint)
            policies[method] = PPO.load(checkpoint, device="cpu")
        # The smoke subset checks all suite code paths. All 20 ID validation cases
        # above selected checkpoints; no final-test cases enter this pipeline.
        seeds = list(VALIDATION_SEEDS[:3])
        all_rows = []
        for method, policy in policies.items():
            rows = []
            for suite in ("id", "curvature", "friction"):
                cases = evaluate(policy, suite, seeds, config=config.env)
                rows.extend(_label(cases, method=method, profile="smoke",
                                   training_seed=None if method == "scripted" else config.seed,
                                   split="validation_smoke_subset"))
            destination = output / "evaluation" / method
            write_csv(destination / "episodes.csv", rows)
            write_json(destination / "summary.json", aggregate(rows))
            all_rows.extend(rows)
        report = write_report([output / "evaluation"], output / "report",
                              training_root=output / "runs")
        estimate = estimate_full_compute(output / "runs", dataset)
        write_json(output / "report/compute-estimate.json", estimate)
        videos, used = [], set()
        selectors = [("success", lambda r: r["success"]),
                     ("contact_loss", lambda r: r["contact_loss"]),
                     ("physical_failure", lambda r: bool(r["failure_reason"]))]
        for index, (requested, predicate) in enumerate(selectors, 1):
            def identity(row):
                return row["method"], row["suite"], row["seed"]
            available = [r for r in all_rows if identity(r) not in used]
            matching = [r for r in available if predicate(r)]
            chosen = max(matching or available, key=lambda r: r["coverage"])
            used.add(identity(chosen))
            actual = (requested if matching else
                      ("timeout" if chosen["truncated"] else "physical_failure"))
            case = f"{chosen['method']}-{chosen['suite']}-{chosen['seed']}"
            filename = f"{index:02d}-{actual}-{case}.mp4"
            video = render_episode(policies[chosen["method"]], chosen["suite"], chosen["seed"],
                                   output / "videos" / filename, method=chosen["method"],
                                   profile="smoke", config=config.env)
            for metric in ("coverage", "elapsed_s", "unsafe_fraction"):
                if not np.isclose(video["metrics"][metric], chosen[metric], atol=1e-6):
                    raise RuntimeError(f"Rendered replay changed {metric}; inspect {filename}")
            video.update(requested_kind=requested, observed_kind=actual,
                         requested_kind_observed=bool(matching))
            videos.append(video)
        write_json(output / "videos/manifest.json", videos)
        write_json(output / "pipeline.json", {
            **provenance, "status": "complete", "profile": "smoke", "training_seeds": [config.seed],
            "evaluation_seeds": seeds, "evaluation_split": "validation_smoke_subset",
            "checkpoints": checkpoints, "dataset": str(dataset), "report": str(report),
            "evaluation_episodes": len(all_rows), "wall_seconds": perf_counter() - started,
            "full_experiment_launched": False,
        })
    except Exception as error:
        write_json(output / "pipeline.json", {**provenance, "status": "failed", "profile": "smoke",
                                              "error": repr(error),
                                              "wall_seconds": perf_counter() - started})
        raise
    print(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke_parser = commands.add_parser("smoke", help="Run the entire small pipeline")
    smoke_parser.add_argument("--config", default="configs/smoke.json")
    smoke_parser.add_argument("--output", type=Path)
    collect_parser = commands.add_parser("collect", help="Collect and quality-gate demonstrations")
    collect_parser.add_argument("--config", default="configs/full.json")
    collect_parser.add_argument("--output", required=True, type=Path)
    train_parser = commands.add_parser("train", help="Fit BC, PPO, or BC+PPO")
    train_parser.add_argument("--config", default="configs/full.json")
    train_parser.add_argument("--method", choices=["bc", "ppo", "bc_ppo"], required=True)
    train_parser.add_argument("--seed", type=int, default=11)
    train_parser.add_argument("--dataset", type=Path)
    train_parser.add_argument("--output", type=Path, required=True)
    eval_parser = commands.add_parser("evaluate", help="Evaluate a checkpoint or scripted baseline")
    eval_parser.add_argument("--config")
    eval_parser.add_argument("--method", choices=["scripted", "bc", "ppo", "bc_ppo"])
    eval_parser.add_argument("--checkpoint", type=Path)
    eval_parser.add_argument("--suite", choices=["id", "curvature", "friction", "all"],
                             default="id")
    eval_parser.add_argument("--split", choices=["validation", "test"], default="test")
    eval_parser.add_argument("--output", type=Path, required=True)
    report_parser = commands.add_parser("report", help="Aggregate paired evaluation CSVs")
    report_parser.add_argument("--input", nargs="+", type=Path, required=True)
    report_parser.add_argument("--output", type=Path, required=True)
    report_parser.add_argument("--runs", type=Path)
    report_parser.add_argument("--require-full", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "smoke":
        timestamp = datetime.now().strftime("pipeline-%Y%m%d-%H%M%S")
        output = args.output or Path("artifacts/smoke") / timestamp
        smoke(load_config(args.config), output)
    elif args.command == "collect":
        print(collect_demonstrations(load_config(args.config), args.output))
    elif args.command == "train":
        config = replace(load_config(args.config), seed=args.seed)
        if args.dataset:
            config = replace(config, dataset=str(args.dataset))
        print(train(args.method, config, args.output))
    elif args.command == "evaluate":
        _evaluate_command(args)
    elif args.command == "report":
        print(write_report(args.input, args.output, training_root=args.runs,
                           require_full=args.require_full))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
