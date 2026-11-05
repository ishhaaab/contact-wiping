"""Resumable full-release orchestration, using the documented CLI commands."""

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from wiping.artifacts import write_json


def choose_output(base: Path, attempt: str) -> Path:
    base = Path(base)
    if not base.exists():
        return base
    destination = base.with_name(f"{base.name}-retry-{attempt}")
    if destination.exists():
        raise FileExistsError(destination)
    return destination


def training_complete(path: Path, method: str, seed: int) -> bool:
    path = Path(path)
    record_path = path / "run.json"
    if not record_path.exists():
        return False
    record = json.loads(record_path.read_text(encoding="utf-8"))
    expected_steps = 0 if method == "bc" else 2_002_944
    if not (record.get("status") == "complete" and record.get("profile") == "full"
            and record.get("method") == method and record.get("seed") == seed
            and record.get("actual_transitions") == expected_steps):
        return False
    return (path / "best.zip").is_file() and (path / "validation_episodes.csv").is_file()


def evaluation_complete(path: Path, method: str, seed: int | None, checkpoint: Path | None):
    import csv

    from wiping.evaluate import normalize_rows

    path = Path(path)
    if not (path / "evaluation.json").exists():
        return False
    record = json.loads((path / "evaluation.json").read_text(encoding="utf-8"))
    if not (record.get("status") == "complete" and record.get("profile") == "full"
            and record.get("method") == method and record.get("training_seed") == seed
            and record.get("split") == "test" and record.get("episodes") == 300):
        return False
    if checkpoint and record.get("checkpoint_sha256") != hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest():
        raise ValueError(f"Completed evaluation belongs to another checkpoint: {path}")
    with (path / "episodes.csv").open(newline="", encoding="utf-8") as file:
        rows = normalize_rows(list(csv.DictReader(file)))
    expected = {(suite, case) for suite in ("id", "curvature", "friction")
                for case in range(2000, 2100)}
    actual = {(r["suite"], r["seed"]) for r in rows}
    if actual != expected or any(r["method"] != method or r["training_seed"] != seed
                                 or r["evaluation_split"] != "test" for r in rows):
        raise ValueError(f"Completed evaluation has inconsistent cases: {path}")
    return True


def run_logged(command: list[str], log: Path, cwd: Path) -> float:
    started = perf_counter()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as file:
        file.write(subprocess.list2cmdline(command) + "\n")
        file.flush()
        subprocess.run(command, cwd=cwd, stdout=file, stderr=subprocess.STDOUT, check=True)
    return perf_counter() - started


def _find_training(base: Path, method: str, seed: int):
    candidates = [base, *sorted(base.parent.glob(f"{seed}-retry-*"))]
    return next((p for p in candidates if training_complete(p, method, seed)), None)


def _find_evaluation(base: Path, method: str, seed: int | None, checkpoint: Path | None):
    candidates = [base, *sorted(base.parent.glob(f"{base.name}-retry-*"))]
    return next((p for p in candidates if evaluation_complete(p, method, seed, checkpoint)), None)


def run_release(control: Path):
    root = Path(__file__).resolve().parents[2]
    control = Path(control).resolve()
    control.mkdir(parents=True, exist_ok=False)
    attempt = control.name
    lock = root / "artifacts/full-release.lock"
    # OS file locking releases automatically if the process dies, unlike a PID file.
    import msvcrt

    lock.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock.open("a+b")
    lock_file.seek(0)
    if not lock_file.read(1):
        lock_file.write(b"0")
        lock_file.flush()
    lock_file.seek(0)
    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
    started = perf_counter()
    state = {"status": "running", "pid": os.getpid(),
             "started_utc": datetime.now(UTC).isoformat(),
             "python": sys.executable, "attempt": attempt, "training": [], "evaluation": [],
             "commands": [], "experiment_settings_changed": False}

    def event(message):
        stamp = datetime.now(UTC).isoformat()
        text = f"{stamp} [{perf_counter() - started:.1f}s] {message}"
        print(text, flush=True)
        with (control / "progress.log").open("a", encoding="utf-8") as file:
            file.write(text + "\n")
        state["elapsed_seconds"] = perf_counter() - started
        write_json(control / "state.json", state)

    def command(arguments, name):
        args = [sys.executable, "-m", "wiping.cli", *map(str, arguments)]
        log = control / f"{name}.log"
        state["commands"].append({"argv": args, "log": str(log), "status": "running"})
        event(f"START {name}")
        seconds = run_logged(args, log, root)
        state["commands"][-1].update(status="complete", wall_seconds=seconds)
        event(f"COMPLETE {name}, {seconds:.1f}s")

    try:
        guard = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        state["automatic_sleep_inhibited"] = bool(guard)
        if not guard:
            raise RuntimeError("Windows refused automatic-sleep inhibition")
        event("Release started; automatic idle sleep inhibited, AC power recommended")
        for seed in (11, 22, 33):
            for method in ("bc", "ppo", "bc_ppo"):
                base = root / "runs/full" / method / str(seed)
                path = _find_training(base, method, seed)
                if path is None:
                    path = choose_output(base, attempt)
                    command(["train", "--method", method, "--seed", seed,
                             "--config", "configs/full.json", "--output", path],
                            f"train-{method}-{seed}")
                    if not training_complete(path, method, seed):
                        raise RuntimeError(f"Training did not meet completion contract: {path}")
                else:
                    event(f"SKIP completed training {method}/{seed}: {path}")
                state["training"].append({"method": method, "seed": seed, "path": str(path)})
                checkpoint = path / "best.zip"
                base_eval = root / "evaluations/full" / method / str(seed)
                evaluation = _find_evaluation(base_eval, method, seed, checkpoint)
                if evaluation is None:
                    evaluation = choose_output(base_eval, attempt)
                    command(["evaluate", "--checkpoint", checkpoint, "--suite", "all",
                             "--output", evaluation], f"evaluate-{method}-{seed}")
                    if not evaluation_complete(evaluation, method, seed, checkpoint):
                        raise RuntimeError(f"Incomplete evaluation: {evaluation}")
                else:
                    event(f"SKIP completed evaluation {method}/{seed}: {evaluation}")
                state["evaluation"].append({"method": method, "seed": seed,
                                             "path": str(evaluation)})
                event(f"PAIR COMPLETE {method}/{seed}")
        baseline = root / "evaluations/full/scripted"
        evaluation = _find_evaluation(baseline, "scripted", None, None)
        if evaluation is None:
            evaluation = choose_output(baseline, attempt)
            command(["evaluate", "--method", "scripted", "--config", "configs/full.json",
                     "--suite", "all", "--output", evaluation], "evaluate-scripted")
            if not evaluation_complete(evaluation, "scripted", None, None):
                raise RuntimeError("Incomplete scripted evaluation")
        state["evaluation"].append({"method": "scripted", "seed": None, "path": str(evaluation)})
        report = choose_output(root / "reports/full", attempt)
        # Explicit inputs exclude incomplete attempts and prevent duplicate final cases on resume.
        inputs = [str(Path(e["path"]) / "episodes.csv") for e in state["evaluation"]]
        canonical = all("-retry-" not in e["path"] for e in state["evaluation"])
        command(["report", "--input", *(["evaluations/full"] if canonical else inputs),
                 "--runs", "runs/full", "--output", report, "--require-full"], "report")
        state.update(status="complete", report=str(report),
                     ended_utc=datetime.now(UTC).isoformat())
        event("FULL COMPARISON COMPLETE; reporting and interpretation ready for session handoff")
    except BaseException as error:
        state.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        if state["commands"] and state["commands"][-1]["status"] == "running":
            state["commands"][-1]["status"] = "failed"
        event(f"STOPPED: {error!r}; no automatic retry or experiment changes")
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True, type=Path,
                        help="New artifacts directory for this orchestration attempt")
    run_release(parser.parse_args().control)
