"""Run provenance and small structured artifact writers."""

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


def write_json(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError("Cannot write an empty metrics table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_provenance(output: Path) -> dict:
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unavailable", None
    versions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    hashes = {}
    for source in sorted((root / "src/wiping").glob("*.py")):
        relative = source.relative_to(root)
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    shutil.copy2(root / "requirements-lock.txt", output / "requirements-lock.txt")
    return {
        "git_revision": revision, "git_dirty": dirty, "source_sha256": hashes,
        "python": platform.python_version(), "packages": versions,
        "hardware": {"platform": platform.platform(), "cpu": platform.processor(),
                     "logical_cpus": os.cpu_count(), "training_device": "cpu",
                     "torch_threads": 1},
    }
