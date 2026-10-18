"""Validated configuration; values are SI unless documented otherwise."""

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

TRAINING_SEEDS = (11, 22, 33)
VALIDATION_SEEDS = tuple(range(1000, 1020))
TEST_SEEDS = tuple(range(2000, 2100))
PROFILES = {"smoke": (8, 8192), "pilot": (40, 200_000), "full": (200, 2_000_000)}


@dataclass(frozen=True)
class EnvConfig:
    max_steps: int = 400

    def __post_init__(self):
        if type(self.max_steps) is not int or not 1 <= self.max_steps <= 400:
            raise ValueError("max_steps must be an integer in [1, 400]")


@dataclass(frozen=True)
class RunConfig:
    profile: str = "smoke"
    seed: int = 11
    demo_episodes: int = 8
    total_steps: int = 8192
    dataset: str | None = None
    env: EnvConfig = field(default_factory=EnvConfig)

    def __post_init__(self):
        if self.profile not in PROFILES:
            raise ValueError(f"Unknown profile: {self.profile}")
        if type(self.seed) is not int or self.seed not in TRAINING_SEEDS:
            raise ValueError(f"Training seed must be one of {TRAINING_SEEDS}")
        if (self.demo_episodes, self.total_steps) != PROFILES[self.profile]:
            raise ValueError("Profile budgets must match the experiment contract")

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> RunConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a JSON object")
    unknown = set(raw) - {f.name for f in fields(RunConfig)}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    try:
        raw["env"] = EnvConfig(**raw.get("env", {}))
        return RunConfig(**raw)
    except TypeError as error:
        raise ValueError(f"Invalid configuration: {error}") from error
