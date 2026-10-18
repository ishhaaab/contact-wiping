"""Validated configuration; values are SI unless documented otherwise."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    max_steps: int = 400

    def __post_init__(self):
        if type(self.max_steps) is not int or not 1 <= self.max_steps <= 400:
            raise ValueError("max_steps must be an integer in [1, 400]")
