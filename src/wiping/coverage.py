"""Geometric wiping coverage on a spherical cap."""

import numpy as np


class CoverageGrid:
    def __init__(self, radius: float):
        self.radius = radius
        x, y = np.meshgrid(np.linspace(-0.075, 0.075, 16),
                           np.linspace(-0.075, 0.075, 16), indexing="xy")
        z = np.sqrt(radius**2 - x**2 - y**2) - radius
        self.centres = np.stack([x, y, z], axis=-1).reshape(256, 3)
        self.covered = np.zeros(256, dtype=bool)

    @property
    def fraction(self) -> float:
        return float(self.covered.mean())

    def update(self, contact_point: np.ndarray, force_n: float) -> float:
        if not 0.5 <= force_n <= 8:
            return 0.0
        distances_sq = np.sum((self.centres - contact_point) ** 2, axis=1)
        new = (distances_sq <= 0.008**2) & ~self.covered
        self.covered |= new
        return float(new.sum() / 256)
