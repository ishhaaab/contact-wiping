"""Imperfect scripted raster demonstrations. These are not human demonstrations."""

import numpy as np

from wiping.env import ACTION_SCALE, WipingEnv


class RasterPolicy:
    def __init__(self, seed: int = 11):
        self.reset(seed)

    def reset(self, seed: int):
        self.rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))
        self.normal_offset_m = float(self.rng.uniform(-0.003, 0.003))
        self._waypoints = None
        self._index = 0

    def _build_waypoints(self, env: WipingEnv):
        points = []
        # Surface waypoints spaced at most 4 mm also follow the cap mid-sweep.
        # Direct interpolation between endpoint heights would cut through the cap.
        previous = None
        for row, y in enumerate(np.arange(13) * 0.012 - 0.075):
            xs = np.linspace(-0.075, 0.075, 39)
            if row % 2:
                xs = xs[::-1]
            if previous is not None:
                for yy in np.linspace(previous[1], y, 4)[1:]:
                    points.append((xs[0], yy))
            points.extend((x, y) for x in xs)
            previous = (xs[-1], y)
        targets = []
        for x, y in points:
            _, normal = env.surface_at(x, y)
            targets.append(env.sphere_centre +
                           (env.radius + 0.012 - 0.002 + self.normal_offset_m) * normal)
        self._waypoints = np.asarray(targets)

    def predict(self, env: WipingEnv) -> np.ndarray:
        if self._waypoints is None:
            self._build_waypoints(env)
        # The tolerance accommodates the specified action noise. It does not
        # inspect coverage, choose offsets, or reject unsuccessful episodes.
        while (self._index < len(self._waypoints) - 1
               and np.linalg.norm(env.target - self._waypoints[self._index]) < 0.0012):
            self._index += 1
        delta = (self._waypoints[self._index] - env.target) / ACTION_SCALE
        action = np.clip(delta, -1, 1) + self.rng.normal(0, 0.03, 3)
        return np.clip(action, -1, 1).astype(np.float32)
