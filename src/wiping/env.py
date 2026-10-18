"""Gymnasium interface for the Cartesian wiping head."""

import gymnasium as gym
import mujoco
import numpy as np

from wiping.config import EnvConfig
from wiping.coverage import CoverageGrid
from wiping.model import build_model, read_normal_force

ACTION_SCALE = np.array([0.004, 0.004, 0.002])
TARGET_LOW = np.array([-0.085, -0.085, -0.040])
TARGET_HIGH = np.array([0.085, 0.085, 0.060])


def step_reward(delta_coverage, forces, action, previous) -> float:
    excess = float(np.maximum(np.asarray(forces) - 8, 0).mean()) if len(forces) else 0.0
    smoothness = float(np.sum((action - previous) ** 2))
    return 10 * delta_coverage - 0.05 * excess / 8 - 0.01 * smoothness - 0.002


class WipingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, config: EnvConfig, render_mode: str | None = None):
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError("Supported render_mode is rgb_array")
        self.config = config
        self.render_mode = render_mode
        self.action_space = gym.spaces.Box(-1, 1, (3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (273,), dtype=np.float32)
        self._renderer = None
        self._done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        suite = (options or {}).get("suite", "id")
        if suite not in ("id", "curvature", "friction"):
            raise ValueError(f"Unknown suite: {suite}")
        self.radius = float(self.np_random.uniform(0.22, 0.30))
        self.friction = float(self.np_random.uniform(0.3, 0.7))
        if suite == "curvature":
            lo, hi = [(0.18, 0.21), (0.31, 0.34)][int(self.np_random.integers(2))]
            self.radius = float(self.np_random.uniform(lo, hi))
        elif suite == "friction":
            lo, hi = [(0.15, 0.25), (0.8, 1.0)][int(self.np_random.integers(2))]
            self.friction = float(self.np_random.uniform(lo, hi))
        self.model = build_model(self.radius, self.friction)
        self.data = mujoco.MjData(self.model)
        self.sphere_centre = np.array([0.0, 0.0, -self.radius])
        self.tool_geom_id = self.model.geom("tool_geom").id
        height, _ = self.surface_at(-0.075, -0.075)
        self.data.qpos[:] = [-0.075, -0.075, height + 0.030]
        self.target = self.data.qpos.copy()
        self.data.ctrl[:] = self.target
        mujoco.mj_forward(self.model, self.data)
        self.coverage = CoverageGrid(self.radius)
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.steps = self.physics_substeps = self.unsafe_substeps = self.excessive_count = 0
        self.force_sum = self.episode_force_max = self.force_mean = self.force_max = 0.0
        self.step_unsafe_fraction = 0.0
        self.success = False
        self.failure_reason = None
        self._done = False
        return self._observation(), self._info()

    def surface_at(self, x: float, y: float) -> tuple[float, np.ndarray]:
        radicand = self.radius**2 - x**2 - y**2
        # Invalid geometry stays nonfinite; it is not clipped to plausible state.
        z = float(np.sqrt(radicand)) if radicand >= 0 else float("nan")
        return z - self.radius, np.array([x, y, z]) / self.radius

    def _observation(self):
        height, normal = self.surface_at(*self.data.qpos[:2])
        return np.concatenate([
            self.data.qpos / 0.1, self.data.qvel / 0.2, self.target / 0.1,
            [np.clip(self.force_mean / 15, 0, 2)], self.previous_action,
            [height / 0.1], normal, self.coverage.covered,
        ]).astype(np.float32)

    def _invalid_state(self) -> str | None:
        if not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()):
            return "nonfinite_physics"
        if np.linalg.norm(self.data.qpos) > 0.15:
            return "out_of_bounds"
        return None

    def step(self, action):
        if self._done:
            raise RuntimeError("reset() is required before stepping a finished episode")
        requested = np.asarray(action, dtype=np.float32)
        if requested.shape != (3,) or not np.isfinite(requested).all():
            raise ValueError("Action must be a finite vector of shape (3,)")
        executed = np.clip(requested, -1, 1)
        clipped = not np.array_equal(executed, requested)
        self.target = np.clip(self.target + ACTION_SCALE * executed, TARGET_LOW, TARGET_HIGH)
        self.data.ctrl[:] = self.target
        before = self.coverage.fraction
        forces = []
        self.failure_reason = self._invalid_state()
        for _ in range(25):
            if self.failure_reason is not None:
                break
            mujoco.mj_step(self.model, self.data)
            force = read_normal_force(self.model, self.data)
            self.physics_substeps += 1
            self.failure_reason = self._invalid_state()
            # MuJoCo can reset unstable qpos internally. Warnings must not disappear.
            if any(warning.number for warning in self.data.warning):
                self.failure_reason = "mujoco_warning"
            if not np.isfinite(force):
                self.failure_reason = "nonfinite_physics"
                break
            forces.append(force)
            self.force_sum += force
            self.episode_force_max = max(self.episode_force_max, force)
            self.unsafe_substeps += int(force > 8)
            self.excessive_count = self.excessive_count + 1 if force > 15 else 0
            if self.excessive_count >= 3:
                self.failure_reason = "excessive_force"
            tool = self.data.geom_xpos[self.tool_geom_id]
            radial = tool - self.sphere_centre
            projection = self.sphere_centre + self.radius * radial / np.linalg.norm(radial)
            self.coverage.update(projection, force)
        self.steps += 1
        self.force_mean = float(np.mean(forces)) if forces else 0.0
        self.force_max = float(max(forces, default=0.0))
        self.step_unsafe_fraction = float(np.mean(np.asarray(forces) > 8)) if forces else 0.0
        reward = step_reward(
            self.coverage.fraction - before, forces, executed, self.previous_action
        )
        self.previous_action = executed.copy()
        self.success = bool(self.failure_reason is None and self.coverage.fraction >= 0.90
                            and self.unsafe_substeps / max(1, self.physics_substeps) <= 0.05)
        terminated = self.failure_reason is not None or self.success
        truncated = not terminated and self.steps >= self.config.max_steps
        reward += 2 if self.success else (-2 if self.failure_reason else 0)
        self._done = terminated or truncated
        info = self._info()
        info.update(action_clipped=clipped, executed_action=executed.tolist())
        return self._observation(), float(reward), bool(terminated), bool(truncated), info

    def _info(self):
        return {
            "coverage": self.coverage.fraction,
            "force_mean_n": self.force_mean,
            "force_max_n": self.force_max,
            "unsafe_fraction": self.unsafe_substeps / max(1, self.physics_substeps),
            "elapsed_s": self.physics_substeps * 0.002,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "radius_m": self.radius,
            "friction": self.friction,
            "physics_substeps": self.physics_substeps,
            "unsafe_substeps": self.unsafe_substeps,
            "episode_force_mean_n": self.force_sum / max(1, self.physics_substeps),
            "episode_force_max_n": self.episode_force_max,
            "step_unsafe_fraction": self.step_unsafe_fraction,
        }

    def render(self):
        if self.render_mode != "rgb_array":
            raise ValueError("Construct the environment with render_mode='rgb_array'")
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0, 0, -0.02]
        camera.distance = 0.42
        camera.azimuth = 135
        camera.elevation = -55
        mujoco.mj_forward(self.model, self.data)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()


    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
