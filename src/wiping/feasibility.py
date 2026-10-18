"""Reproducible, finite-state contact and rendering gate."""

import json
import os
import platform
from pathlib import Path
from time import perf_counter

import imageio.v3 as iio
import mujoco
import numpy as np

from wiping.model import build_model, read_normal_force


def run_physics_gate(output: Path) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    model = build_model(0.26, 0.5)
    data = mujoco.MjData(model)
    data.qpos[:] = [0, 0, 0.04]
    data.ctrl[:] = [0, 0, 0.002]
    mujoco.mj_forward(model, data)
    trace = np.empty((10000, 5))
    for index in range(len(trace)):
        mujoco.mj_step(model, data)
        force = read_normal_force(model, data)
        if not np.isfinite(np.r_[data.qpos, data.qvel, force]).all():
            raise RuntimeError(f"Physics gate failed at substep {index}")
        if any(warning.number for warning in data.warning):
            raise RuntimeError(f"MuJoCo warning during physics gate at substep {index}")
        trace[index] = [data.time, force, *data.qpos]
    np.savetxt(output / "force_trace.csv", trace, delimiter=",",
               header="time_s,force_n,x_m,y_m,z_m", comments="")
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, -0.02]
    camera.distance = 0.45
    camera.azimuth = 135
    camera.elevation = -45
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data, camera=camera)
        iio.imwrite(output / "frame.png", renderer.render())
    summary = {
        "profile": "smoke",
        "substeps": len(trace),
        "all_finite": True,
        "settled_mean_force_n": float(trace[-500:, 1].mean()),
        "max_force_n": float(trace[:, 1].max()),
        "wall_seconds": perf_counter() - start,
        "mujoco_version": mujoco.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "renderer": os.environ.get("MUJOCO_GL", "glfw (default)"),
        "integrator": "Euler",
    }
    if not 0.5 <= summary["settled_mean_force_n"] <= 8:
        raise RuntimeError(f"Position-servo loading gate failed: {summary}")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/smoke/physics"))
    print(json.dumps(run_physics_gate(parser.parse_args().output), indent=2))
