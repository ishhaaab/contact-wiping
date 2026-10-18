"""Catch absent contact, gravity mistakes, frame errors, and unrelated contact sums."""

import mujoco
import numpy as np
import pytest

from wiping.feasibility import run_physics_gate
from wiping.model import build_model, read_normal_force


def test_hover_is_finite_and_has_no_force():
    model = build_model(0.26, 0.5)
    data = mujoco.MjData(model)
    data.qpos[:] = [0, 0, 0.04]
    data.ctrl[:] = data.qpos
    mujoco.mj_forward(model, data)
    for _ in range(1000):
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all()
        assert np.isfinite(data.qvel).all()
        assert 0 <= read_normal_force(model, data) < 1e-4
    # This also catches a tool whose gravity compensation is missing.
    np.testing.assert_allclose(data.qpos, [0, 0, 0.04], atol=1e-6)


@pytest.mark.parametrize("x", [0.0, 0.07])
def test_known_three_newton_static_load(x):
    model = build_model(0.26, 0.5)
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
    data = mujoco.MjData(model)
    normal = np.array([x, 0, np.sqrt(0.26**2 - x**2)]) / 0.26
    data.qpos[:] = np.array([0, 0, -0.26]) + (0.26 + 0.012) * normal
    data.xfrc_applied[model.body("tool").id, :3] = -3 * normal
    mujoco.mj_forward(model, data)
    forces = []
    for _ in range(3000):
        mujoco.mj_step(model, data)
        forces.append(read_normal_force(model, data))
        assert np.isfinite(data.qpos).all()
    measured = np.mean(forces[-500:])
    assert 2.7 <= measured <= 3.3, f"static load measured {measured} N"


def test_unrelated_contacts_do_not_contribute():
    model = mujoco.MjModel.from_xml_string('''
    <mujoco><worldbody>
      <geom name="surface" type="sphere" size=".26" pos="0 0 -.26"/>
      <geom name="floor" type="plane" size="1 1 .1" pos="0 0 -.5"/>
      <body name="tool" pos="0 0 .05" gravcomp="1"><freejoint/>
        <geom name="tool_geom" type="sphere" size=".012" mass=".2"/>
      </body>
      <body pos=".5 0 -.49"><freejoint/>
        <geom name="other" type="sphere" size=".02" mass="1"/>
      </body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon > 0
    force = np.zeros(6)
    mujoco.mj_contactForce(model, data, 0, force)
    assert force[0] > 0
    assert read_normal_force(model, data) == 0


@pytest.mark.parametrize("radius,friction", [(0, 0.5), (-1, 0.5), (0.26, -1)])
def test_invalid_physical_parameters_rejected(radius, friction):
    with pytest.raises(ValueError):
        build_model(radius, friction)


def test_feasibility_trace_and_frame(tmp_path):
    import imageio.v3 as iio

    result = run_physics_gate(tmp_path)
    trace = np.loadtxt(tmp_path / "force_trace.csv", delimiter=",", skiprows=1)
    assert trace.shape == (10000, 5)
    assert np.isfinite(trace).all()
    assert 0.5 < result["settled_mean_force_n"] < 8
    image = iio.imread(tmp_path / "frame.png")
    assert image.shape == (480, 640, 3)
    assert image.std() > 1
