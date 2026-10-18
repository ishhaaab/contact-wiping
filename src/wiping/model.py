"""MuJoCo model and contact-pair measurement."""

import mujoco
import numpy as np


def build_model(radius: float, friction: float) -> mujoco.MjModel:
    if not np.isfinite([radius, friction]).all() or radius <= 0 or friction < 0:
        raise ValueError("radius must be positive and friction nonnegative, both finite")
    xml = f"""
    <mujoco model="cartesian_wiping">
      <option timestep="0.002" gravity="0 0 -9.81" integrator="Euler"/>
      <visual><global offwidth="640" offheight="480"/></visual>
      <default>
        <joint type="slide" damping="5"/>
        <geom condim="3" solref="0.02 1" solimp="0.9 0.95 0.001"
              friction="{friction} 0.005 0.0001"/>
      </default>
      <worldbody>
        <light pos="0 -.3 .6" dir="0 .3 -1"/>
        <geom name="surface" type="sphere" size="{radius}" pos="0 0 {-radius}"
              rgba=".35 .5 .6 1"/>
        <body name="tool" gravcomp="1">
          <joint name="x" axis="1 0 0"/>
          <joint name="y" axis="0 1 0"/>
          <joint name="z" axis="0 0 1"/>
          <geom name="tool_geom" type="sphere" size="0.012" mass="0.20"
                rgba=".95 .55 .1 1"/>
        </body>
      </worldbody>
      <actuator>
        <position name="x_servo" joint="x" kp="300" forcelimited="true"
                  forcerange="-12 12"/>
        <position name="y_servo" joint="y" kp="300" forcelimited="true"
                  forcerange="-12 12"/>
        <position name="z_servo" joint="z" kp="300" forcelimited="true"
                  forcerange="-12 12"/>
      </actuator>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def read_normal_force(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Sum compressive magnitudes in contact coordinates, not world z forces.

    MuJoCo 3.16: contact.frame[0:3] is the contact normal. mj_contactForce
    returns force:torque in that frame, so result[0] is the normal force.
    """
    tool = model.geom("tool_geom").id
    surface = model.geom("surface").id
    total = 0.0
    force = np.zeros(6)
    for index in range(data.ncon):
        pair = data.contact[index].geom
        if (pair[0] == tool and pair[1] == surface) or (
            pair[0] == surface and pair[1] == tool
        ):
            mujoco.mj_contactForce(model, data, index, force)
            if not np.isfinite(force).all():
                return float("nan")
            total += max(0.0, float(force[0]))
    return total
