# Physics feasibility gate

Verified 2026-10-18 on Windows 11, CPython 3.11.17, MuJoCo 3.16.0.

`python -m pytest tests/test_physics.py -q` passes all eight tests. Tests cover finite hovering, gravity compensation, 3 N applied radial loads at the top and off-axis with disabled actuators, unrelated-contact filtering, invalid physical parameters, and the trace/render gate.

`python -m wiping.feasibility --output artifacts/smoke/physics` executed 10,000 substeps. All positions, velocities, and measured forces were finite and MuJoCo emitted no simulation warnings. The enabled position servo settled to 2.969377196325013 N. Peak force was 17.821355102403345 N during the abrupt target change from hovering to loading. This diagnostic deliberately has no episode safety controller. The environment must apply the specified consecutive-substep force termination.

Artifacts: `artifacts/smoke/physics/force_trace.csv`, `frame.png`, `summary.json`. The measured wall time was 1.7454 seconds including rendering. This is a physics diagnostic, not policy training throughput.

Renderer: default GLFW context on native Windows, no `MUJOCO_GL` override. The rendered RGB frame is 640 by 480. Euler is MuJoCo's default integrator and is explicit in the model. No GPU tensor operations are used. CPU PyTorch 2.14.1+cpu is installed.

## API evidence

- Installed `mujoco.mj_contactForce.__doc__`: extracts 6D force:torque in the contact frame.
- [MuJoCo 3.16.0 API](https://mujoco.readthedocs.io/en/3.16.0/APIreference/APIfunctions.html#mj-contactforce).
- [Contact data structure](https://mujoco.readthedocs.io/en/3.16.0/APIreference/APItypes.html#mjcontact): first contact-frame axis is the normal.
- [XML reference](https://mujoco.readthedocs.io/en/3.16.0/XMLreference.html): `gravcomp="1"`, position `kp`, and actuator force limits verified by compiling and running the model.

Force measurement uses element zero of the contact-coordinate wrench and filters by the two named geoms. It does not use world-z force or include other contacts.
