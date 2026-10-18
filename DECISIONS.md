# Implementation decisions and progress

2026-10-18. Source plan: `../../docs/superpowers/plans/2026-10-18-surface-wiping.md`.
Source specification: `../../docs/superpowers/specs/2026-10-18-project-specifications.md`, shared requirements and section A.

## Decisions

1. Ruling: use `MOWITO/projects/contact-wiping` because the latest request adds a `projects` subfolder. This is a new independent Git repository on `implementation/surface-wiping`. Cost if wrong: move the repository; no planning files have been changed.
2. Ruling: run the feasibility gate on native Windows 11 using available CPython 3.11.17. WSL reports no installed distributions. Linux remains the reference platform; native compatibility must be demonstrated, not assumed. Cost if wrong: stop the gate and arrange the reference environment. No simulator or experiment change is authorized by this decision.
3. Ruling: the eight-episode smoke and forty-episode pilot profiles override the full dataset's 200 episodes. Preserve the 80/20 episode-level split using the first floor(0.8*N) for training and the rest for BC validation, with seeds starting at 3000. Full remains exactly 160/40 and 3000..3199. Cost if wrong: regenerate smoke/pilot data; full protocol is unaffected.
4. Ruling: Task 2's twenty scripted episodes run after Task 3 implements the controller, as the plan explicitly requires. Task 3's full dataset and Task 5's full comparison are experimental-release work after the smoke pipeline and compute estimate. Cost if wrong: additional full runs before an experimental release; smoke must never be labelled full evidence.
5. Ruling: use CPU PyTorch and four CPU simulation environments for the small MLP. The GPU is available but the installed CPU build does not use it. Cost if wrong: longer training; throughput will be measured before long runs. Algorithm and network sizes are unchanged.
6. Ruling: the controller uses `SeedSequence(episode_seed, spawn_key=(1,))`, separate from the environment's episode-seeded generator. Reusing the same stream produced a measured radius/normal-offset correlation of 1.0. The regression failed before this fix. Uniform offset and Gaussian noise distributions, episode seeds, physics, and all splits stay as specified. Cost if wrong: regenerate demonstrations. Pre-fix artifacts are retained under `artifacts/diagnostics/pre-rng-fix` and are excluded from training and reported smoke evidence.
7. Ruling: interpolate the raster's surface waypoints at no more than 4 mm before applying the prescribed radial tool-centre formula. Advancing waypoints uses a 1.2 mm target-distance tolerance to accommodate action noise. A single endpoint-height interpolation would cut through a curved cap. Cost if wrong: different scripted coverage; controller discretization is recorded and shared by demonstrations and baseline evaluation.

## Pre-flight interface checks

| Producer -> consumer | Check |
| --- | --- |
| Task 1 model -> Task 2 environment | Three slide joints, named tool/surface geoms, normal force summed only for their contact pair. |
| Task 2 -> Task 3 dataset | 273 float32 observations, 3 clipped actions, terminal observations retained, terminated/truncated kept separately. |
| Task 3 -> Task 4 BC | Variable episode lengths, explicit episode split, transition obs/actions aligned. Smoke/pilot counts resolved above. |
| Task 4 -> Task 5 evaluation | Same MLP actor for BC and PPO, fresh PPO optimizer after BC transfer; validation seeds select checkpoints. |
| Tasks 2/3 -> Task 5 metrics | Coverage and failure metrics survive termination; evaluation includes all attempted episodes. |

No unresolved pre-implementation blocker. Physics and demonstration quality are executable gates.

## Host preflight

- Windows 11 Home Single Language, x86-64, version 10.0.26200.
- AMD Ryzen 7 5800H, 8 cores / 16 logical processors.
- Physical RAM: 14,877,265,920 bytes. Free disk at inspection: 133,281,980,416 bytes.
- NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB VRAM, driver 617.42.
- Available CPython 3.11.17 runtime registered by Astral. Create a project-local venv; no existing simulator or learning dependencies in that runtime.
- `repowise` is not on PATH; use direct tools and concise command output.

## Task status

- [x] Task 1: contact physics, static load, 10,000 substeps, frame, dependency lock.
- [x] Task 2: environment, coverage, reward, safety, reproducibility, SB3 check. Twenty scripted episodes follow in Task 3.
- [x] Task 3 software and smoke: scripted demonstrations, HDF5 split, controller quality gate. Full 200-episode collection follows the smoke pipeline.
- [ ] Task 4: BC, PPO, transfer, smoke training, throughput estimate.
- [ ] Task 5: evaluation, reporting, videos, CLI, clean-install verification.
- [ ] Final whole-repository review and handoff.

## Experimental status

Task 1: seven intended RED failures before model implementation, then seven passing tests. The trace/render test failed before its implementation, then the suite passed 8/8. `ruff check .` passes. Physics details and exact reproduction commands are in `reports/physics-gate.md`; exact dependency versions are in `requirements-lock.txt`.

Task 2: 18 RED failures before implementation, then the full suite passed 26/26 including SB3 `check_env`. Verified exact observation order, reproducibility, contact-gated coverage, reset, clipped actions, reward arithmetic, excessive-force counting across control boundaries, timeout distinction, success bonus, and invalid-state handling before MuJoCo can silently reset it. Profiles preserve the specified budgets. Raw nonfinite state remains visible on a failed transition; training must stop on it. Rendering uses forward kinematics and never changes targets or steps physics.

Task 3: five RED dataset/controller tests and the RNG-stream regression failed before their implementations. Full suite passed 32/32. Corrected smoke data contains eight complete episodes, 3200 transitions, seeds 3000..3007, with six training and two BC-validation episodes. Six of eight reach 0.50 coverage without excessive force, exceeding the 20% gate. Mean final coverage is 0.5322265625; all eight time out and none meet task success. Dataset and SHA256 manifest are under `data/smoke/`; a two-episode real fixture is under `tests/fixtures/`.

Task 2's deferred twenty scripted episodes also ran after the RNG fix: seeds 3010..3029, 8000 transitions, mean coverage 0.497265625, qualifying fraction 0.75, zero task successes, zero physical failures, all timeouts. Raw per-episode results are `reports/smoke/scripted-20.json`. This diagnostic took 14.5311 seconds. It is separate from learning performance and final evaluation.

No training experiments run yet. No physical-robot validation.
