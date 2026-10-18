# Implementation decisions and progress

2026-10-18. Source plan: `../../docs/superpowers/plans/2026-10-18-surface-wiping.md`.
Source specification: `../../docs/superpowers/specs/2026-10-18-project-specifications.md`, shared requirements and section A.

## Decisions

1. Ruling: use `MOWITO/projects/contact-wiping` because the latest request adds a `projects` subfolder. This is a new independent Git repository on `implementation/surface-wiping`. Cost if wrong: move the repository; no planning files have been changed.
2. Ruling: run the feasibility gate on native Windows 11 using available CPython 3.11.17. WSL reports no installed distributions. Linux remains the reference platform; native compatibility must be demonstrated, not assumed. Cost if wrong: stop the gate and arrange the reference environment. No simulator or experiment change is authorized by this decision.
3. Ruling: the eight-episode smoke and forty-episode pilot profiles override the full dataset's 200 episodes. Preserve the 80/20 episode-level split using the first floor(0.8*N) for training and the rest for BC validation, with seeds starting at 3000. Full remains exactly 160/40 and 3000..3199. Cost if wrong: regenerate smoke/pilot data; full protocol is unaffected.
4. Ruling: Task 2's twenty scripted episodes run after Task 3 implements the controller, as the plan explicitly requires. Task 3's full dataset and Task 5's full comparison are experimental-release work after the smoke pipeline and compute estimate. Cost if wrong: additional full runs before an experimental release; smoke must never be labelled full evidence.

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

- [ ] Task 1: contact physics, static load, 10,000 substeps, frame, dependency lock.
- [ ] Task 2: environment, coverage, reward, safety, reproducibility, SB3 check.
- [ ] Task 3: scripted demonstrations, HDF5 split, controller quality gate.
- [ ] Task 4: BC, PPO, transfer, smoke training, throughput estimate.
- [ ] Task 5: evaluation, reporting, videos, CLI, clean-install verification.
- [ ] Final whole-repository review and handoff.

## Experimental status

No experiments run yet. No physical-robot validation.
