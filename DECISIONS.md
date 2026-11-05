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
8. Ruling: implement the basic seeded evaluator during Task 4 because validation checkpoint selection consumes it. Task 5 adds aggregation, plots, videos, and the unified CLI. Cost if wrong: module ordering only. Validation uses exactly seeds 1000..1019; smoke never reads the final-test suite.
9. Ruling: the end-to-end smoke command trains seed 11 by default, uses all twenty ID validation cases for checkpoint selection, and checks each evaluation suite on validation seeds 1000..1002. This execution check does not run or relabel the final-test split. Full release validation requires all 3,000 prescribed episodes across methods, suites, and training seeds. Cost if wrong: additional smoke runs; full evaluation remains unchanged.
10. Ruling: render three actual observed episodes. If the smoke policies produce no success, record that absence in the video manifest and label the replacement video as a timeout or failure. Cost if wrong: the requested success illustration remains unavailable until a policy actually succeeds. No states or outcomes are fabricated.
11. Ruling: the full config's default dataset is `data/demos.h5`, matching the plan's collect/train command pair. Smoke and pilot remain under their named directories; `--dataset` supports alternate locations. Cost if wrong: pass the intended dataset path explicitly. HDF5 profile and split validation prevents training on a mismatched profile.

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
- [x] Task 4: BC, PPO, transfer, smoke training, throughput estimate.
- [x] Task 5 software: evaluation, reporting, videos, CLI, clean-install verification. Full trained comparison remains experimental work.
- [x] Final whole-repository review and handoff. Important findings fixed; one minor deferred.

## Experimental status

Task 1: seven intended RED failures before model implementation, then seven passing tests. The trace/render test failed before its implementation, then the suite passed 8/8. `ruff check .` passes. Physics details and exact reproduction commands are in `reports/physics-gate.md`; exact dependency versions are in `requirements-lock.txt`.

Task 2: 18 RED failures before implementation, then the full suite passed 26/26 including SB3 `check_env`. Verified exact observation order, reproducibility, contact-gated coverage, reset, clipped actions, reward arithmetic, excessive-force counting across control boundaries, timeout distinction, success bonus, and invalid-state handling before MuJoCo can silently reset it. Profiles preserve the specified budgets. Raw nonfinite state remains visible on a failed transition; training must stop on it. Rendering uses forward kinematics and never changes targets or steps physics.

Task 3: five RED dataset/controller tests and the RNG-stream regression failed before their implementations. Full suite passed 32/32. Corrected smoke data contains eight complete episodes, 3200 transitions, seeds 3000..3007, with six training and two BC-validation episodes. Six of eight reach 0.50 coverage without excessive force, exceeding the 20% gate. Mean final coverage is 0.5322265625; all eight time out and none meet task success. Dataset and SHA256 manifest are under `data/smoke/`; a two-episode real fixture is under `tests/fixtures/`.

Task 2's deferred twenty scripted episodes also ran after the RNG fix: seeds 3010..3029, 8000 transitions, mean coverage 0.497265625, qualifying fraction 0.75, zero task successes, zero physical failures, all timeouts. Raw per-episode results are `reports/smoke/scripted-20.json`. This diagnostic took 14.5311 seconds. It is separate from learning performance and final evaluation.

Task 4: five learning/evaluation tests failed before implementation. Full suite passed 37/37 and Ruff passed. Tiny-fixture BC reduced action MSE by more than 50% without modifying critic tensors; transfer preserved deterministic actions to 1e-6; a real four-environment PPO update changed actor weights and retained finite critic parameters.

Seed-11 smoke training completed for BC, PPO, and BC+PPO under `runs/smoke/{method}/11/`. Both RL methods used exactly 8192 transitions. Every learned method evaluated all 20 validation seeds. Mean validation coverage: BC 0.05078125, PPO 0.0, BC+PPO 0.096875. All three had zero successes. These are smoke checks, not full experimental results. A separate tensor comparison confirmed the independently fitted BC and BC+PPO initialization checkpoints are exactly equal.

Measured throughput excluding validation: PPO 506.0400 transitions/s, BC+PPO 444.0391 transitions/s. Full PPO rounds to 2,002,944 transitions per run. Extrapolating three seeds per RL method and 41 validation rounds per run gives about 8.116 hours for six RL runs including validation, before BC fitting, full data collection, final evaluation, and videos. No long run has been launched. Timing is host-specific; the initial test suite ran concurrently with part of the BC smoke stage.

No physical-robot validation.

## 2026-10-19 verification

Task 5: eight new RED tests preceded the aggregation, report, replay-video, and CLI code. The author environment and independently created `.venv-verify` both passed 46 tests. Fresh-environment Ruff passed. Installation used the exact lockfile and editable install without dependency resolution or build isolation. `pip check` found no broken requirements; all 42 exact package pins were compared with installed metadata and matched. Evidence: `reports/smoke/lock-verification.json`.

The clean-environment command `python -m wiping.cli smoke --config configs/smoke.json --output artifacts/smoke/verified-2026-10-19` completed in 151.3429 seconds. It regenerated physics evidence and data, trained all methods, selected checkpoints on twenty validation cases each, and evaluated 36 smoke cases across all methods and suites. It wrote three 401-frame MP4 videos with per-step CSV/plots. The contact-loss video also shows measured contact recovery. No successes or physical-failure terminations occurred in the 36-case smoke evaluation; the other videos are accurately labelled timeouts. This is separate from the safety tests that deliberately trigger physical-failure branches.

Canonical smoke report: `reports/smoke/verified/results.md`. Raw artifacts and source snapshots: `artifacts/smoke/verified-2026-10-19/`. Fresh throughput was 447.3378 transitions/s for PPO and 432.4421 for BC+PPO. Estimated full experiment: 8.6563 hours RL plus validation, 0.2510 hours BC fitting, 0.8159 hours final evaluation, about 161 seconds collection, totaling 9.7678 hours before video rendering. This is a linear extrapolation, not a runtime guarantee. No long training run was launched.

Full 200-episode collection completed: 80,000 transitions, seeds 3000..3199, exact 160/40 episode split, qualifying fraction 0.81, mean final coverage 0.55451171875. All 200 episodes timed out; none succeeded or had a physical-failure termination. Dataset `data/demos.h5`, SHA256 `0ddd2dc0f9346b254c585c5cc1b493a692974a3227a5e828b65798226cebb53a`. Summary: `reports/full-dataset.json`.

## 2026-11-06 final review and corrections

The checkout is now on `main` at the reviewed commit. This session did not switch or rename the branch. The previous virtual environments were absent; `.venv-verify` was recreated from the lockfile. No user changes were reverted.

Final: fixed simulator faults overwritten by excessive-force termination. Warning and out-of-bounds fault-injection regressions failed before the fix. Simulator faults now take precedence, preserving fail-fast diagnostics.

Final: fixed validation/test aggregation. A mixed-split regression failed before the fix. Reports reject mixed evaluation splits and explicitly label their split in JSON and Markdown. Legacy rows without a split remain explicitly `unspecified`.

Final: fixed misleading contact-loss classification. The old 0.5 N threshold detected weak wiping force, not absent contact. A partial-unloading regression failed before correction. Contact loss now requires a full control interval with maximum force below 1e-4 N after prior contact. This corrects an illustrative diagnostic, not physics, observations, actions, reward, algorithms, or evaluation splits. The old 2026-10-19 contact-loss video and recovery claim are superseded. Historical artifacts remain intact for audit.

Final: minor (deferred): the off-axis static-load test is insufficiently steep to reject a world-z-force mutation within its tolerance. Production measurement correctly uses contact-frame force. Stronger mutation coverage is deferred test hardening.

Final: Ruling: retain the algorithms without claiming learning superiority or a full experimental release. Full training, Linux/WSL verification, physical validation, and visual aesthetics were outside the review's measured evidence. Cost if wrong: separate experiments or execution environments are needed, not an optimistic completion claim.

Final review: one fresh-context reviewer found two Important issues and one Minor. Both Important issues and the independently identified diagnostic-label bug enter one regression-tested fix pass. No second reviewer is dispatched. Full training remains unrun.

Final verification: four intended regression failures were observed before the fixes; the full suite then passed 52/52, and Ruff passed. Targeted post-fix checks passed 14/14. `pip check` passes. The corrected complete smoke command finished in 143.6264 seconds under `artifacts/smoke/final-2026-11-06-complete/`, producing 36 evaluation cases, three learned checkpoints, and three videos including actual contact loss. The prior retry without `-complete` was interrupted by the tool's 120-second limit and is retained, not reported as completed. Latest full estimate: 9.5982 hours before rendering. Canonical committed report: `reports/smoke/final/`.

Final: Ruling: preserve the existing current `main` checkout and commit only this session's implementation corrections; do not switch branches, merge, or push. The repository acquired its current branch/remote outside these implementation actions. Cost if wrong: integration can be decided separately; local work remains recoverable.

## Full experimental release request

The user authorized the full three-seed comparison and final evaluation, plus logging-only diagnostics, a resumable detached worker, and automatic sleep inhibition. No experiment settings were changed. Preflight confirmed the full dataset and its gate; original tests passed 52/52. New passive-logging/launcher tests passed, with the complete pre-launch suite at 56/56 and Ruff clean.

Initial PPO value diagnostics distinguish the first-rollout pre-optimizer return/value MSE from the first optimizer pass's SB3 `train/value_loss`. Transition-zero value loss is undefined because no sampled returns exist yet. Initial log-std is recorded before the first rollout. A seeded paired test verifies the logging leaves every trained parameter tensor exactly unchanged.

The detached worker's Windows execution-state request prevents automatic idle sleep only while it runs. No saved power settings are modified. Explicit sleep, lid closure, shutdown, power loss, and thermal shutdown cannot be prevented or guaranteed against; users must keep the laptop plugged in and lid open. Command failures stop the attempt without retrying or patching. Partial artifacts are retained and explicit resumes use fresh siblings.
