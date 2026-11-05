# Contact-aware surface wiping

MuJoCo simulation of a three-axis Cartesian wiping head. Compare an imperfect scripted raster, behaviour cloning, PPO, and BC followed by PPO. Coverage measures geometric contact on a spherical cap. It does not measure dirt removal.

This is an end-effector surrogate with state observations and a known local surface model. It is not a full robot arm, and no physical robot has been tested.

## Setup

Use Python 3.11. Native Windows 11 is the tested host. Linux x86-64 and WSL2 Ubuntu are the specification's reference platforms but have not yet been verified here. The exact tested package versions are in `requirements-lock.txt`, including CPU PyTorch.

PowerShell, from this repository:

```powershell
python3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.venv\Scripts\Activate.ps1
```

On the implementation host, the registered interpreter command is `py -V:Astral/CPython3.11.17` instead of `python3.11`. Use that only for creating the venv. Activation is optional; every command can use `.venv\Scripts\python.exe` directly.

Linux setup uses `python3.11 -m venv .venv` and `source .venv/bin/activate` followed by the same `python -m pip` commands. Headless Linux rendering requires a working OpenGL backend, for example EGL with `MUJOCO_GL=egl`. Windows uses default GLFW and needs an available graphics context.

## Tests and contact diagnostic

```text
python -m pytest -q
python -m ruff check .
python -m wiping.feasibility --output artifacts/smoke/physics
```

The physics diagnostic saves `force_trace.csv`, `frame.png`, and `summary.json`. Its abrupt loading step intentionally differs from an episode's bounded incremental actions. See `reports/physics-gate.md` for measured results.

## Experiment contract

- Observation: 273 float32 values, including tool state, target, contact force, previous action, local surface model, and a 16 by 16 coverage map. No separate radius or friction input.
- Actions: three incremental position-target commands, clipped to `[-1, 1]` and scaled by 4, 4, and 2 mm.
- Physics: 2 ms steps, 25 substeps per action, 400-action episode limit. Contact must be between 0.5 and 8 N to add coverage.
- Success: at least 90% coverage with at most 5% unsafe substeps. Three consecutive substeps over 15 N terminate with failure. Time limits are truncations.
- Scripted demonstrations include normal-offset and action noise. Their random stream is separate from the environment's. Failed episodes are retained.
- BC fits Gaussian action means with MSE, using actor parameters only. BC validation is episode-separated. PPO uses a fresh optimizer after BC weight transfer.
- All learned policies use separate `[256, 256]` Tanh actor and critic networks and no observation normalization. PPO uses four environments.

The full numerical contract is in `../../docs/superpowers/plans/2026-10-18-surface-wiping.md`. Implementation rulings and task evidence are in `DECISIONS.md`.

## Data and splits

| Profile | Demonstrations | BC train/validation episodes | RL transitions per method/seed |
| --- | ---: | ---: | ---: |
| smoke | 8 | 6 / 2 | 8,192 |
| pilot | 40 | 32 / 8 | 200,000 |
| full | 200 | 160 / 40 | 2,000,000 |

SB3 rounds RL budgets to full rollouts of 4,096 transitions. `run.json` records the actual count. BC has up to 100 epochs, with patience 10 on validation MSE.

Training seeds are 11, 22, and 33. Environment-validation seeds 1000..1019 select PPO checkpoints by success rate, then coverage, then lower unsafe fraction. Final-test seeds 2000..2099 are reserved for the final comparison. Demonstration seeds start at 3000; full data uses exactly 3000..3199. The scripted baseline has no learned training seed and is counted once per evaluation suite.

HDF5 stores one group per episode. Observations have `T+1` rows, actions and measurements have `T` rows, and the final observation is retained. The adjacent `.manifest.json` contains SHA256, configuration, seeds, and all final episode results. The training gate requires at least 20% of all demonstrations to reach 50% coverage without excessive-force termination.

`tests/fixtures/smoke-demos.h5` is a two-episode real simulator fixture. It is not a substitute for an experiment dataset. Dataset and run directories are ignored by Git.

## Smoke pipeline

Latest corrected smoke output: `artifacts/smoke/final-2026-11-06-complete/`. The earlier October contact-loss label is withdrawn; the video showed weak contact, not full unloading. See `reports/results.md`. The complete 200-episode dataset is available at `data/demos.h5` with its SHA256 manifest. Full training remains unrun.

```text
python -m wiping.cli smoke --config configs/smoke.json
```

This creates a timestamped directory under `artifacts/smoke/`. It runs the contact diagnostic, collects eight demonstrations, trains all three learned methods with seed 11, selects checkpoints using all twenty ID validation cases, and checks all four methods on three paired validation seeds in every suite. It writes a report, compute estimate, and three labelled replay videos. It never uses final-test seeds.

Pass `--output PATH` to choose a fresh output directory. Collectors, training runs, evaluation runs, and videos refuse to overwrite existing artifacts. For a new dataset location, pass `train --dataset PATH` as well.

The smoke evaluation uses a small validation subset to check the evaluation code. Its table is labelled `smoke`; it is not the required 100-case final comparison. If no successful episode occurs, the video manifest says so and the replacement video is labelled with its actual outcome. The software does not invent a success illustration.

## Individual commands

```text
python -m wiping.cli collect --config configs/full.json --output data/demos.h5
python -m wiping.cli train --method bc_ppo --seed 11 --config configs/full.json --output runs/bc_ppo/11
python -m wiping.cli evaluate --checkpoint runs/bc_ppo/11/best.zip --suite id --output evaluations/bc_ppo/11/id
python -m wiping.cli evaluate --method scripted --config configs/full.json --suite all --output evaluations/scripted
python -m wiping.cli report --input evaluations --output reports/comparison
```

Training methods are `bc`, `ppo`, and `bc_ppo`. Evaluation infers the method, profile, environment settings, and training seed from the checkpoint's adjacent `run.json` and `config.json`. An explicit `--method` must match that record. The scripted baseline requires no checkpoint.

Evaluation defaults to all 100 final-test seeds. Use `--split validation` for smoke/pilot inspection, and `--suite all` to run ID, curvature OOD, and friction OOD together. Curvature OOD changes only radius; friction OOD changes only friction. Logging includes both parameters, but learned policies only receive the observation vector.

## Full experimental release

Read `reports/results.md` and the smoke run's `report/compute-estimate.json` before starting. The measured initial estimate is hours, not minutes. Full training has not been run by the implementation agent.

After collecting `data/demos.h5`, these PowerShell commands run the prescribed comparison:

```powershell
foreach ($seed in @(11, 22, 33)) {
    foreach ($method in @('bc', 'ppo', 'bc_ppo')) {
        python -m wiping.cli train --method $method --seed $seed --config configs/full.json --output "runs/full/$method/$seed"
        if ($LASTEXITCODE -ne 0) { throw "Training failed" }
        python -m wiping.cli evaluate --checkpoint "runs/full/$method/$seed/best.zip" --suite all --output "evaluations/full/$method/$seed"
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed" }
    }
}
python -m wiping.cli evaluate --method scripted --config configs/full.json --suite all --output evaluations/full/scripted
python -m wiping.cli report --input evaluations/full --runs runs/full --output reports/full --require-full
```

`--require-full` rejects missing methods, missing training seeds, missing suites, duplicate cases, and anything short of the complete 3,000-episode final comparison. The baseline contributes 100 cases per suite, once. Each learned method contributes 100 cases for each of three trained models per suite.

Do not use the final tables to redesign rewards or select checkpoints. Checkpoint selection only reads the prescribed validation cases. Retain results even if BC+PPO does not outperform the alternatives.

## Artifact map

For a smoke output directory `ROOT`:

| Path | Contents |
| --- | --- |
| `ROOT/physics/` | 10,000-substep CSV, rendered frame, physics summary |
| `ROOT/data/demos.h5` | Complete smoke demonstration episodes |
| `ROOT/data/demos.manifest.json` | SHA256, split membership, all episode outcomes |
| `ROOT/runs/{method}/11/best.zip` | Selected policy checkpoint |
| `ROOT/runs/{method}/11/last.zip` | Final PPO checkpoint, for RL methods |
| `ROOT/runs/{method}/11/bc_init/` | BC checkpoint, MSE history, fitting metrics |
| `ROOT/runs/{method}/11/run.json` | Timing, actual transitions, seed, Git revision, packages, hardware |
| `ROOT/runs/{method}/11/config.json` | Resolved experiment and algorithm settings |
| `ROOT/runs/{method}/11/source/` | Exact Python source snapshot, including dirty-tree runs |
| `ROOT/runs/{method}/11/validation_episodes.csv` | Every checkpoint-selection episode |
| `ROOT/runs/{method}/11/training_episodes.csv` | RL episode outcomes, including failures |
| `ROOT/evaluation/{method}/episodes.csv` | Paired smoke evaluation cases |
| `ROOT/report/` | Combined episodes, JSON/Markdown summaries, training and coverage/force plots, compute estimate |
| `ROOT/videos/` | Three MP4s, labels, per-step traces and plots, observed-outcome manifest |
| `ROOT/pipeline.json` | Pipeline status, artifact paths, source/hardware provenance |

`summary.json` reports each training seed separately, then the mean and sample standard deviation across models. With one trained model, standard deviation is null. Completion time is conditional on actual success; timeouts retain elapsed time and count against success. Episode mean force includes hovering substeps. Videos are illustrative and do not determine summary metrics.

## Limitations and failure handling

The surface is a rigid convex sphere, the tool is a rigid sphere, friction is a contact-model parameter, and coverage uses an 8 mm geometric footprint. There is no deformable pad, dirt model, articulated-arm dynamics, camera noise, or real force sensor. The policy receives a perfect local surface model.

The raster is deliberately imperfect and has no guarantee of finishing within 400 actions. Its failures stay in BC data and evaluation. Training stops on nonfinite states or MuJoCo warnings and preserves diagnostic files before a vector environment can auto-reset. Excessive-force termination is a recorded task failure, not a discarded simulator error.

Source planning documents and the JD remain outside this repository and have not been edited. Measured implementation results and remaining experimental work are recorded in `reports/results.md`.
