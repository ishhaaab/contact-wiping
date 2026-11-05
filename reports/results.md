# Results status

The repository is a simulation experiment. Smoke checks establish execution and artifact integrity; they do not establish that BC+PPO improves wiping.

## Verified before training

- Eight physics tests passed, including known 3 N radial loading and unrelated-contact filtering.
- A 10,000-substep contact trace and 640 by 480 rendered frame were saved under `artifacts/smoke/physics`.
- The environment and dataset suites passed. Full suite before training: 37 tests, including numerical BC reduction and one PPO update.
- The corrected eight-episode smoke dataset passed the demonstration gate: six episodes reached at least 50% coverage. Mean coverage was 0.5322265625. All eight timed out; none met 90% success. No episodes were discarded.
- Twenty additional scripted diagnostic episodes had mean coverage 0.497265625, a 75% qualifying fraction, and no task successes or physical failures. See `smoke/scripted-20.json`.

The first generated dataset had correlated surface and controller random streams. It was replaced after a failing regression test identified the cause. Historical artifacts remain under `artifacts/diagnostics/pre-rng-fix`; they are not training data or evidence for the corrected pipeline.

## Initial smoke training

One training seed, 11. Each learned method evaluated 20 environment-validation episodes, seeds 1000..1019. These were not final-test cases.

| Method | RL transitions | Mean validation coverage | Successes |
| --- | ---: | ---: | ---: |
| BC | 0 | 0.05078125 | 0 / 20 |
| PPO | 8,192 | 0.0 | 0 / 20 |
| BC+PPO | 8,192 | 0.096875 | 0 / 20 |

BC and BC+PPO initialization policy tensors were independently checked for exact equality. Checkpoints, resolved configurations, source snapshots, package versions, training curves, validation episodes, and timing are under `runs/smoke/{method}/11/`.

PPO measured 506.0400 transitions/s and BC+PPO 444.0391 transitions/s, excluding validation. Six full RL runs, three seeds for each RL method, would take approximately 8.116 hours including 41 validation rounds per run at these rates. This estimate excludes BC fitting, final evaluation, data generation, and rendering. Each full run rounds up to 2,002,944 transitions. No long training run has been launched.

## Clean-install smoke verification, 2026-10-19

The full pipeline completed in a fresh environment installed from the lockfile. All 42 pinned dependencies match. `pytest -q` passed 46 tests, `ruff check .` passed, and `pip check` found no broken requirements.

Command:

```text
python -m wiping.cli smoke --config configs/smoke.json --output artifacts/smoke/verified-2026-10-19
```

Pipeline time was 151.3429 seconds, excluding installation and the preceding tests. It produced three learned checkpoints, 36 paired smoke evaluation episodes, reports, training curves, contact/coverage plots, and three 401-frame MP4s. All 36 evaluations were unsuccessful timeouts. The original contact-loss video label is superseded: its force dropped to 0.325 N but never fully unloaded. Do not present it as physical contact loss or recovery.

- [Smoke evaluation table](smoke/verified/results.md)
- [Raw smoke episodes](smoke/verified/episodes.csv)
- [Summary with sample counts](smoke/verified/summary.json)
- [Compute estimate](smoke/verified/compute-estimate.json)
- [Locked dependency verification](smoke/lock-verification.json)

The smoke table uses validation seeds 1000..1002, one trained model per learned method, and three cases per suite. Final-test seeds remain unused. The complete raw artifact directory is `artifacts/smoke/verified-2026-10-19/`, including `data/`, `runs/`, `physics/`, `videos/`, and the pipeline manifest.

Fresh throughput was 447.3378 transitions/s for PPO and 432.4421 for BC+PPO. The full experiment estimate is **9.77 hours before rendering**, comprising 8.66 hours RL and validation, 0.25 hours BC, 0.82 hours final evaluation, and about 161 seconds of data collection. Actual runtime depends on episode lengths, early stopping, laptop load, and thermal limits. No long training run has been launched.

## Remaining experimental work

The full dataset completed after successful smoke verification: 200 episodes, 80,000 transitions, exact 160/40 split. Of these, 81% passed the demonstration coverage gate. All episodes timed out, none succeeded, none were discarded. See `full-dataset.json` for checksum and timing.

Full three-seed training and the prescribed 3,000-case ID/OOD comparison remain unrun. No full learning-performance or physical-robot validation claim is made.

## Final corrections, 2026-11-06

Fresh-context review identified simulator-fault priority and mixed evaluation-split reporting defects. Both were reproduced with failing regression tests and corrected. Contact-loss diagnostics now require actual unloading below 1e-4 N for a full control interval, rather than merely falling below useful wiping force. The old video/recovery claim is withdrawn.

Final verification: 52 tests passed in 18.22 seconds; Ruff passed. The first rerun was stopped by the tool's 120-second timeout. Its partial artifacts remain under `artifacts/smoke/final-2026-11-06/`. A new run with a longer tool timeout completed in 143.6264 seconds at `artifacts/smoke/final-2026-11-06-complete/`. All 36 evaluation cases are retained. Three labelled videos include actual contact loss; no success or physical-failure example occurred, so the other two are honestly labelled timeouts. The latest estimate is 9.60 hours before rendering for the full experiment. See `smoke/final/results.md` and `smoke/final/compute-estimate.json`.

## Full release in progress

The user authorized full training and final evaluation after preflight reconfirmed 52 passing original tests, Ruff, and the complete full dataset checksum/gate. The logging and resume machinery passed a 56-test pre-launch suite. Training uses the unchanged experiment settings. Passive PPO logging was checked for exact tensor-level invariance in a paired seeded test.

The detached full-release worker was launched with control directory `artifacts/full-release-20261106-005111/`. It runs all nine learned-method/seed pairs in README order, then the scripted baseline and full-protocol report. Final evaluation cases are not used to select checkpoints. See `full-release-progress.md` for logs, resume instructions, and pending reporting steps. No full-performance metrics are available in this status update, and no full result is inferred from smoke data.
