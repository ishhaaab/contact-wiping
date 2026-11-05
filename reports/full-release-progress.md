# Full experimental release progress

## Steps

- [x] Preflight original suite and Ruff; verify full demonstration checksum, seeds, split, and gate.
- [x] Implement passive PPO-start diagnostics and artifact-preserving launcher tests.
- [x] Run final pre-launch test/lint checks and commit logging/launcher provenance.
- [ ] Full training and final evaluation for seeds 11, 22, 33 and methods BC, PPO, BC+PPO.
- [ ] Scripted baseline, once, 100 cases in each suite.
- [ ] Full-protocol report, per-seed diagnostics, learning curves, and honest interpretation.
- [ ] Update README/results, verify outputs, commit, and hand off.

## Preflight evidence

The original 52 tests passed in 16.21 seconds and Ruff passed. The full dataset is complete with 200 episodes, 80,000 transitions, seeds 3000..3199, exact first-160/last-40 split, and an 81% qualifying fraction against the 20% gate. SHA256: `0ddd2dc0f9346b254c585c5cc1b493a692974a3227a5e828b65798226cebb53a`. No recollection is required. No prior full output directories were present.

## Execution contract

Use the existing CLI, full config, method order, seed order, validation checkpoint selection, and final evaluation exactly as documented. No changes to the environment, algorithm settings, networks, seeds, splits, or budgets are authorized. Added logging must not consume RNG or alter tensors. A real PPO test compares all resulting tensors with and without the diagnostic callback.

The launcher prevents Windows automatic idle sleep while it owns the job and restores the thread execution-state request on exit. It does not modify saved power settings or prevent deliberate sleep, lid actions, shutdown, battery exhaustion, or hardware failure. A detached worker survives terminal closure. On actual sleep the process cannot compute; resumption continues it after wake. Keep the laptop plugged in and lid open.

Completed training/evaluation outputs are validated and skipped on explicit resume. Partial outputs remain untouched; reruns use fresh `-retry-ATTEMPT` siblings. A failed subprocess stops the whole attempt immediately and keeps its log/state; there is no automatic retry or patch. Resuming requires a new control directory. Selected retry evaluations are passed explicitly to the report to avoid duplicate or incomplete cases. Without retries the exact README directory-based report command is used.

## Diagnostic definitions

`ppo_start.json` records initial three-axis policy log standard deviations before the first rollout. There is no return target or value loss at transition zero. The file separately reports first-rollout mean squared return-minus-value before optimization and SB3's mean `train/value_loss` from the first PPO optimizer pass. Neither measurement changes training. Existing BC epoch/validation history and training episode CSVs supply all other requested curves and early-stopping diagnostics.

## Status

Pre-launch verification passed: 56 tests in 32.86 seconds, Ruff, and `pip check`. The logging-invariance test confirms exactly equal trained policy tensors with and without passive diagnostics. Four test-first cases verify incomplete-run handling, non-overwrite paths, fail-stop command logging, and numerical logging invariance. Windows accepted a test automatic-sleep inhibition request, which was then released without changing saved power settings.

The full comparison is launched as detached worker PID 26964, using committed logging/launcher revision `4549118`. Windows confirmed the worker's automatic idle-sleep inhibition request. Control directory: `artifacts/full-release-20261106-005111/`. The outer background waiter has no timeout; its completion or failure wakes this session when available.

Progress and per-command elapsed times are recorded in `progress.log` and `state.json` under the control directory. Each train/evaluate/report command has its own log. The detached worker's combined output is `artifacts/full-release-20261106-005111.worker.log`. The initial startup record confirmed the exact README BC/11 command and full-config path. No final-test metrics have been inspected while preparing the reporting code.

The release-analysis generator is tested using explicitly synthetic temporary fixtures only. It writes per-suite/per-seed tables, aggregate sample SD, successful completion counts, complete BC histories/early-stop diagnostics, first-PPO value/log-std diagnostics, and six success/coverage/unsafe learning curves. It refuses to generate a full report until the 3,000-case comparison completes and refuses to overwrite an analysis directory.

After the worker completes successfully, run:

```text
python -m wiping.release_report --control artifacts/full-release-20261106-005111 --output reports/full/analysis
```

Then inspect the actual results, update `reports/results.md` and README status, run tests/lint, and commit the final handoff. If the worker fails, stop and report its preserved logs without patching or automatically retrying. On an explicitly authorized resume, use a fresh control directory with `python -m wiping.release --control artifacts/full-release-NEW-ATTEMPT`; completed pairs are checked and skipped, incomplete attempts are retained and use fresh sibling outputs.

Full results are pending. Do not infer performance from smoke data.
