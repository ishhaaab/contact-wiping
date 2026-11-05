"""Release tables keep counts, uncertainty, and undefined completion times explicit."""

import csv
import json

from wiping.release_report import bc_diagnostics, suite_tables


def case(seed, training_seed, success, coverage):
    return {
        "profile": "full", "evaluation_split": "test", "method": "ppo",
        "training_seed": training_seed, "suite": "id", "seed": seed,
        "coverage": coverage, "force_mean_n": 1, "force_max_n": 2,
        "unsafe_fraction": 0.01, "elapsed_s": 10 if success else 20,
        "success": success, "terminated": success, "truncated": not success,
        "completion_time_s": 10 if success else None, "failure_reason": None,
    }


def test_suite_tables_show_seed_counts_sd_and_no_success_time():
    rows = [case(2000, 11, True, 0.95), case(2001, 11, False, 0.15),
            case(2000, 22, False, 0.2), case(2001, 22, False, 0.4),
            case(2000, 33, False, 0.3), case(2001, 33, False, 0.5)]
    text, summary = suite_tables(rows)
    assert "ID" in text and "mean +/- sample SD" in text
    assert "1/2" in text and "0/2" in text
    assert "NA, 0 successes" in text
    assert "10.0000" in text
    assert summary["groups"][0]["metrics"]["coverage"]["std_across_seeds"] is not None
    assert "0.1667" in text


def test_bc_diagnostics_derive_best_epoch_and_early_stop(tmp_path):
    path = tmp_path / "bc_init"
    path.mkdir()
    history = [{"epoch": i, "train_mse": 0.1, "validation_mse": 0.2 if i == 2 else 0.3}
               for i in range(1, 13)]
    with (path / "bc_history.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    (path / "bc_metrics.json").write_text(json.dumps({"epochs": 12, "best_validation_mse": 0.2}))
    result = bc_diagnostics(tmp_path)
    assert result["best_epoch"] == 2 and result["epochs_trained"] == 12
    assert result["early_stopping_epoch"] == 12
    assert result["epochs_since_improvement"] == 10
    assert len(result["validation_mse_history"]) == 12


def test_complete_release_report_artifacts_and_no_overwrite(tmp_path):
    import pytest

    from wiping.artifacts import write_csv, write_json
    from wiping.release_report import build_release_analysis

    # Synthetic reporting fixture only. These are never experimental measurements.
    rows = [dict(case(seed, training_seed, False, 0.3), method=method, suite=suite)
            for method in ("scripted", "bc", "ppo", "bc_ppo")
            for training_seed in ([None] if method == "scripted" else [11, 22, 33])
            for suite in ("id", "curvature", "friction") for seed in range(2000, 2100)]
    report = tmp_path / "base-report"
    write_csv(report / "episodes.csv", rows)
    training = []
    for method in ("bc", "ppo", "bc_ppo"):
        for seed in (11, 22, 33):
            path = tmp_path / "runs" / method / str(seed)
            write_json(path / "run.json", {"actual_transitions": 0 if method == "bc" else 2_002_944,
                                           "wall_seconds": 1, "selected_checkpoint_transitions": 0})
            if method in ("bc", "bc_ppo"):
                write_json(path / "bc_init/bc_metrics.json",
                           {"epochs": 100, "best_validation_mse": 0.01})
                history = [{"epoch": i, "train_mse": 1 / i, "validation_mse": 1 / i}
                           for i in range(1, 101)]
                write_csv(path / "bc_init/bc_history.csv", history)
            if method in ("ppo", "bc_ppo"):
                write_json(path / "ppo_start.json", {"initial_policy_log_std": [0, 0, 0],
                                                     "pre_update_value_mse": 1.0,
                                                     "first_optimizer_value_loss": 0.5})
                write_csv(path / "training_episodes.csv", [
                    {"transitions": 4096, "success": False, "coverage": 0.3,
                     "unsafe_fraction": 0.01},
                ])
            training.append({"method": method, "seed": seed, "path": str(path)})
    control = tmp_path / "control"
    write_json(control / "state.json", {"status": "complete", "report": str(report),
                                         "training": training, "elapsed_seconds": 10})
    destination = tmp_path / "analysis"
    build_release_analysis(control, destination)
    assert len(list(destination.glob("training-*.png"))) == 6
    assert len(json.loads((destination / "diagnostics.json").read_text())) == 12
    text = (destination / "results.md").read_text()
    assert "0/100" in text and "NA, 0 successes" in text
    assert "PPO-start diagnostics" in text and "Friction OOD" in text
    with pytest.raises(FileExistsError):
        build_release_analysis(control, destination)
