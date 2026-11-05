"""Evaluation preserves failures and passes only observations to learned policies."""

import numpy as np
import pytest

from wiping.config import EnvConfig
from wiping.env import WipingEnv
from wiping.evaluate import aggregate, evaluate, validate_full_protocol


class HoverPolicy:
    def predict(self, obs, deterministic=True):
        assert isinstance(obs, np.ndarray) and obs.shape == (273,)
        return np.zeros(3, dtype=np.float32), None


def test_evaluation_retains_timeouts_and_only_exposes_observations():
    rows = evaluate(HoverPolicy(), "id", [1000, 1001], config=EnvConfig(max_steps=2))
    assert len(rows) == 2 and [r["seed"] for r in rows] == [1000, 1001]
    assert all(r["truncated"] and not r["terminated"] and not r["success"] for r in rows)
    assert all(r["coverage"] == 0 and r["elapsed_s"] == 0.1 for r in rows)


def metric_row(seed=2000, coverage=0.2, success=False, method="ppo", training_seed=11):
    return {
        "profile": "full", "method": method, "training_seed": training_seed,
        "seed": seed, "suite": "id", "coverage": coverage,
        "force_mean_n": 1.0, "force_max_n": 2.0, "unsafe_fraction": 0.0,
        "elapsed_s": 10.0, "success": success, "terminated": success,
        "truncated": not success, "failure_reason": None,
        "completion_time_s": 10.0 if success else None,
    }


def test_aggregation_includes_failures_and_does_not_infer_success():
    rows = [metric_row(2000, 0.9, True), metric_row(2001, 0.1)]
    group = aggregate(rows)["groups"][0]
    assert group["episodes"] == 2 and group["n_training_seeds"] == 1
    assert group["metrics"]["coverage"]["mean"] == 0.5
    assert group["metrics"]["success_rate"]["mean"] == 0.5
    # Even 95% coverage on a timeout must not be relabelled as success.
    row = metric_row(2000, 0.95)
    row.update(success="False", terminated="False", truncated="True")
    assert aggregate([row])["groups"][0]["metrics"]["success_rate"]["mean"] == 0


def test_missing_metrics_and_duplicate_cases_are_rejected():
    row = metric_row()
    del row["unsafe_fraction"]
    with pytest.raises(ValueError, match="Missing"):
        aggregate([row])
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate([metric_row(), metric_row()])


def test_aggregate_reports_seed_means_and_counts_baseline_once():
    rows = [metric_row(coverage=0.2, training_seed=11),
            metric_row(coverage=0.6, training_seed=22),
            metric_row(method="scripted", training_seed=None)]
    groups = {g["method"]: g for g in aggregate(rows)["groups"]}
    assert groups["ppo"]["n_training_seeds"] == 2
    assert groups["ppo"]["metrics"]["coverage"]["mean"] == pytest.approx(0.4)
    assert groups["ppo"]["metrics"]["coverage"]["std_across_seeds"] == pytest.approx(0.282842712)
    assert groups["scripted"]["n_training_seeds"] == 0
    assert groups["scripted"]["episodes"] == 1
    with pytest.raises(ValueError, match="scripted"):
        aggregate([metric_row(method="scripted", training_seed=11)])


def test_full_protocol_requires_all_paired_cases_and_three_training_seeds():
    rows = [dict(metric_row(seed, method=method, training_seed=train_seed), suite=suite)
            for suite in ("id", "curvature", "friction")
            for method in ("scripted", "bc", "ppo", "bc_ppo")
            for train_seed in ([None] if method == "scripted" else [11, 22, 33])
            for seed in range(2000, 2100)]
    validate_full_protocol(rows)
    with pytest.raises(ValueError, match="full"):
        validate_full_protocol(rows[:-1])


def test_id_and_ood_change_only_the_named_parameter():
    env = WipingEnv(EnvConfig())
    for seed in range(1000, 1020):
        _, base = env.reset(seed=seed)
        _, curve = env.reset(seed=seed, options={"suite": "curvature"})
        _, friction = env.reset(seed=seed, options={"suite": "friction"})
        assert 0.22 <= base["radius_m"] <= 0.30 and 0.3 <= base["friction"] <= 0.7
        assert (0.18 <= curve["radius_m"] <= 0.21 or 0.31 <= curve["radius_m"] <= 0.34)
        assert curve["friction"] == base["friction"]
        assert (0.15 <= friction["friction"] <= 0.25 or 0.8 <= friction["friction"] <= 1.0)
        assert friction["radius_m"] == base["radius_m"]
    env.close()


def test_rendered_replay_preserves_metrics_and_writes_video(tmp_path):
    from wiping.reporting import render_episode

    config = EnvConfig(max_steps=2)
    expected = evaluate(HoverPolicy(), "id", [1000], config=config)[0]
    result = render_episode(HoverPolicy(), "id", 1000, tmp_path / "hover.mp4",
                            method="fixture", profile="test_fixture", config=config)
    assert result["metrics"]["coverage"] == expected["coverage"]
    assert result["metrics"]["elapsed_s"] == expected["elapsed_s"]
    assert result["frames"] == 3
    assert (tmp_path / "hover.mp4").stat().st_size > 1000
    assert (tmp_path / "hover.trace.csv").is_file()


def test_report_rejects_empty_input_and_writes_real_metrics(tmp_path):
    from wiping.artifacts import write_csv
    from wiping.reporting import write_report

    with pytest.raises(ValueError):
        write_report([], tmp_path / "empty")
    path = tmp_path / "episodes.csv"
    write_csv(path, [metric_row(2000, 0.9, True), metric_row(2001, 0.1)])
    output = write_report([path], tmp_path / "report")
    assert (output / "summary.json").is_file()
    assert (output / "coverage-force.png").is_file()
    assert "0.5000" in (output / "results.md").read_text()


def test_report_rejects_mixed_evaluation_splits():
    validation = dict(metric_row(seed=1000), evaluation_split="validation")
    final = dict(metric_row(seed=2000), evaluation_split="test")
    with pytest.raises(ValueError, match="split"):
        aggregate([validation, final])


def test_weak_contact_is_not_contact_loss(monkeypatch):
    monkeypatch.setattr("wiping.env.read_normal_force", lambda *_: 0.3)
    row = evaluate(HoverPolicy(), "id", [1000], config=EnvConfig(max_steps=3))[0]
    assert not row["contact_loss"]


def test_partial_unloading_is_not_contact_loss(monkeypatch):
    forces = iter([1.0] * 25 + [0.3] * 25 + [1.0] * 25)
    monkeypatch.setattr("wiping.env.read_normal_force", lambda *_: next(forces))
    row = evaluate(HoverPolicy(), "id", [1000], config=EnvConfig(max_steps=3))[0]
    assert not row["contact_loss"] and not row["contact_recovery"]


def test_actual_unloading_and_recovery_are_recorded(monkeypatch):
    forces = iter([1.0] * 25 + [0.0] * 25 + [1.0] * 25)
    monkeypatch.setattr("wiping.env.read_normal_force", lambda *_: next(forces))
    row = evaluate(HoverPolicy(), "id", [1000], config=EnvConfig(max_steps=3))[0]
    assert row["contact_loss"] and row["contact_recovery"]
