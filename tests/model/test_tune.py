"""tennis.model.tune: search space, seeded determinism, logging, leakage alarm, parallel run, report smoke test."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import optuna
import pytest

from tennis.model import evaluate as ev
from tennis.model import tune


def _strip(records):
    return [(r["number"], r["params"], r["best_iteration"], r["val"]) for r in sorted(records, key=lambda r: r["number"])]


def test_suggest_respects_the_search_space(small_cfg) -> None:
    space = small_cfg["xgb"]["search_space"]
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(30):
        p = tune.suggest(study.ask(), space)
        assert set(p) == set(space)
        for k, (lo, hi) in space.items():
            assert lo <= p[k] <= hi
        assert isinstance(p["max_depth"], int)
    dist = study.trials[0].distributions
    assert dist["learning_rate"].log and not dist["subsample"].log


def test_prepare_data_dev_split_only(synth_features, small_cfg) -> None:
    d = tune.prepare_data(small_cfg, "dev", synth_features)
    assert d.val["date"].max() <= np.datetime64("2021-12-31") and d.train["date"].max() <= np.datetime64("2020-12-31")
    assert d.X_tr.shape == (len(d.train), len(d.features)) and d.X_va.shape == (len(d.val), len(d.features))
    assert d.features == list(small_cfg["model"]["features"])


def test_study_is_deterministic_with_the_seed_and_logs_every_trial(synth_features, small_cfg, tmp_path) -> None:
    d = tune.prepare_data(small_cfg, "dev", synth_features)
    _, r1 = tune.run_study(small_cfg, data=d, n_trials=4, workers=1, out_dir=str(tmp_path / "a"))
    _, r2 = tune.run_study(small_cfg, data=d, n_trials=4, workers=1, out_dir=str(tmp_path / "b"))
    assert _strip(r1) == _strip(r2)
    lines = (tmp_path / "a" / "xgb_trials.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    rec = json.loads(lines[0])
    assert rec["split"] == "dev" and "learning_rate" in rec["params"] and "trial" in rec["params"]
    assert {"val", "best_iteration", "seconds"} <= set(rec["metrics"])
    assert {"log_loss", "accuracy", "brier", "n"} <= set(rec["metrics"]["val"])
    other = copy.deepcopy(small_cfg)
    other["seed"] = small_cfg["seed"] + 1
    _, r3 = tune.run_study(other, data=d, n_trials=4, workers=1, log_name=None)
    assert [r["params"] for r in r3] != [r["params"] for r in r1]


def test_parallel_study_runs(synth_features, small_cfg) -> None:
    d = tune.prepare_data(small_cfg, "dev", synth_features)
    _, recs = tune.run_study(small_cfg, data=d, n_trials=3, workers=2, log_name=None)
    assert sorted(r["number"] for r in recs) == [0, 1, 2]
    # each trial reproduces in-process
    tune._init_worker(d.X_tr, d.y_tr, d.X_va, d.y_va, d.features, small_cfg)
    for r in recs:
        again = tune._run_trial(r["number"], r["params"])
        assert again["val"] == r["val"] and again["best_iteration"] == r["best_iteration"]


def test_leakage_alarm_stops_the_search(synth_features, small_cfg, tmp_path) -> None:
    cfg = copy.deepcopy(small_cfg)
    cfg["model"]["features"] = cfg["model"]["features"] + ["leak_col"]
    df = synth_features.assign(leak_col=synth_features["y"].astype(float))
    d = tune.prepare_data(cfg, "dev", df)
    with pytest.raises(ev.LeakageAlarm):
        tune.run_study(cfg, data=d, n_trials=4, workers=1, out_dir=str(tmp_path))
    lines = (tmp_path / "xgb_trials.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["metrics"]["val"]["accuracy"] > 0.75


def test_budget_stops_asking(synth_features, small_cfg) -> None:
    d = tune.prepare_data(small_cfg, "dev", synth_features)
    _, recs = tune.run_study(small_cfg, data=d, n_trials=4, workers=1, budget_min=0.0, log_name=None)
    assert recs == []


def test_best_record_tie_breaks() -> None:
    recs = [{"number": 0, "val": {"log_loss": 0.6, "accuracy": 0.6}},
            {"number": 1, "val": {"log_loss": 0.5, "accuracy": 0.6}},
            {"number": 2, "val": {"log_loss": 0.5, "accuracy": 0.7}},
            {"number": 3, "val": {"log_loss": 0.5, "accuracy": 0.7}}]
    assert tune.best_record(recs)["number"] == 2


def test_report_smoke(synth_features, small_cfg) -> None:
    d = tune.prepare_data(small_cfg, "dev", synth_features)
    _, recs = tune.run_study(small_cfg, data=d, n_trials=3, workers=1)
    out = tune.run_report(small_cfg, d, recs, runtime_s=1.0, workers=1, n_requested=3)
    root = Path(small_cfg["paths"]["reports"]).parent
    assert (root / "reports" / "xgb_tuning.md").exists()
    for f in ("phase5_calibration_xgb.png", "phase5_feature_importance.png", "phase5_optuna_history.png"):
        assert (root / "figures" / f).exists()
    assert out["model_path"].exists()
    assert len((root / "experiments" / "xgb_final.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert "best_params:" in out["yaml"] and "n_estimators:" in out["yaml"]
    assert set(out["heights"]["feature"]) == {"a_height", "b_height", "d_height"}
