"""Test 20 (Phase 7 final audit): each final test set was scored ONCE, by the frozen model, trained on pre-test rows.

PLAN section 2 rules 5, 6 and 10; DECISIONS D-028 (retrain rule), D-032 (freeze). Complements test 16, which
only requires ONE config hash per final split (a second scoring under the same hash would pass it).

Checks, per final split (replication = Wimbledon 2025, forward = 2026-04-01 .. 2026-06-07):
1. ``experiments/<split>.jsonl`` holds exactly ONE record with a ``test`` key; both final records share one
   config hash; the record's model parameters and tree count are exactly the frozen config values
   (``xgb.best_params`` + ``xgb.replication_n_estimators``; ``xgb.forward_best_params``); the refit had no early
   stopping and its window ends on or before ``val_end``; its validation window lies inside the split's validation
   window and its test window inside the test period.
2. The forward re-tune: every trial in ``experiments/xgb_trials_forward.jsonl`` is a forward-split validation run
   (window inside 2026 Q1, no test metric), finished BEFORE the forward test was scored, and the frozen
   ``forward_best_params`` are exactly the best logged trial (with ``n_estimators`` = its best_iteration + 1).
3. ``evaluate_final.run_final(score=True)`` refuses to score either test again with the real experiment logs,
   before any partition is loaded, and leaves the logs untouched.
4. (realdata) The saved test-prediction CSV holds exactly the split's test partition (same match_ids, same order,
   same labels; 122 Wimbledon rows, 473 forward rows of which 118 Roland Garros), and the logged test metrics
   (overall, per reported tournament, Elo / blend baselines) are recomputed from it.
5. (realdata) The saved booster has the frozen tree count, the allow-listed features and a refit window ending
   by ``val_end``, and reproduces the CSV probabilities on the test partition.
6. (realdata, slow) An independent ``refit_fixed`` on the split's train + validation rows ONLY reproduces the
   saved test probabilities: the final model saw no test row (and no row after ``val_end``).

Catches: re-scoring a test after looking at it, a model trained on test rows or with the test as an
early-stopping set, a CSV / log that does not describe the test partition, tuning that ran after the test was seen.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tennis.config import repo_path

FINAL_SPLITS = ("replication", "forward")
EXPECTED_N_TEST = {"replication": 122, "forward": 473}          # D-017, test 15
EXPECTED_TOURNEY_N = {"forward": {"Roland Garros": 118}}
TOL_METRIC = 1e-6   # the CSV stores p_xgb as float32
TOL_PRED = 1e-6


# ------------------------------------------------------------------------------------------------ helpers
def _exp_dir(cfg: dict[str, Any]) -> Path:
    return Path(repo_path(cfg["paths"]["experiments"]))


def _records(cfg: dict[str, Any], name: str) -> list[dict[str, Any]]:
    path = _exp_dir(cfg) / f"{name}.jsonl"
    if not path.exists():
        pytest.skip(f"{path.name} not logged yet (test not scored)")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _test_records(cfg: dict[str, Any], split: str) -> list[dict[str, Any]]:
    return [r for r in _records(cfg, split) if "test" in (r.get("metrics") or {})]


def _the_record(cfg: dict[str, Any], split: str) -> dict[str, Any]:
    recs = _test_records(cfg, split)
    assert len(recs) == 1, f"{split}: {len(recs)} test records (the test is evaluated exactly once)"
    return recs[0]


def frozen_params(cfg: dict[str, Any], split: str) -> tuple[dict[str, Any], int]:
    """(model params without n_estimators, frozen tree count) for a final split, from config."""
    x = cfg["xgb"]
    if split == "replication":
        p, n = x["best_params"], x.get("replication_n_estimators")
    else:
        p = x.get("forward_best_params") or {}
        n = p.get("n_estimators")
    assert n is not None, f"{split}: no frozen tree count in config"
    return {k: v for k, v in p.items() if k != "n_estimators"}, int(n)


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s[:10])


def _test_period(cfg: dict[str, Any], split: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    s = cfg["splits"][split]
    if split == "replication":
        ve = pd.Timestamp(s["val_end"])
        return ve + pd.Timedelta(days=1), ve + pd.Timedelta(days=14)
    return pd.Timestamp(s["test_start"]), pd.Timestamp(s["test_end"])


def _model_paths(cfg: dict[str, Any], split: str) -> tuple[Path, Path]:
    rec = _the_record(cfg, split)
    model = Path(repo_path(cfg["paths"]["models"])) / rec["params"]["model_path"]
    if not model.exists():
        pytest.skip(f"{model} not present (models/ is gitignored)")
    csv = model.with_name(model.stem + "_test_predictions.csv")
    assert csv.exists(), f"{csv} missing next to the saved final model"
    return model, csv


def _partitions(cfg: dict[str, Any], features_table: pd.DataFrame, split: str) -> dict[str, pd.DataFrame]:
    from tennis.model.splits import time_split
    return time_split(features_table, split, cfg)


# ------------------------------------------------------------------------------------------------ 1. records
@pytest.mark.parametrize("split", FINAL_SPLITS)
def test_exactly_one_test_record_with_the_frozen_settings(cfg: dict, split: str) -> None:
    rec = _the_record(cfg, split)
    assert rec.get("split") == split
    params, n_trees = frozen_params(cfg, split)
    p = rec["params"]
    assert p["model_params"] == params, f"{split}: logged model params differ from the frozen config values"
    assert int(p["n_trees"]) == n_trees, f"{split}: logged {p['n_trees']} trees, config freezes {n_trees}"
    assert p["early_stopping_in_refit"] is False
    refit = rec["metrics"]["test"]["refit"]
    assert refit["early_stopping"] is False and int(refit["fixed_trees"]) == n_trees
    s = cfg["splits"][split]
    val_end = pd.Timestamp(s["val_end"])
    assert _ts(p["refit_window"][1]) <= val_end, f"{split}: refit window {p['refit_window']} passes val_end"
    assert _ts(p["val_window"][0]) >= pd.Timestamp(s["val_start"]) and _ts(p["val_window"][1]) <= val_end
    lo, hi = _test_period(cfg, split)
    assert lo <= _ts(p["test_window"][0]) and _ts(p["test_window"][1]) <= hi, f"{split}: test window {p['test_window']}"
    assert int(p["n_test"]) == EXPECTED_N_TEST[split]
    assert rec["metrics"]["test"]["overall"]["model"]["n"] == EXPECTED_N_TEST[split]
    assert int(rec["metrics"]["val"]["n_trees"]) == n_trees, "validation run did not reproduce the frozen tree count"


def test_both_final_tests_share_one_frozen_config_hash(cfg: dict) -> None:
    hashes = {split: _the_record(cfg, split)["config_hash"] for split in FINAL_SPLITS}
    assert len(set(hashes.values())) == 1, f"final tests scored under different configs: {hashes} (D-032)"


# ------------------------------------------------------------------------------------------------ 2. re-tune
def test_forward_retune_is_validation_only_and_precedes_scoring(cfg: dict) -> None:
    trials = _records(cfg, "xgb_trials_forward")
    assert 0 < len(trials) <= int(cfg["xgb"]["n_trials_forward"])
    s = cfg["splits"]["forward"]
    for t in trials:
        assert t["split"] == "forward"
        assert set(t["metrics"]) <= {"val", "best_iteration", "seconds", "val_window"}, t["metrics"].keys()
        w = t["metrics"]["val_window"]
        assert _ts(w[0]) >= pd.Timestamp(s["val_start"]) and _ts(w[1]) <= pd.Timestamp(s["val_end"]), w
    rec = _the_record(cfg, "forward")
    scored = pd.Timestamp(rec["timestamp"])
    last_trial = max(pd.Timestamp(t["timestamp"]) for t in trials)
    assert last_trial < scored, "a forward tuning trial was logged after the forward test was scored"
    best = min(trials, key=lambda t: (t["metrics"]["val"]["log_loss"], -t["metrics"]["val"]["accuracy"],
                                      int(t["params"]["trial"])))
    params, n_trees = frozen_params(cfg, "forward")
    assert {k: best["params"][k] for k in params} == params, "forward_best_params is not the best logged trial"
    assert int(best["metrics"]["best_iteration"]) + 1 == n_trees
    # the scoring run's validation step reproduced the best trial exactly (same rows, same fit)
    assert rec["metrics"]["val"]["model"]["log_loss"] == pytest.approx(best["metrics"]["val"]["log_loss"], abs=1e-12)


# ------------------------------------------------------------------------------------------------ 3. refusal
@pytest.mark.parametrize("split", FINAL_SPLITS)
def test_rescoring_is_refused_before_loading_anything(cfg: dict, split: str, tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    from tennis.model import evaluate_final as ef

    _the_record(cfg, split)  # skips if not scored yet
    exp = _exp_dir(cfg)
    before = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(exp.glob("*.jsonl"))}

    def boom(*a: Any, **k: Any) -> None:
        raise AssertionError("run_final went past the already-scored check")

    for name in ("partitions", "validate", "score_test", "write_report"):
        monkeypatch.setattr(ef, name, boom)
    monkeypatch.setattr(ef.ev, "log_experiment", boom)
    assert ef.already_scored(cfg, split) is True
    with pytest.raises(RuntimeError, match="already been scored"):
        ef.run_final(cfg, split, score=True, model_dir=tmp_path, report_dir=tmp_path, fig_dir=tmp_path)
    after = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(exp.glob("*.jsonl"))}
    assert after == before


# ------------------------------------------------------------------------------------------------ 4. CSV
@pytest.mark.realdata
@pytest.mark.parametrize("split", FINAL_SPLITS)
def test_prediction_csv_is_the_test_partition_and_reproduces_the_logged_metrics(
        cfg: dict, features_table: pd.DataFrame, split: str) -> None:
    from tennis.model import evaluate as ev

    rec = _the_record(cfg, split)
    _, csv_path = _model_paths(cfg, split)
    csv = pd.read_csv(csv_path)
    parts = _partitions(cfg, features_table, split)
    test = parts["test"]
    assert len(test) == EXPECTED_N_TEST[split]
    assert list(csv["match_id"]) == list(test["match_id"]), f"{split}: CSV rows are not the test partition"
    assert not (set(csv["match_id"]) & (set(parts["train"]["match_id"]) | set(parts["val"]["match_id"])))
    t = test.set_index("match_id").loc[csv["match_id"]]
    assert (t["y"].to_numpy() == csv["y"].to_numpy()).all()
    np.testing.assert_allclose(csv["p_elo"].to_numpy(), t["elo_prob_a"].to_numpy(), rtol=0, atol=1e-12)
    for name, n in EXPECTED_TOURNEY_N.get(split, {}).items():
        assert int((csv["tourney_name"] == name).sum()) == n

    eps = ev.log_loss_eps(cfg)
    test_m = rec["metrics"]["test"]

    def check(logged: dict[str, Any], y: np.ndarray, p: np.ndarray, what: str) -> None:
        got = {"accuracy": ev.accuracy(y, p), "log_loss": ev.log_loss(y, p, eps), "brier": ev.brier(y, p)}
        for k, v in got.items():
            assert abs(v - logged[k]) < TOL_METRIC, f"{split} {what} {k}: CSV gives {v}, logged {logged[k]}"

    y, p = csv["y"].to_numpy(), csv["p_xgb"].to_numpy()
    check(test_m["overall"]["model"], y, p, "model")
    for k in ("elo", "elo_blend_50_50"):
        check(test_m["overall"]["baselines"][k], y, csv[f"p_{k}"].to_numpy(), k)
    for name, sub in (test_m.get("tourneys") or {}).items():
        m = (csv["tourney_name"] == name).to_numpy()
        assert int(m.sum()) == sub["model"]["n"]
        check(sub["model"], y[m], p[m], name)


# ------------------------------------------------------------------------------------------------ 5. booster
@pytest.mark.realdata
@pytest.mark.parametrize("split", FINAL_SPLITS)
def test_saved_final_model_is_the_frozen_refit(cfg: dict, features_table: pd.DataFrame, split: str) -> None:
    from tennis.model import dataset
    from tennis.model.train import load_model, matrix, predict

    model_path, csv_path = _model_paths(cfg, split)
    booster, meta = load_model(model_path)
    params, n_trees = frozen_params(cfg, split)
    assert booster.num_boosted_rounds() == n_trees
    assert meta["fixed_trees"] == n_trees and meta["early_stopping"] is False and meta["augmented"] is False
    assert meta["params"] == params
    assert meta["features"] == dataset.feature_columns(cfg, features_table)
    assert _ts(meta["refit_window"][1]) <= pd.Timestamp(cfg["splits"][split]["val_end"])
    csv = pd.read_csv(csv_path)
    test = _partitions(cfg, features_table, split)["test"].set_index("match_id").loc[csv["match_id"]].reset_index()
    p = predict(booster, matrix(test, meta["features"]))
    np.testing.assert_allclose(p, csv["p_xgb"].to_numpy(), rtol=0, atol=TOL_PRED)


# ------------------------------------------------------------------------------------------------ 6. refit
@pytest.mark.realdata
@pytest.mark.slow
@pytest.mark.parametrize("split", FINAL_SPLITS)
def test_final_model_is_reproduced_from_train_and_validation_rows_only(
        cfg: dict, features_table: pd.DataFrame, split: str) -> None:
    from tennis.model.train import refit_fixed

    _, csv_path = _model_paths(cfg, split)
    csv = pd.read_csv(csv_path)
    parts = _partitions(cfg, features_table, split)
    rows = pd.concat([parts["train"], parts["val"]], ignore_index=True)
    assert pd.to_datetime(rows["date"]).max() <= pd.Timestamp(cfg["splits"][split]["val_end"])
    params, n_trees = frozen_params(cfg, split)
    fit = refit_fixed(rows, params, cfg, n_trees=n_trees, split_name=split, test_df=parts["test"])
    test = parts["test"].set_index("match_id").loc[csv["match_id"]].reset_index()
    np.testing.assert_allclose(fit.predict(test), csv["p_xgb"].to_numpy(), rtol=0, atol=TOL_PRED,
                               err_msg=f"{split}: a refit on pre-test rows does not reproduce the scored model")
