"""tennis.model.evaluate_final + train.refit_fixed (Phases 6/7) on the synthetic 2019-2021 tour.

Synthetic splits: replication = train 2019-2020 / val 2021-01-01..2021-06-20 / test "Grass Slam" 2021
(starts 2021-06-21); forward = train 2019-2020 / val 2021-01..04 / test 2021-05-01..2021-08-31 with
"Clay Slam" reported separately.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from tennis.model import dataset
from tennis.model import evaluate_final as ef
from tennis.model import train as tr
from tennis.model.splits import time_split
from tests.leakage.test_16_test_set_hygiene import check_records

PARAMS = {"learning_rate": 0.1, "max_depth": 3, "min_child_weight": 1.0, "subsample": 0.8, "colsample_bytree": 0.8,
          "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0}


@pytest.fixture
def cfg(small_cfg: dict) -> dict:
    c = copy.deepcopy(small_cfg)
    c["splits"]["replication"] = {"train_start": "2019-01-01", "train_end": "2020-12-31",
                                  "val_start": "2021-01-01", "val_end": "2021-06-20",
                                  "test_tourney": "Grass Slam", "test_year": 2021}
    c["splits"]["forward"] = {"train_start": "2019-01-01", "train_end": "2020-12-31",
                              "val_start": "2021-01-01", "val_end": "2021-04-30",
                              "test_start": "2021-05-01", "test_end": "2021-08-31",
                              "report_tourneys": ["Clay Slam"]}
    c["xgb"]["best_params"] = {**PARAMS, "n_estimators": 689}
    c["xgb"]["n_trials_forward"] = 3
    # start unfrozen: the real config's frozen Phase 6/7 values do not apply to the synthetic tour
    c["xgb"].pop("replication_n_estimators", None)
    c["xgb"].pop("forward_best_params", None)
    return c


def _frozen(cfg: dict, df: pd.DataFrame, split: str) -> dict:
    """cfg with the split's tree count (and forward params) frozen the way the orchestrator would."""
    c = copy.deepcopy(cfg)
    if split == "forward":
        c["xgb"]["forward_best_params"] = dict(PARAMS)
    params, _, _ = ef.model_params(c, split)
    parts = ef.partitions(c, split, df=df)
    n = tr.fit_xgb(parts["train"], parts["val"], params, c, split_name=split).best_iteration + 1
    if split == "forward":
        c["xgb"]["forward_best_params"]["n_estimators"] = n
    else:
        c["xgb"]["replication_n_estimators"] = n
    return c


def _records(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# ------------------------------------------------------------------------------------------ refit_fixed guards
def _refit_parts(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = time_split(df, "replication", cfg)
    return pd.concat([p["train"], p["val"]], ignore_index=True), p["test"]


def test_refit_fixed_has_exactly_n_trees_and_no_eval_set(synth_features, cfg, monkeypatch) -> None:
    rows, test = _refit_parts(synth_features, cfg)
    seen: dict = {}
    real_train = xgb.train

    def spy(params, dtrain, num_boost_round=10, evals=None, **kw):
        seen.update(n_rows=dtrain.num_row(), evals=evals, kw=kw)
        return real_train(params, dtrain, num_boost_round=num_boost_round, evals=evals, **kw)

    monkeypatch.setattr(tr.xgb, "train", spy)
    res = tr.refit_fixed(rows, PARAMS, cfg, n_trees=17, split_name="replication", test_df=test)
    assert res.booster.num_boosted_rounds() == 17 and res.best_iteration == 16
    assert not seen["evals"] and "early_stopping_rounds" not in seen["kw"]
    assert seen["n_rows"] == len(rows)
    assert res.extra["refit_window"][1] <= cfg["splits"]["replication"]["val_end"]
    p = res.predict(test)
    assert len(p) == len(test) and np.isfinite(p).all()


def test_refit_fixed_calls_the_augmentation_guard_with_the_split(synth_features, cfg, monkeypatch) -> None:
    rows, test = _refit_parts(synth_features, cfg)
    calls = []
    real = dataset.maybe_augment_train

    def spy(df, c, split_name=None):
        calls.append((split_name, pd.to_datetime(df["date"]).max()))
        return real(df, c, split_name)

    monkeypatch.setattr(tr.dataset, "maybe_augment_train", spy)
    tr.refit_fixed(rows, PARAMS, cfg, n_trees=3, split_name="replication", test_df=test)
    assert [c[0] for c in calls] == ["replication"]
    assert calls[0][1] <= pd.Timestamp(cfg["splits"]["replication"]["train_end"])


def test_refit_fixed_guards_raise(synth_features, cfg) -> None:
    rows, test = _refit_parts(synth_features, cfg)
    kw = dict(n_trees=3, split_name="replication")
    with pytest.raises(ValueError, match="split_name"):
        tr.refit_fixed(rows, PARAMS, cfg, n_trees=3, split_name=None, test_df=test)  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        tr.refit_fixed(rows, PARAMS, cfg, n_trees=3, split_name="nope", test_df=test)
    with pytest.raises(ValueError, match="n_trees"):
        tr.refit_fixed(rows, PARAMS, cfg, n_trees=0, split_name="replication", test_df=test)
    with pytest.raises(ValueError, match="after val_end"):  # test rows smuggled into the refit
        tr.refit_fixed(pd.concat([rows, test]), PARAMS, cfg, test_df=test, **kw)
    with pytest.raises(ValueError, match="match_id"):  # same match on both sides, dated inside the window
        leak = test.head(3).assign(date=rows["date"].max())
        tr.refit_fixed(pd.concat([rows, leak]), PARAMS, cfg, test_df=test, **kw)
    with pytest.raises(ValueError, match="not dated after"):
        early = rows.head(5)
        tr.refit_fixed(rows.drop(early.index), PARAMS, cfg, test_df=early, **kw)
    with pytest.raises(ValueError, match="flipped"):
        tr.refit_fixed(rows, PARAMS, cfg, test_df=dataset.flip_orientation(test).assign(is_flipped=1), **kw)


# ------------------------------------------------------------------------------------------ partitions / params
@pytest.mark.parametrize("split", ef.FINAL_SPLITS)
def test_validation_only_partitions_never_contain_test_period_rows(synth_features, cfg, split) -> None:
    parts = ef.partitions(cfg, split, df=synth_features)
    assert set(parts) == {"train", "val"}
    end = pd.Timestamp(cfg["splits"][split]["val_end"])
    assert all(p["date"].max() <= end for p in parts.values())
    full = ef.partitions(cfg, split, df=synth_features, include_test=True)
    assert len(full["test"]) and full["test"]["date"].min() > end


def test_model_params(cfg, tmp_path) -> None:
    p, n, _ = ef.model_params(cfg, "replication")
    assert "n_estimators" not in p and n is None
    with pytest.raises(ValueError, match="replication_n_estimators"):
        ef.model_params(cfg, "replication", require_frozen=True)
    with pytest.raises(ValueError, match="--tune"):
        ef.model_params(cfg, "forward", trials_dir=tmp_path)
    with pytest.raises(ValueError, match="forward_best_params"):
        ef.model_params(cfg, "forward", require_frozen=True)
    with pytest.raises(ValueError, match="split must be"):
        ef.model_params(cfg, "dev")
    c = copy.deepcopy(cfg)
    c["xgb"]["forward_best_params"] = {**PARAMS, "n_estimators": 12}
    p, n, _ = ef.model_params(c, "forward", require_frozen=True)
    assert p == PARAMS and n == 12


# ------------------------------------------------------------------------------------------ validation-only run
@pytest.mark.parametrize("split", ef.FINAL_SPLITS)
def test_validation_only_run_never_loads_or_scores_the_test(synth_features, cfg, split, monkeypatch) -> None:
    c = copy.deepcopy(cfg)
    if split == "forward":
        c["xgb"]["forward_best_params"] = dict(PARAMS)
    seen = []
    real = ef.time_split

    def spy(df, name, cc):
        seen.append(pd.to_datetime(df["date"]).max())
        return real(df, name, cc)

    monkeypatch.setattr(ef, "time_split", spy)
    out = ef.run_final(c, split, df=synth_features)
    assert seen and max(seen) <= pd.Timestamp(c["splits"][split]["val_end"])
    assert out["test"] is None and out["record"] is None
    assert out["n_trees"] == out["val"]["best_iteration"] + 1
    assert not (Path(c["paths"]["experiments"]) / f"{split}.jsonl").exists()
    report = (Path(c["paths"]["reports"]) / ef.REPORT_NAME[split]).read_text(encoding="utf-8")
    assert "NOT been scored" in report and "D-028 finding 2" in report


# ------------------------------------------------------------------------------------------ test scoring
def test_scoring_requires_frozen_settings(synth_features, cfg) -> None:
    with pytest.raises(ValueError, match="replication_n_estimators"):
        ef.run_final(cfg, "replication", df=synth_features, score=True)
    assert not (Path(cfg["paths"]["experiments"]) / "replication.jsonl").exists()


def test_frozen_tree_count_must_be_reproduced(synth_features, cfg) -> None:
    c = _frozen(cfg, synth_features, "replication")
    c["xgb"]["replication_n_estimators"] += 1
    with pytest.raises(RuntimeError, match="trees"):
        ef.run_final(c, "replication", df=synth_features, score=True, write=False)
    assert not (Path(c["paths"]["experiments"]) / "replication.jsonl").exists()


@pytest.mark.parametrize("split", ef.FINAL_SPLITS)
def test_test_is_scored_once_and_logged_in_test16_format(synth_features, cfg, split, monkeypatch) -> None:
    c = _frozen(cfg, synth_features, split)
    test_ids = set(ef.partitions(c, split, df=synth_features, include_test=True)["test"]["match_id"])
    es_sets, refit_sets = [], []
    real_fit, real_refit = ef.fit_xgb, ef.refit_fixed

    def fit_spy(train_df, val_df, params, cc, *, split_name, **kw):
        es_sets.append((set(train_df["match_id"]), set(val_df["match_id"])))
        return real_fit(train_df, val_df, params, cc, split_name=split_name, **kw)

    def refit_spy(train_df, params, cc, **kw):
        refit_sets.append(set(train_df["match_id"]))
        return real_refit(train_df, params, cc, **kw)

    monkeypatch.setattr(ef, "fit_xgb", fit_spy)
    monkeypatch.setattr(ef, "refit_fixed", refit_spy)
    out = ef.run_final(c, split, df=synth_features, score=True)

    # no test row is trained on or used for early stopping
    assert len(es_sets) == 1 and len(refit_sets) == 1
    assert not (es_sets[0][0] | es_sets[0][1]) & test_ids
    assert not refit_sets[0] & test_ids
    assert refit_sets[0] == es_sets[0][0] | es_sets[0][1]
    assert out["test"]["refit"]["fixed_trees"] == out["n_trees"]

    log = Path(c["paths"]["experiments"]) / f"{split}.jsonl"
    recs = _records(log)
    assert len(recs) == 1
    rec = recs[0]
    assert rec["split"] == split and "test" in rec["metrics"] and "val" in rec["metrics"]
    t = rec["metrics"]["test"]["overall"]["model"]
    assert t["n"] == len(test_ids) and len(t["accuracy_ci"]) == 2 and "log_loss_ci" in t and "brier_ci" in t
    assert set(rec["metrics"]["test"]["overall"]["diffs"]) >= {"elo", "elo_blend_50_50", "rank_logistic"}
    assert "round" in rec["metrics"]["test"]["by"]
    assert check_records([(log.name, 1, rec)], c) == []
    if split == "forward":
        sub = rec["metrics"]["test"]["tourneys"]["Clay Slam"]["model"]
        assert 0 < sub["n"] < t["n"] and "accuracy_ci" in sub

    # report and model written; a second scoring is refused and logs nothing
    report = (Path(c["paths"]["reports"]) / ef.REPORT_NAME[split]).read_text(encoding="utf-8")
    assert "Single evaluation on the frozen config" in report and "Small sample" in report and "D-031" in report
    assert (Path(c["paths"]["models"]) / out["test"]["model_path"]).exists()
    with pytest.raises(RuntimeError, match="already been scored"):
        ef.run_final(c, split, df=synth_features, score=True)
    assert len(_records(log)) == 1


def test_checker_would_flag_a_second_hash(synth_features, cfg) -> None:
    """Sanity: two test records with different hashes in one file fail test 16's checker."""
    c = _frozen(cfg, synth_features, "replication")
    out = ef.run_final(c, "replication", df=synth_features, score=True, write=False)
    rec = out["record"]
    assert check_records([("replication.jsonl", 1, rec), ("replication.jsonl", 2, {**rec, "config_hash": "x"})], c)


# ------------------------------------------------------------------------------------------ forward re-tuning
def test_forward_tuning_logs_trials_without_test_metrics(synth_features, cfg) -> None:
    recs = ef.tune_forward(cfg, df=synth_features, n_trials=2, workers=1)
    log = Path(cfg["paths"]["experiments"]) / "xgb_trials_forward.jsonl"
    lines = _records(log)
    assert len(recs) == len(lines) == 2
    assert all(r["split"] == "forward" and "test" not in json.dumps(r["metrics"]) for r in lines)
    assert check_records([(log.name, i, r) for i, r in enumerate(lines, 1)], cfg) == []
    vend = pd.Timestamp(cfg["splits"]["forward"]["val_end"])
    assert all(pd.Timestamp(r["metrics"]["val_window"][1]) <= vend for r in lines)
    p, n, src = ef.model_params(cfg, "forward")
    best = min(recs, key=lambda r: (r["val"]["log_loss"], -r["val"]["accuracy"], r["number"]))
    assert p == best["params"] and n == best["best_iteration"] + 1 and "not yet frozen" in src
    with pytest.raises(ValueError, match="n_trials_forward"):
        ef.tune_forward(cfg, df=synth_features, n_trials=4, workers=1)


def test_rescoring_refused_when_predictions_exist_without_a_log_record(synth_features, cfg) -> None:
    """Audit phase7 note 1: a crash after saving the test predictions but before logging must not allow a re-score."""
    c = _frozen(cfg, synth_features, "replication")
    mdir = Path(c["paths"]["models"])
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "xgb_replication_final_deadbeef0000_test_predictions.csv").write_text("match_id\n", encoding="utf-8")
    assert not (Path(c["paths"]["experiments"]) / "replication.jsonl").exists()
    with pytest.raises(RuntimeError, match="already scored"):
        ef.run_final(c, "replication", df=synth_features, score=True, write=False)
    assert not (Path(c["paths"]["experiments"]) / "replication.jsonl").exists()
