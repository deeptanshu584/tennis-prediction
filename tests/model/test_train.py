"""tennis.model.train: allow-list, augmentation guard, determinism, save/load (tiny synthetic runs)."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from tennis.model import dataset
from tennis.model import train as tr
from tennis.model.splits import time_split

PARAMS = {"learning_rate": 0.1, "max_depth": 3, "min_child_weight": 1.0, "subsample": 0.8, "colsample_bytree": 0.8,
          "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0}


def _parts(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = time_split(df, "dev", cfg)
    return p["train"], p["val"]


def test_fit_runs_and_uses_only_the_allow_list(synth_features, small_cfg) -> None:
    train, val = _parts(synth_features, small_cfg)
    res = tr.fit_xgb(train, val, PARAMS, small_cfg, split_name="dev")
    assert res.features == dataset.feature_columns(small_cfg, train)
    assert res.booster.feature_names == res.features
    assert 0 <= res.best_iteration < small_cfg["xgb"]["max_n_estimators"]
    assert res.booster.num_boosted_rounds() == res.best_iteration + 1
    assert len(res.val_pred) == len(val) and np.isfinite(res.val_pred).all()
    assert 0 < res.val_metrics["log_loss"] < 1 and res.val_metrics["n"] == len(val)
    np.testing.assert_array_equal(res.predict(val), res.val_pred)


def test_columns_outside_the_allow_list_are_ignored(synth_features, small_cfg) -> None:
    train, val = _parts(synth_features, small_cfg)
    base = tr.fit_xgb(train, val, PARAMS, small_cfg, split_name="dev")
    leaky_tr, leaky_va = train.assign(zz_leak=train["y"]), val.assign(zz_leak=val["y"])
    leaky = tr.fit_xgb(leaky_tr, leaky_va, PARAMS, small_cfg, split_name="dev")
    assert "zz_leak" not in leaky.features
    np.testing.assert_array_equal(base.val_pred, leaky.val_pred)


def test_banned_column_in_allow_list_raises(synth_features, small_cfg) -> None:
    train, val = _parts(synth_features, small_cfg)
    cfg = copy.deepcopy(small_cfg)
    cfg["model"]["features"] = cfg["model"]["features"] + ["winner_rank"]
    with pytest.raises(ValueError, match="banned"):
        tr.fit_xgb(train, val, PARAMS, cfg, split_name="dev")


def test_fit_is_deterministic(synth_features, small_cfg) -> None:
    train, val = _parts(synth_features, small_cfg)
    a = tr.fit_xgb(train, val, PARAMS, small_cfg, split_name="dev")
    b = tr.fit_xgb(train.copy(), val.copy(), PARAMS, small_cfg, split_name="dev")
    np.testing.assert_array_equal(a.val_pred, b.val_pred)
    assert a.best_iteration == b.best_iteration


def test_augmentation_guard_is_called_with_the_split_name(synth_features, small_cfg, monkeypatch) -> None:
    train, val = _parts(synth_features, small_cfg)
    calls = []
    real = dataset.maybe_augment_train

    def spy(df, cfg, split_name=None):
        calls.append(split_name)
        return real(df, cfg, split_name)

    monkeypatch.setattr(dataset, "maybe_augment_train", spy)
    tr.fit_xgb(train, val, PARAMS, small_cfg, split_name="dev")
    assert calls == ["dev"]


def test_guards_reject_late_training_rows_and_augmented_validation(synth_features, small_cfg) -> None:
    train, val = _parts(synth_features, small_cfg)
    with pytest.raises(ValueError, match="train_end"):
        tr.fit_xgb(pd.concat([train, val]), val, PARAMS, small_cfg, split_name="dev")
    with pytest.raises(ValueError, match="flipped"):
        tr.fit_xgb(train, val.assign(is_flipped=1), PARAMS, small_cfg, split_name="dev")


def test_flip_augmentation_doubles_training_rows_only(synth_features, small_cfg) -> None:
    train, val = _parts(synth_features, small_cfg)
    cfg = copy.deepcopy(small_cfg)
    cfg["features"]["flip_augment_train"] = True
    res = tr.fit_xgb(train, val, PARAMS, cfg, split_name="dev")
    assert res.extra["augmented"] and res.extra["n_train"] == 2 * len(train) and res.extra["n_val"] == len(val)


def test_native_params(small_cfg) -> None:
    p = tr.native_params({**PARAMS, "max_depth": 4.0, "n_estimators": 99}, small_cfg)
    assert p["eta"] == 0.1 and p["alpha"] == 0.0 and p["lambda"] == 1.0 and p["max_depth"] == 4
    assert "n_estimators" not in p and "learning_rate" not in p
    assert p["objective"] == "binary:logistic" and p["eval_metric"] == "logloss"
    assert p["seed"] == small_cfg["seed"] and p["nthread"] == tr.NTHREAD and p["tree_method"] == "hist"


def test_n_estimators_is_capped(synth_features, small_cfg) -> None:
    train, val = _parts(synth_features, small_cfg)
    cfg = copy.deepcopy(small_cfg)
    cfg["xgb"].update(max_n_estimators=3, early_stopping_rounds=50)
    res = tr.fit_xgb(train, val, {**PARAMS, "n_estimators": 1000}, cfg, split_name="dev")
    assert res.booster.num_boosted_rounds() <= 3


def test_save_and_load_round_trip(synth_features, small_cfg) -> None:
    train, val = _parts(synth_features, small_cfg)
    res = tr.fit_xgb(train, val, PARAMS, small_cfg, split_name="dev")
    path = tr.save_model(res, small_cfg, "xgb_test")
    assert path.parent == tr.repo_path(small_cfg["paths"]["models"])
    b, meta = tr.load_model(path)
    assert meta["features"] == res.features and meta["best_iteration"] == res.best_iteration
    assert meta["config_hash"] in path.name
    np.testing.assert_array_equal(tr.predict(b, tr.matrix(val, res.features)), res.val_pred)
