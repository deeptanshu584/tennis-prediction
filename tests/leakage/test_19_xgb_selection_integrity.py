"""Test 19 (Phase 5 audit, final pass): XGBoost selection integrity and train-API guards.

PLAN section 2 rules 5, 6 and 10; audit checklist items 1, 2, 5 and 8 (reports/audits/phase5.md).

Log-level checks (fast, skip if the Phase 5 logs are absent):
- every line of ``experiments/xgb_trials.jsonl`` is a dev-split trial: split ``dev``, a validation window inside
  ``splits.dev`` val dates, one config hash, unique trial numbers, no ``test`` key anywhere;
- ``config.xgb.best_params`` are EXACTLY the parameters of the lowest-logged-log-loss trial (tie-break as
  ``tune.best_record``), and ``n_estimators`` is that trial's ``best_iteration + 1``. A rounded copy is a different
  model: with early stopping it stops elsewhere, so the frozen config would not reproduce the selected trial;
- ``experiments/xgb_final.jsonl``'s refit is that trial and reproduces it; the saved model carries its config hash.

API guards (synthetic data):
- ``train.fit_xgb`` must refuse an early-stopping frame with rows outside the split's validation window. Otherwise a
  test partition can silently drive early stopping (``fit_xgb(train, test, ...)``);
- ``train.refit_fixed`` (Phase 6/7 retrain) refuses ``split_name=None``, rows after ``val_end`` and any match shared
  with the test, and builds exactly ``n_trees`` trees with no evaluation set.

Real data (slow): the saved dev model reproduces its logged validation metrics, and the TUNED parameters with the
frozen tree count and no early stopping score ~0.5 when trained on permuted labels (the Phase 5 report's own
shuffled run early-stopped on the real validation labels after 1 tree, which proves little).
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pytest

from tennis.config import repo_path
from tests.leakage.helpers import SPLITS_MODULE, SPLITS_SKIP, import_or_skip, synthetic_split_cfg

TRAIN_SKIP = "tennis.model.train not built yet (Phase 5)"


# ------------------------------------------------------------------------------------------ log helpers
def _lines(cfg: dict, name: str) -> list[dict[str, Any]]:
    path = Path(repo_path(cfg["paths"]["experiments"])) / f"{name}.jsonl"
    if not path.exists():
        pytest.skip(f"{path.name} not logged yet")
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def _keys(obj: Any, path: str = "") -> Iterator[str]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            yield p
            yield from _keys(v, p)


def _best(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Lowest validation log loss, then higher accuracy, then lower trial number (tune.best_record)."""
    return min(trials, key=lambda r: (r["metrics"]["val"]["log_loss"], -r["metrics"]["val"]["accuracy"],
                                      r["params"]["trial"]))


# ------------------------------------------------------------------------------------------ log checks
def test_trials_are_dev_only_and_never_see_a_test(cfg: dict) -> None:
    trials = _lines(cfg, "xgb_trials")
    dev = cfg["splits"]["dev"]
    lo, hi = pd.Timestamp(dev["val_start"]), pd.Timestamp(dev["val_end"])
    assert {r["split"] for r in trials} == {"dev"}
    assert len({r["config_hash"] for r in trials}) == 1, "trials of one study must share one config hash"
    nums = [r["params"]["trial"] for r in trials]
    assert len(nums) == len(set(nums)), "duplicate trial numbers"
    for r in trials:
        w0, w1 = (pd.Timestamp(x) for x in r["metrics"]["val_window"])
        assert lo <= w0 <= w1 <= hi, (r["params"]["trial"], r["metrics"]["val_window"])
        bad = [k for k in _keys(r) if "test" in k.lower()]
        assert not bad, (r["params"]["trial"], bad)
    assert len({tuple(r["metrics"]["val_window"]) for r in trials}) == 1
    assert len({r["metrics"]["val"]["n"] for r in trials}) == 1


def test_config_best_params_are_exactly_the_logged_argmin(cfg: dict) -> None:
    trials = _lines(cfg, "xgb_trials")
    best = _best(trials)
    bp = dict(cfg["xgb"]["best_params"])
    space = list(cfg["xgb"]["search_space"])
    assert set(space) <= set(bp), f"best_params lacks {set(space) - set(bp)}"
    diff = {k: (bp[k], best["params"][k]) for k in space
            if not math.isclose(float(bp[k]), float(best["params"][k]), rel_tol=1e-12, abs_tol=0.0)}
    assert not diff, (f"config.xgb.best_params differ from trial {best['params']['trial']} (the logged argmin): "
                      f"{diff}. Write full precision: a rounded copy is another model (different early-stopping "
                      "point), so the frozen config does not reproduce the selected trial")
    assert int(bp["n_estimators"]) == int(best["metrics"]["best_iteration"]) + 1


def test_final_refit_is_the_argmin_and_reproduces_it(cfg: dict) -> None:
    trials = _lines(cfg, "xgb_trials")
    final = _lines(cfg, "xgb_final")
    best = _best(trials)
    refits = [r for r in final if r["params"].get("run") == "best_refit"]
    assert refits, "no best_refit line in xgb_final.jsonl"
    for r in refits:
        assert r["split"] == "dev" and not [k for k in _keys(r) if "test" in k.lower()]
        assert r["params"]["trial"] == best["params"]["trial"]
        assert r["metrics"]["best_iteration"] == best["metrics"]["best_iteration"]
        assert abs(r["metrics"]["val"]["log_loss"] - best["metrics"]["val"]["log_loss"]) < 1e-12
        assert r["config_hash"] == best["config_hash"]
        assert best["config_hash"] in r["params"]["model_path"]
    band = cfg["leakage_tests"]["shuffled_label_band"]
    for r in final:
        if r["params"].get("run") == "shuffled_train_labels":
            assert band[0] <= r["metrics"]["val"]["accuracy"] <= band[1]


def _saved_dev_model(cfg: dict) -> tuple[Path, dict[str, Any]]:
    trials = _lines(cfg, "xgb_trials")
    best = _best(trials)
    path = Path(repo_path(cfg["paths"]["models"])) / f"xgb_dev_{best['config_hash']}.json"
    if not path.exists():
        pytest.skip(f"{path} not saved (models/ is gitignored)")
    meta = json.loads(path.with_name(path.stem + "_meta.json").read_text(encoding="utf-8"))
    return path, {"best": best, "meta": meta}


def test_saved_model_metadata_matches_the_selection(cfg: dict) -> None:
    _, d = _saved_dev_model(cfg)
    meta, best = d["meta"], d["best"]
    assert meta["config_hash"] == best["config_hash"]
    assert meta["split"] == "dev" and not meta.get("augmented")
    assert meta["features"] == list(cfg["model"]["features"])
    assert meta["best_iteration"] == best["metrics"]["best_iteration"]
    assert abs(meta["val_metrics"]["log_loss"] - best["metrics"]["val"]["log_loss"]) < 1e-12


# ------------------------------------------------------------------------------------------ API guards (synthetic)
SMALL = {"learning_rate": 0.1, "max_depth": 3, "min_child_weight": 1.0, "subsample": 0.8, "colsample_bytree": 0.8,
         "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0}


@pytest.fixture
def synth(synthetic_build, synthetic_matches, cfg: dict) -> tuple[dict, dict[str, pd.DataFrame]]:
    splits = import_or_skip(SPLITS_MODULE, SPLITS_SKIP)
    feats = synthetic_build[0].copy()
    feats["date"] = pd.to_datetime(feats["date"])
    scfg = synthetic_split_cfg(cfg, synthetic_matches)
    scfg["xgb"] = {**scfg["xgb"], "max_n_estimators": 20, "early_stopping_rounds": 5}
    parts = splits.time_split(feats, "replication", scfg)
    assert len(parts["train"]) and len(parts["val"]) and len(parts.get("test", [])), {k: len(v) for k, v in parts.items()}
    return scfg, parts


def test_fit_xgb_refuses_an_early_stopping_frame_outside_the_val_window(synth) -> None:
    tr = import_or_skip("tennis.model.train", TRAIN_SKIP)
    scfg, parts = synth
    tr.fit_xgb(parts["train"], parts["val"], SMALL, scfg, split_name="replication")  # the legitimate call works
    with pytest.raises(ValueError):
        tr.fit_xgb(parts["train"], parts["test"], SMALL, scfg, split_name="replication")
    with pytest.raises(ValueError):
        tr.fit_xgb(parts["train"], pd.concat([parts["val"], parts["test"]]), SMALL, scfg, split_name="replication")


def test_refit_fixed_guards_and_has_no_eval_set(synth) -> None:
    tr = import_or_skip("tennis.model.train", TRAIN_SKIP)
    if not hasattr(tr, "refit_fixed"):
        pytest.skip("train.refit_fixed not written yet (Phase 6)")
    scfg, parts = synth
    rows = pd.concat([parts["train"], parts["val"]], ignore_index=True)
    test = parts["test"]
    res = tr.refit_fixed(rows, SMALL, scfg, n_trees=7, split_name="replication", test_df=test)
    assert res.booster.num_boosted_rounds() == 7 and len(res.val_pred) == 0
    with pytest.raises((ValueError, TypeError, KeyError)):
        tr.refit_fixed(rows, SMALL, scfg, n_trees=7, split_name=None, test_df=test)
    with pytest.raises(ValueError):  # a test row in the refit rows
        tr.refit_fixed(pd.concat([rows, test.iloc[:1]], ignore_index=True), SMALL, scfg, n_trees=7,
                       split_name="replication", test_df=test)
    with pytest.raises(ValueError):  # a test row in the refit rows, even if the caller passes no test frame
        tr.refit_fixed(pd.concat([rows, test.iloc[:1]], ignore_index=True), SMALL, scfg, n_trees=7,
                       split_name="replication", test_df=test.iloc[:0])
    # the refit must not depend on the test rows in any way (they are only used for the guards)
    other = tr.refit_fixed(rows, SMALL, scfg, n_trees=7, split_name="replication",
                           test_df=test.assign(y=1 - test["y"]))
    X = tr.matrix(test, res.features)
    np.testing.assert_array_equal(tr.predict(res.booster, X), tr.predict(other.booster, X))


# ------------------------------------------------------------------------------------------ real data
@pytest.mark.realdata
@pytest.mark.slow
def test_saved_dev_model_reproduces_logged_validation(features_table: pd.DataFrame, cfg: dict) -> None:
    tr = import_or_skip("tennis.model.train", TRAIN_SKIP)
    splits = import_or_skip(SPLITS_MODULE, SPLITS_SKIP)
    path, d = _saved_dev_model(cfg)
    b, meta = tr.load_model(path)
    parts = splits.time_split(features_table, "dev", cfg)
    assert "test" not in parts
    val = parts["val"]
    p = tr.predict(b, tr.matrix(val, meta["features"]))
    y = val["y"].to_numpy(dtype=float)
    pc = np.clip(p, cfg["evaluation"]["log_loss_eps"], 1 - cfg["evaluation"]["log_loss_eps"])
    ll = float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean())
    assert abs(ll - d["best"]["metrics"]["val"]["log_loss"]) < 1e-6
    assert float(((p > 0.5) == (y == 1)).mean()) <= float(cfg["evaluation"]["leakage_alarm_accuracy"])


@pytest.mark.realdata
@pytest.mark.slow
def test_shuffled_labels_tuned_params_fixed_trees(features_table: pd.DataFrame, cfg: dict) -> None:
    tr = import_or_skip("tennis.model.train", TRAIN_SKIP)
    splits = import_or_skip(SPLITS_MODULE, SPLITS_SKIP)
    xgb = pytest.importorskip("xgboost")
    bp = dict(cfg["xgb"].get("best_params") or {})
    if "n_estimators" not in bp:
        pytest.skip("xgb.best_params not frozen yet")
    parts = splits.time_split(features_table, "dev", cfg)
    train, val = parts["train"], parts["val"]
    feats = list(cfg["model"]["features"])
    y_perm = np.random.default_rng(int(cfg["seed"])).permutation(train["y"].to_numpy(dtype=np.float32))
    dtr = xgb.DMatrix(tr.matrix(train, feats), label=y_perm, missing=np.nan, feature_names=feats, nthread=4)
    b = xgb.train(tr.native_params(bp, cfg, 4), dtr, num_boost_round=int(bp["n_estimators"]))  # no eval set
    p = b.predict(xgb.DMatrix(tr.matrix(val, feats), missing=np.nan, feature_names=feats, nthread=4))
    acc = float(((p > 0.5) == (val["y"].to_numpy() == 1)).mean())
    lo, hi = cfg["leakage_tests"]["shuffled_label_band"]
    assert lo <= acc <= hi, f"tuned params, {bp['n_estimators']} trees, permuted labels: accuracy {acc:.4f}"
