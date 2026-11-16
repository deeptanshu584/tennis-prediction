"""XGBoost training (PLAN section 6 Phase 5, section 7).

- Inputs are exactly :func:`tennis.model.dataset.feature_columns` (the config allow-list). NaN is passed to XGBoost
  as missing (native handling); there is no fitted imputer or scaler, so nothing is learned outside the booster.
- Every fit goes through :func:`tennis.model.dataset.maybe_augment_train` WITH the split name, and the validation
  frame through :func:`tennis.model.dataset.assert_not_augmented` (CONTRACT section 4b, D-021, D-024 R2).
- ``binary:logistic`` / ``logloss`` from ``config.xgb``; early stopping on the validation rows
  (``xgb.early_stopping_rounds``), at most ``xgb.max_n_estimators`` rounds (PLAN section 6 Phase 5).
- Reproducible: ``seed = config.seed`` and a pinned thread count (:data:`NTHREAD`). XGBoost's ``hist`` results
  depend on the thread count, so tuning trials and refits must use the same value.

Caveat: early stopping and model selection both use the validation window, so validation metrics of the selected
model are slightly optimistic. The real out-of-sample checks are the Phase 6/7 test sets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import xgboost as xgb

from tennis.config import config_hash, repo_path
from tennis.model import dataset

TREE_METHOD: str = "hist"
NTHREAD: int = 1
"""Threads per booster. Pinned: hist results change with the thread count (checked on the dev split)."""

# config.xgb.search_space names -> native XGBoost parameter names
_NATIVE: dict[str, str] = {"learning_rate": "eta", "reg_alpha": "alpha", "reg_lambda": "lambda"}


@dataclass
class FitResult:
    """A fitted booster (trimmed to its best iteration) and its validation scores."""
    booster: xgb.Booster
    features: list[str]
    params: dict[str, Any]
    best_iteration: int                      # 0-based; the booster has best_iteration + 1 trees
    val_pred: np.ndarray
    val_metrics: dict[str, float]
    extra: dict[str, Any] = field(default_factory=dict)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """P(A wins) for the rows of ``df`` (never augmented: :func:`assert_not_augmented` is called)."""
        dataset.assert_not_augmented(df)
        return predict(self.booster, matrix(df, self.features))


def native_params(params: Mapping[str, Any], cfg: Mapping[str, Any], nthread: int = NTHREAD) -> dict[str, Any]:
    """Booster parameters: the tuned ``params`` (config names) plus objective, eval_metric, tree method, seed and
    thread count. ``n_estimators`` is not a booster parameter and is ignored here."""
    x = cfg["xgb"]
    if "nthread" in x and int(x["nthread"]) != NTHREAD:  # audit phase5 F4: config and module must agree
        raise ValueError(f"config xgb.nthread={x['nthread']} != train.NTHREAD={NTHREAD}")
    out: dict[str, Any] = {"objective": x["objective"], "eval_metric": x["eval_metric"], "tree_method": x.get("tree_method", TREE_METHOD),
                           "seed": int(cfg["seed"]), "nthread": int(nthread), "verbosity": 0}
    for k, v in params.items():
        if k == "n_estimators":
            continue
        out[_NATIVE.get(k, k)] = int(v) if k == "max_depth" else v
    return out


def matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Allow-listed columns as a float32 array (NaN stays NaN = missing for XGBoost)."""
    return df[features].to_numpy(dtype=np.float32)


def predict(booster: xgb.Booster, X: np.ndarray) -> np.ndarray:
    """Probabilities from all trees of ``booster`` (columns of ``X`` in ``booster.feature_names`` order)."""
    return booster.predict(xgb.DMatrix(X, missing=np.nan, nthread=NTHREAD, feature_names=booster.feature_names))


def val_metrics(y: np.ndarray, p: np.ndarray, cfg: Mapping[str, Any]) -> dict[str, float]:
    """Point metrics on validation rows (log loss clipped with ``evaluation.log_loss_eps``)."""
    from tennis.model import evaluate as ev
    return {"n": int(len(y)), "log_loss": ev.log_loss(y, p, ev.log_loss_eps(cfg)), "accuracy": ev.accuracy(y, p),
            "brier": ev.brier(y, p)}


def train_booster(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray,
                  params: Mapping[str, Any], cfg: Mapping[str, Any], *, nthread: int = NTHREAD,
                  feature_names: list[str] | None = None) -> tuple[xgb.Booster, int]:
    """Low-level fit on prepared arrays with early stopping on (``X_va``, ``y_va``).

    Returns the booster trimmed to ``best_iteration + 1`` trees and ``best_iteration`` (0-based). Callers are
    responsible for the augmentation guard; :func:`fit_xgb` is the checked entry point.
    """
    x = cfg["xgb"]
    n_max = int(min(int(params.get("n_estimators", x["max_n_estimators"])), int(x["max_n_estimators"])))
    dtr = xgb.DMatrix(X_tr, label=y_tr, missing=np.nan, feature_names=feature_names, nthread=nthread)
    dva = xgb.DMatrix(X_va, label=y_va, missing=np.nan, feature_names=feature_names, nthread=nthread)
    b = xgb.train(native_params(params, cfg, nthread), dtr, num_boost_round=n_max, evals=[(dva, "val")],
                  early_stopping_rounds=int(x["early_stopping_rounds"]), verbose_eval=False)
    best = int(b.best_iteration)
    return b[: best + 1], best


def fit_xgb(train_df: pd.DataFrame, val_df: pd.DataFrame, params: Mapping[str, Any], cfg: Mapping[str, Any], *,
            split_name: str, nthread: int = NTHREAD) -> FitResult:
    """Fit XGBoost on ``train_df`` with early stopping on ``val_df`` (PLAN section 6 Phase 5).

    - features: ``dataset.feature_columns(cfg, train_df)`` only;
    - ``train_df`` goes through ``dataset.maybe_augment_train(train_df, cfg, split_name)`` (raises for rows after
      the split's ``train_end``); ``val_df`` through ``dataset.assert_not_augmented``;
    - ``params`` use config names (``learning_rate``, ``max_depth``, ``min_child_weight``, ``subsample``,
      ``colsample_bytree``, ``gamma``, ``reg_alpha``, ``reg_lambda``; optional ``n_estimators`` cap).
    """
    spec = cfg["splits"][split_name]
    vd = pd.to_datetime(val_df["date"])
    out = (vd < pd.Timestamp(spec["val_start"])) | (vd > pd.Timestamp(spec["val_end"]))
    if bool(out.any()):  # audit phase5 F2: the early-stopping set must be the split's validation window
        raise ValueError(f"fit_xgb: {int(out.sum())} early-stopping rows outside split {split_name!r} val window")
    features = dataset.feature_columns(cfg, train_df)
    dataset.assert_not_augmented(val_df)
    tr = dataset.maybe_augment_train(train_df, cfg, split_name)
    y_tr = tr["y"].to_numpy(dtype=np.float32)
    y_va = val_df["y"].to_numpy(dtype=np.float32)
    booster, best = train_booster(matrix(tr, features), y_tr, matrix(val_df, features), y_va, params, cfg,
                                  nthread=nthread, feature_names=features)
    p = predict(booster, matrix(val_df, features))
    return FitResult(booster, features, dict(params), best, p, val_metrics(y_va, p, cfg),
                     {"split": split_name, "n_train": int(len(tr)), "n_val": int(len(val_df)),
                      "augmented": bool(len(tr) != len(train_df))})


def save_model(res: FitResult, cfg: Mapping[str, Any], name: str, out_dir: str | Path | None = None) -> Path:
    """Save ``<models>/<name>_<config hash>.json`` (the booster) plus ``..._meta.json`` (features, params, best
    iteration, validation metrics, config hash). Returns the booster path."""
    h = config_hash(dict(cfg))
    d = Path(out_dir) if out_dir is not None else repo_path(cfg["paths"]["models"])
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}_{h}.json"
    res.booster.save_model(str(path))
    meta = {"config_hash": h, "features": res.features, "params": res.params,
            "native_params": native_params(res.params, cfg), "best_iteration": res.best_iteration,
            "val_metrics": res.val_metrics, **res.extra}
    (d / f"{name}_{h}_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return path


def load_model(path: str | Path) -> tuple[xgb.Booster, dict[str, Any]]:
    """Load a booster saved by :func:`save_model` and its metadata."""
    p = Path(path)
    b = xgb.Booster()
    b.load_model(str(p))
    meta = json.loads(p.with_name(p.stem + "_meta.json").read_text(encoding="utf-8"))
    return b, meta


# ------------------------------------------------------------------------------------------ Phase 6/7 refit
def refit_fixed(train_df: pd.DataFrame, params: Mapping[str, Any], cfg: Mapping[str, Any], *, n_trees: int,
                split_name: str, test_df: pd.DataFrame) -> FitResult:
    """Final refit on train+val with a FIXED number of trees and NO early-stopping set (D-028 retrain rule).

    Args:
        train_df: the refit rows, i.e. the split's training AND validation rows (never test rows).
        params: tuned parameters (config names; ``n_estimators`` is ignored, ``n_trees`` decides).
        n_trees: number of boosting rounds, fixed in advance (``best_iteration + 1`` of a validation run).
        split_name: a key of ``cfg["splits"]``; ``None`` is refused (the guards need the split's dates).
        test_df: the split's test rows. Used ONLY for the disjointness guards below; never trained on and never
            passed to XGBoost as an evaluation / early-stopping set.

    Guards (each raises ``ValueError``):
        - ``split_name`` is None, or ``n_trees`` < 1;
        - any refit row is dated after ``cfg["splits"][split_name]["val_end"]``;
        - a ``match_id`` appears in both ``train_df`` and ``test_df``, or a test row is not dated after every
          refit row;
        - ``test_df`` (or ``train_df``) contains flipped (augmented) rows.

    Augmentation: rows up to the split's ``train_end`` always go through
    ``dataset.maybe_augment_train(rows, cfg, split_name)`` (CONTRACT 4b). The validation-window rows that join
    the refit are appended unaugmented: the D-021 guard forbids flipping rows after ``train_end``. With
    ``features.flip_augment_train: false`` (the current setting) nothing is flipped anyway.

    Returns a :class:`FitResult` whose booster has exactly ``n_trees`` trees; ``best_iteration = n_trees - 1``,
    ``val_pred`` is empty and ``val_metrics`` is ``{}`` (there is no validation set in a final refit).
    """
    if split_name is None:
        raise ValueError("refit_fixed needs split_name (never None): the date guards come from the split")
    spec = cfg["splits"][split_name]  # KeyError for an unknown split
    n = int(n_trees)
    if n < 1:
        raise ValueError(f"n_trees must be >= 1, got {n_trees}")
    if len(train_df) == 0:
        raise ValueError("refit_fixed got no rows")
    dataset.assert_not_augmented(train_df)
    dataset.assert_not_augmented(test_df)
    dates = pd.to_datetime(train_df["date"])
    val_end = pd.Timestamp(spec["val_end"])
    if bool((dates > val_end).any()):
        raise ValueError(f"refit_fixed got {int((dates > val_end).sum())} rows dated after val_end "
                         f"{val_end.date()} (split {split_name!r}; latest {dates.max().date()})")
    shared = set(train_df["match_id"]) & set(test_df["match_id"])
    if shared:
        raise ValueError(f"refit_fixed: {len(shared)} match_id in both the refit rows and the test rows")
    if len(test_df) and not pd.to_datetime(test_df["date"]).min() > dates.max():
        raise ValueError("refit_fixed: a test row is not dated after every refit row")

    features = dataset.feature_columns(cfg, train_df)
    train_end = pd.Timestamp(spec["train_end"])
    early = train_df.loc[(dates <= train_end).to_numpy()]
    late = train_df.loc[(dates > train_end).to_numpy()]
    tr = dataset.maybe_augment_train(early, cfg, split_name)
    if len(late):
        if "is_flipped" in tr.columns:
            late = late.assign(is_flipped=0)
        tr = pd.concat([tr, late], ignore_index=True)
    dtr = xgb.DMatrix(matrix(tr, features), label=tr["y"].to_numpy(dtype=np.float32), missing=np.nan,
                      feature_names=features, nthread=NTHREAD)
    booster = xgb.train(native_params(params, cfg, NTHREAD), dtr, num_boost_round=n, evals=[], verbose_eval=False)
    if booster.num_boosted_rounds() != n:
        raise RuntimeError(f"refit has {booster.num_boosted_rounds()} trees, expected {n}")
    window = [str(dates.min().date()), str(dates.max().date())]
    return FitResult(booster, features, dict(params), n - 1, np.empty(0), {},
                     {"split": split_name, "n_refit": int(len(tr)), "n_refit_rows_original": int(len(train_df)),
                      "refit_window": window, "fixed_trees": n, "early_stopping": False,
                      "augmented": bool(len(tr) != len(train_df))})
