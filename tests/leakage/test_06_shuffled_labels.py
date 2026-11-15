"""Test 6 (PLAN section 9): shuffled labels.

Train a small XGBoost model (fixed parameters) on the dev-split training rows with PERMUTED labels and score it
on the validation rows: accuracy must sit in `leakage_tests.shuffled_label_band`.

Catches: the row structure revealing the label without any real signal — e.g. a feature whose presence,
missingness or scale depends on which side won, an orientation that correlates with the row layout, or
validation labels that are themselves predictable from row position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.leakage.helpers import SPLITS_MODULE, SPLITS_SKIP, import_or_skip, model_features


@pytest.mark.realdata
@pytest.mark.slow
def test_shuffled_labels_dev_split(features_table: pd.DataFrame, cfg: dict) -> None:
    feats = model_features(cfg)
    splits = import_or_skip(SPLITS_MODULE, SPLITS_SKIP)
    xgb = pytest.importorskip("xgboost")
    parts = splits.time_split(features_table, "dev", cfg)
    train, val = parts["train"], parts["val"]
    assert len(train) > 1000 and len(val) > 500, (len(train), len(val))
    rng = np.random.default_rng(int(cfg["seed"]))
    y_perm = rng.permutation(train["y"].astype(int).to_numpy())
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        random_state=int(cfg["seed"]), n_jobs=4,
    )
    model.fit(train[feats].astype(float), y_perm)
    p = model.predict_proba(val[feats].astype(float))[:, 1]
    acc = float(((p > 0.5).astype(int) == val["y"].astype(int).to_numpy()).mean())
    lo, hi = cfg["leakage_tests"]["shuffled_label_band"]
    assert lo <= acc <= hi, f"accuracy with permuted training labels = {acc:.4f}, outside [{lo}, {hi}]"
