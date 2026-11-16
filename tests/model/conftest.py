"""Shared fixtures for the model tests: a small synthetic feature table built by the real engine."""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest

from tennis.config import load_config
from tennis.engine.build_features import build_features
from tests.fixtures.synthetic import make_synthetic_matches


@pytest.fixture(scope="session")
def _synth_features() -> pd.DataFrame:
    cfg = load_config()
    f = build_features(make_synthetic_matches(seed=0, cfg=cfg), cfg)
    f["date"] = pd.to_datetime(f["date"])
    return f


@pytest.fixture
def synth_features(_synth_features: pd.DataFrame) -> pd.DataFrame:
    """Feature table of the 3-season synthetic tour (2019-2021), private copy per test."""
    return _synth_features.copy(deep=True)


@pytest.fixture
def small_cfg(tmp_path: Path) -> dict:
    """Config for fast XGBoost tests: dev split = train 2019-2020 / val 2021, few rounds, outputs in tmp_path."""
    cfg = copy.deepcopy(load_config())
    cfg["splits"]["dev"] = {"train_start": "2019-01-01", "train_end": "2020-12-31",
                            "val_start": "2021-01-01", "val_end": "2021-12-31"}
    cfg["xgb"].update(max_n_estimators=40, early_stopping_rounds=5, n_trials=4)
    cfg["evaluation"]["bootstrap_samples"] = 100
    for k in ("experiments", "reports", "figures", "models"):
        cfg["paths"][k] = str(tmp_path / k)
        (tmp_path / k).mkdir()
    return cfg
