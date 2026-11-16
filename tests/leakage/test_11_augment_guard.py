"""Test 11 (Phase 2 audit): flip augmentation must refuse non-training rows.

PLAN 5.6: "add flipped copies of TRAINING rows only". ``tennis.model.dataset.maybe_augment_train`` documents
"Never call this on validation or test rows" but does not enforce it: called on validation/test rows it silently
doubles them (each real match scored twice, in both orientations, under the same match_id). This test feeds it
rows dated after the latest ``train_end`` of every configured split -- rows that can't be training rows for any
split -- and requires an error.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.leakage.helpers import import_or_skip

DATASET_MODULE = "tennis.model.dataset"


def _rows(dates: list[pd.Timestamp]) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame({
        "match_id": [f"m{i}" for i in range(n)], "date": dates, "a_id": ["1"] * n, "b_id": ["2"] * n,
        "a_elo": [1600.0] * n, "b_elo": [1500.0] * n, "d_elo": [100.0] * n,
        "elo_prob_a": [0.64] * n, "y": [1, 0] * (n // 2) + [1] * (n % 2),
    })


def _latest_train_end(cfg: dict) -> pd.Timestamp:
    return max(pd.Timestamp(s["train_end"]) for s in cfg["splits"].values() if "train_end" in s)


def test_flip_augmentation_refuses_rows_after_every_train_end(cfg: dict) -> None:
    ds = import_or_skip(DATASET_MODULE, "dataset module not built yet")
    cfg["features"]["flip_augment_train"] = True
    late = _latest_train_end(cfg) + pd.Timedelta(days=30)
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        ds.maybe_augment_train(_rows([late, late]), cfg)


def test_flip_augmentation_still_works_on_training_rows(cfg: dict) -> None:
    """The guard must not break the legitimate use: training rows are doubled, flipped copies negate d_."""
    ds = import_or_skip(DATASET_MODULE, "dataset module not built yet")
    cfg["features"]["flip_augment_train"] = True
    early = pd.Timestamp(cfg["splits"]["dev"]["train_start"]) + pd.Timedelta(days=30)
    out = ds.maybe_augment_train(_rows([early, early]), cfg)
    assert len(out) == 4
    assert sorted(out["d_elo"].tolist()) == [-100.0, -100.0, 100.0, 100.0]
