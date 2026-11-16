"""Tests for tennis.model.dataset (feature allow-list, orientation flip, train-only augmentation).

Kept in tests/engine because they test the feature table produced by the engine.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tennis.config import load_config
from tennis.engine.build_features import build_features
from tennis.model.dataset import assert_not_augmented, feature_columns, flip_orientation, maybe_augment_train
from tests.fixtures.synthetic import make_match_row, rows_to_frame


@pytest.fixture
def cfg() -> dict[str, Any]:
    c = copy.deepcopy(load_config())
    c["data"]["feature_first_year"] = 2000
    return c


@pytest.fixture
def table(cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for i in range(12):
        rows.append(make_match_row(tourney_id=f"2020-{i:04d}", tourney_date=pd.Timestamp("2020-01-06")
                                   + pd.Timedelta(days=7 * i), winner_id=str(i % 4), loser_id=str((i + 1) % 4),
                                   stats=True, cfg=cfg))
    return build_features(rows_to_frame(rows), cfg)


def test_feature_columns_returns_allow_list(cfg: dict[str, Any], table: pd.DataFrame) -> None:
    cols = feature_columns(cfg, table)
    assert cols == list(cfg["model"]["features"])


def test_feature_columns_rejects_banned_and_missing(cfg: dict[str, Any], table: pd.DataFrame) -> None:
    for bad in ("w_ace", "winner_rank", "a_score", "minutes"):
        c = copy.deepcopy(cfg)
        c["model"]["features"] = list(c["model"]["features"]) + [bad]
        t = table.assign(**{bad: 0.0})
        with pytest.raises(ValueError, match="banned"):
            feature_columns(c, t)
    c = copy.deepcopy(cfg)
    c["model"]["features"] = list(c["model"]["features"]) + ["a_not_a_feature"]
    with pytest.raises(ValueError, match="missing"):
        feature_columns(c, table)
    c = copy.deepcopy(cfg)
    c["model"]["features"] = ["y"]
    with pytest.raises(ValueError, match="banned"):
        feature_columns(c, table)


def test_flip_orientation(table: pd.DataFrame) -> None:
    f = flip_orientation(table)
    assert (f["a_id"] == table["b_id"]).all() and (f["b_id"] == table["a_id"]).all()
    assert (f["a_name"] == table["b_name"]).all()
    assert (f["y"] == 1 - table["y"]).all()
    np.testing.assert_allclose(f["elo_prob_a"], 1 - table["elo_prob_a"], atol=1e-15)
    np.testing.assert_allclose(f["elo_surface_prob_a"], 1 - table["elo_surface_prob_a"], atol=1e-15)
    for c in table.columns:
        if c.startswith("d_"):
            same = (f[c] == -table[c]) | (f[c].isna() & table[c].isna())
            assert same.all(), c
        if c.startswith("a_"):
            b = "b_" + c[2:]
            assert ((f[c] == table[b]) | (f[c].isna() & table[b].isna())).all(), c
    for c in ("match_id", "date", "round_rank", "surface_Hard", "level_A", "best_of"):
        assert (f[c] == table[c]).all()
    # flipping twice is the identity
    pd.testing.assert_frame_equal(flip_orientation(f), table, check_exact=False, atol=1e-15, rtol=0)
    # input is not mutated
    assert not table["y"].equals(f["y"])


def test_augment_only_when_flag_set(cfg: dict[str, Any], table: pd.DataFrame) -> None:
    c = copy.deepcopy(cfg)
    c["features"]["flip_augment_train"] = False
    assert maybe_augment_train(table, c, "dev") is table
    c["features"]["flip_augment_train"] = True
    aug = maybe_augment_train(table, c, "dev")
    assert len(aug) == 2 * len(table)
    assert aug["y"].mean() == pytest.approx(0.5)
    assert aug["is_flipped"].tolist() == [0] * len(table) + [1] * len(table)
    assert "is_flipped" not in table.columns  # input untouched
    pd.testing.assert_frame_equal(aug.iloc[len(table):].drop(columns="is_flipped").reset_index(drop=True),
                                  flip_orientation(table).reset_index(drop=True))


def test_augment_refuses_rows_after_train_end(cfg: dict[str, Any], table: pd.DataFrame) -> None:
    """D-021: rows dated after cfg.splits[split].train_end can't be augmented (val/test rows)."""
    c = copy.deepcopy(cfg)
    c["features"]["flip_augment_train"] = True
    c["splits"]["dev"]["train_end"] = "2020-01-31"   # the table runs 2020-01-06 .. 2020-03-23
    with pytest.raises(ValueError, match="train_end"):
        maybe_augment_train(table, c, "dev")
    ok = table[table["date"] <= pd.Timestamp("2020-01-31")]
    assert len(maybe_augment_train(ok, c, "dev")) == 2 * len(ok)
    # also refused when the flag is off: the guard does not depend on the flag
    c["features"]["flip_augment_train"] = False
    with pytest.raises(ValueError, match="train_end"):
        maybe_augment_train(table, c, "dev")
    with pytest.raises(KeyError):
        maybe_augment_train(ok, c, "no_such_split")


def test_augment_without_split_name_uses_latest_train_end(cfg: dict[str, Any], table: pd.DataFrame) -> None:
    c = copy.deepcopy(cfg)
    c["features"]["flip_augment_train"] = True
    late = max(pd.Timestamp(s["train_end"]) for s in c["splits"].values() if "train_end" in s)
    assert len(maybe_augment_train(table, c)) == 2 * len(table)
    bad = table.assign(date=late + pd.Timedelta(days=1))
    with pytest.raises(ValueError, match="train_end"):
        maybe_augment_train(bad, c)


def test_augment_refuses_already_augmented(cfg: dict[str, Any], table: pd.DataFrame) -> None:
    c = copy.deepcopy(cfg)
    c["features"]["flip_augment_train"] = True
    aug = maybe_augment_train(table, c, "dev")
    with pytest.raises(ValueError, match="flipped"):
        maybe_augment_train(aug, c, "dev")


def test_assert_not_augmented(cfg: dict[str, Any], table: pd.DataFrame) -> None:
    assert_not_augmented(table)                         # no column: fine
    assert_not_augmented(table.assign(is_flipped=0))    # all zero: fine
    c = copy.deepcopy(cfg)
    c["features"]["flip_augment_train"] = True
    with pytest.raises(ValueError, match="flipped"):
        assert_not_augmented(maybe_augment_train(table, c, "dev"))
