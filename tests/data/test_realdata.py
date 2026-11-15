"""Checks on the real data/interim/matches.parquet (skipped when it hasn't been built)."""

from __future__ import annotations

import pandas as pd
import pytest

from tennis.config import load_config, repo_path
from tennis.ingest.sackmann import MATCH_ORDER, UNIFIED_COLUMNS

CFG = load_config()
MATCHES = repo_path(CFG["paths"]["matches"])

pytestmark = pytest.mark.realdata


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    if not MATCHES.exists():
        pytest.skip(f"{MATCHES} not built (run: .venv/Scripts/python -m tennis.ingest.merge)")
    return pd.read_parquet(MATCHES)


def test_schema(matches):
    assert list(matches.columns) == UNIFIED_COLUMNS
    assert str(matches["tourney_date"].dtype) == "datetime64[ns]"
    assert set(matches["surface"]) <= {"Hard", "Clay", "Grass"}


def test_unique_match_id(matches):
    assert matches["match_id"].is_unique


def test_no_self_matches(matches):
    assert not (matches["winner_id"] == matches["loser_id"]).any()


def test_qualifying_never_training_level(matches):
    q = matches["round"].isin(CFG["data"]["qualifying_rounds"])
    assert (q == matches["is_qualifying"]).all()
    assert not matches.loc[matches["is_qualifying"], "is_training_level"].any()


def test_order_key_is_sorted_position(matches):
    assert (matches["order_key"].to_numpy() == range(len(matches))).all()
    resorted = matches.sort_values(MATCH_ORDER, kind="mergesort")
    assert (resorted.index.to_numpy() == range(len(matches))).all()


def test_tour_level_2023_in_band(matches):
    m = matches
    training = m["is_training_level"] & ~m["is_walkover"] & ~m["is_retirement"]
    n = int((training & (m["tourney_date"].dt.year == 2023)).sum())
    assert 2500 <= n <= 4500, n
