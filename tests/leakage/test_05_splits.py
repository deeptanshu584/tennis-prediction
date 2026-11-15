"""Test 5 (PLAN section 9): time-based splits and duplicates.

For every split in cfg["splits"] (via tennis.model.splits.time_split): max(train date) < min(val date) <
min(test date); no match_id appears in two partitions; partitions are subsets of the input. Each partition is
EXACTLY the rows the CONTRACT rulings define: train/val = rows with *_start <= date <= *_end (inclusive);
dev has no test; replication test = rows with tourney_name == test_tourney and year == test_year;
forward test = rows with date >= test_start (and <= test_end when given). Also: no duplicate match_id in
the match table or the feature table, and no same match entered twice under different ids (same date, players,
round).

Catches: random shuffles / K-fold, overlapping or inverted date ranges, inclusive-boundary double counting,
duplicated matches across merged sources.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from tests.leakage.helpers import SPLITS_MODULE, SPLITS_SKIP, import_or_skip, synthetic_split_cfg

ORDER = ("train", "val", "test")


def _check_split(df: pd.DataFrame, cfg: dict[str, Any], name: str) -> int:
    """Run time_split for one split and check it. Returns the number of non-empty partitions."""
    splits = import_or_skip(SPLITS_MODULE, SPLITS_SKIP)
    spec = cfg["splits"][name]
    parts = splits.time_split(df.copy(), name, cfg)
    assert set(parts) <= set(ORDER) and {"train", "val"} <= set(parts), f"{name}: keys {sorted(parts)}"
    present = [p for p in ORDER if p in parts and len(parts[p])]

    has_test = "test_tourney" in spec or "test_start" in spec
    if not has_test:
        assert "test" not in parts or len(parts["test"]) == 0, f"{name}: no test is configured but test returned"
    for p in present:
        assert parts[p]["match_id"].is_unique, f"{name}.{p} has duplicate match ids"
        assert set(parts[p]["match_id"]) <= set(df["match_id"]), f"{name}.{p} has rows not in the input"

    # exact membership per the CONTRACT rulings
    d = pd.to_datetime(df["date"])
    expected: dict[str, pd.Series] = {}
    for p in ("train", "val"):
        expected[p] = (d >= pd.Timestamp(spec[f"{p}_start"])) & (d <= pd.Timestamp(spec[f"{p}_end"]))
    if "test_tourney" in spec:
        expected["test"] = (df["tourney_name"] == spec["test_tourney"]) & (d.dt.year == int(spec["test_year"]))
    elif "test_start" in spec:
        m = d >= pd.Timestamp(spec["test_start"])
        if "test_end" in spec:
            m &= d <= pd.Timestamp(spec["test_end"])
        expected["test"] = m
    for p, mask in expected.items():
        got = set(parts[p]["match_id"]) if p in parts else set()
        want = set(df.loc[mask, "match_id"])
        assert got == want, (f"{name}.{p}: {len(got - want)} unexpected rows, {len(want - got)} missing rows "
                             f"vs the configured bounds")

    for p, q in zip(present, present[1:]):
        assert pd.to_datetime(parts[p]["date"]).max() < pd.to_datetime(parts[q]["date"]).min(), (
            f"{name}: max({p} date) >= min({q} date)")
    for i, p in enumerate(present):
        for q in present[i + 1:]:
            both = set(parts[p]["match_id"]) & set(parts[q]["match_id"])
            assert not both, f"{name}: {len(both)} match ids in both {p} and {q}"
    return len(present)


def _dup_same_match(m: pd.DataFrame) -> pd.DataFrame:
    """Rows describing the same match under different match_ids (same date, same two players, same round)."""
    lo = m[["winner_id", "loser_id"]].astype(str).min(axis=1)
    hi = m[["winner_id", "loser_id"]].astype(str).max(axis=1)
    key = pd.DataFrame({"d": m["tourney_date"], "lo": lo, "hi": hi, "r": m["round"]})
    return m[key.duplicated(keep=False)]


def test_splits_synthetic(synthetic_matches: pd.DataFrame, cfg: dict) -> None:
    scfg = synthetic_split_cfg(cfg, synthetic_matches)
    em = synthetic_matches[synthetic_matches["is_training_level"] & ~synthetic_matches["is_walkover"]
                           & ~synthetic_matches["is_retirement"]]
    df = pd.DataFrame({"match_id": em["match_id"], "date": em["tourney_date"],
                       "tourney_name": em["tourney_name"], "tourney_id": em["tourney_id"], "y": 1})
    df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)  # time_split must not rely on row order
    for name in scfg["splits"]:
        assert _check_split(df, scfg, name) >= 2, name


def test_no_duplicates_synthetic(synthetic_matches: pd.DataFrame) -> None:
    assert synthetic_matches["match_id"].is_unique
    assert _dup_same_match(synthetic_matches).empty


def test_no_duplicates_synthetic_features(synthetic_build) -> None:
    features, _ = synthetic_build
    assert features["match_id"].is_unique


@pytest.mark.realdata
def test_no_duplicates_real_matches(real_matches: pd.DataFrame) -> None:
    dup = real_matches["match_id"].duplicated(keep=False)
    assert not dup.any(), f"{int(dup.sum())} rows share a match_id"
    same = _dup_same_match(real_matches[~real_matches["is_walkover"]])
    assert same.empty, (
        f"{len(same)} rows look like the same match under different ids (date, players, round), e.g.\n"
        + same[["match_id", "source", "tourney_id", "tourney_date", "round", "winner_id", "loser_id"]]
        .head(10).to_string())


@pytest.mark.realdata
def test_splits_real(features_table: pd.DataFrame, cfg: dict) -> None:
    assert features_table["match_id"].is_unique, "duplicate match_id in features.parquet"
    for name in cfg["splits"]:
        _check_split(features_table, cfg, name)
