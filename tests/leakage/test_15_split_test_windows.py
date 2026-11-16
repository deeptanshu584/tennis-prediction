"""Test 15 (Phase 5 audit): the real test windows are exactly the ones the human fixed, and the dev split can't reach them.

PLAN section 2 rules 5-6, CONTRACT section 4, DECISIONS D-004 / D-005 / D-017.

- Config: the replication test is Wimbledon 2025 with validation ending 2025-06-29; the forward test is pinned to
  2026-04-01 .. 2026-06-07 (D-017; `odds.max_date` equals `test_end`). The dev split and the Elo-tuning window
  (`elo_harness.tune_end`) end before every test period.
- Real data (features.parquet): the replication test partition is every Wimbledon-2025 row and nothing else (one
  tourney_id, one start date after val_end), and no Wimbledon-2025 row sits in its train/val; the forward test
  partition is exactly the rows dated in the pinned window; no test row of either split is in any dev partition,
  and every dev partition ends before the first test date.

Catches: a test window that silently moved or grew, a dev/tuning window that reaches a test period, a test
tournament leaking into its own split's train/val.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from tests.leakage.helpers import SPLITS_MODULE, SPLITS_SKIP, import_or_skip

# Human decisions (DECISIONS D-005, D-017). A change here must come with a DECISIONS entry.
REPLICATION_TEST = ("Wimbledon", 2025)
REPLICATION_VAL_END = pd.Timestamp("2025-06-29")
FORWARD_TEST = (pd.Timestamp("2026-04-01"), pd.Timestamp("2026-06-07"))


def _first_test_date(cfg: dict[str, Any]) -> pd.Timestamp:
    rep, fwd = cfg["splits"]["replication"], cfg["splits"]["forward"]
    return min(pd.Timestamp(rep["val_end"]) + pd.Timedelta(days=1), pd.Timestamp(fwd["test_start"]))


def test_test_windows_are_pinned(cfg: dict) -> None:
    rep, fwd = cfg["splits"]["replication"], cfg["splits"]["forward"]
    assert (rep["test_tourney"], int(rep["test_year"])) == REPLICATION_TEST
    assert pd.Timestamp(rep["val_end"]) == REPLICATION_VAL_END
    assert "test_end" in fwd, "forward.test_end must stay pinned (D-017): added data must not enter the test"
    assert (pd.Timestamp(fwd["test_start"]), pd.Timestamp(fwd["test_end"])) == FORWARD_TEST
    assert pd.Timestamp(cfg["odds"]["max_date"]) == FORWARD_TEST[1]


def test_dev_and_elo_tuning_end_before_every_test_period(cfg: dict) -> None:
    first = _first_test_date(cfg)
    dev = cfg["splits"]["dev"]
    assert "test_tourney" not in dev and "test_start" not in dev, "the dev split must have no test partition"
    for key in ("train_end", "val_end"):
        assert pd.Timestamp(dev[key]) < first, f"dev.{key} reaches a test period (first test date {first.date()})"
    h = cfg["elo_harness"]
    assert pd.Timestamp(h["tune_end"]) < first, "Elo tuning window reaches a test period"
    assert pd.Timestamp(h["tune_end"]) < pd.Timestamp(cfg["splits"]["replication"]["val_start"]), (
        "Elo tuning window overlaps the replication validation window")


@pytest.mark.realdata
def test_replication_test_is_exactly_wimbledon_2025(features_table: pd.DataFrame, cfg: dict) -> None:
    splits = import_or_skip(SPLITS_MODULE, SPLITS_SKIP)
    parts = splits.time_split(features_table, "replication", cfg)
    test = parts["test"]
    name, year = REPLICATION_TEST
    all_w = features_table[(features_table["tourney_name"] == name) & (features_table["date"].dt.year == year)]
    assert len(test) >= 100, f"Wimbledon {year} test has only {len(test)} rows"
    assert set(test["match_id"]) == set(all_w["match_id"])
    assert test["tourney_id"].nunique() == 1 and test["date"].nunique() == 1, "test is not one tournament"
    assert test["date"].min() > REPLICATION_VAL_END
    for p in ("train", "val"):
        assert not set(parts[p]["match_id"]) & set(all_w["match_id"]), f"Wimbledon {year} rows in replication.{p}"
        assert parts[p]["date"].max() < test["date"].min()


@pytest.mark.realdata
def test_forward_test_is_exactly_the_pinned_window(features_table: pd.DataFrame, cfg: dict) -> None:
    splits = import_or_skip(SPLITS_MODULE, SPLITS_SKIP)
    parts = splits.time_split(features_table, "forward", cfg)
    test = parts["test"]
    lo, hi = FORWARD_TEST
    d = features_table["date"]
    want = set(features_table.loc[(d >= lo) & (d <= hi), "match_id"])
    assert len(test) > 0 and set(test["match_id"]) == want
    assert test["date"].min() >= lo and test["date"].max() <= hi
    assert parts["val"]["date"].max() < lo


@pytest.mark.realdata
def test_no_test_row_reachable_from_dev(features_table: pd.DataFrame, cfg: dict) -> None:
    splits = import_or_skip(SPLITS_MODULE, SPLITS_SKIP)
    dev = splits.time_split(features_table, "dev", cfg)
    assert "test" not in dev
    dev_ids = set().union(*(set(p["match_id"]) for p in dev.values()))
    first = _first_test_date(cfg)
    for p, part in dev.items():
        assert part["date"].max() < first, f"dev.{p} has rows dated {part['date'].max().date()} >= {first.date()}"
    for name in ("replication", "forward"):
        test_ids = set(splits.time_split(features_table, name, cfg)["test"]["match_id"])
        assert not dev_ids & test_ids, f"{len(dev_ids & test_ids)} {name} test rows are in the dev split"
