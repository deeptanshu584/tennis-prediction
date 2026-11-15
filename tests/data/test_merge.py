"""Tests for src/tennis/ingest/merge.py: source priority, de-duplication, order_key."""

from __future__ import annotations

import pytest

from tennis.ingest.merge import merge_sources
from tennis.ingest.sackmann import UNIFIED_COLUMNS, unify
from tests.data.helpers import raw_row, typed_frame


def _frame(cfg, rows, source):
    u, _ = unify(typed_frame(rows), cfg, source=source)
    return u


def test_priority_and_dedup_on_match_id(cfg):
    sack = _frame(cfg, [raw_row(match_num="1", winner_id="1", loser_id="2", score="6-4 6-4")], "sackmann")
    # same match_id (players swapped => same hash), different score text: Sackmann must win
    tml = _frame(cfg, [
        raw_row(match_num="1", winner_id="2", loser_id="1", score="4-6 4-6"),
        raw_row(match_num="2", winner_id="3", loser_id="4", tourney_date="20230102"),
    ], "tml")
    merged, contrib = merge_sources({"tml": tml, "sackmann": sack}, cfg)  # dict order must not matter
    assert len(merged) == 2
    row = merged[merged["match_num"] == 1].iloc[0]
    assert row["source"] == "sackmann" and row["winner_id"] == "1"
    c = contrib.set_index("source")
    assert c.loc["sackmann", "rows_kept"] == 1 and c.loc["tml", "rows_kept"] == 1
    assert c.loc["tml", "dup_match_id"] == 1
    assert list(merged.columns) == UNIFIED_COLUMNS
    assert merged["order_key"].tolist() == [0, 1]


def test_dedup_on_d009_key_across_sources(cfg):
    sack = _frame(cfg, [raw_row(match_num="10", winner_id="1", loser_id="2")], "sackmann")
    tml = _frame(cfg, [raw_row(match_num="99", winner_id="1", loser_id="2")], "tml")  # same match, other num
    merged, contrib = merge_sources({"sackmann": sack, "tml": tml}, cfg)
    assert len(merged) == 1 and merged["source"].item() == "sackmann"
    assert contrib.set_index("source").loc["tml", "dup_key"] == 1


def test_unknown_source_raises(cfg):
    f = _frame(cfg, [raw_row()], "sackmann")
    with pytest.raises(ValueError, match="unknown source"):
        merge_sources({"espn": f.assign(source="espn")}, cfg)


def test_missing_columns_raise(cfg):
    f = _frame(cfg, [raw_row()], "sackmann").drop(columns=["is_walkover"])
    with pytest.raises(ValueError, match="missing unified columns"):
        merge_sources({"sackmann": f}, cfg)


def test_source_column_must_match_key(cfg):
    f = _frame(cfg, [raw_row()], "sackmann")
    with pytest.raises(ValueError, match="different `source`"):
        merge_sources({"tml": f}, cfg)


def test_merged_order_key_is_position_in_match_order(cfg):
    sack = _frame(cfg, [raw_row(match_num="5", tourney_date="20230109", winner_id="1", loser_id="2")], "sackmann")
    tml = _frame(cfg, [raw_row(match_num="1", tourney_date="20230102", winner_id="3", loser_id="4")], "tml")
    merged, _ = merge_sources({"sackmann": sack, "tml": tml}, cfg)
    assert merged["source"].tolist() == ["tml", "sackmann"]
    assert merged["order_key"].tolist() == [0, 1]
    assert merged["order_key"].dtype == "int64"
