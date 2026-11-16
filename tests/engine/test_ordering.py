"""Unit tests for the authoritative match order (CONTRACT section 2, PLAN section 2 rule 2)."""

from __future__ import annotations

import pandas as pd
import pytest

from tennis.config import load_config
from tennis.engine.ordering import ORDER_COLUMNS, check_round_rank, sort_matches
from tests.fixtures.synthetic import make_match_row, rows_to_frame


@pytest.fixture
def cfg() -> dict:
    return load_config()


def _frame(cfg: dict) -> pd.DataFrame:
    rows = [
        # same date, two tourney ids; qualifying numbered higher than main draw
        make_match_row(tourney_id="2020-0002", tourney_date="2020-01-06", winner_id="1", loser_id="2",
                       round_="R32", match_num=1, cfg=cfg),
        make_match_row(tourney_id="2020-0002", tourney_date="2020-01-06", winner_id="3", loser_id="4",
                       round_="Q1", match_num=300, cfg=cfg),
        make_match_row(tourney_id="2020-0001", tourney_date="2020-01-06", winner_id="5", loser_id="6",
                       round_="F", match_num=1, cfg=cfg),
        make_match_row(tourney_id="2020-0001", tourney_date="2020-01-06", winner_id="7", loser_id="8",
                       round_="SF", match_num=2, cfg=cfg),
        make_match_row(tourney_id="2019-0009", tourney_date="2019-12-30", winner_id="9", loser_id="10",
                       round_="F", match_num=5, cfg=cfg),
    ]
    return rows_to_frame(rows)


def test_order_columns_are_the_contract_ones() -> None:
    assert ORDER_COLUMNS == ("tourney_date", "tourney_id", "round_rank", "match_num", "match_id")


def test_sort_matches_uses_date_tid_round_matchnum(cfg: dict) -> None:
    df = _frame(cfg)
    out = sort_matches(df)
    got = list(zip(out["tourney_id"], out["round"]))
    assert got == [("2019-0009", "F"), ("2020-0001", "SF"), ("2020-0001", "F"),
                   ("2020-0002", "Q1"), ("2020-0002", "R32")]


def test_sort_ignores_input_order_and_order_key(cfg: dict) -> None:
    df = _frame(cfg)
    a = sort_matches(df)
    shuffled = df.sample(frac=1.0, random_state=3).reset_index(drop=True)
    shuffled["order_key"] = list(range(len(shuffled)))[::-1]
    b = sort_matches(shuffled)
    assert a["match_id"].tolist() == b["match_id"].tolist()


def test_sort_does_not_mutate_input(cfg: dict) -> None:
    df = _frame(cfg)
    before = df.copy(deep=True)
    sort_matches(df)
    pd.testing.assert_frame_equal(df, before)


def test_match_id_breaks_full_ties(cfg: dict) -> None:
    rows = [make_match_row(tourney_id="2020-0001", tourney_date="2020-01-06", winner_id=w, loser_id=l,
                           round_="R32", match_num=1, cfg=cfg) for w, l in (("1", "2"), ("3", "4"), ("5", "6"))]
    df = rows_to_frame(rows)
    out = sort_matches(df)
    assert out["match_id"].tolist() == sorted(df["match_id"])


def test_duplicate_match_id_raises(cfg: dict) -> None:
    df = _frame(cfg)
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="match_id"):
        sort_matches(dup)


def test_check_round_rank(cfg: dict) -> None:
    df = _frame(cfg)
    check_round_rank(df, cfg)
    bad = df.copy()
    bad.loc[0, "round_rank"] = 99
    with pytest.raises(ValueError, match="round_rank"):
        check_round_rank(bad, cfg)
