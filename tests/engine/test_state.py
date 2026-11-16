"""Unit tests for per-player state and its small update functions (PLAN 5.1, CONTRACT section 2)."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from tennis.engine.elo import expected_score, update_rating
from tennis.engine.state import (
    count_recent,
    days_since,
    elo_after_match,
    elo_deltas,
    layoff_played,
    m_value,
    new_player,
    ols_slope,
    record_elo,
    record_match,
    record_result,
    surface_count,
    surface_rating,
    winrate,
)

D = dt.date


def _p(**kw):
    args = dict(start_rating=1500.0, grad_window=20, results_len=50, rolling_n=50)
    args.update(kw)
    return new_player(**args)


# ---------------------------------------------------------------- fresh player


def test_new_player_snapshot_values() -> None:
    p = _p()
    assert p.elo == 1500.0
    assert list(p.elo_hist) == [1500.0]
    assert p.matches_played == 0
    assert surface_count(p, "Clay") == 0
    assert surface_rating(p, "Clay", 1500.0) == 1500.0
    assert p.last_day is None
    assert len(p.results) == 0 and len(p.stats) == 0
    assert math.isnan(days_since(p.last_day, 100))
    assert math.isnan(ols_slope(list(p.elo_hist)))


# ---------------------------------------------------------------- Elo pieces


def test_alcaraz_djokovic_13_95() -> None:
    new_w, new_l = elo_after_match(2063.0, 2120.0, 24.0, 24.0)
    assert new_w - 2063.0 == pytest.approx(13.95, abs=0.01)
    assert new_l - 2120.0 == pytest.approx(-13.95, abs=0.01)


def test_elo_after_match_uses_each_players_own_k() -> None:
    new_w, new_l = elo_after_match(1500.0, 1500.0, 40.0, 10.0)
    assert new_w == 1520.0
    assert new_l == 1495.0
    assert new_w == update_rating(1500.0, expected_score(1500.0, 1500.0), 1.0, 40.0)


def test_elo_deltas_match_update_and_use_given_ratings() -> None:
    dw, dl = elo_deltas(2063.0, 2120.0, 24.0, 24.0)
    assert dw == pytest.approx(13.95, abs=0.01) and dl == pytest.approx(-13.95, abs=0.01)
    new_w, new_l = elo_after_match(1600.0, 1500.0, 30.0, 20.0)
    dw, dl = elo_deltas(1600.0, 1500.0, 30.0, 20.0)
    assert 1600.0 + dw == new_w and 1500.0 + dl == new_l   # bitwise the same arithmetic
    # deltas depend only on the ratings passed in (pre-block snapshot), not on any current rating
    assert elo_deltas(1500.0, 1500.0, 10.0, 10.0) == (5.0, -5.0)


def test_record_elo_sets_ratings_and_appends_history() -> None:
    p = _p(grad_window=3)
    record_elo(p, "Clay", 1510.0, 1520.0)
    record_elo(p, "Hard", 1505.0, 1490.0)
    assert p.elo == 1505.0
    assert surface_rating(p, "Clay", 1500.0) == 1520.0
    assert surface_rating(p, "Hard", 1500.0) == 1490.0
    assert surface_rating(p, "Grass", 1500.0) == 1500.0
    assert list(p.elo_hist) == [1500.0, 1510.0, 1505.0]
    record_elo(p, "Hard", 1501.0, 1480.0)
    assert list(p.elo_hist) == [1510.0, 1505.0, 1501.0]  # window 3


@pytest.mark.parametrize("vals", [
    [1500.0, 1510.0],
    [1500.0, 1512.3, 1498.7, 1530.1, 1525.0],
    list(1500 + np.cumsum(np.random.default_rng(1).normal(0, 15, 20))),
    [2100.123456, 2100.123457, 2100.123455],
])
def test_ols_slope_equals_polyfit(vals: list[float]) -> None:
    expect = np.polyfit(np.arange(len(vals), dtype=float), np.asarray(vals, dtype=float), 1)[0]
    assert ols_slope(vals) == pytest.approx(expect, abs=1e-9)


def test_ols_slope_short_is_nan() -> None:
    assert math.isnan(ols_slope([]))
    assert math.isnan(ols_slope([1500.0]))


# ---------------------------------------------------------------- m and layoff list


def test_m_value_modes() -> None:
    assert m_value(1.0, "event_matches_so_far", 0) == 1.0
    assert m_value(1.0, "event_matches_so_far", 3) == 4.0
    assert m_value(5.0, "constant", 3) == 5.0
    with pytest.raises(ValueError):
        m_value(1.0, "event_total", 3)


def test_layoff_played_one_entry_per_tourney_id() -> None:
    p = _p()
    # two matches in tourney A (qualifying + main), then B and C on the same start date
    record_match(p, "Hard", "A", D(2020, 1, 6), 0, 14)
    record_match(p, "Hard", "A", D(2020, 1, 6), 0, 14)
    record_match(p, "Hard", "B", D(2020, 1, 13), 7, 14)
    record_match(p, "Hard", "C", D(2020, 1, 13), 7, 14)
    assert list(p.tourneys) == ["A", "B", "C"]
    # current tourney already present -> not appended again
    assert layoff_played(p.tourneys, "C", D(2020, 1, 13)) == [D(2020, 1, 6), D(2020, 1, 13), D(2020, 1, 13)]
    # new tourney -> its start date appended
    assert layoff_played(p.tourneys, "E", D(2020, 3, 2)) == [D(2020, 1, 6), D(2020, 1, 13), D(2020, 1, 13),
                                                              D(2020, 3, 2)]
    assert layoff_played({}, "X", D(2020, 1, 1)) == [D(2020, 1, 1)]


def test_event_count_includes_qualifying_in_same_tourney() -> None:
    p = _p()
    assert p.event_matches.get("T", 0) == 0
    record_match(p, "Hard", "T", D(2020, 1, 6), 0, 14)   # Q1
    record_match(p, "Hard", "T", D(2020, 1, 6), 0, 14)   # Q2
    assert p.event_matches["T"] == 2
    assert m_value(1.0, "event_matches_so_far", p.event_matches.get("T", 0)) == 3.0
    assert p.event_matches.get("U", 0) == 0


# ---------------------------------------------------------------- counts, dates, results


def test_record_match_counts() -> None:
    p = _p()
    record_match(p, "Clay", "T1", D(2020, 4, 6), 100, 14)
    record_match(p, "Clay", "T1", D(2020, 4, 6), 100, 14)
    record_match(p, "Hard", "T2", D(2020, 4, 13), 107, 14)
    assert p.matches_played == 3
    assert surface_count(p, "Clay") == 2
    assert surface_count(p, "Hard") == 1
    assert surface_count(p, "Grass") == 0
    assert p.last_day == 107
    assert days_since(p.last_day, 121) == 14.0


def test_count_recent_window_inclusive() -> None:
    days = [80, 86, 100, 100, 107]
    assert count_recent(days, 114, 14) == 3     # >= 100
    assert count_recent(days, 107, 14) == 3     # >= 93
    assert count_recent(days, 100, 14) == 4     # >= 86 (inclusive)
    assert count_recent(days, 200, 14) == 0
    assert count_recent([], 10, 14) == 0


def test_record_match_prunes_old_recent_days_but_keeps_window() -> None:
    p = _p()
    for d in (0, 10, 20, 30):
        record_match(p, "Hard", f"T{d}", D(2020, 1, 1) + dt.timedelta(days=d), d, 14)
    assert count_recent(p.recent_days, 30, 14) == 2   # 20 and 30
    assert count_recent(p.recent_days, 34, 14) == 2   # 20 and 30
    assert count_recent(p.recent_days, 44, 14) == 1


def test_winrate_last_k() -> None:
    p = _p(results_len=5)
    assert math.isnan(winrate(p.results, 5))
    for r in (1, 0, 1, 1):
        record_result(p, r)
    assert winrate(p.results, 5) == 3 / 4        # min(k, available)
    assert winrate(p.results, 2) == 1.0
    for r in (0, 0):
        record_result(p, r)
    assert list(p.results) == [0, 1, 1, 0, 0]
    assert winrate(p.results, 5) == 2 / 5
    assert winrate(p.results, 3) == 1 / 3
