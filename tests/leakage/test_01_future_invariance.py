"""Test 1 (PLAN section 9): future invariance.

Scramble everything about the matches after a cutoff (swap winner/loser with all winner_/loser_ attributes and
w_/l_ stats, permute and blank the stats, rewrite score/minutes/ranks, toggle some retirement/walkover flags),
rebuild, and require every feature row at or before the cutoff to be IDENTICAL (exact, NaN-aware).
Also: drop the future entirely -> same rows; the builder must not mutate its input.

Catches: any feature that reads a later match — whole-table groupby/rolling/expanding/merge, end-of-event
aggregates, same-tournament look-ahead (mid-event cutoff), a global normalisation, or a sort that is not by
(tourney_date, tourney_id, round_rank, match_num).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from tests.leakage.helpers import assert_frames_identical, build, get_builder, match_position, scramble_future


def _rows_upto(features: pd.DataFrame, matches: pd.DataFrame, cutoff_key: int) -> pd.DataFrame:
    keys = pd.Series(match_position(matches).to_numpy(), index=matches["match_id"].to_numpy())
    return features[features["match_id"].map(keys) <= cutoff_key]


def _mid_event_cutoff(matches: pd.DataFrame, level: str = "G") -> int:
    """Position of the last R16 match of a mid-history event of the given level (later rounds are 'future')."""
    ev = matches[(matches["tourney_level"] == level) & ~matches["is_qualifying"]]
    dates = sorted(ev["tourney_date"].unique())
    if not dates:
        ev = matches[matches["is_training_level"]]
        dates = sorted(ev["tourney_date"].unique())
    date = dates[len(dates) // 2]
    one = ev[ev["tourney_date"] == date]
    one = one[one["tourney_id"] == one["tourney_id"].min()]
    r16 = one[one["round"] == "R16"]
    return int(match_position(matches)[(r16 if len(r16) else one).index].max())


def _date_cutoff(matches: pd.DataFrame) -> int:
    """Position of the last match dated on or before the middle start date (all matches dated <= d are 'past')."""
    dates = sorted(matches["tourney_date"].unique())
    d = pd.Timestamp(dates[len(dates) // 2])
    return int(match_position(matches)[matches["tourney_date"] <= d].max())


def _check_invariance(matches: pd.DataFrame, cfg: dict[str, Any], cutoff_key: int) -> None:
    """cutoff_key = position in the authoritative order (match_position); later rows are the 'future'."""
    base = build(matches, cfg)
    past_rows = _rows_upto(base, matches, cutoff_key)
    assert len(past_rows) > 50, "cutoff leaves too few rows to be meaningful"
    future = match_position(matches) > cutoff_key
    assert future.sum() > 50

    scrambled = scramble_future(matches, future)
    assert scrambled.loc[~future].equals(matches.loc[~future])
    rebuilt = build(scrambled, cfg)
    assert_frames_identical(past_rows, _rows_upto(rebuilt, matches, cutoff_key),
                            f"future scrambled after {cutoff_key}")

    truncated = build(matches.loc[~future], cfg)
    assert_frames_identical(past_rows, _rows_upto(truncated, matches, cutoff_key),
                            f"future removed after {cutoff_key}")


@pytest.mark.parametrize("kind", ["date", "mid_event"])
def test_future_invariance_synthetic(synthetic_matches: pd.DataFrame, cfg: dict, kind: str) -> None:
    get_builder()
    cutoff = _date_cutoff(synthetic_matches) if kind == "date" else _mid_event_cutoff(synthetic_matches)
    _check_invariance(synthetic_matches, cfg, cutoff)


def test_future_invariance_synthetic_nondefault_flags(synthetic_matches: pd.DataFrame, cfg: dict) -> None:
    """Same check with every retirement flag flipped, the layoff bonus on and the other m_mode."""
    get_builder()
    for flag in ("retirement_updates_elo", "retirement_updates_stats", "retirement_updates_h2h",
                 "retirement_updates_form"):
        cfg["cleaning"][flag] = not cfg["cleaning"][flag]
    cfg["elo"]["overall"]["bonus_after_layoff"] = True
    cfg["elo"]["surface"]["bonus_after_layoff"] = True
    cfg["elo"]["m_mode"] = "constant" if cfg["elo"]["m_mode"] == "event_matches_so_far" else "event_matches_so_far"
    cfg["features"]["rolling_n"] = 7
    _check_invariance(synthetic_matches, cfg, _mid_event_cutoff(synthetic_matches, "M"))


def test_builder_does_not_mutate_input(synthetic_matches: pd.DataFrame, cfg: dict) -> None:
    fn = get_builder()
    before = synthetic_matches.copy(deep=True)
    fn(synthetic_matches, cfg)
    pd.testing.assert_frame_equal(synthetic_matches, before)


def test_input_row_order_is_irrelevant(synthetic_matches: pd.DataFrame, cfg: dict) -> None:
    """Shuffling the input rows must not change the output (the builder sorts by the match order)."""
    a = build(synthetic_matches, cfg)
    b = build(synthetic_matches.sample(frac=1.0, random_state=7).reset_index(drop=True), cfg)
    assert_frames_identical(a, b, "shuffled input rows")


def test_stale_order_key_is_irrelevant(synthetic_matches: pd.DataFrame, cfg: dict) -> None:
    """CONTRACT section 2: the builder sorts by the 5 authoritative columns and never by order_key alone.

    Inputs routinely carry a stale order_key (a filtered or concatenated table, as in tests 1 and 8), so reversing
    the order_key column must not change the output.
    """
    a = build(synthetic_matches, cfg)
    stale = synthetic_matches.copy()
    stale["order_key"] = (len(stale) - 1 - stale["order_key"]).astype("int64")
    b = build(stale, cfg)
    assert_frames_identical(a, b, "reversed order_key column")


@pytest.mark.realdata
@pytest.mark.slow
@pytest.mark.parametrize("kind", ["date", "mid_event"])
def test_future_invariance_real(real_universe: pd.DataFrame, cfg: dict, kind: str) -> None:
    get_builder()
    cutoff = _date_cutoff(real_universe) if kind == "date" else _mid_event_cutoff(real_universe)
    _check_invariance(real_universe, cfg, cutoff)
