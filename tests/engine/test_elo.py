"""Unit tests for the pure Elo module (PLAN 5.2-5.4, CONTRACT section 3). Synthetic numbers only."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from tennis.config import load_config
from tennis.engine.elo import (
    EloParams,
    expected_score,
    gaps_between_tourneys,
    k_bonus_after_layoff,
    k_factor,
    update_rating,
)

VANILLA = dict(k_factor=24.0, div_number=0.0, base_k=0.0, max_k=0.0, bonus_after_layoff=False)
DYNAMIC = dict(k_factor=24.0, div_number=800.0, base_k=0.0, max_k=100.0, bonus_after_layoff=False)


def _dates_from_gaps(gaps_most_recent_first: list[int], end: dt.date = dt.date(2024, 6, 1)) -> list[dt.date]:
    """Build a chronological list of dates whose gaps (most recent first) are exactly the given ones."""
    dates = [end]
    for g in gaps_most_recent_first:
        dates.append(dates[-1] - dt.timedelta(days=g))
    return dates[::-1]


# ---------------------------------------------------------------- expected score / update (PLAN 5.2)


def test_alcaraz_beats_djokovic_gain_is_13_95() -> None:
    e_a = expected_score(2063.0, 2120.0)
    new_a = update_rating(2063.0, e_a, 1.0, 24.0)
    assert new_a - 2063.0 == pytest.approx(13.95, abs=0.01)


def test_djokovic_loses_same_amount_with_equal_k() -> None:
    e_a = expected_score(2063.0, 2120.0)
    e_b = expected_score(2120.0, 2063.0)
    gain = update_rating(2063.0, e_a, 1.0, 24.0) - 2063.0
    loss = update_rating(2120.0, e_b, 0.0, 24.0) - 2120.0
    assert loss == pytest.approx(-gain, abs=1e-9)
    assert loss == pytest.approx(-13.95, abs=0.01)


@pytest.mark.parametrize("r_a, r_b", [(1500, 1500), (2063, 2120), (1200, 2400), (1800.5, 1650.25)])
def test_expected_score_symmetry(r_a: float, r_b: float) -> None:
    assert expected_score(r_a, r_b) + expected_score(r_b, r_a) == pytest.approx(1.0, abs=1e-12)


def test_expected_score_equal_ratings_is_half() -> None:
    assert expected_score(1500.0, 1500.0) == 0.5
    assert expected_score(2222.0, 2222.0) == 0.5


def test_expected_score_400_points_is_ten_to_one() -> None:
    assert expected_score(1900.0, 1500.0) == pytest.approx(10 / 11)


def test_update_rating_loss_and_zero_k() -> None:
    assert update_rating(1500.0, 0.5, 0.0, 32.0) == pytest.approx(1484.0)
    assert update_rating(1500.0, 0.3, 1.0, 0.0) == 1500.0


# ---------------------------------------------------------------- dynamic K (PLAN 5.3)


@pytest.mark.parametrize("ne", [0, 1, 5, 99, 10_000])
def test_vanilla_config_gives_fixed_k_24(ne: int) -> None:
    assert k_factor(ne, 1.0, **VANILLA) == 24.0


def test_vanilla_ignores_m_even_if_nonpositive() -> None:
    # div_number == 0 short-circuits before the Ne + m guard.
    assert k_factor(0, 0.0, **VANILLA) == 24.0


def test_dynamic_k_new_player_is_capped_by_max_k() -> None:
    assert k_factor(0, 1.0, **DYNAMIC) == 100.0  # min(100, 800 / 1)
    assert k_factor(0, 1.0, **{**DYNAMIC, "max_k": 1000.0}) == 800.0


def test_dynamic_k_after_99_matches_is_8() -> None:
    assert k_factor(99, 1.0, **DYNAMIC) == pytest.approx(8.0)


def test_dynamic_k_base_k_floor() -> None:
    # 800 / (999 + 1) = 0.8 -> floored to base_k.
    assert k_factor(999, 1.0, **{**DYNAMIC, "base_k": 16.0}) == 16.0
    assert k_factor(999, 1.0, **DYNAMIC) == pytest.approx(0.8)


def test_dynamic_k_max_k_cap() -> None:
    assert k_factor(4, 1.0, **{**DYNAMIC, "max_k": 32.0}) == 32.0  # 800/5 = 160 -> 32
    assert k_factor(49, 1.0, **{**DYNAMIC, "max_k": 32.0}) == 16.0  # 800/50 = 16 under the cap


def test_dynamic_k_uses_m() -> None:
    assert k_factor(5, 5.0, **DYNAMIC) == pytest.approx(80.0)  # 800 / 10


def test_dynamic_k_max_k_zero_forces_zero() -> None:
    # Documented author quirk: max_k == 0 with div_number != 0 makes K = 0. Not "fixed" silently.
    assert k_factor(10, 1.0, **{**DYNAMIC, "max_k": 0.0}) == 0.0


@pytest.mark.parametrize("ne, m", [(0, 0.0), (0, -1.0), (2, -5.0)])
def test_dynamic_k_rejects_nonpositive_denominator(ne: int, m: float) -> None:
    with pytest.raises(ValueError, match="ne \\+ m"):
        k_factor(ne, m, **DYNAMIC)


def test_k_factor_applies_layoff_bonus_when_enabled() -> None:
    played = _dates_from_gaps([200] + [10] * 15)  # 16 gaps, most recent long -> 1.5
    on = {**VANILLA, "bonus_after_layoff": True}
    assert k_factor(50, 1.0, **on, played=played) == pytest.approx(36.0)
    assert k_factor(50, 1.0, **VANILLA, played=played) == 24.0  # flag off -> no bonus
    dyn_on = {**DYNAMIC, "bonus_after_layoff": True}
    assert k_factor(99, 1.0, **dyn_on, played=played) == pytest.approx(12.0)  # 8 * 1.5


def test_k_factor_passes_layoff_params() -> None:
    played = _dates_from_gaps([60] + [10] * 15)  # 60 < default threshold 100, but >= 50
    on = {**VANILLA, "bonus_after_layoff": True}
    assert k_factor(5, 1.0, **on, played=played) == pytest.approx(24.0)  # all short -> floor 1.0
    layoff = dict(threshold=50, min_gaps=15, start_bonus=1.5, step=0.05, floor=1.0)
    assert k_factor(5, 1.0, **on, played=played, layoff=layoff) == pytest.approx(36.0)


def test_k_factor_bonus_with_too_few_tourneys_is_neutral() -> None:
    on = {**VANILLA, "bonus_after_layoff": True}
    assert k_factor(0, 1.0, **on) == 24.0  # no played dates
    assert k_factor(0, 1.0, **on, played=_dates_from_gaps([300] * 3)) == 24.0


# ---------------------------------------------------------------- gaps


def test_gaps_most_recent_first() -> None:
    played = [dt.date(2024, 1, 1), dt.date(2024, 1, 15), dt.date(2024, 3, 1)]
    assert gaps_between_tourneys(played) == [46, 14]


def test_gaps_unsorted_input_is_sorted() -> None:
    played = [dt.date(2024, 3, 1), dt.date(2024, 1, 1), dt.date(2024, 1, 15)]
    assert gaps_between_tourneys(played) == [46, 14]


def test_gaps_empty_and_single() -> None:
    assert gaps_between_tourneys([]) == []
    assert gaps_between_tourneys([dt.date(2024, 1, 1)]) == []


@pytest.mark.parametrize(
    "played",
    [
        [dt.date(2023, 12, 25), dt.date(2024, 1, 1), dt.date(2024, 3, 1)],
        [dt.datetime(2023, 12, 25), dt.datetime(2024, 1, 1, 18, 30), dt.datetime(2024, 3, 1)],
        [np.datetime64("2023-12-25"), np.datetime64("2024-01-01"), np.datetime64("2024-03-01")],
        [np.datetime64("2023-12-25T00:00:00.000000000"), np.datetime64("2024-01-01T00:00:00.000000000"),
         np.datetime64("2024-03-01T00:00:00.000000000")],
        [pd.Timestamp("2023-12-25"), pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-01")],
        [20231225, 20240101, 20240301],
        [np.int64(20231225), np.int64(20240101), np.int64(20240301)],
        pd.to_datetime(pd.Series(["2023-12-25", "2024-01-01", "2024-03-01"])).values,  # datetime64[ns] ndarray
        pd.DatetimeIndex(["2023-12-25", "2024-01-01", "2024-03-01"]),
    ],
)
def test_gaps_date_type_variants(played: object) -> None:
    assert gaps_between_tourneys(played) == [60, 7]  # type: ignore[arg-type]


def test_gaps_across_leap_day() -> None:
    assert gaps_between_tourneys([20240228, 20240301]) == [2]


def test_gaps_reject_bad_input() -> None:
    with pytest.raises((TypeError, ValueError)):
        gaps_between_tourneys(["not a date", "nor this"])  # type: ignore[list-item]
    with pytest.raises((TypeError, ValueError)):
        gaps_between_tourneys([True, False])  # type: ignore[list-item]


# ---------------------------------------------------------------- layoff bonus (PLAN 5.4)


@pytest.mark.parametrize("n_gaps", [0, 1, 14, 15])
def test_bonus_with_at_most_15_gaps_is_one(n_gaps: int) -> None:
    played = _dates_from_gaps([365] * n_gaps)  # even huge layoffs do not count yet
    assert k_bonus_after_layoff(played) == 1.0


def test_bonus_long_most_recent_gap_is_start_bonus() -> None:
    played = _dates_from_gaps([150] + [7] * 15)  # 16 gaps
    assert k_bonus_after_layoff(played) == pytest.approx(1.5)


def test_bonus_exactly_threshold_counts_as_long() -> None:
    played = _dates_from_gaps([100] + [7] * 15)  # g < threshold is short, so 100 is long
    assert k_bonus_after_layoff(played) == pytest.approx(1.5)


def test_bonus_many_short_gaps_hits_floor() -> None:
    played = _dates_from_gaps([7] * 16)
    assert k_bonus_after_layoff(played) == 1.0
    played = _dates_from_gaps([7] * 40)
    assert k_bonus_after_layoff(played) == 1.0


def test_bonus_three_short_then_long_is_1_35() -> None:
    played = _dates_from_gaps([7, 14, 21, 200] + [7] * 12)  # 16 gaps
    assert k_bonus_after_layoff(played) == pytest.approx(1.35)


def test_bonus_old_long_gaps_after_ten_short_ones_reach_floor() -> None:
    played = _dates_from_gaps([7] * 10 + [300] + [7] * 5)  # 1.5 - 10 * 0.05 = 1.0
    assert k_bonus_after_layoff(played) == pytest.approx(1.0)


def test_bonus_unsorted_input_is_sorted() -> None:
    played = _dates_from_gaps([7, 14, 21, 200] + [7] * 12)
    shuffled = played[::2] + played[1::2]
    assert shuffled != played
    assert k_bonus_after_layoff(shuffled) == pytest.approx(1.35)


def test_bonus_accepts_int_dates() -> None:
    played = [int(d.strftime("%Y%m%d")) for d in _dates_from_gaps([7, 14, 21, 200] + [7] * 12)]
    assert k_bonus_after_layoff(played) == pytest.approx(1.35)


def test_bonus_custom_parameters() -> None:
    played = _dates_from_gaps([7, 7, 200] + [7] * 3)  # 6 gaps
    assert k_bonus_after_layoff(played) == 1.0  # default min_gaps 15
    assert k_bonus_after_layoff(played, min_gaps=5) == pytest.approx(1.4)
    assert k_bonus_after_layoff(played, min_gaps=5, start_bonus=2.0, step=0.1) == pytest.approx(1.8)
    assert k_bonus_after_layoff(played, min_gaps=5, threshold=5) == pytest.approx(1.5)
    assert k_bonus_after_layoff(_dates_from_gaps([7] * 20), min_gaps=5, floor=0.9) == pytest.approx(0.9)


# ---------------------------------------------------------------- EloParams


@pytest.mark.parametrize("which", ["overall", "surface"])
def test_elo_params_from_real_config(which: str) -> None:
    cfg = load_config()
    p = EloParams.from_config(cfg, which)
    sec = cfg["elo"][which]
    lay = cfg["elo"]["layoff"]
    assert p.k_factor == sec["k_factor"]
    assert p.div_number == sec["div_number"]
    assert p.m == sec["m"]
    assert p.base_k == sec["base_k"]
    assert p.max_k == sec["max_k"]
    assert p.bonus_after_layoff == sec["bonus_after_layoff"]
    assert p.start_rating == cfg["elo"]["start_rating"]
    assert p.layoff == {
        "threshold": lay["threshold_days"],
        "min_gaps": lay["min_gaps"],
        "start_bonus": lay["start_bonus"],
        "step": lay["step"],
        "floor": lay["floor"],
    }


def test_elo_params_k_matches_k_factor() -> None:
    cfg = load_config()
    p = EloParams.from_config(cfg, "overall")
    expected = k_factor(
        10, p.m, k_factor=p.k_factor, div_number=p.div_number, base_k=p.base_k, max_k=p.max_k,
        bonus_after_layoff=p.bonus_after_layoff, played=(), layoff=p.layoff,
    )
    assert p.k(10) == expected
    assert p.k(10, m=5.0) == k_factor(
        10, 5.0, k_factor=p.k_factor, div_number=p.div_number, base_k=p.base_k, max_k=p.max_k,
        bonus_after_layoff=p.bonus_after_layoff, played=(), layoff=p.layoff,
    )


def test_elo_params_vanilla_from_synthetic_config() -> None:
    cfg = {
        "elo": {
            "start_rating": 1500.0,
            "overall": {**VANILLA, "m": 1.0, "bonus_after_layoff": True},
            "layoff": {"threshold_days": 100, "min_gaps": 15, "start_bonus": 1.5, "step": 0.05, "floor": 1.0},
        }
    }
    p = EloParams.from_config(cfg, "overall")
    assert p.k(0) == 24.0
    assert p.k(0, played=_dates_from_gaps([7, 14, 21, 200] + [7] * 12)) == pytest.approx(24.0 * 1.35)


def test_elo_params_rejects_unknown_section() -> None:
    with pytest.raises(ValueError):
        EloParams.from_config(load_config(), "clay")  # type: ignore[arg-type]
