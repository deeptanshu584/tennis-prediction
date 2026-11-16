"""Unit tests for the rolling serve/return stats buffer (PLAN 5.5, CONTRACT section 2)."""

from __future__ import annotations

import math
import random

import pytest

from tennis.engine.rolling import RATE_FEATURES, STAT_FIELDS, StatsBuffer, rates_from_sums, valid_stats

NAN = float("nan")


def _stats(ace=5, df=2, svpt=60, i1=36, w1=27, w2=12, sv=10, bps=2, bpf=3) -> tuple[float, ...]:
    return tuple(float(x) for x in (ace, df, svpt, i1, w1, w2, sv, bps, bpf))


def test_stat_fields_order() -> None:
    assert STAT_FIELDS == ("ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced")
    assert len(RATE_FEATURES) == 13


def test_valid_stats_requires_all_18_and_positive_svpt() -> None:
    good = _stats()
    assert valid_stats(good, good)
    assert not valid_stats(good, _stats(svpt=0))
    assert not valid_stats(_stats(svpt=0), good)
    missing = list(good)
    missing[0] = NAN
    assert not valid_stats(tuple(missing), good)
    assert not valid_stats(good, tuple(missing))
    assert not valid_stats(good, (None,) * 9)


def test_empty_buffer_rates_are_nan() -> None:
    buf = StatsBuffer(maxlen=50)
    assert len(buf) == 0
    rates = buf.rates()
    assert set(rates) == set(RATE_FEATURES)
    assert all(math.isnan(v) for v in rates.values())


def test_rates_single_match_formulas() -> None:
    own = _stats(ace=5, df=2, svpt=60, i1=36, w1=27, w2=12, bps=2, bpf=3)
    opp = _stats(ace=8, df=4, svpt=70, i1=40, w1=30, w2=15, bps=4, bpf=6)
    r = rates_from_sums(own, opp)
    assert r["p_ace"] == 5 / 60
    assert r["p_df"] == 2 / 60
    assert r["p_1stIn"] == 36 / 60
    assert r["p_1stWon"] == 27 / 36
    assert r["p_2ndWon"] == 12 / 24
    assert r["p_bpSaved"] == 2 / 3
    assert r["p_rpw"] == (70 - 30 - 15) / 70
    assert r["p_retAceAgainst"] == 8 / 70
    assert r["p_ret1stWon"] == (40 - 30) / 40
    assert r["p_ret2ndWon"] == (70 - 40 - 15) / 30
    assert r["p_bpConv"] == (6 - 4) / 6
    assert r["p_totalPtsWon"] == (27 + 12 + 25) / (60 + 70)
    spw = (27 + 12) / 60
    assert r["dominance_ratio"] == pytest.approx(((70 - 30 - 15) / 70) / (1 - spw), abs=1e-15)


def test_zero_denominators_give_nan() -> None:
    own = _stats(i1=60, w1=40, w2=0, bps=0, bpf=0)   # no second serves, no break points faced
    opp = _stats(svpt=50, i1=50, w1=50, w2=0, bps=0, bpf=0)
    r = rates_from_sums(own, opp)
    assert math.isnan(r["p_2ndWon"])
    assert math.isnan(r["p_bpSaved"])
    assert math.isnan(r["p_ret2ndWon"])
    assert math.isnan(r["p_bpConv"])
    assert r["p_ace"] == 5 / 60


def test_dominance_nan_when_all_serve_points_won() -> None:
    own = _stats(svpt=40, i1=30, w1=30, w2=10)   # spw = 1 -> 1 - spw = 0
    r = rates_from_sums(own, _stats())
    assert math.isnan(r["dominance_ratio"])


def test_rates_are_sums_over_sums_not_mean_of_percentages() -> None:
    buf = StatsBuffer(maxlen=50)
    buf.push(_stats(ace=10, svpt=100, i1=60, w1=40, w2=20), _stats())
    buf.push(_stats(ace=0, svpt=10, i1=6, w1=4, w2=2), _stats())
    r = buf.rates()
    sums = (10 + 0) / (100 + 10)
    mean_of_pcts = (10 / 100 + 0 / 10) / 2
    assert r["p_ace"] == pytest.approx(sums, abs=1e-15)
    assert abs(sums - mean_of_pcts) > 0.01
    assert r["p_ace"] != pytest.approx(mean_of_pcts)


def test_buffer_evicts_oldest_and_sums_stay_exact() -> None:
    rng = random.Random(0)
    buf = StatsBuffer(maxlen=3)
    entries = []
    for _ in range(20):
        own = _stats(ace=rng.randint(0, 20), svpt=rng.randint(40, 120))
        opp = _stats(ace=rng.randint(0, 20), svpt=rng.randint(40, 120))
        buf.push(own, opp)
        entries.append((own, opp))
        last = entries[-3:]
        assert len(buf) == len(last)
        own_sum = tuple(sum(e[0][j] for e in last) for j in range(9))
        opp_sum = tuple(sum(e[1][j] for e in last) for j in range(9))
        assert tuple(buf.own_sums) == own_sum
        assert tuple(buf.opp_sums) == opp_sum
        brute = rates_from_sums(own_sum, opp_sum)
        got = buf.rates()
        for k in RATE_FEATURES:
            assert (math.isnan(got[k]) and math.isnan(brute[k])) or got[k] == brute[k], k


def test_buffer_maxlen_must_be_positive() -> None:
    with pytest.raises(ValueError):
        StatsBuffer(maxlen=0)
