"""Test 7 (PLAN section 9): Elo arithmetic.

2063 beats 2120 with K = 24 -> the winner gains +13.95 (+-0.01) and the loser loses the same (tennis.engine.elo).
The author's "vanilla" setting (div_number = 0) gives a fixed K = 24.
Engine replay: two fresh players, fixed K = 24 — the emitted rows show the ratings the formula predicts, so the
engine applies the update after (never before) writing the row.

Catches: a wrong expected-score formula (sign, scale 400, base 10), K applied to the wrong player, a pre-match
update (rating already includes the current result).
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.fixtures.synthetic import make_match_row, rows_to_frame
from tests.leakage.helpers import ELO_MODULE, ELO_SKIP, build, cfg_with, get_builder, import_or_skip


def _expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def test_alcaraz_beats_djokovic() -> None:
    elo = import_or_skip(ELO_MODULE, ELO_SKIP)
    e_a = elo.expected_score(2063.0, 2120.0)
    assert e_a == pytest.approx(_expected(2063.0, 2120.0), abs=1e-12)
    gain = elo.update_rating(2063.0, e_a, 1.0, 24.0) - 2063.0
    assert gain == pytest.approx(13.95, abs=0.01)
    e_b = elo.expected_score(2120.0, 2063.0)
    assert e_a + e_b == pytest.approx(1.0, abs=1e-12)
    loss = elo.update_rating(2120.0, e_b, 0.0, 24.0) - 2120.0
    assert loss == pytest.approx(-13.95, abs=0.01)


def test_vanilla_k_is_fixed() -> None:
    elo = import_or_skip(ELO_MODULE, ELO_SKIP)
    for ne in (0, 1, 10, 500):
        k = elo.k_factor(ne, 1.0, k_factor=24.0, div_number=0.0, base_k=0.0, max_k=0.0,
                         bonus_after_layoff=False)
        assert k == pytest.approx(24.0)
    # chess formula K = min(max_k, max(base_k, div / (Ne + m)))
    assert elo.k_factor(9, 1.0, k_factor=24.0, div_number=800.0, base_k=0.0, max_k=100.0,
                        bonus_after_layoff=False) == pytest.approx(80.0)
    assert elo.k_factor(0, 1.0, k_factor=24.0, div_number=800.0, base_k=0.0, max_k=100.0,
                        bonus_after_layoff=False) == pytest.approx(100.0)
    assert elo.k_factor(999, 1.0, k_factor=24.0, div_number=800.0, base_k=16.0, max_k=100.0,
                        bonus_after_layoff=False) == pytest.approx(16.0)


def test_engine_replays_the_formula(cfg: dict) -> None:
    get_builder()
    c = cfg_with(cfg, **{
        "elo.overall.div_number": 0.0, "elo.overall.k_factor": 24.0, "elo.overall.bonus_after_layoff": False,
        "elo.surface.div_number": 0.0, "elo.surface.k_factor": 24.0, "elo.surface.bonus_after_layoff": False,
        "data.feature_first_year": 2000,
    })
    x, y, z = "900001", "900002", "900003"
    rows = [
        make_match_row(tourney_id="2020-9001", tourney_date="2020-01-06", winner_id=x, loser_id=y, cfg=c),
        make_match_row(tourney_id="2020-9002", tourney_date="2020-01-13", winner_id=y, loser_id=z, cfg=c),
        make_match_row(tourney_id="2020-9003", tourney_date="2020-01-20", winner_id=z, loser_id=x,
                       surface="Clay", cfg=c),
    ]
    matches = rows_to_frame(rows)
    feats = build(matches, c).set_index("match_id")
    assert len(feats) == 3

    def elo_of(mid: str, pid: str, col: str) -> float:
        r = feats.loc[mid]
        return float(r[f"a_{col}"] if r["a_id"] == pid else r[f"b_{col}"])

    start = float(c["elo"]["start_rating"])
    m1, m2, m3 = matches["match_id"]
    # match 1: both fresh
    assert elo_of(m1, x, "elo") == start and elo_of(m1, y, "elo") == start
    # after match 1: x +12, y -12 (equal ratings, K = 24)
    rx, ry, rz = start + 12.0, start - 12.0, start
    assert elo_of(m2, y, "elo") == pytest.approx(ry, abs=1e-9)
    assert elo_of(m2, z, "elo") == pytest.approx(rz, abs=1e-9)
    # after match 2: y beats z
    e = _expected(ry, rz)
    ry, rz = ry + 24.0 * (1 - e), rz - 24.0 * (1 - e)
    assert elo_of(m3, z, "elo") == pytest.approx(rz, abs=1e-9)
    assert elo_of(m3, x, "elo") == pytest.approx(rx, abs=1e-9)
    # surface Elo: match 3 is the first Clay match for both -> start rating
    assert elo_of(m3, z, "elo_surface") == start and elo_of(m3, x, "elo_surface") == start
    r3 = feats.loc[m3]
    p_a = _expected(elo_of(m3, r3["a_id"], "elo"), elo_of(m3, r3["b_id"], "elo"))
    assert float(r3["elo_prob_a"]) == pytest.approx(p_a, abs=1e-12)
    assert isinstance(pd.Timestamp(r3["date"]), pd.Timestamp)
