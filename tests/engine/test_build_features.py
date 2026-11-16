"""Scenario tests for the chronological feature builder (CONTRACT section 2, PLAN section 2 and 5).

Small hand-written match tables; every expected value is computed by hand from the CONTRACT definitions.
"""

from __future__ import annotations

import copy
import hashlib
import math
from typing import Any

import pandas as pd
import pytest

from tennis.config import load_config
from tennis.engine.build_features import build_features, orient_a_is_winner, output_columns
from tennis.engine.elo import expected_score
from tests.fixtures.synthetic import STAT_FIELDS, make_match_row, rows_to_frame

START = 1500.0


# Unit tests pin the Elo settings they compute by hand (the pre-tuning defaults: K = 800/(Ne+m), cap 100, no
# floor, no layoff bonus), so they don't depend on the tuned values in config.yaml (DECISIONS D-027).
PINNED_ELO = {"k_factor": 24.0, "div_number": 800.0, "m": 1.0, "base_k": 0.0, "max_k": 100.0,
              "bonus_after_layoff": False}


@pytest.fixture
def cfg() -> dict[str, Any]:
    c = copy.deepcopy(load_config())
    c["data"]["feature_first_year"] = 2000
    for which in ("overall", "surface"):
        c["elo"][which] = dict(PINNED_ELO)
    return c


def _fixed_k(cfg: dict[str, Any], k: float = 24.0) -> dict[str, Any]:
    c = copy.deepcopy(cfg)
    for which in ("overall", "surface"):
        c["elo"][which]["div_number"] = 0.0
        c["elo"][which]["k_factor"] = k
        c["elo"][which]["bonus_after_layoff"] = False
    return c


def _row(cfg: dict[str, Any], **kw: Any) -> dict[str, Any]:
    kw.setdefault("tourney_date", "2020-01-06")
    return make_match_row(cfg=cfg, **kw)


def _val(feats: pd.DataFrame, mid: str, pid: str, col: str) -> Any:
    r = feats.set_index("match_id").loc[mid]
    return r[f"a_{col}"] if r["a_id"] == pid else r[f"b_{col}"]


def _set_stats(row: dict[str, Any], w: dict[str, float], l: dict[str, float]) -> dict[str, Any]:
    for s in STAT_FIELDS:
        row[f"w_{s}"] = float(w[s])
        row[f"l_{s}"] = float(l[s])
    return row


def _st(ace, df, svpt, i1, w1, w2, sv, bps, bpf) -> dict[str, float]:
    return dict(zip(STAT_FIELDS, (ace, df, svpt, i1, w1, w2, sv, bps, bpf)))


# ---------------------------------------------------------------- columns and orientation


def test_output_columns_generated_from_config(cfg: dict[str, Any]) -> None:
    cols = output_columns(cfg)
    for c in ("match_id", "date", "tourney_id", "tourney_name", "a_id", "b_id", "a_name", "b_name", "y",
              "surface_Hard", "surface_Clay", "surface_Grass", "round_rank", "best_of", "elo_prob_a",
              "elo_surface_prob_a", "a_winrate_last_5", "d_winrate_last_50", "a_p_totalPtsWon", "d_dominance_ratio"):
        assert c in cols, c
    for lv in cfg["data"]["training_levels"]:
        assert f"level_{lv}" in cols
    c2 = copy.deepcopy(cfg)
    c2["features"]["winrate_k"] = [3]
    c2["data"]["training_levels"] = ["G"]
    cols2 = output_columns(c2)
    assert "a_winrate_last_3" in cols2 and "a_winrate_last_5" not in cols2
    assert "level_G" in cols2 and "level_A" not in cols2
    assert len(cols) == len(set(cols))
    assert not any(c.startswith(("w_", "l_")) or "winner" in c or "loser" in c for c in cols)


def test_orientation_hash() -> None:
    for mid in ("abc", "0123456789abcdef", "ffff"):
        h = int(hashlib.sha256(f"42:{mid}".encode()).hexdigest()[:8], 16)
        assert orient_a_is_winner(42, mid) == (h % 2 == 0)


def test_y_follows_hash_and_a_is_winner_iff_y(cfg: dict[str, Any]) -> None:
    rows = [_row(cfg, tourney_id=f"2020-{i:04d}", winner_id=str(i), loser_id=str(100 + i)) for i in range(40)]
    m = rows_to_frame(rows)
    f = build_features(m, cfg)
    assert len(f) == 40
    winners = m.set_index("match_id")["winner_id"]
    for _, r in f.iterrows():
        assert r["y"] == int(orient_a_is_winner(cfg["seed"], r["match_id"]))
        assert (r["a_id"] == winners[r["match_id"]]) == (r["y"] == 1)
    assert 0 < f["y"].mean() < 1


def test_d_is_a_minus_b(cfg: dict[str, Any]) -> None:
    rows = [
        _row(cfg, tourney_id="2020-0001", winner_id="1", loser_id="2", stats=True),
        _row(cfg, tourney_id="2020-0002", tourney_date="2020-01-13", winner_id="1", loser_id="3", stats=True),
        _row(cfg, tourney_id="2020-0003", tourney_date="2020-01-20", winner_id="3", loser_id="1", stats=True),
    ]
    f = build_features(rows_to_frame(rows), cfg)
    for c in f.columns:
        if c.startswith("d_"):
            a, b, d = f["a_" + c[2:]].astype(float), f["b_" + c[2:]].astype(float), f[c].astype(float)
            ok = (d == a - b) | (d.isna() & (a.isna() | b.isna()))
            assert ok.all(), c


# ---------------------------------------------------------------- snapshot before update


def test_first_match_snapshot_is_fresh_state(cfg: dict[str, Any]) -> None:
    m = rows_to_frame([_row(cfg, tourney_id="2020-0001", winner_id="1", loser_id="2", stats=True)])
    f = build_features(m, cfg)
    mid = m["match_id"].iloc[0]
    for pid in ("1", "2"):
        assert _val(f, mid, pid, "matches_played") == 0
        assert _val(f, mid, pid, "matches_surface_played") == 0
        assert _val(f, mid, pid, "elo") == START
        assert _val(f, mid, pid, "elo_surface") == START
        assert math.isnan(_val(f, mid, pid, "elo_grad"))
        assert _val(f, mid, pid, "n_stat_matches") == 0
        assert _val(f, mid, pid, "h2h_wins") == 0
        assert math.isnan(_val(f, mid, pid, "days_since_last_match"))
        assert _val(f, mid, pid, "matches_last_14_days") == 0
        for k in cfg["features"]["winrate_k"]:
            assert math.isnan(_val(f, mid, pid, f"winrate_last_{k}"))
        for c in ("p_ace", "p_rpw", "p_totalPtsWon", "dominance_ratio"):
            assert math.isnan(_val(f, mid, pid, c))
    assert f["elo_prob_a"].iloc[0] == 0.5


def test_second_match_sees_only_the_first(cfg: dict[str, Any]) -> None:
    c = _fixed_k(cfg)
    w = _st(10, 2, 80, 50, 40, 18, 12, 3, 4)
    l = _st(4, 5, 70, 40, 25, 12, 11, 5, 9)
    r1 = _set_stats(_row(c, tourney_id="2020-0001", winner_id="1", loser_id="2"), w, l)
    r2 = _row(c, tourney_id="2020-0002", tourney_date="2020-01-13", winner_id="2", loser_id="1", stats=True)
    m = rows_to_frame([r1, r2])
    f = build_features(m, c)
    mid2 = m["match_id"].iloc[1]
    assert _val(f, mid2, "1", "elo") == START + 12.0
    assert _val(f, mid2, "2", "elo") == START - 12.0
    assert _val(f, mid2, "1", "matches_played") == 1
    assert _val(f, mid2, "1", "winrate_last_5") == 1.0
    assert _val(f, mid2, "2", "winrate_last_5") == 0.0
    assert _val(f, mid2, "1", "h2h_wins") == 1
    assert _val(f, mid2, "2", "h2h_wins") == 0
    assert _val(f, mid2, "1", "days_since_last_match") == 7.0
    assert _val(f, mid2, "1", "matches_last_14_days") == 1
    assert _val(f, mid2, "1", "n_stat_matches") == 1
    # player 1's own serve = w_ stats of match 1, opponent = l_ stats
    assert _val(f, mid2, "1", "p_ace") == 10 / 80
    assert _val(f, mid2, "2", "p_ace") == 4 / 70
    assert _val(f, mid2, "1", "p_rpw") == (70 - 25 - 12) / 70
    assert _val(f, mid2, "2", "p_bpConv") == (4 - 3) / 4
    assert _val(f, mid2, "1", "elo_grad") == pytest.approx(12.0, abs=1e-12)
    assert _val(f, mid2, "2", "elo_grad") == pytest.approx(-12.0, abs=1e-12)


def test_static_attributes_from_row(cfg: dict[str, Any]) -> None:
    r = _row(cfg, tourney_id="2020-0001", winner_id="1", loser_id="2")
    r.update(winner_rank=5.0, loser_rank=float("nan"), winner_rank_points=4000.0, loser_rank_points=10.0,
             winner_age=24.5, loser_age=31.0, winner_ht=190.0, loser_ht=float("nan"),
             winner_hand="L", loser_hand="U")
    m = rows_to_frame([r])
    f = build_features(m, cfg)
    mid = m["match_id"].iloc[0]
    assert _val(f, mid, "1", "rank") == 5.0 and math.isnan(_val(f, mid, "2", "rank"))
    assert _val(f, mid, "1", "rank_points") == 4000.0
    assert _val(f, mid, "2", "age") == 31.0
    fill = cfg["features"]["static_fill"]
    # D-020: missing height / hand / age are filled with config constants (no missingness signal)
    assert _val(f, mid, "1", "height") == 190.0 and _val(f, mid, "2", "height") == fill["height"]
    assert _val(f, mid, "1", "is_left") == 1.0 and _val(f, mid, "2", "is_left") == fill["is_left"]
    assert math.isnan(f["d_rank"].iloc[0])           # rank missingness is real pre-match information
    assert f["d_height"].iloc[0] == pytest.approx(
        (190.0 - fill["height"]) * (1 if f["a_id"].iloc[0] == "1" else -1))


def test_static_fill_age_and_never_nan(cfg: dict[str, Any]) -> None:
    c = copy.deepcopy(cfg)
    c["features"]["static_fill"] = {"height": 180.0, "is_left": 0.5, "age": 30.0}
    r = _row(c, tourney_id="2020-0001", winner_id="1", loser_id="2")
    r.update(winner_age=float("nan"), loser_age=22.0, winner_ht=float("nan"), loser_ht=float("nan"),
             winner_hand=None, loser_hand="A")
    m = rows_to_frame([r])
    f = build_features(m, c)
    mid = m["match_id"].iloc[0]
    assert _val(f, mid, "1", "age") == 30.0 and _val(f, mid, "2", "age") == 22.0
    assert _val(f, mid, "1", "height") == 180.0 and _val(f, mid, "2", "height") == 180.0
    assert _val(f, mid, "1", "is_left") == 0.5 and _val(f, mid, "2", "is_left") == 0.5
    for x in ("height", "is_left", "age"):
        assert not f[[f"a_{x}", f"b_{x}", f"d_{x}"]].isna().any().any()


# ---------------------------------------------------------------- what is emitted / what updates state


def test_walkovers_are_dropped_completely(cfg: dict[str, Any]) -> None:
    rows = [
        _row(cfg, tourney_id="2020-0001", winner_id="1", loser_id="2", score="W/O"),
        _row(cfg, tourney_id="2020-0002", tourney_date="2020-01-13", winner_id="1", loser_id="2"),
    ]
    m = rows_to_frame(rows)
    f, k_log = build_features(m, cfg, record_k=True)
    assert f["match_id"].tolist() == [m["match_id"].iloc[1]]
    mid = m["match_id"].iloc[1]
    assert _val(f, mid, "1", "matches_played") == 0
    assert _val(f, mid, "1", "elo") == START
    assert set(k_log["match_id"]) == {mid}


def test_qualifying_updates_state_but_emits_no_row(cfg: dict[str, Any]) -> None:
    c = _fixed_k(cfg)
    rows = [
        _row(c, tourney_id="2020-0001", winner_id="1", loser_id="2", round_="Q1", match_num=300),
        _row(c, tourney_id="2020-0001", winner_id="1", loser_id="3", round_="R32", match_num=1),
    ]
    m = rows_to_frame(rows)
    f = build_features(m, c)
    assert f["match_id"].tolist() == [m["match_id"].iloc[1]]
    mid = m["match_id"].iloc[1]
    assert _val(f, mid, "1", "matches_played") == 1
    assert _val(f, mid, "1", "elo") == START + 12.0
    assert _val(f, mid, "1", "days_since_last_match") == 0.0


def test_challenger_updates_state_but_emits_no_row(cfg: dict[str, Any]) -> None:
    rows = [
        _row(cfg, tourney_id="2020-0001", winner_id="1", loser_id="2", level="C"),
        _row(cfg, tourney_id="2020-0002", tourney_date="2020-01-13", winner_id="2", loser_id="1"),
    ]
    m = rows_to_frame(rows)
    f = build_features(m, cfg)
    assert len(f) == 1
    assert _val(f, m["match_id"].iloc[1], "1", "matches_played") == 1


def test_rows_before_feature_first_year_update_state_only(cfg: dict[str, Any]) -> None:
    c = copy.deepcopy(cfg)
    c["data"]["feature_first_year"] = 2020
    rows = [
        _row(c, tourney_id="2019-0001", tourney_date="2019-12-30", winner_id="1", loser_id="2"),
        _row(c, tourney_id="2020-0002", tourney_date="2020-01-06", winner_id="1", loser_id="2"),
    ]
    m = rows_to_frame(rows)
    f = build_features(m, c)
    assert f["match_id"].tolist() == [m["match_id"].iloc[1]]
    assert _val(f, m["match_id"].iloc[1], "1", "h2h_wins") == 1


def _ret_scenario(c: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stats = _st(3, 1, 30, 20, 15, 5, 5, 1, 2)
    rows = [
        _set_stats(_row(c, tourney_id="2020-0001", winner_id="1", loser_id="2", score="6-4 2-1 RET"),
                   stats, stats),
        _row(c, tourney_id="2020-0002", tourney_date="2020-01-13", winner_id="2", loser_id="1"),
    ]
    m = rows_to_frame(rows)
    f, k_log = build_features(m, c, record_k=True)
    return m, f, k_log


def test_retirement_default_flags(cfg: dict[str, Any]) -> None:
    assert cfg["cleaning"]["retirement_updates_elo"] is False
    m, f, k_log = _ret_scenario(cfg)
    ret_mid, mid = m["match_id"].tolist()
    assert f["match_id"].tolist() == [mid]               # RET never a label
    assert _val(f, mid, "1", "elo") == START              # no Elo move
    assert math.isnan(_val(f, mid, "1", "elo_grad"))      # no history entry
    assert _val(f, mid, "1", "matches_played") == 1       # counts still increase
    assert _val(f, mid, "1", "matches_surface_played") == 1
    assert _val(f, mid, "1", "n_stat_matches") == 1       # retirement_updates_stats: true
    assert _val(f, mid, "1", "h2h_wins") == 0             # retirement_updates_h2h: false
    assert math.isnan(_val(f, mid, "1", "winrate_last_5"))  # retirement_updates_form: false
    assert _val(f, mid, "1", "days_since_last_match") == 7.0
    # k_log has the RET match with the K that would have applied (debut: 800 / (0 + 1) capped at 100)
    kr = k_log[k_log["match_id"] == ret_mid]
    assert len(kr) == 2 and (kr["k_overall"] == 100.0).all() and (kr["k_surface"] == 100.0).all()


def test_retirement_flags_flipped(cfg: dict[str, Any]) -> None:
    c = _fixed_k(cfg)
    for flag in ("retirement_updates_elo", "retirement_updates_h2h", "retirement_updates_form"):
        c["cleaning"][flag] = True
    c["cleaning"]["retirement_updates_stats"] = False
    m, f, _ = _ret_scenario(c)
    mid = m["match_id"].iloc[1]
    assert _val(f, mid, "1", "elo") == START + 12.0
    assert _val(f, mid, "1", "n_stat_matches") == 0
    assert _val(f, mid, "1", "h2h_wins") == 1
    assert _val(f, mid, "1", "winrate_last_5") == 1.0


# ---------------------------------------------------------------- Elo specifics


def test_engine_elo_13_95_after_warmup(cfg: dict[str, Any]) -> None:
    """Fixed K = 24 replay: the update applied after match 1 is exactly the PLAN 5.2 formula."""
    c = _fixed_k(cfg)
    rows = [
        _row(c, tourney_id="2020-0001", winner_id="1", loser_id="2"),
        _row(c, tourney_id="2020-0002", tourney_date="2020-01-13", winner_id="1", loser_id="2"),
        _row(c, tourney_id="2020-0003", tourney_date="2020-01-20", winner_id="2", loser_id="1"),
    ]
    m = rows_to_frame(rows)
    f = build_features(m, c)
    m2, m3 = m["match_id"].iloc[1], m["match_id"].iloc[2]
    r1, r2 = _val(f, m2, "1", "elo"), _val(f, m2, "2", "elo")
    e = expected_score(r1, r2)
    assert _val(f, m3, "1", "elo") == pytest.approx(r1 + 24 * (1 - e), abs=1e-12)
    assert _val(f, m3, "2", "elo") == pytest.approx(r2 - 24 * (1 - e), abs=1e-12)
    # and the canonical number
    e_a = expected_score(2063.0, 2120.0)
    assert 24 * (1 - e_a) == pytest.approx(13.95, abs=0.01)


def test_surface_elo_uses_only_same_surface_matches(cfg: dict[str, Any]) -> None:
    c = _fixed_k(cfg)
    rows = [
        _row(c, tourney_id="2020-0001", winner_id="1", loser_id="2", surface="Clay"),
        _row(c, tourney_id="2020-0002", tourney_date="2020-01-13", winner_id="1", loser_id="2", surface="Hard"),
        _row(c, tourney_id="2020-0003", tourney_date="2020-01-20", winner_id="2", loser_id="1", surface="Clay"),
    ]
    m = rows_to_frame(rows)
    f = build_features(m, c)
    m2, m3 = m["match_id"].iloc[1], m["match_id"].iloc[2]
    # match 2 (Hard): first Hard match -> surface Elo at start, overall moved by the Clay win
    assert _val(f, m2, "1", "elo_surface") == START
    assert _val(f, m2, "1", "elo") == START + 12.0
    assert _val(f, m2, "1", "matches_surface_played") == 0
    # match 3 (Clay): surface Elo reflects only the Clay match 1
    assert _val(f, m3, "1", "elo_surface") == START + 12.0
    assert _val(f, m3, "2", "elo_surface") == START - 12.0
    assert _val(f, m3, "1", "matches_surface_played") == 1
    assert _val(f, m3, "1", "matches_played") == 2
    r = f.set_index("match_id").loc[m3]
    a_s = r["a_elo_surface"]
    b_s = r["b_elo_surface"]
    assert r["elo_surface_prob_a"] == pytest.approx(expected_score(a_s, b_s), abs=1e-15)


def test_m_counts_prior_matches_in_the_same_tourney_id(cfg: dict[str, Any]) -> None:
    """Dynamic K, m_mode=event_matches_so_far, m=1: K = 800 / (Ne + 1 + event matches so far)."""
    c = copy.deepcopy(cfg)
    c["elo"]["overall"]["max_k"] = 1000.0
    c["elo"]["surface"]["max_k"] = 1000.0
    rows = [
        _row(c, tourney_id="2020-0001", winner_id="1", loser_id="9", round_="R32", match_num=1),
        _row(c, tourney_id="2020-0002", tourney_date="2020-01-13", winner_id="1", loser_id="2", round_="Q1",
             match_num=300),
        _row(c, tourney_id="2020-0002", tourney_date="2020-01-13", winner_id="1", loser_id="3", round_="Q2",
             match_num=301),
        _row(c, tourney_id="2020-0002", tourney_date="2020-01-13", winner_id="1", loser_id="4", round_="R32",
             match_num=1),
    ]
    m = rows_to_frame(rows)
    _, k_log = build_features(m, c, record_k=True)
    k1 = k_log[k_log["player_id"] == "1"].set_index("match_id")
    ids = m["match_id"].tolist()
    assert k1.loc[ids[0], "k_overall"] == 800 / (0 + 1 + 0)
    assert k1.loc[ids[1], "k_overall"] == 800 / (1 + 1 + 0)
    assert k1.loc[ids[2], "k_overall"] == 800 / (2 + 1 + 1)
    assert k1.loc[ids[3], "k_overall"] == 800 / (3 + 1 + 2)   # qualifying counts
    assert k1.loc[ids[3], "k_surface"] == 800 / (3 + 1 + 2)   # same event count for surface
    c2 = copy.deepcopy(c)
    c2["elo"]["m_mode"] = "constant"
    _, k_log2 = build_features(m, c2, record_k=True)
    k2 = k_log2[k_log2["player_id"] == "1"].set_index("match_id")
    assert k2.loc[ids[3], "k_overall"] == 800 / (3 + 1)


def test_k_log_one_row_per_processed_match_per_player(cfg: dict[str, Any]) -> None:
    rows = [
        _row(cfg, tourney_id="2020-0001", winner_id="1", loser_id="2"),
        _row(cfg, tourney_id="2020-0001", winner_id="3", loser_id="4", match_num=2, score="6-1 RET"),
        _row(cfg, tourney_id="2020-0001", winner_id="5", loser_id="6", match_num=3, score="W/O"),
        _row(cfg, tourney_id="2020-0001", winner_id="7", loser_id="8", match_num=4, round_="Q1", level="A"),
    ]
    m = rows_to_frame(rows)
    _, k_log = build_features(m, cfg, record_k=True)
    assert list(k_log.columns) == ["match_id", "player_id", "k_overall", "k_surface"]
    assert len(k_log) == 6
    assert set(k_log["player_id"]) == {"1", "2", "3", "4", "7", "8"}


def test_layoff_bonus_uses_one_entry_per_tourney(cfg: dict[str, Any]) -> None:
    """With the layoff bonus on, K after a long break exceeds K without it; qualifying + main draw of the same
    tourney_id count once in the played list."""
    c = copy.deepcopy(cfg)
    c["elo"]["m_mode"] = "constant"
    for which in ("overall", "surface"):
        c["elo"][which].update(div_number=0.0, k_factor=24.0, bonus_after_layoff=True)

    def scenario(n_tourneys: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        rows = []
        date = pd.Timestamp("2018-01-01")
        for i in range(n_tourneys):  # weekly tourneys, each with a Q1 and an R32 for player 1
            tid = f"2018-{i:04d}"
            rows.append(_row(c, tourney_id=tid, tourney_date=date, winner_id="1", loser_id=f"q{i}", round_="Q1",
                             match_num=300))
            rows.append(_row(c, tourney_id=tid, tourney_date=date, winner_id="1", loser_id=f"r{i}"))
            date += pd.Timedelta(days=7)
        comeback = date + pd.Timedelta(days=200)
        rows.append(_row(c, tourney_id="2018-9999", tourney_date=comeback, winner_id="1", loser_id="z"))
        m = rows_to_frame(rows)
        _, k_log = build_features(m, c, record_k=True)
        return m, k_log[k_log["player_id"] == "1"].set_index("match_id")

    # 20 tourneys + comeback: 21 entries, 20 gaps (> 15), most recent gap 207 days -> bonus 1.5
    m, k1 = scenario(20)
    assert k1.loc[m["match_id"].iloc[-1], "k_overall"] == pytest.approx(24.0 * 1.5)
    # R32 of the 20th tourney: 20 entries, 19 short gaps -> 1.5 - 19 * 0.05 floored at 1.0
    assert k1.loc[m["match_id"].iloc[-2], "k_overall"] == pytest.approx(24.0)
    assert k1.loc[m["match_id"].iloc[1], "k_overall"] == pytest.approx(24.0)
    # 9 tourneys + comeback: 10 entries -> 9 gaps (<= 15) -> no bonus. Counting Q1 and R32 separately would
    # give 18 + 1 entries, 18 gaps and a 1.5 bonus.
    m, k1 = scenario(9)
    assert k1.loc[m["match_id"].iloc[-1], "k_overall"] == pytest.approx(24.0)


def test_does_not_mutate_input_and_ignores_row_order(cfg: dict[str, Any]) -> None:
    rows = [
        _row(cfg, tourney_id="2020-0001", winner_id="1", loser_id="2", round_="SF", match_num=1),
        _row(cfg, tourney_id="2020-0001", winner_id="1", loser_id="3", round_="F", match_num=2),
        _row(cfg, tourney_id="2020-0001", winner_id="3", loser_id="4", round_="SF", match_num=0),
    ]
    m = rows_to_frame(rows)
    before = m.copy(deep=True)
    f1 = build_features(m, cfg)
    pd.testing.assert_frame_equal(m, before)
    f2 = build_features(m.iloc[::-1].reset_index(drop=True), cfg)
    pd.testing.assert_frame_equal(f1.sort_values("match_id").reset_index(drop=True),
                                  f2.sort_values("match_id").reset_index(drop=True))
    fid = m["match_id"].iloc[1]
    assert _val(f1, fid, "1", "matches_played") == 1
    assert _val(f1, fid, "3", "matches_played") == 1
