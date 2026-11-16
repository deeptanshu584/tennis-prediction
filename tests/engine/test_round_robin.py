"""Round-robin blocks (CONTRACT section 2 "Round-robin blocks", DECISIONS D-019).

Inside an RR round, match_num is not chronological, so every match of a block (consecutive matches with the same
tourney_id, round RR, level not exempt) is snapshotted, and gets its K and expected score, from the state BEFORE
the block. The block's updates are then applied in order with deltas K * (S - E) from that pre-block snapshot.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import pandas as pd
import pytest

from tennis.config import load_config
from tennis.engine.build_features import build_features
from tennis.engine.elo import expected_score
from tests.fixtures.synthetic import make_match_row, rows_to_frame

RR_TID = "2020-0605"
RR_DATE = "2020-11-16"
GROUP = [("1", "2"), ("3", "4"), ("1", "3"), ("2", "4"), ("1", "4"), ("2", "3")]   # all six meetings
PRE_COUNTS = {"1": 2, "2": 1, "3": 2, "4": 1}   # matches each player played before the block (warm-up)


@pytest.fixture
def cfg() -> dict[str, Any]:
    c = copy.deepcopy(load_config())
    c["data"]["feature_first_year"] = 2000
    # Pinned Elo settings (independent of the tuned config, DECISIONS D-027): K = 800/(Ne+m), no floor, no bonus.
    for which in ("overall", "surface"):
        c["elo"][which] = {"k_factor": 24.0, "div_number": 800.0, "m": 1.0, "base_k": 0.0, "max_k": 1000.0,
                           "bonus_after_layoff": False}
    return c


def _row(c: dict[str, Any], **kw: Any) -> dict[str, Any]:
    return make_match_row(cfg=c, **kw)


def _table(c: dict[str, Any], match_nums: list[int], level: str = "F", swap: bool = False) -> pd.DataFrame:
    rows = [  # warm-up so every player has non-trivial pre-block state
        _row(c, tourney_id="2020-0001", tourney_date="2020-11-02", winner_id="1", loser_id="2", stats=True),
        _row(c, tourney_id="2020-0001", tourney_date="2020-11-02", winner_id="3", loser_id="4", match_num=2,
             stats=True),
        _row(c, tourney_id="2020-0001", tourney_date="2020-11-02", winner_id="1", loser_id="3", round_="SF",
             stats=True),
    ]
    for (w, l), num in zip(GROUP, match_nums):
        if swap:
            w, l = l, w
        rows.append(_row(c, tourney_id=RR_TID, tourney_date=RR_DATE, winner_id=w, loser_id=l, round_="RR",
                         match_num=num, level=level, stats=True))
    rows.append(_row(c, tourney_id=RR_TID, tourney_date=RR_DATE, winner_id="1", loser_id="2", round_="F",
                     match_num=99, level=level))
    return rows_to_frame(rows)


def _player_rows(f: pd.DataFrame, m: pd.DataFrame, rnd: str = "RR") -> dict[tuple, dict[str, Any]]:
    """Per-player features of the rows of round `rnd`, keyed by (sorted player pair, player)."""
    ids = set(m.loc[m["round"] == rnd, "match_id"])
    names = [c[2:] for c in f.columns if c.startswith("a_") and c not in ("a_id", "a_name")]
    out = {}
    for _, r in f[f["match_id"].isin(ids)].iterrows():
        pair = tuple(sorted((r["a_id"], r["b_id"])))
        for side in ("a", "b"):
            out[(pair, r[f"{side}_id"])] = {n: r[f"{side}_{n}"] for n in names}
    return out


def _val(f: pd.DataFrame, mid: str, pid: str, col: str) -> Any:
    r = f.set_index("match_id").loc[mid]
    return r[f"a_{col}"] if r["a_id"] == pid else r[f"b_{col}"]


def _same(x: Any, y: Any) -> bool:
    return (isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y)) or x == y


def test_block_rows_use_pre_block_state(cfg: dict[str, Any]) -> None:
    m = _table(cfg, [1, 2, 3, 4, 5, 6])
    f = build_features(m, cfg)
    rr = _player_rows(f, m)
    assert len(rr) == 12
    for (pair, pid), v in rr.items():
        assert v["matches_played"] == PRE_COUNTS[pid], (pair, pid)
        assert v["matches_surface_played"] == PRE_COUNTS[pid]
        assert v["matches_last_14_days"] == PRE_COUNTS[pid]
        assert v["days_since_last_match"] == 14.0
        assert v["n_stat_matches"] == PRE_COUNTS[pid]
    p1 = [v for (pair, pid), v in rr.items() if pid == "1"]
    assert len(p1) == 3
    for v in p1[1:]:
        for k in ("elo", "elo_surface", "elo_grad", "winrate_last_5", "n_stat_matches", "p_ace", "p_rpw"):
            assert _same(v[k], p1[0][k]), k
    # head-to-head inside the block is pre-block: only the warm-up meetings count
    assert rr[(("1", "3"), "1")]["h2h_wins"] == 1
    assert rr[(("1", "2"), "1")]["h2h_wins"] == 1
    assert rr[(("2", "4"), "2")]["h2h_wins"] == 0


@pytest.mark.parametrize("perm", [[6, 5, 4, 3, 2, 1], [3, 1, 6, 2, 5, 4], [2, 4, 1, 6, 3, 5]])
def test_reordering_match_num_inside_block_changes_nothing(cfg: dict[str, Any], perm: list[int]) -> None:
    m0 = _table(cfg, [1, 2, 3, 4, 5, 6])
    m1 = _table(cfg, perm)
    f0, k0 = build_features(m0, cfg, record_k=True)
    f1, k1 = build_features(m1, cfg, record_k=True)
    a, b = _player_rows(f0, m0), _player_rows(f1, m1)
    assert a.keys() == b.keys() and len(a) == 12
    for key in a:
        for n in a[key]:
            assert _same(a[key][n], b[key][n]), (key, n, a[key][n], b[key][n])
    # the logged K of a block is the pre-block K, whatever the order
    for kl, mm in ((k0, m0), (k1, m1)):
        rr_ids = set(mm.loc[mm["round"] == "RR", "match_id"])
        k_rr = kl[kl["match_id"].isin(rr_ids)]
        assert len(k_rr) == 12
        assert k_rr.groupby("player_id")["k_overall"].nunique().eq(1).all()
    # post-block ratings are equal up to float summation order
    fa, fb = _player_rows(f0, m0, "F"), _player_rows(f1, m1, "F")
    assert fa.keys() == fb.keys()
    for key in fa:
        assert fa[key]["elo"] == pytest.approx(fb[key]["elo"], abs=1e-9)
        assert fa[key]["elo_surface"] == pytest.approx(fb[key]["elo_surface"], abs=1e-9)
        assert fa[key]["matches_played"] == fb[key]["matches_played"]


def test_block_rows_ignore_results_inside_the_block(cfg: dict[str, Any]) -> None:
    """Like leakage test 10: swapping every RR result leaves every RR row's pre-match features unchanged."""
    m0 = _table(cfg, [1, 2, 3, 4, 5, 6])
    m1 = _table(cfg, [1, 2, 3, 4, 5, 6], swap=True)
    a, b = _player_rows(build_features(m0, cfg), m0), _player_rows(build_features(m1, cfg), m1)
    assert a.keys() == b.keys()
    for key in a:
        for n in a[key]:
            assert _same(a[key][n], b[key][n]), (key, n)


def test_block_elo_deltas_come_from_pre_block_snapshot(cfg: dict[str, Any]) -> None:
    m = _table(cfg, [1, 2, 3, 4, 5, 6])
    f, k_log = build_features(m, cfg, record_k=True)
    rr = _player_rows(f, m)
    pre = {pid: v["elo"] for (_, pid), v in rr.items()}
    pre_s = {pid: v["elo_surface"] for (_, pid), v in rr.items()}
    kl = k_log.set_index(["match_id", "player_id"])
    rr_rows = m[m["round"] == "RR"].sort_values("match_num")
    for _, r in rr_rows.iterrows():  # K = 800 / (pre-block Ne + 1 + event count 0)
        for pid in (r["winner_id"], r["loser_id"]):
            assert kl.loc[(r["match_id"], pid), "k_overall"] == 800 / (PRE_COUNTS[pid] + 1)
            assert kl.loc[(r["match_id"], pid), "k_surface"] == 800 / (PRE_COUNTS[pid] + 1)
    expect, expect_s = dict(pre), dict(pre_s)
    for _, r in rr_rows.iterrows():
        w, l, mid = r["winner_id"], r["loser_id"], r["match_id"]
        expect[w] += kl.loc[(mid, w), "k_overall"] * (1 - expected_score(pre[w], pre[l]))
        expect[l] += kl.loc[(mid, l), "k_overall"] * (0 - expected_score(pre[l], pre[w]))
        expect_s[w] += kl.loc[(mid, w), "k_surface"] * (1 - expected_score(pre_s[w], pre_s[l]))
        expect_s[l] += kl.loc[(mid, l), "k_surface"] * (0 - expected_score(pre_s[l], pre_s[w]))
    final = _player_rows(f, m, "F")
    assert len(final) == 2
    for (_, pid), v in final.items():
        assert v["elo"] == pytest.approx(expect[pid], abs=1e-9)
        assert v["elo_surface"] == pytest.approx(expect_s[pid], abs=1e-9)
        assert v["matches_played"] == PRE_COUNTS[pid] + 3
    # the final's K uses Ne and the event count after the block (3 RR matches each)
    fid = m.loc[m["round"] == "F", "match_id"].iloc[0]
    assert kl.loc[(fid, "1"), "k_overall"] == 800 / (5 + 1 + 3)


def test_davis_cup_rr_is_processed_match_by_match(cfg: dict[str, Any]) -> None:
    assert "D" in cfg["features"]["rr_block_exempt_levels"]
    m = _table(cfg, [1, 2, 3, 4, 5, 6], level="D")
    f = build_features(m, cfg)
    rr_ids = m.loc[m["round"] == "RR"].sort_values("match_num")["match_id"].tolist()
    p1 = (rr_ids[0], rr_ids[2], rr_ids[4])  # player 1 plays RR matches 1, 3, 5
    assert [_val(f, mid, "1", "matches_played") for mid in p1] == [2, 3, 4]
    c2 = copy.deepcopy(cfg)
    c2["features"]["rr_block_exempt_levels"] = []
    f2 = build_features(m, c2)
    assert [_val(f2, mid, "1", "matches_played") for mid in p1] == [2, 2, 2]


def test_blocks_are_split_by_tourney_id(cfg: dict[str, Any]) -> None:
    rows = [
        _row(cfg, tourney_id="2020-0700", tourney_date=RR_DATE, winner_id="1", loser_id="2", round_="RR",
             level="F"),
        _row(cfg, tourney_id="2020-0701", tourney_date=RR_DATE, winner_id="1", loser_id="3", round_="RR",
             level="F"),
    ]
    m = rows_to_frame(rows)
    f = build_features(m, cfg)
    assert _val(f, m["match_id"].iloc[1], "1", "matches_played") == 1


def test_non_rr_rounds_are_not_blocked(cfg: dict[str, Any]) -> None:
    rows = [  # same tourney and round, player 1 twice: knock-out rounds stay match by match
        _row(cfg, tourney_id="2020-0800", tourney_date=RR_DATE, winner_id="1", loser_id="2", round_="R32",
             match_num=1),
        _row(cfg, tourney_id="2020-0800", tourney_date=RR_DATE, winner_id="1", loser_id="3", round_="R32",
             match_num=2),
    ]
    m = rows_to_frame(rows)
    f = build_features(m, cfg)
    assert _val(f, m["match_id"].iloc[1], "1", "matches_played") == 1


def test_block_with_retirement_and_walkover(cfg: dict[str, Any]) -> None:
    """A RET inside a block logs its pre-block K and emits no row; a walkover is dropped before the pass."""
    m = _table(cfg, [1, 2, 3, 4, 5, 6])
    rr = m["round"] == "RR"
    ret = m.index[rr & (m["match_num"] == 1)][0]
    wo = m.index[rr & (m["match_num"] == 2)][0]
    m.loc[ret, "score"] = "6-1 RET"
    m.loc[ret, "is_retirement"] = True
    m.loc[wo, "score"] = "W/O"
    m.loc[wo, "is_walkover"] = True
    f, k_log = build_features(m, cfg, record_k=True)
    assert m.loc[ret, "match_id"] not in set(f["match_id"])
    assert m.loc[wo, "match_id"] not in set(f["match_id"])
    assert m.loc[ret, "match_id"] in set(k_log["match_id"])
    assert m.loc[wo, "match_id"] not in set(k_log["match_id"])
    for (_, pid), v in _player_rows(f, m).items():
        assert v["matches_played"] == PRE_COUNTS[pid]
    # player 1: RET (no Elo move) + 2 rated RR wins -> 3 counted matches, final Elo moved by 2 deltas only
    final = _player_rows(f, m, "F")
    assert final[(("1", "2"), "1")]["matches_played"] == PRE_COUNTS["1"] + 3
