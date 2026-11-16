"""Elo tuning harness (tennis.model.elo_harness): exact agreement with the feature engine, the incremental layoff
bonus, round-robin blocks, scoring scope, the grid and multiprocessing."""

from __future__ import annotations

import copy
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from tennis.config import load_config, repo_path
from tennis.engine.build_features import build_features
from tennis.engine.elo import k_bonus_after_layoff
from tennis.engine.state import layoff_played
from tennis.model import elo_harness as eh
from tests.fixtures.synthetic import make_synthetic_matches


@pytest.fixture(scope="module")
def base_cfg() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def synth(base_cfg: dict) -> pd.DataFrame:
    return make_synthetic_matches(seed=0, cfg=base_cfg)


def _variant(cfg: dict, **kw) -> dict:
    c = copy.deepcopy(cfg)
    for w in ("overall", "surface"):
        for k in ("div_number", "m", "base_k", "max_k", "bonus_after_layoff", "k_factor"):
            if k in kw:
                c["elo"][w][k] = kw[k]
    if "min_gaps" in kw:
        c["elo"]["layoff"]["min_gaps"] = kw["min_gaps"]
    if "m_mode" in kw:
        c["elo"]["m_mode"] = kw["m_mode"]
    if "upd_ret" in kw:
        c["cleaning"]["retirement_updates_elo"] = kw["upd_ret"]
    return c


VARIANTS = [
    {},
    {"bonus_after_layoff": True},
    {"bonus_after_layoff": True, "min_gaps": 3, "m_mode": "constant", "m": 5.0, "base_k": 8.0, "max_k": 32.0},
    {"div_number": 1200.0, "m": 10.0, "base_k": 16.0, "max_k": 64.0, "bonus_after_layoff": True, "min_gaps": 5},
    {"div_number": 0.0, "k_factor": 24.0, "bonus_after_layoff": True, "min_gaps": 3},
    {"upd_ret": True},
]


# ---------------------------------------------------------------------------------- engine agreement
@pytest.mark.parametrize("variant", VARIANTS, ids=[str(v) or "default" for v in VARIANTS])
def test_harness_equals_engine_on_synthetic(synth: pd.DataFrame, base_cfg: dict, variant: dict) -> None:
    """K per match side equals the engine's k_log, and pre-match ratings equal a_elo/b_elo(_surface), exactly."""
    cfg = _variant(base_cfg, **variant)
    feats, k_log = build_features(synth, cfg, record_k=True)
    prep = eh.prepare(synth, cfg)
    pos = pd.Series(np.arange(len(prep.match_id)), index=prep.match_id)
    assert len(k_log) == 2 * len(prep.match_id)
    ki = pos.loc[k_log["match_id"]].to_numpy()
    k_is_w = k_log["player_id"].to_numpy() == prep.winner_id[ki]
    fi = pos.loc[feats["match_id"]].to_numpy()
    a_is_w = feats["a_id"].to_numpy() == prep.winner_id[fi]
    for which, col, kcol in (("overall", "elo", "k_overall"), ("surface", "elo_surface", "k_surface")):
        s = eh.current_setting(cfg, which)
        kw, kl = eh.k_arrays(prep, s)
        np.testing.assert_array_equal(np.where(k_is_w, kw[ki], kl[ki]), k_log[kcol].to_numpy())
        rw, rl = eh.run_setting(prep, s)
        np.testing.assert_array_equal(np.where(a_is_w, rw[fi], rl[fi]), feats[f"a_{col}"].to_numpy())
        np.testing.assert_array_equal(np.where(a_is_w, rl[fi], rw[fi]), feats[f"b_{col}"].to_numpy())
        p_w = eh.win_prob(rw[fi], rl[fi])          # winner's expected score
        p_l = eh.win_prob(rl[fi], rw[fi])          # loser's (as the engine computes it when A lost)
        prob_col = "elo_prob_a" if which == "overall" else "elo_surface_prob_a"
        np.testing.assert_array_equal(np.where(a_is_w, p_w, p_l), feats[prob_col].to_numpy())
        np.testing.assert_allclose(p_w + p_l, 1.0, rtol=0, atol=1e-12)


def test_synthetic_exercises_bonus_and_rr_blocks(synth: pd.DataFrame, base_cfg: dict) -> None:
    prep = eh.prepare(synth, base_cfg)
    assert (prep.bonus_w > 1).any() or (prep.bonus_l > 1).any()
    assert (prep.block_end - np.arange(len(prep.block_end)) > 1).any()


@pytest.mark.realdata
@pytest.mark.slow
def test_harness_matches_feature_table(base_cfg: dict) -> None:
    """Under the current config the harness reproduces features.parquet a_elo/b_elo and the surface ratings."""
    mpath, fpath = repo_path(base_cfg["paths"]["matches"]), repo_path(base_cfg["paths"]["features"])
    if not (mpath.exists() and fpath.exists()):
        pytest.skip("real data absent")
    prep = eh.prepare(pd.read_parquet(mpath), base_cfg)
    f = pd.read_parquet(fpath, columns=["match_id", "a_id", "a_elo", "b_elo", "a_elo_surface", "b_elo_surface"])
    pos = pd.Series(np.arange(len(prep.match_id)), index=prep.match_id)
    fi = pos.loc[f["match_id"]].to_numpy()
    a_is_w = f["a_id"].to_numpy() == prep.winner_id[fi]
    for which, col in (("overall", "elo"), ("surface", "elo_surface")):
        rw, rl = eh.run_setting(prep, eh.current_setting(base_cfg, which))
        np.testing.assert_allclose(np.where(a_is_w, rw[fi], rl[fi]), f[f"a_{col}"].to_numpy(), rtol=0, atol=1e-9)
        np.testing.assert_allclose(np.where(a_is_w, rl[fi], rw[fi]), f[f"b_{col}"].to_numpy(), rtol=0, atol=1e-9)
    # scoring scope: never a test period
    scored = prep.season[prep.score]
    assert scored.min() >= 2015 and scored.max() <= 2024


# -------------------------------------------------------------------------------------- layoff bonus
LAYOFFS = [
    {"threshold": 100, "min_gaps": 15, "start_bonus": 1.5, "step": 0.05, "floor": 1.0},
    {"threshold": 30, "min_gaps": 2, "start_bonus": 1.5, "step": 0.05, "floor": 1.0},
    {"threshold": 14, "min_gaps": 0, "start_bonus": 2.0, "step": 0.3, "floor": 1.1},
]


@pytest.mark.parametrize("layoff", LAYOFFS)
def test_incremental_layoff_bonus_equals_engine(layoff: dict) -> None:
    """Random careers (several matches per event, same-day events, long layoffs): the O(1) tracker equals
    k_bonus_after_layoff(layoff_played(...)) before every match."""
    rng = np.random.default_rng(7)
    for career in range(40):
        lt = eh.LayoffTracker(float(layoff["threshold"]))
        tourneys: dict[str, dt.date] = {}
        day = dt.date(2000, 1, 3)
        for e in range(int(rng.integers(1, 60))):
            day += dt.timedelta(days=int(rng.choice([0, 7, 14, 21, 35, 99, 100, 101, 200])))
            tid = f"t{e}"
            for _ in range(int(rng.integers(1, 4))):
                played = layoff_played(tourneys, tid, day)
                want = k_bonus_after_layoff(played, **layoff)
                g, run = lt.peek(0, day.toordinal(), tid not in tourneys)
                assert eh.layoff_bonus(g, run, layoff) == want, (career, e, played)
                if tid not in tourneys:
                    lt.add(0, day.toordinal())
                    tourneys[tid] = day


# ----------------------------------------------------------------------------------------- RR blocks
def _mini(rows: list[tuple]) -> pd.DataFrame:
    """Minimal unified table: (tourney_id, date, round, match_num, winner, loser, is_ret)."""
    cfg = load_config()
    order = {r: i for i, r in enumerate(cfg["data"]["round_order"])}
    out = []
    for tid, d, rnd, num, w, l, ret in rows:
        out.append({"match_id": f"{tid}-{num}", "tourney_id": tid, "tourney_date": pd.Timestamp(d), "round": rnd,
                    "round_rank": order[rnd], "match_num": num, "winner_id": w, "loser_id": l, "surface": "Hard",
                    "tourney_level": "F", "is_walkover": False, "is_retirement": ret, "is_qualifying": False,
                    "is_training_level": True})
    return pd.DataFrame(out)


def test_rr_block_uses_pre_block_ratings(base_cfg: dict) -> None:
    m = _mini([("T1", "2020-01-06", "RR", 1, "a", "b", False), ("T1", "2020-01-06", "RR", 2, "a", "c", False),
               ("T1", "2020-01-06", "F", 3, "a", "b", False)])
    prep = eh.prepare(m, base_cfg)
    assert prep.block_end.tolist() == [2, 2, 3]
    rw, rl = eh.run_setting(prep, eh.current_setting(base_cfg, "overall"))
    start = base_cfg["elo"]["start_rating"]
    assert rw[0] == rw[1] == start and rl[1] == start          # a's second RR match sees the pre-block rating
    assert rw[2] > start and rl[2] < start
    assert prep.n_event_w.tolist() == [0, 0, 2]                # in-event count is pre-block too


def test_retirement_counts_but_does_not_move_elo(base_cfg: dict) -> None:
    m = _mini([("T1", "2020-01-06", "R16", 1, "a", "b", True), ("T1", "2020-01-06", "QF", 2, "a", "c", False)])
    prep = eh.prepare(m, base_cfg)
    rw, _ = eh.run_setting(prep, eh.current_setting(base_cfg, "overall"))
    assert rw[1] == base_cfg["elo"]["start_rating"]
    assert prep.ne_overall_w.tolist() == [0, 1] and prep.n_event_w.tolist() == [0, 1]
    assert prep.score.tolist() == [False, False]                # 2020 < tune_start; RET never scored anyway


# ------------------------------------------------------------------------------------------- scoring
def test_scoring_scope(synth: pd.DataFrame, base_cfg: dict) -> None:
    cfg = copy.deepcopy(base_cfg)
    cfg["elo_harness"].update(warmup_matches=300, tune_start="2020-01-01", tune_end="2020-12-31")
    prep = eh.prepare(synth, cfg)
    ms = synth.set_index("match_id").loc[prep.match_id]
    sc = prep.score
    assert sc.any()
    assert not sc[:300].any()
    assert (prep.season[sc] == 2020).all()
    assert ms["is_training_level"].to_numpy()[sc].all()
    assert not ms["is_qualifying"].to_numpy()[sc].any() and not ms["is_retirement"].to_numpy()[sc].any()
    want = (ms["is_training_level"] & ~ms["is_qualifying"] & ~ms["is_retirement"]
            & (ms["tourney_date"].dt.year == 2020)).to_numpy() & (np.arange(len(sc)) >= 300)
    np.testing.assert_array_equal(sc, want)


def test_score_setting_metrics(synth: pd.DataFrame, base_cfg: dict) -> None:
    cfg = copy.deepcopy(base_cfg)
    cfg["elo_harness"].update(warmup_matches=0, tune_start="2020-01-01", tune_end="2021-12-31")
    prep = eh.prepare(synth, cfg)
    r = eh.score_setting(prep, eh.current_setting(cfg, "overall"), 1e-15)
    rw, rl = eh.run_setting(prep, eh.current_setting(cfg, "overall"))
    p = eh.win_prob(rw[prep.score], rl[prep.score])
    assert r["n"] == int(prep.score.sum())
    assert r["log_loss"] == pytest.approx(float(np.mean(-np.log(p))))
    assert r["brier"] == pytest.approx(float(np.mean((1 - p) ** 2)))
    assert r["accuracy"] == pytest.approx(float(np.mean(np.where(p > 0.5, 1.0, np.where(p < 0.5, 0.0, 0.5)))))
    assert set(r["seasons"]) == {"2020", "2021"}
    assert sum(v["n"] for v in r["seasons"].values()) == r["n"]


# ---------------------------------------------------------------------------------------------- grid
def test_grid_settings(base_cfg: dict) -> None:
    g = base_cfg["elo_harness"]["grid"]
    for which in eh.WHICH:
        gs = eh.grid_settings(base_cfg, which)
        n_dyn = (len(g["div_number"]) * len(g["m"]) * len(g["base_k"]) * len(g["max_k"])
                 * len(g["bonus_after_layoff"]))
        assert len(gs) == n_dyn + len(g["fixed_k"]) * len(g["bonus_after_layoff"])
        assert len(set(gs)) == len(gs) and all(s.which == which for s in gs)
        assert eh.current_setting(base_cfg, which) in gs
        fixed = [s for s in gs if s.div_number == 0]
        assert {(s.k_factor, s.bonus_after_layoff) for s in fixed} == {(24.0, False), (24.0, True)}


def test_edge_probes(base_cfg: dict) -> None:
    g = base_cfg["elo_harness"]["grid"]
    corner = eh.EloSetting("overall", 24.0, max(g["div_number"]), min(g["m"]), max(g["base_k"]), max(g["max_k"]), True)
    probes = eh.edge_probes(base_cfg, corner)
    d, b, x = sorted(g["div_number"]), sorted(g["base_k"]), sorted(g["max_k"])
    want_div, want_base, want_max = d[-1] + d[-1] - d[-2], b[-1] + b[-1] - b[-2], x[-1] + x[-1] - x[-2]
    assert {(p.div_number, p.base_k, p.max_k) for p in probes} == {
        (want_div, corner.base_k, corner.max_k), (corner.div_number, want_base, corner.max_k),
        (corner.div_number, corner.base_k, want_max), (want_div, want_base, want_max)}
    assert all(p.m >= 1 for p in probes)
    interior = eh.EloSetting("overall", 24.0, d[1], sorted(g["m"])[1], b[1], x[1], False)
    assert eh.edge_probes(base_cfg, interior) == []
    assert eh.edge_probes(base_cfg, eh.EloSetting("overall", 24.0, 0.0, 1.0, 0.0, 100.0, True)) == []


def test_run_grid_parallel_equals_serial(synth: pd.DataFrame, base_cfg: dict) -> None:
    cfg = copy.deepcopy(base_cfg)
    cfg["elo_harness"].update(warmup_matches=0, tune_start="2020-01-01", tune_end="2021-12-31")
    prep = eh.prepare(synth, cfg)
    settings = eh.grid_settings(cfg, "overall")[:3] + eh.grid_settings(cfg, "surface")[-2:]
    serial = eh.run_grid(prep, settings, 1e-15, processes=1)
    parallel = eh.run_grid(prep, settings, 1e-15, processes=2)
    assert [s for s, _ in parallel] == settings
    assert serial == parallel
