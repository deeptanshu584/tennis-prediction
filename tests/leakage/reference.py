"""Independent, deliberately simple reference implementation of the CONTRACT.md section 2 features.

Used by test 2 (slow recompute). It shares NO code with `tennis.engine`:

- Elo (overall + surface, dynamic K, layoff bonus, m modes) is replayed with plain dicts in one chronological loop,
  written from PLAN.md 5.2-5.4 and CONTRACT.md. The pre-match snapshot of match i is the state after replaying
  matches 0..i-1, i.e. exactly "replay the strictly earlier matches".
- Every other player feature is recomputed by FILTERING the strictly earlier matches (``ms.iloc[:i]``) for the
  player, with no incremental state at all.

Contract rulings mirrored here:
- Authoritative order (tourney_date, tourney_id, round_rank, match_num, match_id); match_id is the final tiebreaker.
- Dynamic K with Ne + m <= 0 raises ValueError (config keeps m >= 1, so this is unreachable).
- Layoff `played`: one entry per distinct prior tourney_id; the current tourney is added only if its id is absent.
- k_log covers RET/DEF matches too (the K that would have applied).
- Column names come from config: winrate_last_{k}, level_{L}, surface_{S}.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import pandas as pd

STAT_FIELDS: tuple[str, ...] = (
    "ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced",
)
RATE_FEATURES: tuple[str, ...] = (
    "p_ace", "p_df", "p_1stIn", "p_1stWon", "p_2ndWon", "p_bpSaved", "p_rpw", "p_retAceAgainst",
    "p_ret1stWon", "p_ret2ndWon", "p_bpConv", "p_totalPtsWon", "dominance_ratio",
)
SORT_COLUMNS: list[str] = ["tourney_date", "tourney_id", "round_rank", "match_num", "match_id"]
NAN = float("nan")


# ---------------------------------------------------------------------------------------------
# column lists
# ---------------------------------------------------------------------------------------------
def player_feature_names(cfg: dict[str, Any]) -> list[str]:
    """The player features `f` of CONTRACT section 2 (each appears as a_f, b_f, d_f)."""
    return (
        ["elo", "elo_surface", "elo_grad", "matches_played", "matches_surface_played"]
        + [f"winrate_last_{k}" for k in cfg["features"]["winrate_k"]]
        + ["h2h_wins", "h2h_surface_wins", "n_stat_matches"]
        + list(RATE_FEATURES)
        + ["rank", "rank_points", "age", "height", "is_left", "days_since_last_match", "matches_last_14_days"]
    )


def surfaces(cfg: dict[str, Any]) -> list[str]:
    """Normalised surfaces, in config order (distinct values of data.surface_map)."""
    return list(dict.fromkeys(cfg["data"]["surface_map"].values()))


def context_feature_names(cfg: dict[str, Any]) -> list[str]:
    return (
        [f"surface_{s}" for s in surfaces(cfg)]
        + [f"level_{lv}" for lv in cfg["data"]["training_levels"]]
        + ["round_rank", "best_of", "elo_prob_a", "elo_surface_prob_a"]
    )


NON_FEATURE_COLUMNS: list[str] = ["match_id", "date", "tourney_id", "tourney_name", "a_id", "b_id", "a_name",
                                  "b_name", "y"]


def all_output_columns(cfg: dict[str, Any]) -> list[str]:
    cols = list(NON_FEATURE_COLUMNS) + context_feature_names(cfg)
    for f in player_feature_names(cfg):
        cols += [f"a_{f}", f"b_{f}", f"d_{f}"]
    return cols


# ---------------------------------------------------------------------------------------------
# basic rules
# ---------------------------------------------------------------------------------------------
def a_is_winner(seed: int, match_id: str) -> bool:
    """CONTRACT orientation coin flip."""
    return int(hashlib.sha256(f"{seed}:{match_id}".encode("utf-8")).hexdigest()[:8], 16) % 2 == 0


def chronological(matches: pd.DataFrame) -> pd.DataFrame:
    """Non-walkover matches in match order, index reset to the position in the pass."""
    ms = matches.loc[~matches["is_walkover"].astype(bool)].copy()
    ms["tourney_date"] = pd.to_datetime(ms["tourney_date"])
    for c in ("winner_id", "loser_id", "tourney_id", "match_id"):
        ms[c] = ms[c].astype(str)
    return ms.sort_values(SORT_COLUMNS, kind="mergesort").reset_index(drop=True)


def emitted(ms: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    """Boolean mask over `chronological()` rows that must produce a feature row."""
    return (
        ms["is_training_level"].astype(bool)
        & ~ms["is_retirement"].astype(bool)
        & (ms["tourney_date"].dt.year >= int(cfg["data"]["feature_first_year"]))
    )


def _isnan(x: Any) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x)) or bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def _div(num: float, den: float) -> float:
    return NAN if den == 0 else num / den


# ---------------------------------------------------------------------------------------------
# Elo replay
# ---------------------------------------------------------------------------------------------
def _layoff_bonus(played: list[pd.Timestamp], lay: dict[str, Any]) -> float:
    gaps = [(played[i] - played[i - 1]).days for i in range(1, len(played))]
    gaps.reverse()  # most recent first
    if len(gaps) <= int(lay["min_gaps"]):
        return 1.0
    bonus = float(lay["start_bonus"])
    for g in gaps:
        if g < lay["threshold_days"]:
            bonus -= float(lay["step"])
        else:
            break
    return max(float(lay["floor"]), bonus)


def _k(ne: int, m: float, p: dict[str, Any], played: list[pd.Timestamp], lay: dict[str, Any]) -> float:
    if float(p["div_number"]) == 0:
        k = float(p["k_factor"])
    else:
        den = ne + m
        if den <= 0:
            raise ValueError(f"dynamic K needs ne + m > 0, got ne={ne}, m={m}")
        k = min(float(p["max_k"]), max(float(p["base_k"]), float(p["div_number"]) / den))
    if p["bonus_after_layoff"]:
        k *= _layoff_bonus(played, lay)
    return k


def _expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def elo_replay(ms: pd.DataFrame, cfg: dict[str, Any], targets: Iterable[str]) -> dict[str, dict[str, dict]]:
    """Replay Elo over `chronological()` rows; return pre-match snapshots for the target match ids.

    snapshot[match_id][player_id] = {elo, elo_surface, hist (last window values), k_overall, k_surface}
    """
    targets = set(targets)
    e = cfg["elo"]
    start = float(e["start_rating"])
    mode = e["m_mode"]
    assert mode in ("constant", "event_matches_so_far"), mode
    lay = e["layoff"]
    window = int(cfg["features"]["elo_grad_window"])
    upd_ret = bool(cfg["cleaning"]["retirement_updates_elo"])

    rating: dict[str, float] = {}
    srating: dict[tuple[str, str], float] = {}
    hist: dict[str, list[float]] = {}
    n_all: dict[str, int] = defaultdict(int)
    n_surf: dict[tuple[str, str], int] = defaultdict(int)
    n_event: dict[tuple[str, str], int] = defaultdict(int)
    tourneys: dict[str, dict[str, pd.Timestamp]] = defaultdict(dict)  # insertion order = chronological
    snaps: dict[str, dict[str, dict]] = {}

    cols = ["match_id", "winner_id", "loser_id", "surface", "tourney_id", "tourney_date", "is_retirement"]
    for mid, w, l, s, tid, d, is_ret in ms[cols].itertuples(index=False, name=None):
        for p in (w, l):
            rating.setdefault(p, start)
            srating.setdefault((p, s), start)
            hist.setdefault(p, [start])
        k: dict[str, tuple[float, float]] = {}
        for p in (w, l):
            played = list(tourneys[p].values())
            if tid not in tourneys[p]:
                played.append(d)
            m_o, m_s = float(e["overall"]["m"]), float(e["surface"]["m"])
            if mode == "event_matches_so_far":  # CONTRACT: m = elo.<which>.m + matches already played in event
                m_o += n_event[(p, tid)]
                m_s += n_event[(p, tid)]
            k[p] = (_k(n_all[p], m_o, e["overall"], played, lay),
                    _k(n_surf[(p, s)], m_s, e["surface"], played, lay))
        if mid in targets:
            snaps[mid] = {
                p: {"elo": rating[p], "elo_surface": srating[(p, s)], "hist": list(hist[p][-window:]),
                    "k_overall": k[p][0], "k_surface": k[p][1]}
                for p in (w, l)
            }
        if (not bool(is_ret)) or upd_ret:
            rw, rl = rating[w], rating[l]
            new_w = rw + k[w][0] * (1.0 - _expected(rw, rl))
            new_l = rl + k[l][0] * (0.0 - _expected(rl, rw))
            sw, sl = srating[(w, s)], srating[(l, s)]
            new_sw = sw + k[w][1] * (1.0 - _expected(sw, sl))
            new_sl = sl + k[l][1] * (0.0 - _expected(sl, sw))
            rating[w], rating[l] = new_w, new_l
            srating[(w, s)], srating[(l, s)] = new_sw, new_sl
            hist[w].append(new_w)
            hist[l].append(new_l)
        for p in (w, l):
            n_all[p] += 1
            n_surf[(p, s)] += 1
            n_event[(p, tid)] += 1
            tourneys[p].setdefault(tid, d)
    return snaps


def ols_slope(values: list[float]) -> float:
    """Slope of a least-squares line through (0, v0), (1, v1), ...; NaN if fewer than 2 values."""
    n = len(values)
    if n < 2:
        return NAN
    x = np.arange(n, dtype=float)
    y = np.asarray(values, dtype=float)
    xm, ym = x.mean(), y.mean()
    return float(((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum())


# ---------------------------------------------------------------------------------------------
# filter-based player features
# ---------------------------------------------------------------------------------------------
def _stats_ok(df: pd.DataFrame) -> pd.Series:
    cols = [f"{side}_{s}" for side in ("w", "l") for s in STAT_FIELDS]
    return df[cols].notna().all(axis=1) & (df["w_svpt"] > 0) & (df["l_svpt"] > 0)


def player_features(prior: pd.DataFrame, row: pd.Series, p: str, opp: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Features of player `p` for match `row`, computed only from `prior` (= strictly earlier matches)."""
    clean = cfg["cleaning"]
    fcfg = cfg["features"]
    surf = row["surface"]
    date = row["tourney_date"]
    mine = prior[(prior["winner_id"] == p) | (prior["loser_id"] == p)]
    f: dict[str, Any] = {}

    f["matches_played"] = len(mine)
    f["matches_surface_played"] = int((mine["surface"] == surf).sum())

    form = mine if clean["retirement_updates_form"] else mine[~mine["is_retirement"].astype(bool)]
    results = (form["winner_id"] == p).astype(float).to_numpy()
    for k in fcfg["winrate_k"]:
        last = results[-int(k):]
        f[f"winrate_last_{k}"] = float(last.sum() / len(last)) if len(last) else NAN

    h2h = mine[(mine["winner_id"] == p) & (mine["loser_id"] == opp)]
    if not clean["retirement_updates_h2h"]:
        h2h = h2h[~h2h["is_retirement"].astype(bool)]
    f["h2h_wins"] = len(h2h)
    f["h2h_surface_wins"] = int((h2h["surface"] == surf).sum())

    ok = _stats_ok(mine)
    if not clean["retirement_updates_stats"]:
        ok &= ~mine["is_retirement"].astype(bool)
    buf = mine[ok].tail(int(fcfg["rolling_n"]))
    f["n_stat_matches"] = len(buf)
    won = (buf["winner_id"] == p).to_numpy()
    own, opp_ = {}, {}
    for s in STAT_FIELDS:
        w_vals, l_vals = buf[f"w_{s}"].to_numpy(float), buf[f"l_{s}"].to_numpy(float)
        own[s] = float(np.where(won, w_vals, l_vals).sum())    # the player's own serve counts
        opp_[s] = float(np.where(won, l_vals, w_vals).sum())   # the opponent's serve counts
    rpw_num = opp_["svpt"] - opp_["1stWon"] - opp_["2ndWon"]
    f["p_ace"] = _div(own["ace"], own["svpt"])
    f["p_df"] = _div(own["df"], own["svpt"])
    f["p_1stIn"] = _div(own["1stIn"], own["svpt"])
    f["p_1stWon"] = _div(own["1stWon"], own["1stIn"])
    f["p_2ndWon"] = _div(own["2ndWon"], own["svpt"] - own["1stIn"])
    f["p_bpSaved"] = _div(own["bpSaved"], own["bpFaced"])
    f["p_rpw"] = _div(rpw_num, opp_["svpt"])
    f["p_retAceAgainst"] = _div(opp_["ace"], opp_["svpt"])
    f["p_ret1stWon"] = _div(opp_["1stIn"] - opp_["1stWon"], opp_["1stIn"])
    f["p_ret2ndWon"] = _div(opp_["svpt"] - opp_["1stIn"] - opp_["2ndWon"], opp_["svpt"] - opp_["1stIn"])
    f["p_bpConv"] = _div(opp_["bpFaced"] - opp_["bpSaved"], opp_["bpFaced"])
    f["p_totalPtsWon"] = _div(own["1stWon"] + own["2ndWon"] + rpw_num, own["svpt"] + opp_["svpt"])
    spw = _div(own["1stWon"] + own["2ndWon"], own["svpt"])
    rpw = f["p_rpw"]
    f["dominance_ratio"] = NAN if (_isnan(spw) or _isnan(rpw)) else _div(rpw, 1.0 - spw)

    side = "winner" if row["winner_id"] == p else "loser"
    f["rank"] = row[f"{side}_rank"]
    f["rank_points"] = row[f"{side}_rank_points"]
    f["age"] = row[f"{side}_age"]
    f["height"] = row[f"{side}_ht"]
    hand = row[f"{side}_hand"]
    f["is_left"] = 1.0 if hand == "L" else (0.0 if hand == "R" else NAN)

    if len(mine):
        f["days_since_last_match"] = float((date - mine["tourney_date"].iloc[-1]).days)
    else:
        f["days_since_last_match"] = NAN
    window_start = date - pd.Timedelta(days=int(fcfg["recent_days_window"]))
    f["matches_last_14_days"] = int((mine["tourney_date"] >= window_start).sum())
    return f


# ---------------------------------------------------------------------------------------------
# full rows
# ---------------------------------------------------------------------------------------------
def reference_rows(matches: pd.DataFrame, cfg: dict[str, Any], match_ids: Iterable[str] | None = None
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reference feature rows (A/B oriented, CONTRACT columns) and reference K values.

    `match_ids` = which emitted matches to compute (default: all emitted). Returns (rows, k_ref) where k_ref has
    columns match_id, player_id, k_overall, k_surface.
    """
    ms = chronological(matches)
    em = emitted(ms, cfg)
    if match_ids is None:
        match_ids = ms.loc[em, "match_id"].tolist()
    match_ids = set(match_ids)
    snaps = elo_replay(ms, cfg, match_ids)
    seed = int(cfg["seed"])
    feats = player_feature_names(cfg)
    rows, krows = [], []
    # narrow, numpy-backed copy for the filters (same rows, same order; only for speed)
    hist_cols = (["winner_id", "loser_id", "surface", "tourney_date", "is_retirement"]
                 + [f"{side}_{s}" for side in ("w", "l") for s in STAT_FIELDS])
    narrow = ms[hist_cols].copy()
    for c in ("winner_id", "loser_id", "surface"):
        narrow[c] = narrow[c].astype(object)
    narrow["is_retirement"] = narrow["is_retirement"].astype(bool)
    positions = ms.index[ms["match_id"].isin(match_ids)]
    for i in positions:
        row = ms.loc[i]
        prior = narrow.iloc[:i]  # strictly earlier matches only
        mid, w, l = row["match_id"], row["winner_id"], row["loser_id"]
        a_w = a_is_winner(seed, mid)
        a, b = (w, l) if a_w else (l, w)
        fa = player_features(prior, row, a, b, cfg)
        fb = player_features(prior, row, b, a, cfg)
        sa, sb = snaps[mid][a], snaps[mid][b]
        for fx, sx in ((fa, sa), (fb, sb)):
            fx["elo"] = sx["elo"]
            fx["elo_surface"] = sx["elo_surface"]
            fx["elo_grad"] = ols_slope(sx["hist"])
        side_a = "winner" if a_w else "loser"
        side_b = "loser" if a_w else "winner"
        out: dict[str, Any] = {
            "match_id": mid, "date": row["tourney_date"], "tourney_id": row["tourney_id"],
            "tourney_name": row["tourney_name"], "a_id": a, "b_id": b,
            "a_name": row[f"{side_a}_name"], "b_name": row[f"{side_b}_name"], "y": int(a_w),
        }
        for s in surfaces(cfg):
            out[f"surface_{s}"] = int(row["surface"] == s)
        for lv in cfg["data"]["training_levels"]:
            out[f"level_{lv}"] = int(row["tourney_level"] == lv)
        out["round_rank"] = int(row["round_rank"])
        out["best_of"] = int(row["best_of"])
        out["elo_prob_a"] = _expected(sa["elo"], sb["elo"])
        out["elo_surface_prob_a"] = _expected(sa["elo_surface"], sb["elo_surface"])
        for f in feats:
            va, vb = fa[f], fb[f]
            out[f"a_{f}"] = va
            out[f"b_{f}"] = vb
            out[f"d_{f}"] = NAN if (_isnan(va) or _isnan(vb)) else va - vb
        rows.append(out)
        for p in (a, b):
            krows.append({"match_id": mid, "player_id": p, "k_overall": snaps[mid][p]["k_overall"],
                          "k_surface": snaps[mid][p]["k_surface"]})
    return pd.DataFrame(rows), pd.DataFrame(krows)


# ---------------------------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------------------------
def values_match(x: Any, y: Any, tol: float = 1e-9) -> bool:
    """NaN-aware equality: numbers within abs/rel tolerance `tol`, everything else exactly."""
    nx, ny = _isnan(x), _isnan(y)
    if nx or ny:
        return nx and ny
    if isinstance(x, (pd.Timestamp, np.datetime64)) or isinstance(y, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(x) == pd.Timestamp(y)
    if isinstance(x, (str, bytes)) or isinstance(y, (str, bytes)):
        return str(x) == str(y)
    try:
        return math.isclose(float(x), float(y), rel_tol=tol, abs_tol=tol)
    except (TypeError, ValueError):
        return x == y


def compare_rows(engine: pd.DataFrame, ref: pd.DataFrame, columns: list[str], key: str = "match_id",
                 tol: float = 1e-9) -> list[str]:
    """List of human-readable mismatches between engine and reference rows (aligned on `key`)."""
    e = engine.set_index(key)
    r = ref.set_index(key)
    problems = []
    for mid in r.index:
        if mid not in e.index:
            problems.append(f"{mid}: missing from engine output")
            continue
        for c in columns:
            if c == key:
                continue
            ev, rv = e.at[mid, c], r.at[mid, c]
            if not values_match(ev, rv, tol):
                problems.append(f"{mid} {c}: engine={ev!r} reference={rv!r}")
    return problems
