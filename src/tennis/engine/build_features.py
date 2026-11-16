"""The chronological feature builder (PLAN sections 2 and 5, CONTRACT section 2).

One pass over every non-walkover match in the authoritative order. For each match:

1. read both players' state (and the K and expected score each would use),
2. write the feature row (only for training-level, non-retirement matches from ``data.feature_first_year`` on),
3. THEN update state with the result and the match statistics.

Round-robin blocks (CONTRACT section 2, DECISIONS D-019): consecutive matches with the same tourney_id in round RR
(level not in ``features.rr_block_exempt_levels``) are one block. Steps 1-2 run for every match of the block on
the state before the block; step 3 then runs for each match in order, with Elo deltas K * (S - E) from that
pre-block snapshot. An ordinary match is a block of one, so the same two-phase loop covers both cases.

No feature is computed with groupby / rolling / merge / expanding over the table.

CLI::

    .venv/Scripts/python -m tennis.engine.build_features      # matches.parquet -> data/processed/features.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from tennis.config import load_config, repo_path
from tennis.engine.elo import EloParams, expected_score
from tennis.engine.h2h import H2H
from tennis.engine.ordering import check_round_rank, sort_matches
from tennis.engine.rolling import RATE_FEATURES, STAT_FIELDS, valid_stats
from tennis.engine.state import (
    PlayerState,
    count_recent,
    days_since,
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

NAN = float("nan")
NON_FEATURE_COLUMNS: tuple[str, ...] = (
    "match_id", "date", "tourney_id", "tourney_name", "a_id", "b_id", "a_name", "b_name", "y",
)
K_LOG_COLUMNS: tuple[str, ...] = ("match_id", "player_id", "k_overall", "k_surface")


# --------------------------------------------------------------------------- column names (from config)


def surfaces(cfg: Mapping[str, Any]) -> list[str]:
    """Normalised surfaces: distinct values of ``data.surface_map`` in config order."""
    return list(dict.fromkeys(cfg["data"]["surface_map"].values()))


def player_feature_names(cfg: Mapping[str, Any]) -> list[str]:
    """Player features ``f`` of CONTRACT section 2 (each appears as ``a_f``, ``b_f``, ``d_f``)."""
    return (
        ["elo", "elo_surface", "elo_grad", "matches_played", "matches_surface_played"]
        + [f"winrate_last_{k}" for k in cfg["features"]["winrate_k"]]
        + ["h2h_wins", "h2h_surface_wins", "n_stat_matches"]
        + list(RATE_FEATURES)
        + ["rank", "rank_points", "age", "height", "is_left", "days_since_last_match", "matches_last_14_days"]
    )


def context_feature_names(cfg: Mapping[str, Any]) -> list[str]:
    """Match-level context features (surface and level one-hots, round, best-of, Elo probabilities)."""
    return (
        [f"surface_{s}" for s in surfaces(cfg)]
        + [f"level_{lv}" for lv in cfg["data"]["training_levels"]]
        + ["round_rank", "best_of", "elo_prob_a", "elo_surface_prob_a"]
    )


def output_columns(cfg: Mapping[str, Any]) -> list[str]:
    """All columns of the feature table, in output order."""
    cols = list(NON_FEATURE_COLUMNS) + context_feature_names(cfg)
    for f in player_feature_names(cfg):
        cols += [f"a_{f}", f"b_{f}", f"d_{f}"]
    return cols


# --------------------------------------------------------------------------- small helpers


def orient_a_is_winner(seed: int, match_id: str) -> bool:
    """CONTRACT coin flip: A is the winner iff ``int(sha256(f"{seed}:{match_id}")[:8], 16)`` is even."""
    return int(hashlib.sha256(f"{seed}:{match_id}".encode("utf-8")).hexdigest()[:8], 16) % 2 == 0


def _isnan(x: Any) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _hand_is_left(hand: Any, fill: float) -> float:
    """1.0 for 'L', 0.0 for 'R', ``fill`` otherwise (unknown / ambidextrous / missing; D-020)."""
    if hand == "L":
        return 1.0
    if hand == "R":
        return 0.0
    return fill


def _float_or_nan(x: Any) -> float:
    return NAN if _isnan(x) else float(x)


def _float_or_fill(x: Any, fill: float) -> float:
    return fill if _isnan(x) else float(x)


def static_fill(cfg: Mapping[str, Any]) -> tuple[float, float, float]:
    """(height, is_left, age) constants for missing static attributes, from ``features.static_fill`` (D-020)."""
    f = cfg["features"]["static_fill"]
    return float(f["height"]), float(f["is_left"]), float(f["age"])


def _snapshot(st: PlayerState, pid: str, opp: str, surface: str, today: int, h2h: H2H, start: float,
              winrate_k: list[int], recent_window: int, static: tuple[Any, Any, Any, Any, Any],
              fill: tuple[float, float, float]) -> list[Any]:
    """Pre-match player features in :func:`player_feature_names` order. Reads state only.

    Rank and rank points keep NaN (unranked is real pre-match information); height, is_left and age are filled
    with the ``features.static_fill`` constants (their missingness reflects the later career, D-020).
    """
    rates = st.stats.rates()
    rank, rank_points, age, height, hand = static
    fill_height, fill_left, fill_age = fill
    return (
        [st.elo, surface_rating(st, surface, start), ols_slope(st.elo_hist), st.matches_played,
         surface_count(st, surface)]
        + [winrate(st.results, k) for k in winrate_k]
        + [h2h.wins(pid, opp), h2h.surface_wins(pid, opp, surface), len(st.stats)]
        + [rates[r] for r in RATE_FEATURES]
        + [_float_or_nan(rank), _float_or_nan(rank_points), _float_or_fill(age, fill_age),
           _float_or_fill(height, fill_height), _hand_is_left(hand, fill_left), days_since(st.last_day, today),
           count_recent(st.recent_days, today, recent_window)]
    )


def _k_pair(st: PlayerState, params: tuple[EloParams, EloParams], mode: str, surface: str, tourney_id: str,
            start_date: Any) -> tuple[float, float]:
    """(overall K, surface K) for one player before this match. Uses only prior matches (CONTRACT section 2)."""
    p_o, p_s = params
    n_event = st.event_matches.get(tourney_id, 0)
    need_played = p_o.bonus_after_layoff or p_s.bonus_after_layoff
    played = layoff_played(st.tourneys, tourney_id, start_date) if need_played else ()
    k_o = p_o.k(st.matches_played, played, m_value(p_o.m, mode, n_event))
    k_s = p_s.k(surface_count(st, surface), played, m_value(p_s.m, mode, n_event))
    return k_o, k_s


# --------------------------------------------------------------------------- the builder


@dataclass
class PassResult:
    """Output of :func:`run_pass`: the final state plus the emitted rows (empty when ``emit_rows=False``)."""

    players: dict[str, PlayerState]
    h2h: H2H
    emitted_pos: list[int] = field(default_factory=list)       # positions (in the sorted table) of emitted rows
    a_won: list[bool] = field(default_factory=list)
    a_vals: list[list[Any]] = field(default_factory=list)      # player features of A, per emitted row
    b_vals: list[list[Any]] = field(default_factory=list)
    prob: list[float] = field(default_factory=list)            # elo_prob_a
    sprob: list[float] = field(default_factory=list)           # elo_surface_prob_a
    k_rows: list[tuple[str, str, float, float]] = field(default_factory=list)


def build_features(matches: pd.DataFrame, cfg: Mapping[str, Any], *, record_k: bool = False
                   ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Build the A/B-oriented feature table from the unified match table (CONTRACT section 2).

    Args:
        matches: unified winner/loser match table (CONTRACT section 1). Not mutated; any row order.
        cfg: the config dict (``tennis.config.load_config()``).
        record_k: also return ``k_log`` with one row per processed (non-walkover) match per player:
            ``match_id, player_id, k_overall, k_surface`` (RET/DEF included, with the K that would have applied).

    Returns:
        ``features`` or ``(features, k_log)``.
    """
    check_round_rank(matches, cfg)
    ms = sort_matches(matches.loc[~matches["is_walkover"].astype(bool)])
    res = run_pass(ms, cfg, record_k=record_k)
    features = _assemble(ms, res, cfg)
    if not record_k:
        return features
    k_log = pd.DataFrame(res.k_rows, columns=list(K_LOG_COLUMNS))
    return features, k_log


def run_pass(ms: pd.DataFrame, cfg: Mapping[str, Any], *, record_k: bool = False, emit_rows: bool = True
             ) -> PassResult:
    """The one chronological pass over ``ms`` (already sorted by the authoritative order, walkovers removed).

    For each block: read the pre-block state (K, Elo deltas, and the feature row of every emitted match) -> THEN
    update state. ``emit_rows=False`` skips writing feature rows (the state updates are identical); it is used
    by :mod:`tennis.engine.snapshot` to get the final state cheaply.
    """
    clean = cfg["cleaning"]
    upd_elo = bool(clean["retirement_updates_elo"])
    upd_stats = bool(clean["retirement_updates_stats"])
    upd_h2h = bool(clean["retirement_updates_h2h"])
    upd_form = bool(clean["retirement_updates_form"])
    fcfg = cfg["features"]
    winrate_k = [int(k) for k in fcfg["winrate_k"]]
    rolling_n = int(fcfg["rolling_n"])
    grad_window = int(fcfg["elo_grad_window"])
    recent_window = int(fcfg["recent_days_window"])
    params = (EloParams.from_config(cfg, "overall"), EloParams.from_config(cfg, "surface"))
    start = params[0].start_rating
    mode = str(cfg["elo"]["m_mode"])
    m_value(1.0, mode, 0)  # validate the mode up front
    seed = int(cfg["seed"])
    first_year = int(cfg["data"]["feature_first_year"])
    fill = static_fill(cfg)
    exempt = [str(x) for x in fcfg["rr_block_exempt_levels"]]

    # ---- plain Python sequences, one per column (fast iteration, no per-row pandas indexing)
    dates64 = ms["tourney_date"].to_numpy().astype("datetime64[D]")
    days = dates64.astype("int64").tolist()
    pydates = dates64.astype(object).tolist()  # datetime.date, for the layoff list
    years = pd.DatetimeIndex(ms["tourney_date"]).year.to_numpy()
    emit = (ms["is_training_level"].astype(bool).to_numpy() & ~ms["is_retirement"].astype(bool).to_numpy()
            & (years >= first_year)).tolist()
    is_ret = ms["is_retirement"].astype(bool).tolist()
    mids = ms["match_id"].astype(str).tolist()
    wids = ms["winner_id"].astype(str).tolist()
    lids = ms["loser_id"].astype(str).tolist()
    surf = ms["surface"].astype(str).tolist()
    tids = ms["tourney_id"].astype(str).tolist()
    # row-wise flag: this match belongs to a round-robin block (CONTRACT section 2, D-019)
    in_rr_block = ((ms["round"].astype(str) == "RR").to_numpy()
                   & ~ms["tourney_level"].astype(str).isin(exempt).to_numpy()).tolist()
    w_stats = ms[[f"w_{s}" for s in STAT_FIELDS]].to_numpy(dtype=float).tolist()
    l_stats = ms[[f"l_{s}" for s in STAT_FIELDS]].to_numpy(dtype=float).tolist()

    def static(side: str) -> list[tuple[Any, ...]]:
        cols = [f"{side}_rank", f"{side}_rank_points", f"{side}_age", f"{side}_ht", f"{side}_hand"]
        return list(zip(*(ms[c].tolist() for c in cols)))

    w_static, l_static = static("winner"), static("loser")

    players: dict[str, PlayerState] = {}
    h2h = H2H()
    results_len = max(winrate_k) if winrate_k else 1

    def get(pid: str) -> PlayerState:
        st = players.get(pid)
        if st is None:
            st = new_player(start, grad_window, results_len, rolling_n)
            players[pid] = st
        return st

    res = PassResult(players, h2h)
    emitted_pos, a_won_list, a_vals_rows, b_vals_rows = res.emitted_pos, res.a_won, res.a_vals, res.b_vals
    prob, sprob, k_rows = res.prob, res.sprob, res.k_rows

    n = len(mids)
    i = 0
    while i < n:
        # the block [i, j): one match, or a maximal run of RR matches of the same tourney_id
        j = i + 1
        if in_rr_block[i]:
            while j < n and in_rr_block[j] and tids[j] == tids[i]:
                j += 1

        # 1-2. read the (pre-block) state: K, Elo deltas and the emitted row of every match in the block
        pending: list[tuple[int, PlayerState, PlayerState, float, float, float, float]] = []
        for t in range(i, j):
            w, l, s, tid, day, d = wids[t], lids[t], surf[t], tids[t], days[t], pydates[t]
            sw, sl = get(w), get(l)
            kw_o, kw_s = _k_pair(sw, params, mode, s, tid, d)
            kl_o, kl_s = _k_pair(sl, params, mode, s, tid, d)
            if record_k:  # logged even for RET/DEF (the K that would have applied)
                k_rows.append((mids[t], w, kw_o, kw_s))
                k_rows.append((mids[t], l, kl_o, kl_s))
            dw, dl = elo_deltas(sw.elo, sl.elo, kw_o, kl_o)
            dsw, dsl = elo_deltas(surface_rating(sw, s, start), surface_rating(sl, s, start), kw_s, kl_s)
            pending.append((t, sw, sl, dw, dl, dsw, dsl))
            if emit_rows and emit[t]:
                fw = _snapshot(sw, w, l, s, day, h2h, start, winrate_k, recent_window, w_static[t], fill)
                fl = _snapshot(sl, l, w, s, day, h2h, start, winrate_k, recent_window, l_static[t], fill)
                a_w = orient_a_is_winner(seed, mids[t])
                fa, fb = (fw, fl) if a_w else (fl, fw)
                emitted_pos.append(t)
                a_won_list.append(a_w)
                a_vals_rows.append(fa)
                b_vals_rows.append(fb)
                prob.append(expected_score(fa[0], fb[0]))
                sprob.append(expected_score(fa[1], fb[1]))

        # 3. THEN update state, match by match in authoritative order
        for t, sw, sl, dw, dl, dsw, dsl in pending:
            w, l, s, tid, day, d = wids[t], lids[t], surf[t], tids[t], days[t], pydates[t]
            ret = is_ret[t]
            if not ret or upd_elo:
                record_elo(sw, s, sw.elo + dw, surface_rating(sw, s, start) + dsw)
                record_elo(sl, s, sl.elo + dl, surface_rating(sl, s, start) + dsl)
            if (not ret or upd_stats) and valid_stats(w_stats[t], l_stats[t]):
                sw.stats.push(w_stats[t], l_stats[t])
                sl.stats.push(l_stats[t], w_stats[t])
            if not ret or upd_h2h:
                h2h.record_win(w, l, s)
            if not ret or upd_form:
                record_result(sw, 1)
                record_result(sl, 0)
            record_match(sw, s, tid, d, day, recent_window)
            record_match(sl, s, tid, d, day, recent_window)
        i = j
    return res


def _assemble(ms: pd.DataFrame, res: PassResult, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Assemble the feature table once, from the sorted matches and the emitted rows of :func:`run_pass`."""
    surf_names = surfaces(cfg)
    levels = list(cfg["data"]["training_levels"])
    pf_names = player_feature_names(cfg)
    n_pf = len(pf_names)
    a_vals_rows, b_vals_rows = res.a_vals, res.b_vals
    pos = np.asarray(res.emitted_pos, dtype=np.int64)
    a_won = np.asarray(res.a_won, dtype=bool)
    sub = ms.iloc[pos]
    w_arr, l_arr = sub["winner_id"].astype(str).to_numpy(), sub["loser_id"].astype(str).to_numpy()
    wn_arr, ln_arr = sub["winner_name"].to_numpy(dtype=object), sub["loser_name"].to_numpy(dtype=object)
    out: dict[str, Any] = {
        "match_id": sub["match_id"].astype(str).to_numpy(),
        "date": sub["tourney_date"].to_numpy(),
        "tourney_id": sub["tourney_id"].astype(str).to_numpy(),
        "tourney_name": sub["tourney_name"].to_numpy(dtype=object),
        "a_id": np.where(a_won, w_arr, l_arr),
        "b_id": np.where(a_won, l_arr, w_arr),
        "a_name": np.where(a_won, wn_arr, ln_arr),
        "b_name": np.where(a_won, ln_arr, wn_arr),
        "y": a_won.astype(np.int64),
    }
    surf_arr = sub["surface"].astype(str).to_numpy()
    level_arr = sub["tourney_level"].astype(str).to_numpy()
    for sname in surf_names:
        out[f"surface_{sname}"] = (surf_arr == sname).astype(np.int64)
    for lv in levels:
        out[f"level_{lv}"] = (level_arr == lv).astype(np.int64)
    out["round_rank"] = sub["round_rank"].to_numpy(dtype=np.int64)
    out["best_of"] = sub["best_of"].to_numpy(dtype=np.int64)
    out["elo_prob_a"] = np.asarray(res.prob, dtype=float)
    out["elo_surface_prob_a"] = np.asarray(res.sprob, dtype=float)
    for j, f in enumerate(pf_names):
        av = [r[j] for r in a_vals_rows]
        bv = [r[j] for r in b_vals_rows]
        dv = [NAN if (_isnan(x) or _isnan(y)) else x - y for x, y in zip(av, bv)]
        out[f"a_{f}"] = av
        out[f"b_{f}"] = bv
        out[f"d_{f}"] = dv
    assert len(out) == len(NON_FEATURE_COLUMNS) + len(context_feature_names(cfg)) + 3 * n_pf
    features = pd.DataFrame(out, columns=output_columns(cfg))
    return _fix_dtypes(features, pf_names)


_INT_FEATURES: frozenset[str] = frozenset({
    "matches_played", "matches_surface_played", "h2h_wins", "h2h_surface_wins", "n_stat_matches",
    "matches_last_14_days",
})


def _fix_dtypes(df: pd.DataFrame, pf_names: list[str]) -> pd.DataFrame:
    """Count features as int64, every other player feature as float64 (also when the table is empty)."""
    for f in pf_names:
        dt_ = np.int64 if f in _INT_FEATURES else np.float64
        for p in ("a_", "b_", "d_"):
            df[p + f] = df[p + f].astype(dt_)
    return df


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    """Read matches.parquet, build the features, write features.parquet; print row count and runtime."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None, help="path to config.yaml (default: repo config)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    src = repo_path(cfg["paths"]["matches"])
    dst = repo_path(cfg["paths"]["features"])
    t0 = time.perf_counter()
    matches = pd.read_parquet(src)
    t1 = time.perf_counter()
    features = build_features(matches, cfg)
    t2 = time.perf_counter()
    dst.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(dst, index=False)
    t3 = time.perf_counter()
    print(f"read {len(matches):,} matches from {src} in {t1 - t0:.1f}s")
    print(f"built {len(features):,} feature rows x {features.shape[1]} columns in {t2 - t1:.1f}s")
    print(f"wrote {dst} in {t3 - t2:.1f}s (total {t3 - t0:.1f}s)")


if __name__ == "__main__":
    main()
