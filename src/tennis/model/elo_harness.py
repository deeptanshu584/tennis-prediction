"""Elo hyper-parameter tuning harness (PLAN section 6 Phase 4; the author's ``run_ELO_evaluation``).

One walk over every non-walkover match of ``data/interim/matches.parquet`` (1990 on) in the authoritative order:
predict each match with the pre-match Elo probability, THEN update. The Elo rules are the engine's (CONTRACT
section 2), reusing :mod:`tennis.engine.ordering`, :mod:`tennis.engine.state` and :mod:`tennis.engine.elo`:

- walkovers are dropped; RET/DEF matches count as played (Ne, event counts, layoff list) but don't move Elo unless
  ``cleaning.retirement_updates_elo``;
- K = ``k_factor`` if ``div_number == 0`` else ``min(max_k, max(base_k, div_number / (Ne + m)))``, times the layoff
  bonus when enabled; ``m`` follows ``elo.m_mode`` (``event_matches_so_far``: base m + matches the player already
  played in this tourney_id, qualifying included; D-008);
- round-robin blocks (D-019): every match of a block is predicted from, and gets its K / expected score from, the
  state before the block; the block's updates are then applied in order.

Speed. Ne (overall and per surface), the in-event count and the layoff multiplier depend only on the match history,
never on the Elo hyper-parameters, so :func:`prepare` computes them ONCE per match side. The layoff multiplier is
incremental (per player: distinct-tournament count, last tournament start date, and the run of consecutive short
gaps at the recent end), equal to :func:`tennis.engine.elo.k_bonus_after_layoff` but O(1) per match. Each grid
point then costs a vectorised K and one rating walk. Overall and surface Elo are independent (neither update reads
the other), so they're tuned in separate walks, each scored on its own probability.

Scoring: matches that are training-level, not qualifying, not RET/DEF, dated in [``elo_harness.tune_start``,
``tune_end``] (2015-2024; never a test period), and past the first ``elo_harness.warmup_matches`` processed
matches. Metrics: log loss of the Elo probability (primary, clipped with ``evaluation.log_loss_eps``), accuracy
(half credit at p = 0.5) and Brier, on the winner-oriented probability (all three are symmetric under an A/B flip).

CLI::

    .venv/Scripts/python -m tennis.model.elo_harness     # full grid -> experiments/elo_tuning.jsonl,
                                                          # reports/elo_tuning.md, reports/figures/phase4_*.png
"""

from __future__ import annotations

import itertools
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from multiprocessing import Pool
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from tennis.config import config_hash, load_config, repo_path
from tennis.engine.elo import EloParams
from tennis.engine.ordering import check_round_rank, sort_matches
from tennis.engine.state import elo_deltas, m_value
from tennis.engine.elo import expected_score
from tennis.model import evaluate as ev

WHICH: tuple[str, ...] = ("overall", "surface")


# ------------------------------------------------------------------------------------------------ settings
@dataclass(frozen=True)
class EloSetting:
    """One grid point for one Elo flavour (fields as in ``config.elo.<overall|surface>``)."""
    which: str
    k_factor: float
    div_number: float
    m: float
    base_k: float
    max_k: float
    bonus_after_layoff: bool

    @property
    def label(self) -> str:
        bonus = "bonus" if self.bonus_after_layoff else "no bonus"
        if self.div_number == 0:
            return f"fixed K={self.k_factor:g}, {bonus}"
        return f"div={self.div_number:g} m={self.m:g} base={self.base_k:g} max={self.max_k:g}, {bonus}"

    def config_block(self) -> dict[str, Any]:
        """The ``elo.<which>`` YAML block for this setting."""
        return {"k_factor": float(self.k_factor), "div_number": float(self.div_number), "m": float(self.m),
                "base_k": float(self.base_k), "max_k": float(self.max_k),
                "bonus_after_layoff": bool(self.bonus_after_layoff)}


def current_setting(cfg: Mapping[str, Any], which: str) -> EloSetting:
    """The setting currently in ``config.elo.<which>``."""
    p = EloParams.from_config(cfg, which)  # type: ignore[arg-type]
    return EloSetting(which, p.k_factor, p.div_number, p.m, p.base_k, p.max_k, p.bonus_after_layoff)


def grid_settings(cfg: Mapping[str, Any], which: str) -> list[EloSetting]:
    """``elo_harness.grid``: the dynamic-K product (div_number x m x base_k x max_k x bonus), then fixed K
    (``div_number = 0``, ``k_factor`` in ``fixed_k``) with the bonus off/on. Fields a fixed K doesn't use keep the
    current config values."""
    g = cfg["elo_harness"]["grid"]
    cur = current_setting(cfg, which)
    out = [EloSetting(which, cur.k_factor, float(d), float(m), float(b), float(x), bool(bo))
           for d, m, b, x, bo in itertools.product(g["div_number"], g["m"], g["base_k"], g["max_k"],
                                                     g["bonus_after_layoff"])]
    out += [EloSetting(which, float(k), 0.0, cur.m, cur.base_k, cur.max_k, bool(bo))
            for k in g.get("fixed_k", []) for bo in g["bonus_after_layoff"]]
    return out


# ------------------------------------------------------------------------------------------- layoff bonus
@lru_cache(maxsize=64)
def _bonus_table(threshold: float, min_gaps: int, start_bonus: float, step: float, floor: float,
                 max_run: int = 1000) -> tuple[float, ...]:
    """``table[r]`` = the bonus after ``r`` consecutive short gaps at the recent end, computed with the same
    sequential subtraction as :func:`k_bonus_after_layoff` (so the floats are identical)."""
    out = []
    b = start_bonus
    for r in range(max_run + 1):
        out.append(max(floor, b))
        b -= step
    return tuple(out)


def layoff_bonus(n_gaps: int, run: int, layoff: Mapping[str, float]) -> float:
    """Layoff multiplier from the gap count and the run of short gaps at the recent end (equals
    ``k_bonus_after_layoff(played, **layoff)`` for the corresponding ``played`` list)."""
    if n_gaps <= int(layoff["min_gaps"]):
        return 1.0
    t = _bonus_table(float(layoff["threshold"]), int(layoff["min_gaps"]), float(layoff["start_bonus"]),
                     float(layoff["step"]), float(layoff["floor"]))
    return t[min(run, len(t) - 1)]


class LayoffTracker:
    """Per-player incremental layoff state: distinct tournaments played, last tournament start day, and the
    number of consecutive gaps shorter than ``threshold`` at the recent end of the gap list.

    Tournaments must be fed in match order (start dates non-decreasing), as the chronological pass does; then the
    first-appearance order of tourney_ids is chronological and the gaps are consecutive differences.
    """

    __slots__ = ("threshold", "n", "last", "run")

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.n: dict[int, int] = {}
        self.last: dict[int, int] = {}
        self.run: dict[int, int] = {}

    def peek(self, player: int, day: int, new_tourney: bool) -> tuple[int, int]:
        """(number of gaps, short-gap run) of the layoff ``played`` list for a match on ``day``; ``new_tourney`` is
        True when the player has no prior match in this tourney_id (its start date is then appended)."""
        n = self.n.get(player, 0)
        run = self.run.get(player, 0)
        if not new_tourney:
            return max(n - 1, 0), run
        if n == 0:
            return 0, 0
        return n, (run + 1 if day - self.last[player] < self.threshold else 0)

    def add(self, player: int, day: int) -> None:
        """Record a new tournament (first match of the player in a tourney_id) starting on ``day``."""
        n = self.n.get(player, 0)
        if n:
            self.run[player] = self.run.get(player, 0) + 1 if day - self.last[player] < self.threshold else 0
        self.n[player] = n + 1
        self.last[player] = day


# ------------------------------------------------------------------------------------------------ prepare
@dataclass
class Prepared:
    """Per-match arrays (authoritative order) that don't depend on the Elo hyper-parameters.

    ``w``/``l`` = winner/loser side. ``ent_*`` are rating-entity indices (player for overall Elo,
    player x surface for surface Elo); ``ne_*`` prior match counts (overall / on this surface); ``n_event``
    matches already played in this tourney_id; ``bonus`` the layoff multiplier; ``block_end[i]`` the end
    (exclusive) of the block starting at ``i`` (``i + 1`` outside RR blocks); ``update`` whether the match moves
    Elo; ``score`` whether it is scored; ``season`` its tourney_date year.
    """
    match_id: np.ndarray
    ent_overall_w: np.ndarray
    ent_overall_l: np.ndarray
    ent_surface_w: np.ndarray
    ent_surface_l: np.ndarray
    ne_overall_w: np.ndarray
    ne_overall_l: np.ndarray
    ne_surface_w: np.ndarray
    ne_surface_l: np.ndarray
    n_event_w: np.ndarray
    n_event_l: np.ndarray
    bonus_w: np.ndarray
    bonus_l: np.ndarray
    block_end: np.ndarray
    update: np.ndarray
    score: np.ndarray
    season: np.ndarray
    winner_id: np.ndarray
    loser_id: np.ndarray
    n_players: int
    n_surfaces: int
    start_rating: float
    m_mode: str


def prepare(matches: pd.DataFrame, cfg: Mapping[str, Any]) -> Prepared:
    """Sort, drop walkovers and precompute the hyper-parameter-free per-match arrays (one chronological pass)."""
    check_round_rank(matches, cfg)
    ms = sort_matches(matches.loc[~matches["is_walkover"].astype(bool)])
    n = len(ms)
    h = cfg["elo_harness"]
    layoff = EloParams.from_config(cfg, "overall").layoff
    mode = str(cfg["elo"]["m_mode"])
    m_value(1.0, mode, 0)  # validate

    pid_codes, pid_uniques = pd.factorize(pd.concat([ms["winner_id"], ms["loser_id"]]).astype(str))
    w_p, l_p = pid_codes[:n].tolist(), pid_codes[n:].tolist()
    surf_codes, surf_uniques = pd.factorize(ms["surface"].astype(str))
    s_c = surf_codes.tolist()
    n_s = len(surf_uniques)
    tid_codes = pd.factorize(ms["tourney_id"].astype(str))[0].tolist()
    days = ms["tourney_date"].to_numpy().astype("datetime64[D]").astype("int64").tolist()

    exempt = [str(x) for x in cfg["features"]["rr_block_exempt_levels"]]
    in_rr = ((ms["round"].astype(str) == "RR").to_numpy()
             & ~ms["tourney_level"].astype(str).isin(exempt).to_numpy()).tolist()

    ne_o = {}  # player -> matches
    ne_s = {}  # player*n_s + surface -> matches
    ev_cnt: dict[tuple[int, int], int] = {}
    lt = LayoffTracker(float(layoff["threshold"]))
    arrs = {k: np.zeros(n, dtype=np.int64) for k in ("neow", "neol", "nesw", "nesl", "nevw", "nevl")}
    bw = np.ones(n)
    bl = np.ones(n)
    block_end = np.zeros(n, dtype=np.int64)

    i = 0
    while i < n:
        j = i + 1
        if in_rr[i]:
            while j < n and in_rr[j] and tid_codes[j] == tid_codes[i]:
                j += 1
        for t in range(i, j):  # read pre-block state
            block_end[t] = j
            tid, day, s = tid_codes[t], days[t], s_c[t]
            for p, kneo, knes, knev, barr in ((w_p[t], "neow", "nesw", "nevw", bw), (l_p[t], "neol", "nesl", "nevl", bl)):
                arrs[kneo][t] = ne_o.get(p, 0)
                arrs[knes][t] = ne_s.get(p * n_s + s, 0)
                ne_ev = ev_cnt.get((p, tid), 0)
                arrs[knev][t] = ne_ev
                g, run = lt.peek(p, day, ne_ev == 0)
                barr[t] = layoff_bonus(g, run, layoff)
        for t in range(i, j):  # then update, in order (as record_match)
            tid, day, s = tid_codes[t], days[t], s_c[t]
            for p in (w_p[t], l_p[t]):
                ne_o[p] = ne_o.get(p, 0) + 1
                ne_s[p * n_s + s] = ne_s.get(p * n_s + s, 0) + 1
                key = (p, tid)
                if key not in ev_cnt:
                    lt.add(p, day)
                    ev_cnt[key] = 0
                ev_cnt[key] += 1
        i = j

    is_ret = ms["is_retirement"].astype(bool).to_numpy()
    update = ~is_ret | bool(cfg["cleaning"]["retirement_updates_elo"])
    d = ms["tourney_date"]
    score = (ms["is_training_level"].astype(bool).to_numpy() & ~ms["is_qualifying"].astype(bool).to_numpy()
             & ~is_ret & (d >= pd.Timestamp(h["tune_start"])).to_numpy() & (d <= pd.Timestamp(h["tune_end"])).to_numpy()
             & (np.arange(n) >= int(h["warmup_matches"])))
    wp, lp, sc = np.asarray(w_p), np.asarray(l_p), np.asarray(s_c)
    return Prepared(
        match_id=ms["match_id"].astype(str).to_numpy(),
        ent_overall_w=wp, ent_overall_l=lp, ent_surface_w=wp * n_s + sc, ent_surface_l=lp * n_s + sc,
        ne_overall_w=arrs["neow"], ne_overall_l=arrs["neol"], ne_surface_w=arrs["nesw"], ne_surface_l=arrs["nesl"],
        n_event_w=arrs["nevw"], n_event_l=arrs["nevl"], bonus_w=bw, bonus_l=bl, block_end=block_end,
        update=update, score=score, season=pd.DatetimeIndex(d).year.to_numpy(),
        winner_id=ms["winner_id"].astype(str).to_numpy(), loser_id=ms["loser_id"].astype(str).to_numpy(),
        n_players=len(pid_uniques), n_surfaces=n_s,
        start_rating=float(cfg["elo"]["start_rating"]), m_mode=mode,
    )


# ------------------------------------------------------------------------------------------- K and walk
def k_arrays(prep: Prepared, s: EloSetting) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised :func:`tennis.engine.elo.k_factor` for the winner and loser side of every match."""
    out = []
    for side in ("w", "l"):
        ne = getattr(prep, f"ne_{s.which}_{side}").astype(float)
        if s.div_number == 0:
            k = np.full(len(ne), float(s.k_factor))
        else:
            ev_ = getattr(prep, f"n_event_{side}") if prep.m_mode == "event_matches_so_far" else 0
            m = float(s.m) + ev_
            k = np.minimum(s.max_k, np.maximum(s.base_k, s.div_number / (ne + m)))
        if s.bonus_after_layoff:
            k = k * getattr(prep, f"bonus_{side}")
        out.append(k)
    return out[0], out[1]


def walk(ent_w: Sequence[int], ent_l: Sequence[int], k_w: Sequence[float], k_l: Sequence[float],
         update: Sequence[bool], block_end: Sequence[int], start: float, n_entities: int
         ) -> tuple[np.ndarray, np.ndarray]:
    """Chronological Elo pass. Returns the pre-match (pre-block) ratings of the winner and loser of every match.

    Deltas use :func:`tennis.engine.state.elo_deltas` on the pre-block ratings and are added to the current
    ratings in match order, exactly as the feature builder does.
    """
    ent_w, ent_l = list(ent_w), list(ent_l)
    k_w, k_l = list(map(float, k_w)), list(map(float, k_l))
    update, block_end = list(map(bool, update)), list(map(int, block_end))
    r = [float(start)] * n_entities
    n = len(ent_w)
    rw = [0.0] * n
    rl = [0.0] * n
    i = 0
    while i < n:
        j = block_end[i]
        if j == i + 1:
            a, b = ent_w[i], ent_l[i]
            ra, rb = r[a], r[b]
            rw[i], rl[i] = ra, rb
            if update[i]:
                dw, dl = elo_deltas(ra, rb, k_w[i], k_l[i])
                r[a] = ra + dw
                r[b] = rb + dl
        else:
            pend = []
            for t in range(i, j):
                a, b = ent_w[t], ent_l[t]
                rw[t], rl[t] = r[a], r[b]
                pend.append((a, b) + elo_deltas(r[a], r[b], k_w[t], k_l[t]))
            for t, (a, b, dw, dl) in zip(range(i, j), pend):
                if update[t]:
                    r[a] = r[a] + dw
                    r[b] = r[b] + dl
        i = j
    return np.asarray(rw), np.asarray(rl)


def win_prob(rw: np.ndarray, rl: np.ndarray) -> np.ndarray:
    """Winner's pre-match expected score, element by element with :func:`tennis.engine.elo.expected_score`."""
    return np.fromiter((expected_score(a, b) for a, b in zip(rw.tolist(), rl.tolist())), float, len(rw))


def run_setting(prep: Prepared, s: EloSetting) -> tuple[np.ndarray, np.ndarray]:
    """Pre-match (winner, loser) ratings of every match for one setting."""
    kw, kl = k_arrays(prep, s)
    n_ent = prep.n_players * (prep.n_surfaces if s.which == "surface" else 1)
    return walk(getattr(prep, f"ent_{s.which}_w"), getattr(prep, f"ent_{s.which}_l"), kw, kl, prep.update,
                prep.block_end, prep.start_rating, n_ent)


def scored_win_prob(prep: Prepared, s: EloSetting) -> np.ndarray:
    """Winner's pre-match Elo probability on the scored matches (for paired comparisons)."""
    rw, rl = run_setting(prep, s)
    return win_prob(rw[prep.score], rl[prep.score])


def edge_probes(cfg: Mapping[str, Any], best: EloSetting) -> list[EloSetting]:
    """Settings one grid step beyond each edge the (dynamic-K) best sits on: ``div_number``, ``base_k`` or
    ``max_k`` at the grid maximum (step = the last grid spacing), ``m`` at the minimum when the step down keeps
    ``m >= 1``; plus all those steps together when there are several. Empty for a fixed-K best."""
    if best.div_number == 0:
        return []
    g = cfg["elo_harness"]["grid"]
    steps: dict[str, float] = {}
    for f in ("div_number", "base_k", "max_k"):
        vals = sorted(float(v) for v in g[f])
        if len(vals) > 1 and getattr(best, f) == vals[-1]:
            steps[f] = vals[-1] + (vals[-1] - vals[-2])
    mv = sorted(float(v) for v in g["m"])
    if len(mv) > 1 and best.m == mv[0] and mv[0] - (mv[1] - mv[0]) >= 1:
        steps["m"] = mv[0] - (mv[1] - mv[0])
    out = [replace(best, **{f: v}) for f, v in steps.items()]
    if len(steps) > 1:
        out.append(replace(best, **steps))
    return out


def score_setting(prep: Prepared, s: EloSetting, eps: float) -> dict[str, Any]:
    """Metrics of one setting on the scored matches: overall and per season."""
    rw, rl = run_setting(prep, s)
    m = prep.score
    p = win_prob(rw[m], rl[m])
    y = np.ones(len(p))
    out: dict[str, Any] = {"n": int(m.sum()), "log_loss": ev.log_loss(y, p, eps), "accuracy": ev.accuracy(y, p),
                           "brier": ev.brier(y, p), "seasons": {}}
    seasons = prep.season[m]
    for yr in np.unique(seasons):
        k = seasons == yr
        out["seasons"][str(int(yr))] = {"n": int(k.sum()), "log_loss": ev.log_loss(y[k], p[k], eps),
                                        "accuracy": ev.accuracy(y[k], p[k]), "brier": ev.brier(y[k], p[k])}
    return out


# ------------------------------------------------------------------------------------- multiprocessing
_PREP: Prepared | None = None
_EPS: float = 1e-15


def _init_worker(prep: Prepared, eps: float) -> None:
    global _PREP, _EPS
    _PREP, _EPS = prep, eps


def _worker(s: EloSetting) -> tuple[EloSetting, dict[str, Any]]:
    assert _PREP is not None
    return s, score_setting(_PREP, s, _EPS)


def run_grid(prep: Prepared, settings: Sequence[EloSetting], eps: float, processes: int | None = None
             ) -> list[tuple[EloSetting, dict[str, Any]]]:
    """Score every setting, in parallel (``processes`` defaults to CPU count - 1); results in input order."""
    procs = processes if processes is not None else max(1, (os.cpu_count() or 2) - 1)
    if procs <= 1:
        return [(s, score_setting(prep, s, eps)) for s in settings]
    with Pool(procs, initializer=_init_worker, initargs=(prep, eps)) as pool:
        return pool.map(_worker, settings, chunksize=1)


# ------------------------------------------------------------------------------------------------- report
def _rank(results: list[tuple[EloSetting, dict[str, Any]]], which: str) -> list[tuple[EloSetting, dict[str, Any]]]:
    """Settings of one flavour, best first: log loss, then accuracy (higher), then label for determinism."""
    rs = [r for r in results if r[0].which == which]
    return sorted(rs, key=lambda r: (r[1]["log_loss"], -r[1]["accuracy"], r[0].label))


def _heatmap(results: list[tuple[EloSetting, dict[str, Any]]], cfg: Mapping[str, Any], path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = cfg["elo_harness"]["grid"]
    rows = [(d, m) for d in g["div_number"] for m in g["m"]]
    cols = [(b, x) for b in g["base_k"] for x in g["max_k"]]
    bonuses = list(g["bonus_after_layoff"])
    fig, axes = plt.subplots(len(WHICH), len(bonuses), figsize=(6.4 * len(bonuses), 5.6 * len(WHICH)),
                             squeeze=False)
    ink, muted = "#2b2b2b", "#6b6b6b"
    for i, which in enumerate(WHICH):
        rs = {(s.div_number, s.m, s.base_k, s.max_k, s.bonus_after_layoff): r["log_loss"]
              for s, r in results if s.which == which and s.div_number != 0}
        vals = np.array(list(rs.values()))
        lo, hi = vals.min(), np.quantile(vals, 0.9)
        for j, bo in enumerate(bonuses):
            ax = axes[i][j]
            mat = np.array([[rs[(d, m, b, x, bo)] for b, x in cols] for d, m in rows])
            im = ax.imshow(mat, cmap="Blues_r", vmin=lo, vmax=hi, aspect="auto")
            best = np.unravel_index(np.argmin(mat), mat.shape)
            for (r_, c_), v in np.ndenumerate(mat):
                dark = (v - lo) / max(hi - lo, 1e-12) < 0.45
                ax.text(c_, r_, f"{v:.4f}", ha="center", va="center", fontsize=6.5,
                        color="white" if dark else ink, fontweight="bold" if (r_, c_) == best else "normal")
            ax.add_patch(plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False, ec="#eb6834", lw=2))
            ax.set_xticks(range(len(cols)), [f"b{b:g}\nmax{x:g}" for b, x in cols], fontsize=7, color=muted)
            ax.set_yticks(range(len(rows)), [f"div {d:g}, m {m:g}" for d, m in rows], fontsize=7.5, color=muted)
            ax.set_title(f"{which} Elo, layoff bonus {'on' if bo else 'off'}: log loss 2015–2024",
                         loc="left", fontsize=10, color=ink)
            for sp in ax.spines.values():
                sp.set_visible(False)
            fig.colorbar(im, ax=ax, shrink=0.8).ax.tick_params(labelsize=7, colors=muted)
    fig.text(0.01, 0.005, "Columns: base_k / max_k. Darker = lower (better) log loss; orange box = best cell in the "
             "panel; colour scale capped at the 90th percentile per flavour.", fontsize=8, color=muted)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _season_plot(series: Mapping[str, Mapping[str, Any]], path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#2b2b2b", "#6b6b6b", "#e6e6e3"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, metric, lab in ((axes[0], "log_loss", "log loss (lower is better)"),
                            (axes[1], "accuracy", "accuracy (higher is better)")):
        for i, (name, res) in enumerate(series.items()):
            yrs = sorted(res["seasons"])
            ax.plot([int(y) for y in yrs], [res["seasons"][y][metric] for y in yrs], color=ev.SERIES_COLORS[i % 4],
                    marker=ev.SERIES_MARKERS[i % 4], lw=2, ms=6, mec="white", mew=1.2, label=name)
        ax.set_title(lab, loc="left", fontsize=10, color=ink)
        ax.grid(color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=muted, labelsize=8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Elo per season 2015–2024: tuned vs current default", x=0.01, ha="left", fontsize=11, color=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _yaml_block(which: str, s: EloSetting) -> str:
    b = s.config_block()
    lines = [f"  {which}:"] + [f"    {k}: {str(v).lower() if isinstance(v, bool) else v}" for k, v in b.items()]
    return "\n".join(lines)


def write_report(results: list[tuple[EloSetting, dict[str, Any]]], cfg: Mapping[str, Any], runtime: dict[str, float],
                 prep: Prepared, probe_results: Mapping[str, list[tuple[EloSetting, dict[str, Any]]]]
                 ) -> dict[str, Any]:
    """Figures + ``reports/elo_tuning.md``. Returns the chosen settings and default results."""
    n_matches, n_scored = len(prep.match_id), int(prep.score.sum())
    figs = repo_path(cfg["paths"]["figures"])
    figs.mkdir(parents=True, exist_ok=True)
    _heatmap(results, cfg, figs / "phase4_elo_logloss_heatmap.png")
    h = cfg["elo_harness"]
    alarm = float(cfg["evaluation"]["leakage_alarm_accuracy"])
    chosen, defaults = {}, {}
    L = ["# Phase 4 — Elo tuning\n",
         f"Generated by `python -m tennis.model.elo_harness` · config hash `{config_hash(dict(cfg))}` · git "
         f"`{ev.git_commit() or 'n/a'}`. One JSON line per grid point in `experiments/elo_tuning.jsonl`.\n",
         f"**Walk:** {n_matches:,} non-walkover matches 1990 → 2026 (all levels incl. Challenger and qualifying, "
         "RET/DEF included as played but not moving Elo) in the authoritative order, with the engine's rules "
         f"(m_mode `{cfg['elo']['m_mode']}`, RR blocks from pre-block state, layoff bonus as "
         "`k_bonus_after_layoff`). The harness reproduces the feature table's `a_elo`/`a_elo_surface` exactly "
         "under the current config (test `test_harness_matches_feature_table`).\n",
         f"**Scored:** {n_scored:,} matches: training-level, not qualifying, not RET/DEF, tourney_date in "
         f"{h['tune_start']} … {h['tune_end']}, after the first {int(h['warmup_matches']):,} processed matches. "
         "No test period is touched (Wimbledon 2025; 2026-04-01 → 2026-06-07). Metric: log loss of the Elo "
         "probability (primary), accuracy (half credit at p = 0.5) and Brier as tie-breaker/context.\n",
         "**Caveat (D-004):** the tuning window 2015–2024 contains the dev validation window 2023–2024 used by "
         "Phase 5. That is selection overlap on validation data only (Elo hyper-parameters chosen with 2023–24 "
         "in view), not test contamination; the Phase 5 validation numbers for Elo are therefore slightly "
         "optimistic relative to a truly unseen window.\n",
         f"**Grid:** {len([r for r in results if r[0].which == 'overall'])} settings per flavour "
         "(div_number × m × base_k × max_k × bonus, plus fixed K = 24 with bonus off/on), overall and surface "
         f"tuned separately. Runtime: prepare {runtime['prepare']:.1f}s, grid {runtime['grid']:.1f}s on "
         f"{runtime['processes']} processes.\n"]

    for which in WHICH:
        ranked = _rank(results, which)
        cur = current_setting(cfg, which)
        default = next(r for r in results if r[0] == cur)
        chosen[which], defaults[which] = ranked[0], default
        L.append(f"## {which.capitalize()} Elo — top 15 of {len(ranked)}\n")
        rows = []
        for i, (s, r) in enumerate(ranked[:15], 1):
            rows.append([i, s.label, f"{r['log_loss']:.5f}", f"{r['accuracy']:.4f}", f"{r['brier']:.5f}"])
        rows.append(["…", "", "", "", ""])
        pos = next(i for i, x in enumerate(ranked, 1) if x[0] == cur)
        rows.append([pos, f"**current default** ({cur.label})", f"{default[1]['log_loss']:.5f}",
                     f"{default[1]['accuracy']:.4f}", f"{default[1]['brier']:.5f}"])
        rows.append([len(ranked), f"worst ({ranked[-1][0].label})", f"{ranked[-1][1]['log_loss']:.5f}",
                     f"{ranked[-1][1]['accuracy']:.4f}", f"{ranked[-1][1]['brier']:.5f}"])
        L.append(ev._md_table(["#", "Setting", "Log loss", "Accuracy", "Brier"], rows) + "\n")
        # paired bootstrap: best minus a few reference settings, on the same scored matches
        best_s = ranked[0][0]
        fixed = [x for x in ranked if x[0].div_number == 0]
        refs = [("runner-up", ranked[1][0]), ("best fixed K", fixed[0][0] if fixed else None),
                ("best without layoff bonus", next((x[0] for x in ranked if not x[0].bonus_after_layoff), None)),
                ("current default", cur)]
        p_best = scored_win_prob(prep, best_s)
        y1 = np.ones(len(p_best))
        rows = []
        for name, s_ref in refs:
            if s_ref is None or s_ref == best_s:
                continue
            d, lo, hi = ev.paired_bootstrap_diff(y1, p_best, scored_win_prob(prep, s_ref), "log_loss", cfg=cfg)
            da, alo, ahi = ev.paired_bootstrap_diff(y1, p_best, scored_win_prob(prep, s_ref), "accuracy", cfg=cfg)
            rows.append([f"{name} ({s_ref.label})", f"{d:+.5f} [{lo:+.5f}, {hi:+.5f}]",
                         f"{da:+.4f} [{alo:+.4f}, {ahi:+.4f}]"])
        L.append("Best minus reference, paired bootstrap over the scored matches "
                 f"({int(cfg['evaluation']['bootstrap_samples']):,} resamples; negative Δ log loss = best is better):\n")
        L.append(ev._md_table(["Reference", "Δ log loss", "Δ accuracy"], rows) + "\n")

        probes = probe_results.get(which, [])
        if probes:
            L.append(f"**Beyond-grid probe** — the best sits on a grid edge, so one step past each edge was "
                     "scored (diagnostic only; not selectable under the configured grid):\n")
            rows = [[s.label, f"{r['log_loss']:.5f}", f"{r['log_loss'] - ranked[0][1]['log_loss']:+.5f}",
                     f"{r['accuracy']:.4f}"] for s, r in probes]
            L.append(ev._md_table(["Probe", "Log loss", "Δ vs grid best", "Accuracy"], rows) + "\n")

        L.append(f"### {which.capitalize()} Elo — effect of each hyper-parameter (dynamic-K settings)\n")
        dyn = [(s, r) for s, r in results if s.which == which and s.div_number != 0]
        rows = []
        for field in ("div_number", "m", "base_k", "max_k", "bonus_after_layoff"):
            for v in sorted({getattr(s, field) for s, _ in dyn}):
                ll = [r["log_loss"] for s, r in dyn if getattr(s, field) == v]
                acc = [r["accuracy"] for s, r in dyn if getattr(s, field) == v]
                rows.append([field, v, len(ll), f"{np.mean(ll):.5f}", f"{np.min(ll):.5f}", f"{np.mean(acc):.4f}"])
        for bo in (False, True):
            fx = [r for s, r in results if s.which == which and s.div_number == 0 and s.bonus_after_layoff == bo]
            for r in fx:
                rows.append(["fixed K = 24", bo, 1, f"{r['log_loss']:.5f}", f"{r['log_loss']:.5f}",
                             f"{r['accuracy']:.4f}"])
        L.append(ev._md_table(["Hyper-parameter", "Value", "Settings", "Mean log loss", "Best log loss",
                               "Mean accuracy"], rows) + "\n")

    L.append("![log-loss heatmap](figures/phase4_elo_logloss_heatmap.png)\n")

    L.append("## Chosen settings\n")
    L.append("Best log loss per flavour (accuracy breaks ties). Per season, vs the current default:\n")
    series = {}
    for which in WHICH:
        s, r = chosen[which]
        ds, dr = defaults[which]
        series[f"{which} tuned ({s.label})"] = r
        series[f"{which} default"] = dr
        rows = []
        for yr in sorted(r["seasons"]):
            a, b = r["seasons"][yr], dr["seasons"][yr]
            rows.append([yr, f"{a['n']:,}", f"{a['log_loss']:.4f}", f"{b['log_loss']:.4f}",
                         f"{a['log_loss'] - b['log_loss']:+.4f}", f"{a['accuracy']:.4f}", f"{b['accuracy']:.4f}"])
        rows.append(["2015–2024", f"{r['n']:,}", f"**{r['log_loss']:.4f}**", f"{dr['log_loss']:.4f}",
                     f"{r['log_loss'] - dr['log_loss']:+.4f}", f"**{r['accuracy']:.4f}**", f"{dr['accuracy']:.4f}"])
        L.append(f"### {which.capitalize()}: {s.label}\n")
        L.append(ev._md_table(["Season", "n", "Log loss tuned", "Log loss default", "Δ", "Acc tuned", "Acc default"],
                              rows) + "\n")
    _season_plot(series, figs / "phase4_elo_seasons.png")
    L.append("![per-season](figures/phase4_elo_seasons.png)\n")
    notes = []
    for which in WHICH:
        s = chosen[which][0]
        if s.div_number != 0:
            k_floor_at = s.div_number / s.base_k - s.m if s.base_k > 0 else float("inf")
            notes.append(f"- {which}: K = max({s.base_k:g}, {s.div_number:g}/(Ne + m)) stops shrinking after about "
                         f"{max(k_floor_at, 0):.0f} prior matches, so established players keep K ≈ {s.base_k:g} "
                         "(the current default lets K fall towards 0, e.g. 800/501 ≈ 1.6 at 500 matches, so ratings "
                         "barely react to form). base_k is the dominant hyper-parameter; m and max_k matter little.")
        pr = probe_results.get(which, [])
        if pr:
            bp = min(pr, key=lambda x: x[1]["log_loss"])
            notes.append(f"- {which}: the grid best sits on a grid edge; the best probe ({bp[0].label}) improves log "
                         f"loss by a further {chosen[which][1]['log_loss'] - bp[1]['log_loss']:.5f}. The optimum lies "
                         "just outside the configured grid (orchestrator: consider extending div_number / base_k "
                         "upward and re-running).")
    L.append("**Findings:**\n\n" + "\n".join(notes) + "\n")
    L.append("Config block for the orchestrator (`config.yaml`, under `elo:`):\n")
    L.append("```yaml\n" + _yaml_block("overall", chosen["overall"][0]) + "\n"
             + _yaml_block("surface", chosen["surface"][0]) + "\n```\n")
    every = list(results) + [x for v in probe_results.values() for x in v]
    over = [f"{s.which} {s.label}: {k} {v:.4f}" for s, r in every
            for k, v in [("2015–24", r["accuracy"])] + [(y, x["accuracy"]) for y, x in r["seasons"].items()]
            if v > alarm]
    L.append(f"**Leakage alarm:** {'NONE above ' + str(alarm) if not over else 'ALARM: ' + '; '.join(over[:20])}. "
             "Every full-window and per-season accuracy of every setting was checked.\n")
    L.append("**Next (orchestrator):** write the block to config.yaml, regenerate features.parquet, re-run "
             "leakage tests 1–2 and the Phase 3 baselines (they used the old Elo).\n")
    (repo_path(cfg["paths"]["reports"]) / "elo_tuning.md").write_text("\n".join(L), encoding="utf-8")
    return {"chosen": chosen, "defaults": defaults, "alarm": over}


def main(processes: int | None = None) -> dict[str, Any]:
    """Run the full grid for overall and surface Elo; log, report, and print the chosen settings."""
    cfg = load_config()
    eps = float(cfg["evaluation"]["log_loss_eps"])
    t0 = time.perf_counter()
    matches = pd.read_parquet(repo_path(cfg["paths"]["matches"]))
    prep = prepare(matches, cfg)
    t1 = time.perf_counter()
    settings = [s for w in WHICH for s in grid_settings(cfg, w)]
    procs = processes if processes is not None else max(1, (os.cpu_count() or 2) - 1)
    results = run_grid(prep, settings, eps, procs)
    t2 = time.perf_counter()
    alarm = float(cfg["evaluation"]["leakage_alarm_accuracy"])
    probe_results = {w: [(s, score_setting(prep, s, eps)) for s in edge_probes(cfg, _rank(results, w)[0][0])]
                     for w in WHICH}
    for probe, rs in ((False, results), (True, [x for v in probe_results.values() for x in v])):
        for s, r in rs:
            params = {"elo": s.which, **asdict(s), "label": s.label, "m_mode": prep.m_mode, "probe": probe,
                      "warmup_matches": int(cfg["elo_harness"]["warmup_matches"]),
                      "window": [cfg["elo_harness"]["tune_start"], cfg["elo_harness"]["tune_end"]]}
            metrics = {"val": {k: r[k] for k in ("n", "log_loss", "accuracy", "brier")}, "seasons": r["seasons"]}
            ev.log_experiment("elo_tuning", split="elo_tune_2015_2024", params=params, metrics=metrics, cfg=cfg)
    runtime = {"prepare": t1 - t0, "grid": t2 - t1, "processes": procs}
    out = write_report(results, cfg, runtime, prep, probe_results)
    for which in WHICH:
        s, r = out["chosen"][which]
        d = out["defaults"][which][1]
        print(f"{which:8s} best {s.label}: ll={r['log_loss']:.5f} acc={r['accuracy']:.4f} | default "
              f"ll={d['log_loss']:.5f} acc={d['accuracy']:.4f}")
    print(f"prepare {t1 - t0:.1f}s, grid {t2 - t1:.1f}s on {procs} processes ({len(settings)} settings)")
    if out["alarm"]:
        raise ev.LeakageAlarm(f"accuracy above {alarm}: {out['alarm'][:5]}")
    return out


if __name__ == "__main__":
    main()
