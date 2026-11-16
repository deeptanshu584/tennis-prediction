"""Final engine state and read-only feature rows for hypothetical matches (PLAN section 6 Phase 8).

:func:`build_state` runs the SAME chronological pass as :func:`tennis.engine.build_features.build_features`
(:func:`~tennis.engine.build_features.run_pass`, without writing feature rows) and returns the state after the last
match: every player's :class:`~tennis.engine.state.PlayerState`, the head-to-head counts, and each player's latest
static attributes (rank, rank points, age, height, hand, name) from their last processed match row.

:func:`query_rows` builds feature rows for hypothetical matches (``p1`` vs ``p2`` on a surface / level / round /
best-of / date) from that state WITHOUT updating it: every query reads the same final state, and no query can see
another. Columns and semantics are those of ``build_features`` with A = ``p1`` and B = ``p2``:

- ``a_/b_/d_`` player features, context one-hots, ``round_rank``, ``best_of``, ``elo_prob_a``,
  ``elo_surface_prob_a``; the ``features.static_fill`` constants for missing height / hand / age (D-020).
- Day-based features (``days_since_last_match``, ``matches_last_14_days``) use the query ``date``, read as the
  tournament START date like ``tourney_date`` (default: the day after the last tourney_date in the data).
- Static attributes default to the player's last known values: rank and rank points as of their last processed
  match (NOT the ranking on the query date), height and hand as recorded, and age projected to the query date
  (``last age + days elapsed / 365.25``). A query may override them with the optional columns ``p1_rank``,
  ``p1_rank_points``, ``p1_age``, ``p1_ht``, ``p1_hand`` (same for ``p2``); a NaN override means "use default".
- Nothing event-dependent applies: K-factors (``m``), layoff bonuses and round-robin blocks only matter for
  state updates, and queries never update state.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from tennis.config import config_hash, repo_path
from tennis.engine.build_features import (
    _fix_dtypes,
    _snapshot,
    output_columns,
    player_feature_names,
    run_pass,
    static_fill,
    surfaces,
)
from tennis.engine.elo import expected_score
from tennis.engine.h2h import H2H
from tennis.engine.ordering import check_round_rank, sort_matches
from tennis.engine.state import PlayerState, new_player

NAN = float("nan")
DAYS_PER_YEAR: float = 365.25
STATIC_FIELDS: tuple[str, ...] = ("rank", "rank_points", "age", "ht", "hand")
QUERY_COLUMNS: tuple[str, ...] = ("p1", "p2", "surface", "level", "round", "best_of")
STATE_FORMAT: int = 1  # bump when EngineState changes shape (invalidates cached pickles)


@dataclass(frozen=True)
class PlayerInfo:
    """A player's latest static attributes, from their last processed (non-walkover) match row."""

    name: str
    rank: float
    rank_points: float
    age: float          # age at ``day`` (NaN if unknown)
    ht: float
    hand: Any
    day: int            # tourney_date (days since epoch) of that match row


@dataclass
class EngineState:
    """The state after the last processed match. Treat as read-only: :func:`query_rows` never mutates it."""

    players: dict[str, PlayerState]
    h2h: H2H
    info: dict[str, PlayerInfo]
    names: dict[str, str]                 # player_id -> most recent name spelling (walkover rows included)
    name_day: dict[str, int]              # player_id -> tourney_date (days) of the last row naming the player
    last_day: int                         # last tourney_date (days since epoch) processed
    n_matches: int                        # processed (non-walkover) matches
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def cutoff_date(self) -> dt.date:
        """The last tourney_date in the data (a START date; that event's matches may run a week or two later)."""
        return day_to_date(self.last_day)

    @property
    def default_query_date(self) -> dt.date:
        """The day after :attr:`cutoff_date`: the default "now" for queries."""
        return day_to_date(self.last_day + 1)


def day_to_date(day: int) -> dt.date:
    """Days since the epoch -> ``datetime.date``."""
    return dt.date(1970, 1, 1) + dt.timedelta(days=int(day))


def to_day(x: Any) -> int:
    """A date-like value -> days since the epoch (the engine's day unit)."""
    return int(np.datetime64(pd.Timestamp(x).date(), "D").astype("int64"))


def _isnan(x: Any) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _f(x: Any) -> float:
    return NAN if _isnan(x) else float(x)


# --------------------------------------------------------------------------- building the final state


def build_state(matches: pd.DataFrame, cfg: Mapping[str, Any]) -> EngineState:
    """Run the one chronological pass over ``matches`` and return the final state.

    Walkovers are dropped before the pass exactly as in ``build_features``, and the player-state / H2H updates are
    the same code (:func:`~tennis.engine.build_features.run_pass` with ``emit_rows=False``). ``matches`` is not
    mutated.
    """
    check_round_rank(matches, cfg)
    ms = sort_matches(matches.loc[~matches["is_walkover"].astype(bool)])
    if len(ms) == 0:
        raise ValueError("build_state got no non-walkover matches")
    res = run_pass(ms, cfg, emit_rows=False)

    # latest static attributes: a loop in authoritative order, the last row wins (winner and loser of one row are
    # different players, so the order inside a row doesn't matter)
    days = ms["tourney_date"].to_numpy().astype("datetime64[D]").astype("int64").tolist()
    info: dict[str, PlayerInfo] = {}
    sides = []
    for side in ("winner", "loser"):
        cols = [f"{side}_id", f"{side}_name"] + [f"{side}_{f}" for f in STATIC_FIELDS]
        sides.append(list(zip(*(ms[c].tolist() for c in cols))))
    for t, day in enumerate(days):
        for rows in sides:
            pid, name, rank, pts, age, ht, hand = rows[t]
            info[str(pid)] = PlayerInfo(str(name), _f(rank), _f(pts), _f(age), _f(ht), hand, int(day))

    names, name_day = latest_names(matches)
    return EngineState(res.players, res.h2h, info, names, name_day, int(days[-1]), int(len(ms)))


def latest_names(matches: pd.DataFrame) -> tuple[dict[str, str], dict[str, int]]:
    """``player_id -> most recent name spelling`` (and the day of that row) over every row, walkovers included."""
    ms = sort_matches(matches)
    days = ms["tourney_date"].to_numpy().astype("datetime64[D]").astype("int64").tolist()
    names: dict[str, str] = {}
    name_day: dict[str, int] = {}
    cols = [ms["winner_id"].astype(str).tolist(), ms["winner_name"].tolist(),
            ms["loser_id"].astype(str).tolist(), ms["loser_name"].tolist()]
    for wid, wname, lid, lname, day in zip(*cols, days):
        for pid, name in ((wid, wname), (lid, lname)):
            if isinstance(name, str) and name:
                names[pid] = name
                name_day[pid] = int(day)
    return names, name_day


# --------------------------------------------------------------------------- the on-disk cache


def matches_fingerprint(path: str | Path) -> dict[str, Any]:
    """Cheap identity of the matches file (size and modification time): a changed file invalidates the cache."""
    st = Path(path).stat()
    return {"path": str(Path(path).resolve()), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def engine_fingerprint() -> str:
    """Hash of the engine source files: a code change invalidates the cached state too."""
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("*.py")):
        h.update(f.name.encode("utf-8"))
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def state_cache_path(cfg: Mapping[str, Any]) -> Path:
    """``<paths.models>/state_<config hash>.pkl``."""
    return repo_path(cfg["paths"]["models"]) / f"state_{config_hash(dict(cfg))}.pkl"


def load_or_build_state(cfg: Mapping[str, Any], *, rebuild: bool = False, verbose: bool = True) -> EngineState:
    """The final state for ``paths.matches``, cached at :func:`state_cache_path`.

    The cache is used only if it was built with the same config hash, :data:`STATE_FORMAT`, engine source
    (:func:`engine_fingerprint`) and matches-file fingerprint; otherwise (or with ``rebuild=True``) the state is
    rebuilt and the cache rewritten.
    """
    src = repo_path(cfg["paths"]["matches"])
    if not src.exists():
        raise FileNotFoundError(f"{src} not found (built by the Phase 1 ingestion)")
    want = {"config_hash": config_hash(dict(cfg)), "format": STATE_FORMAT, "engine": engine_fingerprint(),
            "matches": matches_fingerprint(src)}
    cache = state_cache_path(cfg)
    if cache.exists() and not rebuild:
        try:
            with cache.open("rb") as fh:
                state = pickle.load(fh)
        except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, TypeError):
            state = None  # unreadable or from an older code version: rebuild
        if isinstance(state, EngineState) and all(state.meta.get(k) == v for k, v in want.items()):
            return state
    if verbose:
        print(f"building the engine state from {src} (one chronological pass, about a minute) ...")
    state = build_state(pd.read_parquet(src), cfg)
    state.meta.update(want)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(cache)
    if verbose:
        print(f"cached the state at {cache}")
    return state


# --------------------------------------------------------------------------- read-only query rows


def _static_for(state: EngineState, pid: str, q: Mapping[str, Any], prefix: str, today: int
                ) -> tuple[Any, Any, Any, Any, Any]:
    """(rank, rank_points, age, height, hand) for ``pid`` on day ``today``: overrides first, else last known."""
    inf = state.info.get(pid)
    rank = pts = age = ht = NAN
    hand: Any = None
    if inf is not None:
        rank, pts, ht, hand = inf.rank, inf.rank_points, inf.ht, inf.hand
        age = NAN if _isnan(inf.age) else inf.age + (today - inf.day) / DAYS_PER_YEAR
    vals = {"rank": rank, "rank_points": pts, "age": age, "ht": ht, "hand": hand}
    for f in STATIC_FIELDS:
        key = f"{prefix}_{f}"
        if key in q:
            v = q[key]
            if f == "hand":
                if isinstance(v, str) and v:
                    vals[f] = v
            elif not _isnan(v):
                vals[f] = float(v)
    return vals["rank"], vals["rank_points"], vals["age"], vals["ht"], vals["hand"]


def query_rows(state: EngineState, queries: pd.DataFrame, cfg: Mapping[str, Any], *,
               allow_new_players: bool = False) -> pd.DataFrame:
    """Feature rows for hypothetical matches, read from ``state`` without updating it.

    Args:
        state: from :func:`build_state` / :func:`load_or_build_state`. Not mutated.
        queries: one row per match with columns ``p1``, ``p2`` (player ids), ``surface``, ``level``, ``round``,
            ``best_of``, optional ``date`` (tournament start date; default ``state.default_query_date``) and the
            optional static overrides described in the module docstring. A = ``p1``, B = ``p2``.
        cfg: the config the state was built with.
        allow_new_players: if False (default) an id with no processed match raises ``KeyError``. If True, such a
            player gets a fresh first-appearance state (never stored), as ``build_features`` would give a debutant.

    Returns:
        A frame with the ``build_features`` columns except ``y`` (``match_id`` = ``query_<i>``,
        ``tourney_id``/``tourney_name`` = ``query``), one row per query in input order.

    Raises:
        ValueError: unknown surface / level / round, best_of not in {3, 5}, p1 == p2, or a date before the
            state's last tourney_date (the state would already contain later matches).
        KeyError: an unknown player id (unless ``allow_new_players``).
    """
    missing = [c for c in QUERY_COLUMNS if c not in queries.columns]
    if missing:
        raise ValueError(f"queries lack columns {missing}")
    fcfg = cfg["features"]
    winrate_k = [int(k) for k in fcfg["winrate_k"]]
    recent_window = int(fcfg["recent_days_window"])
    start = float(cfg["elo"]["start_rating"])
    fill = static_fill(cfg)
    surf_names = surfaces(cfg)
    levels = [str(x) for x in cfg["data"]["training_levels"]]
    round_order = [str(x) for x in cfg["data"]["round_order"]]
    pf_names = player_feature_names(cfg)
    results_len = max(winrate_k) if winrate_k else 1

    def player(pid: str) -> PlayerState:
        st = state.players.get(pid)
        if st is not None:
            return st
        if not allow_new_players:
            raise KeyError(f"player id {pid!r} has no processed match in the data")
        return new_player(start, int(fcfg["elo_grad_window"]), results_len, int(fcfg["rolling_n"]))

    out: dict[str, list[Any]] = {c: [] for c in output_columns(cfg) if c != "y"}
    for i, q in enumerate(queries.to_dict("records")):
        a, b = str(q["p1"]), str(q["p2"])
        surface, level, rnd, best_of = str(q["surface"]), str(q["level"]), str(q["round"]), int(q["best_of"])
        if a == b:
            raise ValueError(f"query {i}: p1 and p2 are the same player ({a})")
        if surface not in surf_names:
            raise ValueError(f"query {i}: surface {surface!r} not in {surf_names}")
        if level not in levels:
            raise ValueError(f"query {i}: level {level!r} not in data.training_levels {levels}")
        if rnd not in round_order:
            raise ValueError(f"query {i}: round {rnd!r} not in data.round_order {round_order}")
        if best_of not in (3, 5):
            raise ValueError(f"query {i}: best_of must be 3 or 5, got {best_of}")
        date = q.get("date")
        no_date = date is None or (isinstance(date, str) and not date.strip()) or (
            not isinstance(date, str) and bool(pd.isna(date)))
        today = state.last_day + 1 if no_date else to_day(date)
        if today < state.last_day:
            raise ValueError(f"query {i}: date {day_to_date(today)} is before the data cut-off "
                             f"{state.cutoff_date}; the final state already contains later matches")
        sa, sb = player(a), player(b)
        fa = _snapshot(sa, a, b, surface, today, state.h2h, start, winrate_k, recent_window,
                       _static_for(state, a, q, "p1", today), fill)
        fb = _snapshot(sb, b, a, surface, today, state.h2h, start, winrate_k, recent_window,
                       _static_for(state, b, q, "p2", today), fill)
        row: dict[str, Any] = {
            "match_id": f"query_{i}", "date": pd.Timestamp(day_to_date(today)), "tourney_id": "query",
            "tourney_name": "query", "a_id": a, "b_id": b,
            "a_name": state.names.get(a, a), "b_name": state.names.get(b, b),
        }
        for s in surf_names:
            row[f"surface_{s}"] = int(surface == s)
        for lv in levels:
            row[f"level_{lv}"] = int(level == lv)
        row["round_rank"] = round_order.index(rnd)
        row["best_of"] = best_of
        row["elo_prob_a"] = expected_score(fa[0], fb[0])
        row["elo_surface_prob_a"] = expected_score(fa[1], fb[1])
        for j, f in enumerate(pf_names):
            x, y = fa[j], fb[j]
            row[f"a_{f}"], row[f"b_{f}"] = x, y
            row[f"d_{f}"] = NAN if (_isnan(x) or _isnan(y)) else x - y
        for c in out:
            out[c].append(row[c])

    df = pd.DataFrame(out, columns=list(out))
    df["date"] = pd.to_datetime(df["date"])
    for c in ("round_rank", "best_of", *(f"surface_{s}" for s in surf_names), *(f"level_{lv}" for lv in levels)):
        df[c] = df[c].astype(np.int64)
    for c in ("elo_prob_a", "elo_surface_prob_a"):
        df[c] = df[c].astype(np.float64)
    return _fix_dtypes(df, pf_names)
