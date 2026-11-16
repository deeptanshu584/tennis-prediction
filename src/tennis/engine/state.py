"""Per-player state for the chronological pass, plus small update functions (PLAN 5.1, CONTRACT section 2).

The builder READS a player's state to write a feature row and only THEN calls the ``record_*`` functions with the
match result. Each function touches only the state passed to it (no globals), so each is easy to test on its own.
The read-side helpers (``winrate``, ``ols_slope``, ``count_recent``, ``layoff_played`` ...) are pure.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from tennis.engine.elo import expected_score, update_rating
from tennis.engine.rolling import StatsBuffer

NAN = float("nan")
M_MODES: tuple[str, ...] = ("constant", "event_matches_so_far")


@dataclass(slots=True)
class PlayerState:
    """Everything known about one player before their next match.

    Static attributes (rank, age, height, hand) are not stored: they come from the match row (CONTRACT section 2).
    """

    elo: float
    elo_surface: dict[str, float]
    elo_hist: deque[float]                 # last ``grad_window`` overall-Elo values, starting with the start rating
    results: deque[int]                    # last results (1 win, 0 loss) for winrate_last_k
    stats: StatsBuffer                     # last ``rolling_n`` matches with complete stats
    matches_played: int = 0                # prior non-walkover matches (incl. RET/DEF, qualifying, Challenger)
    surface_played: dict[str, int] = field(default_factory=dict)
    tourneys: dict[str, dt.date] = field(default_factory=dict)       # tourney_id -> start date, in match order
    event_matches: dict[str, int] = field(default_factory=dict)      # tourney_id -> matches played in it so far
    last_day: int | None = None            # tourney_date (days since epoch) of the last prior match
    recent_days: deque[int] = field(default_factory=deque)           # tourney_dates (days) of recent prior matches


def new_player(start_rating: float, grad_window: int, results_len: int, rolling_n: int) -> PlayerState:
    """State of a player at first appearance: Elo history ``[start_rating]``, empty buffers, zero counts."""
    if grad_window < 1:
        raise ValueError(f"elo_grad_window must be >= 1, got {grad_window}")
    return PlayerState(
        elo=float(start_rating),
        elo_surface={},
        elo_hist=deque([float(start_rating)], maxlen=int(grad_window)),
        results=deque(maxlen=max(1, int(results_len))),
        stats=StatsBuffer(int(rolling_n)),
    )


# --------------------------------------------------------------------------- read side (pure)


def surface_rating(state: PlayerState, surface: str, start_rating: float) -> float:
    """The player's Elo on ``surface`` (``start_rating`` before their first match on it)."""
    return state.elo_surface.get(surface, start_rating)


def surface_count(state: PlayerState, surface: str) -> int:
    """Prior matches on ``surface``."""
    return state.surface_played.get(surface, 0)


def winrate(results: Sequence[int], k: int) -> float:
    """Mean of the last ``k`` results over ``min(k, available)``; NaN if there are none."""
    n = min(int(k), len(results))
    if n == 0:
        return NAN
    if n == len(results):
        return sum(results) / n
    return sum(list(results)[-n:]) / n


def ols_slope(values: Sequence[float]) -> float:
    """Least-squares slope through ``(0, v0), (1, v1), ...`` (equals ``np.polyfit(x, v, 1)[0]``); NaN if n < 2."""
    n = len(values)
    if n < 2:
        return NAN
    xm = (n - 1) / 2.0
    ym = math.fsum(values) / n
    num = 0.0
    for i, v in enumerate(values):
        num += (i - xm) * (v - ym)
    den = n * (n * n - 1) / 12.0  # sum((i - xm)^2) for i = 0..n-1
    return num / den


def count_recent(days: Iterable[int], today: int, window: int) -> int:
    """Number of prior match dates ``d`` with ``d >= today - window`` (all in days)."""
    lo = today - window
    return sum(1 for d in days if d >= lo)


def days_since(last_day: int | None, today: int) -> float:
    """Days between the last prior match's tourney_date and ``today``; NaN if no prior match."""
    return NAN if last_day is None else float(today - last_day)


def m_value(base_m: float, mode: str, n_event: int) -> float:
    """``m`` in K = div / (Ne + m) (CONTRACT section 2, DECISIONS D-008).

    ``event_matches_so_far``: ``base_m + matches the player already played in this tourney_id``;
    ``constant``: ``base_m``. Never anything that depends on later matches.
    """
    if mode == "event_matches_so_far":
        return float(base_m) + n_event
    if mode == "constant":
        return float(base_m)
    raise ValueError(f"elo.m_mode must be one of {M_MODES}, got {mode!r}")


def layoff_played(tourneys: Mapping[str, dt.date], tourney_id: str, start_date: dt.date) -> list[dt.date]:
    """Layoff ``played`` list: one start date per prior tourney_id (match order), plus the current tournament's
    start date if its tourney_id is not already there (CONTRACT section 2)."""
    played = list(tourneys.values())
    if tourney_id not in tourneys:
        played.append(start_date)
    return played


def elo_deltas(r_winner: float, r_loser: float, k_winner: float, k_loser: float) -> tuple[float, float]:
    """Rating changes ``(K_w * (1 - E_w), K_l * (0 - E_l))`` computed from the given (pre-match or pre-block)
    ratings. Adding them to the current ratings is the PLAN 5.2 update; inside a round-robin block the ratings and
    K's are the pre-block ones (CONTRACT section 2, "Round-robin blocks")."""
    return (k_winner * (1.0 - expected_score(r_winner, r_loser)),
            k_loser * (0.0 - expected_score(r_loser, r_winner)))


def elo_after_match(r_winner: float, r_loser: float, k_winner: float, k_loser: float) -> tuple[float, float]:
    """New (winner, loser) ratings; each player uses their own K (PLAN 5.2)."""
    new_w = update_rating(r_winner, expected_score(r_winner, r_loser), 1.0, k_winner)
    new_l = update_rating(r_loser, expected_score(r_loser, r_winner), 0.0, k_loser)
    return new_w, new_l


# --------------------------------------------------------------------------- write side (after the row)


def record_elo(state: PlayerState, surface: str, new_overall: float, new_surface: float) -> None:
    """Store the post-match ratings and append the new overall rating to the Elo history."""
    state.elo = new_overall
    state.elo_surface[surface] = new_surface
    state.elo_hist.append(new_overall)


def record_match(state: PlayerState, surface: str, tourney_id: str, start_date: dt.date, day: int,
                 recent_window: int) -> None:
    """Count one processed (non-walkover) match: totals, surface count, event count, tourney list, dates.

    ``day`` is the tourney_date in days. Dates older than ``day - recent_window`` are pruned from
    ``recent_days``; this is safe because later matches never have an earlier tourney_date.
    """
    state.matches_played += 1
    state.surface_played[surface] = state.surface_played.get(surface, 0) + 1
    state.event_matches[tourney_id] = state.event_matches.get(tourney_id, 0) + 1
    if tourney_id not in state.tourneys:
        state.tourneys[tourney_id] = start_date
    state.last_day = day
    rd = state.recent_days
    rd.append(day)
    lo = day - recent_window
    while rd and rd[0] < lo:
        rd.popleft()


def record_result(state: PlayerState, won: int) -> None:
    """Append a result (1 win, 0 loss) to the form deque."""
    state.results.append(int(won))
