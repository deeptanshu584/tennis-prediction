"""Pure Elo arithmetic for the tennis feature engine (PLAN 5.2-5.4, CONTRACT section 3).

Everything here is a pure function of its arguments: no player state, no data frames. The feature builder owns
state and calls these functions only AFTER a match's feature row has been written.

    E_A  = 1 / (1 + 10 ** ((R_B - R_A) / 400))
    R_A' = R_A + K_A * (S_A - E_A)

Dynamic K follows the author's code (PLAN 5.3); the layoff bonus is his logic verbatim (PLAN 5.4), with its
constants exposed as parameters whose defaults are the plan's values. Real settings come from ``config.yaml``
via :meth:`EloParams.from_config`.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping

import numpy as np

DateLike = dt.date | dt.datetime | np.datetime64 | int
"""A date: ``datetime.date``/``datetime.datetime``/``pandas.Timestamp``, ``numpy.datetime64`` or a YYYYMMDD int."""

ELO_SECTIONS: tuple[str, ...] = ("overall", "surface")

# Mapping from config.yaml `elo.layoff` keys to k_bonus_after_layoff keyword names.
_LAYOFF_CONFIG_KEYS: dict[str, str] = {
    "threshold_days": "threshold",
    "min_gaps": "min_gaps",
    "start_bonus": "start_bonus",
    "step": "step",
    "floor": "floor",
}


# --------------------------------------------------------------------------- core Elo


def expected_score(r_a: float, r_b: float) -> float:
    """Probability that a player rated ``r_a`` beats one rated ``r_b`` (logistic, 400-point scale)."""
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def update_rating(r: float, expected: float, score: float, k: float) -> float:
    """Return the new rating ``r + k * (score - expected)``.

    Args:
        r: rating before the match.
        expected: the player's expected score from :func:`expected_score`.
        score: actual result, 1.0 for a win and 0.0 for a loss.
        k: this player's own K-factor (each player in a match may have a different K).
    """
    return r + k * (score - expected)


# --------------------------------------------------------------------------- layoff bonus


def _to_date(x: Any) -> dt.date:
    """Convert one date-like value to ``datetime.date`` (time of day is discarded).

    Accepts ``datetime.date``, ``datetime.datetime`` (incl. ``pandas.Timestamp``), ``numpy.datetime64`` of any unit,
    and YYYYMMDD integers (Python or numpy). Missing values (NaT) raise ``ValueError``; bools, floats, strings and
    anything else raise ``TypeError``.
    """
    if isinstance(x, (bool, np.bool_)):
        raise TypeError(f"bool is not a date: {x!r}")
    if isinstance(x, np.datetime64):
        if np.isnat(x):
            raise ValueError("missing date (NaT) in played dates")
        return x.astype("datetime64[D]").item()
    if isinstance(x, dt.datetime):  # pandas.Timestamp and pandas.NaT subclass datetime
        if x != x:  # NaT is the only datetime unequal to itself
            raise ValueError("missing date (NaT) in played dates")
        return x.date()
    if isinstance(x, dt.date):
        return x
    if isinstance(x, (int, np.integer)):
        v = int(x)
        return dt.date(v // 10000, (v // 100) % 100, v % 100)  # ValueError if not a valid YYYYMMDD
    raise TypeError(f"unsupported date type {type(x).__name__}: {x!r}")


def gaps_between_tourneys(played: Iterable[DateLike]) -> list[int]:
    """Gaps in whole days between consecutive tournament start dates, MOST RECENT FIRST.

    The dates are converted to calendar dates and sorted chronologically before differencing, so input order does
    not matter. Duplicate dates are kept and produce 0-day gaps; the caller is expected to pass one date per
    tournament (CONTRACT section 2, "Layoff played").

    Example: ``[2024-01-01, 2024-01-15, 2024-03-01] -> [46, 14]``.
    """
    dates = sorted(_to_date(d) for d in played)
    gaps = [(later - earlier).days for earlier, later in zip(dates, dates[1:])]
    return gaps[::-1]


def k_bonus_after_layoff(
    played: Iterable[DateLike],
    threshold: float = 100,
    min_gaps: int = 15,
    start_bonus: float = 1.5,
    step: float = 0.05,
    floor: float = 1.0,
) -> float:
    """K multiplier for a player returning from a layoff (the author's logic, PLAN 5.4, verbatim).

    Walk the gaps from the most recent backwards: start at ``start_bonus`` and subtract ``step`` for each gap shorter
    than ``threshold`` days, stopping at the first gap of at least ``threshold`` days. The result is floored at
    ``floor``. Players with ``min_gaps`` gaps or fewer (i.e. at most ``min_gaps + 1`` tournaments) get exactly 1.0,
    no matter how long their layoffs were.

    Args:
        played: start dates of the player's prior tournaments plus the current one (any order, any date-like type).
        threshold: a gap of at least this many days counts as a layoff.
        min_gaps: the bonus only applies when there are MORE than this many gaps.
        start_bonus: multiplier when the most recent gap is itself a layoff.
        step: decrement per short gap before the first layoff.
        floor: lower bound of the multiplier.
    """
    gaps = gaps_between_tourneys(played)
    if len(gaps) <= min_gaps:
        return 1.0
    bonus = start_bonus
    for g in gaps:
        if g < threshold:
            bonus -= step
        else:
            break
    return max(floor, bonus)


# --------------------------------------------------------------------------- dynamic K


def k_factor(
    ne: int,
    m: float,
    *,
    k_factor: float,
    div_number: float,
    base_k: float,
    max_k: float,
    bonus_after_layoff: bool,
    played: Iterable[DateLike] = (),
    layoff: Mapping[str, float] | None = None,
) -> float:
    """K-factor for one player in one match (PLAN 5.3, the author's ``updateStats`` logic).

    ::

        if div_number == 0: K = k_factor
        else:               K = min(max_k, max(base_k, div_number / (ne + m)))
        if bonus_after_layoff: K *= k_bonus_after_layoff(played, **layoff)

    Args:
        ne: matches the player has played before this one (overall, or on this surface for surface Elo).
        m: offset in the denominator. Must never depend on how far the player goes in the current event
            (leakage rule 9): use a constant or the matches already played in this event.
        k_factor: fixed K, used only when ``div_number == 0``.
        div_number: numerator of the dynamic K (chess uses 800); 0 switches dynamic K off.
        base_k: lower bound of the dynamic K.
        max_k: upper bound of the dynamic K. NOTE: the author's vanilla setup has ``max_k=0`` together with
            ``div_number=0``, so the cap is never reached there. ``max_k == 0`` with ``div_number != 0`` gives
            ``K = 0`` (ratings never move). That is the formula as written and is deliberately not special-cased.
        bonus_after_layoff: multiply K by :func:`k_bonus_after_layoff` of ``played``.
        played: tournament start dates for the layoff bonus (prior tournaments plus the current one).
        layoff: keyword overrides for :func:`k_bonus_after_layoff` (``threshold``, ``min_gaps``, ``start_bonus``,
            ``step``, ``floor``). ``None`` uses that function's defaults.

    Raises:
        ValueError: if ``div_number != 0`` and ``ne + m <= 0`` (division by zero or a negative K).
    """
    if div_number == 0:
        k = k_factor
    else:
        denom = ne + m
        if denom <= 0:
            raise ValueError(f"dynamic K needs ne + m > 0, got ne={ne}, m={m}")
        k = min(max_k, max(base_k, div_number / denom))
    if bonus_after_layoff:
        k *= k_bonus_after_layoff(played, **(layoff or {}))
    return k


# --------------------------------------------------------------------------- config convenience


@dataclass(frozen=True)
class EloParams:
    """K-factor settings for one Elo flavour (overall or surface), read from ``config.yaml``."""

    start_rating: float
    k_factor: float
    div_number: float
    m: float
    base_k: float
    max_k: float
    bonus_after_layoff: bool
    layoff: dict[str, float]

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], which: Literal["overall", "surface"]) -> EloParams:
        """Build from ``cfg["elo"][which]`` plus ``cfg["elo"]["layoff"]`` and ``cfg["elo"]["start_rating"]``.

        The layoff key ``threshold_days`` is renamed to ``threshold`` to match :func:`k_bonus_after_layoff`.
        """
        if which not in ELO_SECTIONS:
            raise ValueError(f"which must be one of {ELO_SECTIONS}, got {which!r}")
        elo = cfg["elo"]
        sec = elo[which]
        layoff = {arg: elo["layoff"][key] for key, arg in _LAYOFF_CONFIG_KEYS.items()}
        return cls(
            start_rating=float(elo["start_rating"]),
            k_factor=float(sec["k_factor"]),
            div_number=float(sec["div_number"]),
            m=float(sec["m"]),
            base_k=float(sec["base_k"]),
            max_k=float(sec["max_k"]),
            bonus_after_layoff=bool(sec["bonus_after_layoff"]),
            layoff=layoff,
        )

    def k(self, ne: int, played: Iterable[DateLike] = (), m: float | None = None) -> float:
        """K-factor for a player with ``ne`` prior matches; ``m`` overrides the configured constant
        (e.g. for ``elo.m_mode == "event_matches_so_far"``)."""
        return k_factor(
            ne,
            self.m if m is None else m,
            k_factor=self.k_factor,
            div_number=self.div_number,
            base_k=self.base_k,
            max_k=self.max_k,
            bonus_after_layoff=self.bonus_after_layoff,
            played=played,
            layoff=self.layoff,
        )
