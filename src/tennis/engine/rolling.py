"""Rolling serve/return statistics buffer (PLAN 5.5, CONTRACT section 2).

Each buffer entry holds a player's own nine serve counts and the opponent's nine serve counts for one prior match
with complete stats. Rates are SUMS of numerators divided by SUMS of denominators over the buffer (never means of
per-match percentages); a zero denominator or an empty buffer gives NaN.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Sequence

STAT_FIELDS: tuple[str, ...] = ("ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced")
RATE_FEATURES: tuple[str, ...] = (
    "p_ace", "p_df", "p_1stIn", "p_1stWon", "p_2ndWon", "p_bpSaved", "p_rpw", "p_retAceAgainst",
    "p_ret1stWon", "p_ret2ndWon", "p_bpConv", "p_totalPtsWon", "dominance_ratio",
)
NAN = float("nan")

# positions inside a 9-tuple of serve counts
ACE, DF, SVPT, IN1, WON1, WON2, SVGMS, BPS, BPF = range(9)


def _missing(x: Any) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def valid_stats(w: Sequence[Any], l: Sequence[Any]) -> bool:
    """True if all nine serve fields are present for BOTH players and both served at least one point."""
    if any(_missing(x) for x in w) or any(_missing(x) for x in l):
        return False
    return w[SVPT] > 0 and l[SVPT] > 0


def _div(num: float, den: float) -> float:
    return NAN if den == 0 else num / den


def rates_from_sums(own: Sequence[float], opp: Sequence[float]) -> dict[str, float]:
    """The 13 PLAN 5.5 rates from summed own / opponent serve counts (pure function).

    ``own`` and ``opp`` are 9-sequences in :data:`STAT_FIELDS` order. With an empty buffer (all zeros) every rate
    is NaN, because every denominator is zero.
    """
    rpw_num = opp[SVPT] - opp[WON1] - opp[WON2]
    r = {
        "p_ace": _div(own[ACE], own[SVPT]),
        "p_df": _div(own[DF], own[SVPT]),
        "p_1stIn": _div(own[IN1], own[SVPT]),
        "p_1stWon": _div(own[WON1], own[IN1]),
        "p_2ndWon": _div(own[WON2], own[SVPT] - own[IN1]),
        "p_bpSaved": _div(own[BPS], own[BPF]),
        "p_rpw": _div(rpw_num, opp[SVPT]),
        "p_retAceAgainst": _div(opp[ACE], opp[SVPT]),
        "p_ret1stWon": _div(opp[IN1] - opp[WON1], opp[IN1]),
        "p_ret2ndWon": _div(opp[SVPT] - opp[IN1] - opp[WON2], opp[SVPT] - opp[IN1]),
        "p_bpConv": _div(opp[BPF] - opp[BPS], opp[BPF]),
        "p_totalPtsWon": _div(own[WON1] + own[WON2] + rpw_num, own[SVPT] + opp[SVPT]),
    }
    spw = _div(own[WON1] + own[WON2], own[SVPT])  # serve points won rate
    rpw = r["p_rpw"]
    r["dominance_ratio"] = NAN if (math.isnan(spw) or math.isnan(rpw)) else _div(rpw, 1.0 - spw)
    return r


class StatsBuffer:
    """Last ``maxlen`` matches with stats, with running sums of own and opponent counts.

    The running sums are exact because serve counts are integers stored as floats.
    """

    __slots__ = ("entries", "maxlen", "own_sums", "opp_sums")

    def __init__(self, maxlen: int) -> None:
        if maxlen < 1:
            raise ValueError(f"rolling buffer length must be >= 1, got {maxlen}")
        self.maxlen = int(maxlen)
        self.entries: deque[tuple[tuple[float, ...], tuple[float, ...]]] = deque()
        self.own_sums = [0.0] * 9
        self.opp_sums = [0.0] * 9

    def __len__(self) -> int:
        return len(self.entries)

    def push(self, own: Sequence[float], opp: Sequence[float]) -> None:
        """Append one match (own counts, opponent counts), evicting the oldest when full."""
        if len(self.entries) == self.maxlen:
            old_own, old_opp = self.entries.popleft()
            for j in range(9):
                self.own_sums[j] -= old_own[j]
                self.opp_sums[j] -= old_opp[j]
        own_t, opp_t = tuple(own), tuple(opp)
        self.entries.append((own_t, opp_t))
        for j in range(9):
            self.own_sums[j] += own_t[j]
            self.opp_sums[j] += opp_t[j]

    def rates(self) -> dict[str, float]:
        """Current rates over the buffer (all NaN when empty)."""
        if not self.entries:
            return dict.fromkeys(RATE_FEATURES, NAN)
        return rates_from_sums(self.own_sums, self.opp_sums)
