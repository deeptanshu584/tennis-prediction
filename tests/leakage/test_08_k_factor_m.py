"""Test 8 (PLAN section 9): the K-factor's m never looks ahead within an event.

With record_k=True, remove the later rounds of one event (keep rounds up to R16 / QF) and rebuild: the K logged
for every retained match of that event and every earlier match must be unchanged. Run for both
`elo.m_mode` values with dynamic K (div_number 800), with and without the layoff bonus.

Catches: m (or Ne, or the layoff 'played' list) computed from how far the player eventually goes in the event,
from the event's total match count, or from any later match.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from tests.leakage.helpers import build, cfg_with, get_builder, match_position


def _events(matches: pd.DataFrame) -> list[tuple[str, int]]:
    """(tourney_id, last kept round_rank) for a mid-history Slam and Masters, and a 16-draw 250."""
    years = sorted(matches["tourney_date"].dt.year.unique())
    out = []
    for level, year, keep in (("G", years[1], "R16"), ("M", years[-1], "R16"), ("A", years[1], "QF")):
        ev = matches[(matches["tourney_level"] == level) & (matches["tourney_date"].dt.year == year)
                     & matches["is_qualifying"].eq(False)]
        tids = sorted(ev.groupby("tourney_id").filter(lambda g: (g["round"] == keep).any())["tourney_id"].unique())
        tid = tids[len(tids) // 2]
        keep_rank = int(matches.loc[matches["round"] == keep, "round_rank"].iloc[0])
        out.append((tid, keep_rank))
    return out


def _klog(matches: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    _, k = build(matches, cfg, record_k=True)
    k = k[["match_id", "player_id", "k_overall", "k_surface"]].copy()
    k["match_id"] = k["match_id"].astype(str)
    k["player_id"] = k["player_id"].astype(str)
    return k.sort_values(["match_id", "player_id"]).reset_index(drop=True)


@pytest.mark.parametrize("bonus", [False, True], ids=["no_layoff_bonus", "layoff_bonus"])
@pytest.mark.parametrize("m_mode", ["constant", "event_matches_so_far"])
def test_k_unchanged_when_later_rounds_removed(synthetic_matches: pd.DataFrame, cfg: dict, m_mode: str,
                                               bonus: bool) -> None:
    get_builder()
    c = cfg_with(cfg, **{
        "elo.m_mode": m_mode,
        "elo.overall.div_number": 800.0, "elo.surface.div_number": 800.0,
        "elo.overall.bonus_after_layoff": bonus, "elo.surface.bonus_after_layoff": bonus,
    })
    full = _klog(synthetic_matches, c)
    pos = match_position(synthetic_matches)
    for tid, keep_rank in _events(synthetic_matches):
        in_event = synthetic_matches["tourney_id"] == tid
        removed = in_event & (synthetic_matches["round_rank"] > keep_rank)
        assert removed.sum() >= 3, tid
        first_removed = pos[removed].min()
        retained = synthetic_matches.loc[~removed & (pos < first_removed), "match_id"]
        assert (in_event & ~removed).sum() > 0
        trunc = _klog(synthetic_matches.loc[~removed], c)
        a = full[full["match_id"].isin(retained)].reset_index(drop=True)
        b = trunc[trunc["match_id"].isin(retained)].reset_index(drop=True)
        assert len(a) > 0 and set(retained) & set(a["match_id"]), "k_log has no rows for the retained matches"
        pd.testing.assert_frame_equal(a, b, check_exact=True,
                                      obj=f"K log for matches up to {tid} round_rank<={keep_rank}")
