"""Authoritative match order (PLAN section 2 rule 2, CONTRACT section 2).

Matches are processed in the order ``(tourney_date, tourney_id, round_rank, match_num, match_id)``.
``match_id`` is the final tiebreaker, so the order is total. Input row order and the convenience column
``order_key`` are never trusted.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

ORDER_COLUMNS: tuple[str, ...] = ("tourney_date", "tourney_id", "round_rank", "match_num", "match_id")


def check_round_rank(matches: pd.DataFrame, cfg: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` unless ``round_rank`` equals the index of ``round`` in ``cfg.data.round_order``."""
    order = {r: i for i, r in enumerate(cfg["data"]["round_order"])}
    expected = matches["round"].map(order)
    bad = expected.isna() | (expected != matches["round_rank"])
    if bool(bad.any()):
        ex = matches.loc[bad, ["match_id", "round", "round_rank"]].head(3).to_dict("records")
        raise ValueError(f"round_rank disagrees with config data.round_order on {int(bad.sum())} rows, e.g. {ex}")


def sort_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``matches`` sorted by the authoritative order, with a fresh 0..n-1 index.

    Raises ``ValueError`` if ``match_id`` is not unique (the order would not be total).
    """
    if not matches["match_id"].is_unique:
        dup = matches.loc[matches["match_id"].duplicated(), "match_id"].head(3).tolist()
        raise ValueError(f"match_id is not unique (e.g. {dup}); the match order would not be total")
    out = matches.copy()
    out["tourney_date"] = pd.to_datetime(out["tourney_date"])
    return out.sort_values(list(ORDER_COLUMNS), kind="mergesort").reset_index(drop=True)
