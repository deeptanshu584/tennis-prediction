"""Helpers for the data-ingestion tests (tests/data only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tennis.ingest.sackmann import SACKMANN_COLUMNS, STAT_SUFFIXES, type_columns

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sackmann"

GOOD_W = {"ace": 5, "df": 2, "svpt": 60, "1stIn": 40, "1stWon": 30, "2ndWon": 10, "SvGms": 10, "bpSaved": 2,
          "bpFaced": 3}
GOOD_L = {"ace": 3, "df": 4, "svpt": 65, "1stIn": 38, "1stWon": 25, "2ndWon": 12, "SvGms": 10, "bpSaved": 4,
          "bpFaced": 7}


def raw_row(stats: bool = True, w: dict | None = None, l: dict | None = None, **kw: Any) -> dict[str, Any]:
    """One raw (all-string) Sackmann row with sensible defaults; keyword args override columns."""
    r: dict[str, Any] = {c: None for c in SACKMANN_COLUMNS}
    r.update(
        tourney_id="2023-0001", tourney_name="Testopen", surface="Hard", draw_size="32", tourney_level="A",
        tourney_date="20230102", match_num="1", winner_id="100001", winner_name="Player A", winner_hand="R",
        winner_ioc="ESP", loser_id="100002", loser_name="Player B", loser_hand="R", loser_ioc="FRA",
        score="6-4 6-4", best_of="3", round="R32", minutes="90",
    )
    if stats:
        for s in STAT_SUFFIXES:
            r[f"w_{s}"] = str((w or GOOD_W)[s])
            r[f"l_{s}"] = str((l or GOOD_L)[s])
    r.update({k: (None if v is None else str(v)) for k, v in kw.items()})
    return r


def typed_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Raw rows -> typed frame (through the real `type_columns`)."""
    raw = pd.DataFrame(rows, columns=SACKMANN_COLUMNS).astype(object)
    return type_columns(raw)
