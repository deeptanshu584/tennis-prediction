"""Merge per-source unified match frames into data/interim/matches.parquet.

Priority comes from `data.source_priority` (sackmann > tml > scraped). Duplicates across sources are removed on
`match_id` and on the D-009 key (tourney_id, round, winner_id, loser_id, score), keeping the higher-priority
source (then the first row in match order). Results are never altered.

CLI (Phase 1: Sackmann only):
    .venv/Scripts/python -m tennis.ingest.merge
builds data/interim/matches.parquet, data/interim/player_map.csv and reports/data/ingest_sackmann.md.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tennis.config import load_config, repo_path
from tennis.ingest import sackmann
from tennis.ingest.player_map import sackmann_player_map
from tennis.ingest.sackmann import MATCH_ORDER, UNIFIED_COLUMNS, assign_order_key, dup_key


def merge_sources(
    frames: Mapping[str, pd.DataFrame], cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge unified frames keyed by source name. Returns (merged, contributions).

    `contributions` has one row per source: rows_in, dup_match_id (dropped: match_id already supplied by a
    higher-priority source or earlier row), dup_key (dropped on the D-009 key), rows_kept.
    Raises on an unknown source, missing columns, or a duplicate match_id surviving the merge.
    """
    priority = list(cfg["data"]["source_priority"])
    parts = []
    for name, df in frames.items():
        if name not in priority:
            raise ValueError(f"unknown source {name!r}; expected one of {priority}")
        missing = [c for c in UNIFIED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"source {name!r} is missing unified columns {missing}")
        if (df["source"] != name).any():
            raise ValueError(f"source {name!r} frame has rows with a different `source` value")
        parts.append(df[UNIFIED_COLUMNS].assign(_priority=priority.index(name)))
    if not parts:
        raise ValueError("no frames to merge")

    allm = pd.concat(parts, ignore_index=True)
    allm = allm.sort_values(
        ["_priority", *MATCH_ORDER], kind="mergesort"
    )
    dup_id = allm["match_id"].duplicated(keep="first")
    after_id = allm[~dup_id]
    dup_on_key = dup_key(after_id).duplicated(keep="first")
    merged = after_id[~dup_on_key]

    contributions = pd.DataFrame(
        {
            "rows_in": allm.groupby("source").size(),
            "dup_match_id": allm[dup_id].groupby("source").size(),
            "dup_key": after_id[dup_on_key].groupby("source").size(),
            "rows_kept": merged.groupby("source").size(),
        }
    ).fillna(0).astype(int).rename_axis("source").reset_index()

    merged = assign_order_key(merged.drop(columns="_priority"))[UNIFIED_COLUMNS]
    if merged["match_id"].duplicated().any():
        raise AssertionError("duplicate match_id after merge")
    return merged, contributions


def main() -> None:
    """Build matches.parquet + player_map.csv from Sackmann and write reports/data/ingest_sackmann.md."""
    cfg = load_config()
    unified, audit = sackmann.build_sackmann(cfg)
    players = sackmann.load_players(cfg)
    pmap = sackmann_player_map(unified, players)  # raises on unmatched ids

    merged, contributions = merge_sources({"sackmann": unified}, cfg)

    out = repo_path(cfg["paths"]["matches"])
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    pmap_path = repo_path(cfg["paths"]["player_map"])
    pmap.to_csv(pmap_path, index=False)

    report = sackmann.render_report(merged, audit, contributions, pmap, cfg)
    rpath = repo_path(cfg["paths"]["reports"]) / "data" / "ingest_sackmann.md"
    rpath.parent.mkdir(parents=True, exist_ok=True)
    rpath.write_text(report, encoding="utf-8")
    print(f"wrote {out} ({len(merged):,} rows), {pmap_path} ({len(pmap):,} players), {rpath}")


if __name__ == "__main__":
    main()
