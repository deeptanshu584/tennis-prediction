"""Sackmann `tennis_atp` ingestion: raw CSVs -> unified winner/loser match table (CONTRACT.md section 1).

Pipeline (`build_sackmann`):
    load raw CSVs as strings -> type columns deliberately -> classify scores -> normalise surface ->
    round_rank / match_id -> data-quality flags -> drop self-matches (D-010) and exact
    duplicates (D-009) -> sort into match order and set order_key (0-based int64 position).

Nothing under data/raw/ is modified. Results (winner/loser) are never changed; problems are flagged.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tennis.config import repo_path

SOURCE = "sackmann"

STAT_SUFFIXES: list[str] = ["ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced"]
STAT_COLS: list[str] = [f"w_{s}" for s in STAT_SUFFIXES] + [f"l_{s}" for s in STAT_SUFFIXES]

# The 49 columns of PLAN.md section 4.2, in file order.
SACKMANN_COLUMNS: list[str] = [
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level", "tourney_date", "match_num",
    "winner_id", "winner_seed", "winner_entry", "winner_name", "winner_hand", "winner_ht", "winner_ioc",
    "winner_age",
    "loser_id", "loser_seed", "loser_entry", "loser_name", "loser_hand", "loser_ht", "loser_ioc", "loser_age",
    "score", "best_of", "round", "minutes",
    *STAT_COLS,
    "winner_rank", "winner_rank_points", "loser_rank", "loser_rank_points",
]

STR_COLS: list[str] = [
    "tourney_id", "tourney_name", "surface", "tourney_level", "round", "score",
    "winner_id", "winner_entry", "winner_name", "winner_hand", "winner_ioc",
    "loser_id", "loser_entry", "loser_name", "loser_hand", "loser_ioc",
]
INT_COLS: list[str] = ["draw_size", "match_num", "best_of"]
FLOAT_COLS: list[str] = [
    "winner_seed", "winner_ht", "winner_age", "winner_rank", "winner_rank_points",
    "loser_seed", "loser_ht", "loser_age", "loser_rank", "loser_rank_points",
    "minutes", *STAT_COLS,
]

FLAG_COLS: list[str] = [
    "is_walkover", "is_retirement", "is_qualifying", "is_training_level", "surface_imputed", "stats_invalid",
]
UNIFIED_COLUMNS: list[str] = ["match_id", "source", *SACKMANN_COLUMNS, "round_rank", "order_key", *FLAG_COLS]

# D-009 duplicate key.
DUP_KEY: list[str] = ["tourney_id", "round", "winner_id", "loser_id", "score"]


# --------------------------------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------------------------------
def read_raw_csv(path: str | Path) -> pd.DataFrame:
    """Read one Sackmann CSV with every column as a string (empty cell -> NaN; 'NA' stays a string)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8")
    missing = [c for c in SACKMANN_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in SACKMANN_COLUMNS]
    if missing or extra:
        raise ValueError(f"{path}: unexpected columns (missing={missing}, extra={extra})")
    return df[SACKMANN_COLUMNS]


def raw_file_paths(cfg: dict[str, Any]) -> list[Path]:
    """Tour-level and (if configured) qual_chall files for data.first_year .. data.last_sackmann_year."""
    d = repo_path(cfg["paths"]["sackmann_dir"])
    years = range(int(cfg["data"]["first_year"]), int(cfg["data"]["last_sackmann_year"]) + 1)
    paths: list[Path] = []
    for y in years:
        paths.append(d / f"atp_matches_{y}.csv")
        if cfg["data"].get("include_qual_chall", True):
            paths.append(d / f"atp_matches_qual_chall_{y}.csv")
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing Sackmann files: {missing}")
    return paths


def load_raw(cfg: dict[str, Any]) -> pd.DataFrame:
    """Concatenate all configured raw match files (strings), adding a `source_file` column for auditing."""
    frames = []
    for p in raw_file_paths(cfg):
        df = read_raw_csv(p)
        df["source_file"] = p.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_players(cfg: dict[str, Any]) -> pd.DataFrame:
    """atp_players.csv as strings (player_id is a string id)."""
    path = repo_path(cfg["paths"]["sackmann_dir"]) / "atp_players.csv"
    players = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8")
    if players["player_id"].duplicated().any():
        raise ValueError("atp_players.csv has duplicate player_id values")
    return players


# --------------------------------------------------------------------------------------------------------------
# Typing
# --------------------------------------------------------------------------------------------------------------
def type_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Type the string frame deliberately. Any non-numeric value in a numeric column raises.

    Strings are whitespace-stripped (empty -> NaN). IDs stay strings. Integer columns must be fully populated.
    """
    df = raw.copy()
    for c in STR_COLS:
        s = df[c].astype(object).str.strip()
        df[c] = s.where(s.notna() & (s != ""), np.nan)
    for c in INT_COLS:
        if df[c].isna().any():
            raise ValueError(f"column {c} has {int(df[c].isna().sum())} nulls; expected an integer everywhere")
        num = pd.to_numeric(df[c].str.strip(), errors="raise")
        if (num % 1 != 0).any():
            raise ValueError(f"column {c} has non-integer values")
        df[c] = num.astype("int64")
    for c in FLOAT_COLS:
        df[c] = pd.to_numeric(df[c].str.strip(), errors="raise").astype("float64")
    df["tourney_date"] = pd.to_datetime(
        df["tourney_date"].str.strip(), format="%Y%m%d", errors="raise"
    ).astype("datetime64[ns]")  # CONTRACT: datetime64[ns] (pandas 3 would default to [us])
    for c in ("tourney_id", "winner_id", "loser_id", "winner_name", "loser_name", "tourney_level", "round"):
        if df[c].isna().any():
            raise ValueError(f"column {c} has {int(df[c].isna().sum())} nulls")
    return df


# --------------------------------------------------------------------------------------------------------------
# Score vocabulary
# --------------------------------------------------------------------------------------------------------------
_SET_RE = re.compile(r"^(\d+)-(\d+)(?:\((\d+)\))?\??$")  # 6-4, 7-6(5), 6-4? (uncertain score marker)
_MTB_RE = re.compile(r"^\[(\d+)-(\d+)\]$")                # [10-7] match tie-break
_WALKOVER_TOKENS = {"w/o", "walkover"}
_RET_TOKENS = {"ret", "re", "ret'd"}
_DEF_TOKENS = {"def", "def.", "default"}
_ABANDON_TOKENS = {"abd", "abandoned", "unfinished"}
_ABANDON_FILLER = {"played", "and"}                       # only inside "Played and unfinished/abandoned"
_PLACEHOLDER_TOKENS = {">"}                               # carries no information

# Category precedence when several markers appear in one score.
SCORE_CATEGORIES = ["normal", "walkover", "default", "retirement", "abandoned", "null", "no_score"]


@dataclass(frozen=True)
class ScoreInfo:
    """Parsed score. `games` counts games in set scores; `has_mtb` = a match tie-break was played."""

    category: str
    games: int
    has_mtb: bool
    tokens: tuple[str, ...] = field(default_factory=tuple)

    @property
    def points_in_score(self) -> bool:
        """True if the score itself shows at least one game or a match tie-break."""
        return self.games > 0 or self.has_mtb


def classify_score(score: Any) -> ScoreInfo:
    """Classify a raw Sackmann score string. Unknown tokens raise ValueError (vocabulary must be explicit).

    Categories: normal, walkover (W/O, Walkover), default (DEF, Def., Default), retirement (RET, RE, Ret'd),
    abandoned (ABD, 'Played and unfinished', 'Played and abandoned'), null (missing/empty score),
    no_score (only placeholder tokens such as '>').
    """
    if score is None or (isinstance(score, float) and np.isnan(score)) or pd.isna(score):
        return ScoreInfo("null", 0, False, ())
    text = str(score).strip()
    if not text:
        return ScoreInfo("null", 0, False, ())
    tokens = tuple(text.split())
    games = 0
    has_mtb = False
    markers: set[str] = set()
    filler_seen = False
    for tok in tokens:
        low = tok.lower()
        m = _SET_RE.match(tok)
        if m:
            games += int(m.group(1)) + int(m.group(2))
            continue
        if _MTB_RE.match(tok):
            has_mtb = True
            continue
        if low in _WALKOVER_TOKENS:
            markers.add("walkover")
        elif low in _DEF_TOKENS:
            markers.add("default")
        elif low in _RET_TOKENS:
            markers.add("retirement")
        elif low in _ABANDON_TOKENS:
            markers.add("abandoned")
        elif low in _ABANDON_FILLER:
            filler_seen = True
        elif low in _PLACEHOLDER_TOKENS:
            markers.add("placeholder")
        else:
            raise ValueError(f"unknown score token {tok!r} in score {text!r}")
    if filler_seen and "abandoned" not in markers:
        raise ValueError(f"'Played'/'and' outside an abandoned-match phrase in score {text!r}")
    for cat in ("walkover", "default", "retirement", "abandoned"):
        if cat in markers:
            return ScoreInfo(cat, games, has_mtb, tokens)
    if games == 0 and not has_mtb:
        return ScoreInfo("no_score", 0, False, tokens)
    return ScoreInfo("normal", games, has_mtb, tokens)


def points_played_evidence(df: pd.DataFrame) -> pd.Series:
    """True where stats or duration show the match was started: w_svpt>0 or l_svpt>0 or minutes>0."""
    return (df["w_svpt"].fillna(0) > 0) | (df["l_svpt"].fillna(0) > 0) | (df["minutes"].fillna(0) > 0)


def score_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Return `score_category`, `is_walkover`, `is_retirement` for each row of a typed frame.

    is_walkover: explicit W/O/Walkover, OR no point is shown in the score (null/empty/placeholder/'0-0 RET'/
    bare 'DEF') AND there is no stats/duration evidence that points were played.
    is_retirement: RET/DEF/abandoned-type finish (an unfinished match is never a label). The two flags are
    independent; the engine drops walkovers first.
    """
    uniq = pd.Series(df["score"].dropna().unique())
    info = {s: classify_score(s) for s in uniq}
    null_info = classify_score(None)
    infos = [info.get(s, null_info) if isinstance(s, str) else null_info for s in df["score"]]
    category = pd.Series([i.category for i in infos], index=df.index)
    points_in_score = pd.Series([i.points_in_score for i in infos], index=df.index)
    evidence = points_played_evidence(df)
    is_walkover = (category == "walkover") | (~points_in_score & ~evidence)
    is_retirement = category.isin(["retirement", "default", "abandoned"])
    return pd.DataFrame(
        {"score_category": category, "is_walkover": is_walkover.astype(bool), "is_retirement": is_retirement},
        index=df.index,
    )


# --------------------------------------------------------------------------------------------------------------
# Surface, round, ids, order
# --------------------------------------------------------------------------------------------------------------
def normalise_surface(surface: pd.Series, surface_map: dict[str, str]) -> tuple[pd.Series, pd.Series]:
    """Map raw surfaces through `data.surface_map`. Null -> 'Hard' with imputed=True (D-011). Unknown -> raise."""
    imputed = surface.isna()
    unknown = sorted(set(surface.dropna().unique()) - set(surface_map))
    if unknown:
        raise ValueError(f"unknown surface values: {unknown}")
    out = surface.map(surface_map)
    out = out.where(~imputed, "Hard")
    return out.astype(object), imputed.astype(bool)


def round_rank(rounds: pd.Series, round_order: list[str]) -> pd.Series:
    """Index of each round code in `data.round_order`. Unknown codes raise."""
    unknown = sorted(set(rounds.dropna().unique()) - set(round_order))
    if unknown or rounds.isna().any():
        raise ValueError(f"unknown round codes: {unknown} (nulls: {int(rounds.isna().sum())})")
    lookup = {r: i for i, r in enumerate(round_order)}
    return rounds.map(lookup).astype("int64")


def make_match_id(tourney_id: str, match_num: int, winner_id: str, loser_id: str) -> str:
    """CONTRACT section 1: sha1(f"{tourney_id}|{match_num}|{min(ids)}|{max(ids)}")[:16], ids as strings."""
    a, b = str(winner_id), str(loser_id)
    lo, hi = (a, b) if a <= b else (b, a)
    return hashlib.sha1(f"{tourney_id}|{int(match_num)}|{lo}|{hi}".encode("utf-8")).hexdigest()[:16]


def make_match_ids(df: pd.DataFrame) -> pd.Series:
    """Vectorised `make_match_id` over a frame."""
    return pd.Series(
        [make_match_id(t, m, w, l) for t, m, w, l in
         zip(df["tourney_id"], df["match_num"], df["winner_id"], df["loser_id"])],
        index=df.index, dtype=object,
    )


MATCH_ORDER: list[str] = ["tourney_date", "tourney_id", "round_rank", "match_num", "match_id"]


def sort_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Sort into the authoritative match order (CONTRACT section 1/2):
    (tourney_date, tourney_id, round_rank, match_num, match_id), match_id being the final tiebreaker."""
    return df.sort_values(MATCH_ORDER, kind="mergesort")


def assign_order_key(df: pd.DataFrame) -> pd.DataFrame:
    """Sort into match order, reset the index and set `order_key` = 0-based int64 position (CONTRACT section 1)."""
    out = sort_matches(df).reset_index(drop=True)
    out["order_key"] = np.arange(len(out), dtype="int64")
    return out


def order_ties(df: pd.DataFrame) -> pd.DataFrame:
    """Rows sharing (tourney_date, tourney_id, round_rank, match_num) with another row (match_id breaks them)."""
    return df[df.duplicated(MATCH_ORDER[:4], keep=False)]


# --------------------------------------------------------------------------------------------------------------
# Data-quality rules
# --------------------------------------------------------------------------------------------------------------
def invalid_stats_mask(df: pd.DataFrame) -> pd.Series:
    """D-012: impossible serve counts for either player, or any negative stat value."""
    bad = (df[STAT_COLS] < 0).any(axis=1)
    for p in ("w_", "l_"):
        svpt, first_in = df[f"{p}svpt"], df[f"{p}1stIn"]
        bad |= first_in > svpt
        bad |= df[f"{p}1stWon"] > first_in
        bad |= df[f"{p}2ndWon"] > svpt - first_in
        bad |= df[f"{p}bpSaved"] > df[f"{p}bpFaced"]
    return bad.fillna(False).astype(bool)


def apply_invalid_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Set all 18 stat columns to NaN where stats are impossible and add `stats_invalid`. Results untouched."""
    out = df.copy()
    bad = invalid_stats_mask(out)
    out.loc[bad, STAT_COLS] = np.nan
    out["stats_invalid"] = bad
    return out


def split_self_matches(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """D-010: return (kept, dropped) where dropped has winner_id == loser_id."""
    mask = df["winner_id"] == df["loser_id"]
    return df[~mask], df[mask]


def split_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """D-009: drop rows repeating (tourney_id, round, winner_id, loser_id, score), keeping the first in match
    order. Null scores compare equal. Returns (kept, dropped); `dropped` gets `duplicate_of` (kept match_id)."""
    ordered = sort_matches(df)
    key = dup_key(ordered)
    dup = key.duplicated(keep="first")
    dropped = ordered[dup].copy()
    first_id = ordered["match_id"].groupby(key).first()
    dropped["duplicate_of"] = key[dup].map(first_id).astype(object)
    return ordered[~dup], dropped


def dup_key(df: pd.DataFrame) -> pd.Series:
    """The D-009 key as one string per row; nulls become '<NA>' so null scores compare equal."""
    parts = [df[c].astype(object).where(df[c].notna(), "<NA>").astype(str) for c in DUP_KEY]
    key = parts[0]
    for p in parts[1:]:
        key = key + "\x1f" + p
    return key


def split_excluded_tourneys(df: pd.DataFrame, names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """D-022: remove every row whose tourney_name equals one of `names` exactly. Returns (kept, dropped)."""
    mask = df["tourney_name"].isin(list(names))
    return df[~mask], df[mask]


def apply_level_overrides(
    df: pd.DataFrame, overrides: list[dict[str, str]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """D-022: set tourney_level = override level where tourney_name FULLY matches `name_regex`.

    Returns (frame, changes, hits):
    - changes: one row per (year, tourney_id, tourney_name, old_level, new_level) actually changed, with row counts.
    - hits: per override, rows matched and rows changed.
    A name matched by two overrides with different levels raises. Only `tourney_level` is touched.
    """
    out = df.copy()
    new_level = pd.Series(pd.NA, index=out.index, dtype=object)
    hit_rows = []
    for ov in overrides:
        rx, lvl = str(ov["name_regex"]), str(ov["level"])
        match = out["tourney_name"].astype(str).str.fullmatch(rx)
        clash = match & new_level.notna() & (new_level != lvl)
        if clash.any():
            names = sorted(out.loc[clash, "tourney_name"].unique())
            raise ValueError(f"level_overrides conflict for tourney names {names}")
        new_level = new_level.mask(match, lvl)
        changed = match & (out["tourney_level"] != lvl)
        hit_rows.append({"name_regex": rx, "level": lvl, "rows matched": int(match.sum()),
                         "rows changed": int(changed.sum())})
    changed = new_level.notna() & (out["tourney_level"] != new_level)
    ch = out[changed]
    changes = (
        pd.DataFrame({"year": ch["tourney_date"].dt.year, "tourney_id": ch["tourney_id"],
                      "tourney_name": ch["tourney_name"], "old_level": ch["tourney_level"],
                      "new_level": new_level[changed]})
        .value_counts().rename("rows").reset_index().sort_values(["year", "tourney_id"]).reset_index(drop=True)
    )
    out.loc[changed, "tourney_level"] = new_level[changed]
    hits = pd.DataFrame(hit_rows, columns=["name_regex", "level", "rows matched", "rows changed"])
    return out, changes, hits


def same_match_key(df: pd.DataFrame) -> pd.Series:
    """(tourney_date, round, unordered player pair) as a string: two rows with this key describe the same
    meeting, because two players cannot meet twice in one round of one event."""
    w, l = df["winner_id"].astype(str), df["loser_id"].astype(str)
    lo = w.where(w <= l, l)
    hi = l.where(w <= l, w)
    return df["tourney_date"].astype(str) + "\x1f" + df["round"].astype(str) + "\x1f" + lo + "\x1f" + hi


def conflicting_results(df: pd.DataFrame) -> pd.DataFrame:
    """Non-walkover rows that share `same_match_key` with another non-walkover row: one meeting recorded
    twice, typically with the winner swapped and a different score, so the source contradicts itself.
    Returns those rows in match order (all members of each group)."""
    nw = df[~df["is_walkover"]]
    return sort_matches(nw[same_match_key(nw).duplicated(keep=False)])


def walkover_shadows(df: pd.DataFrame) -> pd.DataFrame:
    """Walkover rows sharing `same_match_key` with a played row (harmless: the engine drops walkovers)."""
    key = same_match_key(df)
    played = set(key[~df["is_walkover"]])
    return df[df["is_walkover"] & key.isin(played)]


# --------------------------------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------------------------------
def unify(typed: pd.DataFrame, cfg: dict[str, Any], source: str = SOURCE) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Typed Sackmann-format frame -> unified frame + audit dict (dropped rows, flag counts, vocabulary).

    Works for any source in Sackmann format (`source` sets the `source` column).
    """
    data_cfg = cfg["data"]
    df = typed.copy()
    audit: dict[str, Any] = {"rows_in": len(df)}

    audit["carpet_rows"] = int((df["surface"] == "Carpet").sum())
    df["surface"], df["surface_imputed"] = normalise_surface(df["surface"], data_cfg["surface_map"])
    df["round_rank"] = round_rank(df["round"], data_cfg["round_order"])
    df["match_id"] = make_match_ids(df)
    df["source"] = source

    # D-022: excluded tourneys (exact tourney_name) are removed entirely; level labels corrected by name.
    df, excluded = split_excluded_tourneys(df, data_cfg.get("excluded_tourneys", []))
    audit["excluded_tourneys"] = excluded
    df, level_changes, override_hits = apply_level_overrides(df, data_cfg.get("level_overrides", []))
    audit["level_changes"] = level_changes
    audit["level_override_hits"] = override_hits

    flags = score_flags(df)
    df["is_walkover"] = flags["is_walkover"]
    df["is_retirement"] = flags["is_retirement"]
    df["score_category"] = flags["score_category"]
    df["is_qualifying"] = df["round"].isin(data_cfg["qualifying_rounds"])
    df["is_training_level"] = df["tourney_level"].isin(data_cfg["training_levels"]) & ~df["is_qualifying"]

    audit["stats_invalid_rows"] = df[invalid_stats_mask(df)].copy()
    df = apply_invalid_stats(df)

    df, self_matches = split_self_matches(df)
    df, duplicates = split_exact_duplicates(df)
    audit["self_matches"] = self_matches
    audit["duplicates"] = duplicates

    # Same meeting recorded twice with contradictory results: always reported; dropped (all rows of each
    # group) only when `cleaning.drop_conflicting_results` is true -- an orchestrator decision. Default: keep.
    conflicts = conflicting_results(df)
    drop_conflicts = bool(cfg.get("cleaning", {}).get("drop_conflicting_results", False))
    audit["conflicts"] = conflicts
    audit["conflicts_dropped"] = drop_conflicts
    if drop_conflicts:
        df = df[~df["match_id"].isin(conflicts["match_id"])]
    audit["walkover_shadows"] = walkover_shadows(df)

    if df["match_id"].duplicated().any():
        dups = df.loc[df["match_id"].duplicated(keep=False), ["match_id", "tourney_id", "match_num"]]
        raise ValueError(f"duplicate match_id after cleaning:\n{dups}")

    df = assign_order_key(df)
    audit["order_ties"] = order_ties(df)
    # Per-row audit info keyed by match_id (the merged table may be in a different row order).
    audit["score_category"] = df.set_index("match_id")["score_category"]
    audit["source_file"] = (df.set_index("match_id")["source_file"] if "source_file" in df.columns
                            else pd.Series("", index=df["match_id"]))
    for c in FLAG_COLS:
        df[c] = df[c].astype(bool)
    return df[UNIFIED_COLUMNS], audit


def score_vocabulary(scores: pd.Series) -> pd.DataFrame:
    """Count every non-set-score token in a score column (null scores shown as '<null>')."""
    s = scores.astype(object).where(scores.notna(), "<null>")
    toks = s.astype(str).str.split().explode()
    toks = toks[~toks.str.match(r"^\d+-\d+(\(\d+\))?$") & ~toks.str.match(r"^\[\d+-\d+\]$")]
    return toks.value_counts().rename_axis("token").reset_index(name="rows")


def build_sackmann(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load + type + unify all configured Sackmann files. Returns (unified, audit)."""
    raw = load_raw(cfg)
    typed = type_columns(raw)
    unified, audit = unify(typed, cfg)
    audit["raw_files"] = sorted(raw["source_file"].unique())
    audit["raw_scores"] = raw["score"]
    return unified, audit


# --------------------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------------------
_TOKEN_MEANING: dict[str, tuple[str, str]] = {
    "W/O": ("walkover", "is_walkover=True (no result; the engine drops it)"),
    "Walkover": ("walkover", "is_walkover=True"),
    "RET": ("retirement", "is_retirement=True; also walkover if no point was played"),
    "RE": ("retirement (truncated RET)", "is_retirement=True"),
    "Ret'd": ("retirement", "is_retirement=True"),
    "DEF": ("default", "is_retirement=True (RET/DEF); also walkover if no point was played"),
    "Def.": ("default", "is_retirement=True; also walkover if no point was played"),
    "Default": ("default", "is_retirement=True"),
    "ABD": ("abandoned (Davis Cup dead rubber)", "is_retirement=True: unfinished, never a label"),
    "Played": ("part of 'Played and unfinished/abandoned'", "is_retirement=True: unfinished, never a label"),
    "and": ("part of 'Played and unfinished/abandoned'", "(phrase filler)"),
    "unfinished": ("abandoned", "is_retirement=True: unfinished, never a label"),
    "abandoned": ("abandoned", "is_retirement=True: unfinished, never a label"),
    ">": ("placeholder, no information", "no point shown -> walkover unless stats/minutes show play"),
    "<null>": ("missing score", "no point shown -> walkover unless stats/minutes show play"),
    "<set score>?": ("uncertain-score marker on a completed match", "normal result; '?' ignored"),
    "[a-b] match tie-break": ("match tie-break / super tie-break", "normal result; counts as play"),
}

_SHOW = ["match_id", "tourney_name", "tourney_date", "round", "winner_name", "loser_name", "score"]


def _fmt(v: Any) -> str:
    """Format one markdown table cell."""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return ""
        return f"{int(v):,}" if float(v).is_integer() else f"{v:.3g}"
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return ""
    return str(v).replace("|", "\\|")


def _md_table(df: pd.DataFrame, index: bool = False) -> str:
    """Render a DataFrame as a GitHub markdown table."""
    if index:  # index values (years, levels) are labels, not quantities: no thousands separators
        df = df.copy()
        if isinstance(df.index, pd.MultiIndex):
            df.index = df.index.set_levels([lv.astype(str) for lv in df.index.levels])
        else:
            df.index = df.index.astype(str)
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in d.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def _token_label(tok: str) -> str:
    """Collapse '6-4?'-style tokens and match tie-breaks into one vocabulary label each."""
    if re.match(r"^\d+-\d+(\(\d+\))?\?$", tok):
        return "<set score>?"
    if _MTB_RE.match(tok):
        return "[a-b] match tie-break"
    return tok


def _stat_reason(r: pd.Series) -> str:
    """Human-readable reason a row failed the D-012 check."""
    rs = []
    for p in ("w_", "l_"):
        sv, fi = r[f"{p}svpt"], r[f"{p}1stIn"]
        if fi > sv:
            rs.append(f"{p}1stIn {fi:.0f} > {p}svpt {sv:.0f}")
        if r[f"{p}1stWon"] > fi:
            rs.append(f"{p}1stWon {r[f'{p}1stWon']:.0f} > {p}1stIn {fi:.0f}")
        if r[f"{p}2ndWon"] > sv - fi:
            rs.append(f"{p}2ndWon {r[f'{p}2ndWon']:.0f} > {p}svpt-1stIn {sv - fi:.0f}")
        if r[f"{p}bpSaved"] > r[f"{p}bpFaced"]:
            rs.append(f"{p}bpSaved {r[f'{p}bpSaved']:.0f} > {p}bpFaced {r[f'{p}bpFaced']:.0f}")
    negs = [c for c in STAT_COLS if r[c] < 0]
    if negs:
        rs.append("negative " + ", ".join(f"{c}={r[c]:.0f}" for c in negs))
    return "; ".join(rs)


def training_mask(m: pd.DataFrame) -> pd.Series:
    """Rows the engine will emit (ignoring the feature_first_year cut): training level, not WO, not RET/DEF."""
    return m["is_training_level"] & ~m["is_walkover"] & ~m["is_retirement"]


def render_report(
    matches: pd.DataFrame,
    audit: dict[str, Any],
    contributions: pd.DataFrame,
    player_map: pd.DataFrame,
    cfg: dict[str, Any],
) -> str:
    """Markdown ingestion report for reports/data/ingest_sackmann.md."""
    m = matches
    data_cfg = cfg["data"]
    year = m["tourney_date"].dt.year.rename("year")
    training = training_mask(m)
    has_stats = m[STAT_COLS].notna().all(axis=1)
    cat = pd.Series(audit["score_category"].reindex(m["match_id"]).values, index=m.index)
    evidence = points_played_evidence(m)
    src = pd.Series(audit["source_file"].reindex(m["match_id"]).fillna("").values, index=m.index)
    out: list[str] = []
    w = out.append

    w("# Ingestion report: Sackmann `tennis_atp` -> `data/interim/matches.parquet`\n")
    w(f"Generated by `.venv/Scripts/python -m tennis.ingest.merge` on {pd.Timestamp.now():%Y-%m-%d %H:%M}. "
      f"Source: `{cfg['paths']['sackmann_dir']}` (read in place; nothing under `data/raw/` is modified). "
      f"Loaded `atp_matches_YYYY.csv` + `atp_matches_qual_chall_YYYY.csv` for {data_cfg['first_year']}-"
      f"{data_cfg['last_sackmann_year']} ({len(audit['raw_files'])} files) and `atp_players.csv`. "
      "Futures files are not loaded. All CSVs are read as strings, then typed column by column (a non-numeric "
      "value in a numeric column raises).\n")
    w("Years in this report are `tourney_date` years: events starting in late December count in the year they "
      "start, although Sackmann files them under the next season.\n")

    # ---- 1. Summary
    w("## 1. Summary\n")
    n_self, n_dup = len(audit["self_matches"]), len(audit["duplicates"])
    summ = pd.DataFrame([
        ("raw rows read", audit["rows_in"]),
        ("dropped: self-matches (D-010)", n_self),
        ("dropped: excluded tourneys (D-022)", len(audit.get("excluded_tourneys", []))),
        ("dropped: exact duplicates (D-009)", n_dup),
        ("contradictory duplicate results (section 4.3)",
         f"{len(audit.get('conflicts', []))} rows, "
         f"{'dropped (D-014)' if audit.get('conflicts_dropped') else 'kept'}"),
        ("rows in matches.parquet", len(m)),
        ("unique match_id", m["match_id"].nunique()),
        ("players (all found in atp_players.csv)", len(player_map)),
        ("tour-level training rows, all years", int(training.sum())),
        ("tour-level training rows, 2000+", int((training & (year >= 2000)).sum())),
        ("tourney_date range", f"{m['tourney_date'].min():%Y-%m-%d} to {m['tourney_date'].max():%Y-%m-%d}"),
    ], columns=["item", "value"])
    w(_md_table(summ) + "\n")

    # ---- 2. Flags
    w("## 2. Flag counts (rows in matches.parquet)\n")
    flags = pd.DataFrame([
        ("is_walkover", int(m["is_walkover"].sum())),
        ("is_retirement (RET/DEF/abandoned)", int(m["is_retirement"].sum())),
        ("both (retired/defaulted with no point played)", int((m["is_walkover"] & m["is_retirement"]).sum())),
        ("is_qualifying", int(m["is_qualifying"].sum())),
        ("is_training_level", int(m["is_training_level"].sum())),
        ("surface_imputed (null surface -> Hard, D-011)", int(m["surface_imputed"].sum())),
        ("stats_invalid (D-012)", int(m["stats_invalid"].sum())),
        ("Carpet -> Hard (raw rows, before drops)", audit["carpet_rows"]),
    ], columns=["flag", "rows"])
    w(_md_table(flags) + "\n")

    # ---- 3. Score vocabulary
    w("## 3. Score vocabulary and decisions\n")
    w("Every non-set token in the raw `score` column (raw rows, before drops; set scores such as `6-4` and "
      "`7-6(5)` are not listed). A token outside this vocabulary makes `classify_score` raise, so new "
      "vocabulary can't slip through silently.\n")
    voc = score_vocabulary(audit["raw_scores"])
    voc["token"] = voc["token"].map(_token_label)
    voc = voc.groupby("token", as_index=False)["rows"].sum().sort_values("rows", ascending=False)
    voc["meaning"] = voc["token"].map(lambda t: _TOKEN_MEANING.get(t, ("?", "?"))[0])
    voc["decision"] = voc["token"].map(lambda t: _TOKEN_MEANING.get(t, ("?", "?"))[1])
    w(_md_table(voc) + "\n")
    w("Rules (`classify_score` + `score_flags` in `src/tennis/ingest/sackmann.py`):\n")
    w("- **Walkover** (`is_walkover`): the score contains `W/O` or `Walkover` (case-insensitive), **or** the score "
      "shows no point (null/empty, only `>`, `0-0 RET`, bare `RET`/`DEF`/`Def.`) **and** nothing else shows play "
      "(`w_svpt > 0` or `l_svpt > 0` or `minutes > 0`). A null score is no evidence that points were played, so it "
      "is walkover-like unless the stats or minutes say otherwise. Walkover rows stay in matches.parquet so they "
      "remain auditable; the engine drops them (CONTRACT section 2).")
    w("- **Retirement** (`is_retirement`): RET / RE / Ret'd, DEF / Def. / Default, and abandoned matches (ABD, "
      "'Played and unfinished', 'Played and abandoned'). An abandoned Davis Cup rubber is an unfinished match, so "
      "it is treated like a retirement: never a label. The two flags are independent.")
    w("- Set scores with a trailing `?` and match tie-breaks `[a-b]` are normal completed results.\n")
    tab = (pd.DataFrame({"score category": cat, "is_walkover": m["is_walkover"],
                         "is_retirement": m["is_retirement"]})
           .value_counts().rename("rows").reset_index().sort_values("rows", ascending=False))
    w("Score category x flags:\n")
    w(_md_table(tab) + "\n")
    no_pts = ~m["score"].fillna("").str.contains(r"[1-9]")
    kept = m[~m["is_walkover"] & no_pts]
    w(f"Rows whose score shows no point but whose stats/minutes show play (kept as retirements, not walkovers): "
      f"{len(kept)}\n")
    if len(kept):
        w(_md_table(kept[_SHOW + ["w_svpt", "l_svpt", "minutes"]]) + "\n")
    nulls_sc = m[m["score"].isna()]
    w(f"Null scores: {len(nulls_sc)} rows, {int(points_played_evidence(nulls_sc).sum())} with stats/minutes "
      "evidence of play, so all are walkover-like. By level: "
      + ", ".join(f"{k} {v}" for k, v in nulls_sc["tourney_level"].value_counts().items())
      + ". By round: " + ", ".join(f"{k} {v}" for k, v in nulls_sc["round"].value_counts().items()) + ".\n")

    # ---- 4. Dropped rows
    w("## 4. Dropped rows\n")
    show = ["match_id", "tourney_id", "tourney_name", "tourney_date", "round", "match_num",
            "winner_id", "winner_name", "loser_id", "loser_name", "score"]
    w(f"### 4.1 Self-matches (D-010): {n_self} rows\n")
    w("Reason: winner_id == loser_id, so no valid result exists.\n")
    w(_md_table(audit["self_matches"][show]) + "\n" if n_self else "None.\n")
    w(f"### 4.2 Exact duplicates (D-009): {n_dup} rows\n")
    w("Reason: same (tourney_id, round, winner_id, loser_id, score) as an earlier row in match order. "
      "`duplicate_of` is the kept row's match_id.\n")
    w(_md_table(audit["duplicates"][show + ["duplicate_of"]]) + "\n" if n_dup else "None.\n")

    conf = audit.get("conflicts", pd.DataFrame())
    dropped_conf = audit.get("conflicts_dropped", False)
    w(f"### 4.3 Contradictory duplicate results: {len(conf)} rows "
      f"({'DROPPED' if dropped_conf else 'KEPT, flagged here'})\n")
    w("Non-walkover rows where the same two players meet in the same round of an event starting on the same "
      "date more than once. Two players can't meet twice in one round, so the source records one meeting "
      "twice, here with the **winner swapped and a different score** (so D-009 doesn't catch them). The "
      "source contradicts itself, so no reliable result exists. They are dropped only if "
      "`cleaning.drop_conflicting_results: true` (D-014; with the flag false the rows are kept "
      "unchanged). While they're kept, the leakage suite's `test_no_duplicates_real_matches` (PLAN section 2, "
      "rule 8) fails on them.\n")
    if len(conf):
        w(_md_table(conf[["match_id", "tourney_id", "tourney_name", "tourney_date", "round", "match_num",
                          "winner_name", "loser_name", "score", "w_svpt", "l_svpt", "minutes"]]) + "\n")
    shadows = audit.get("walkover_shadows", pd.DataFrame())
    w(f"Also: {len(shadows)} walkover row(s) record the same meeting as a played row. Kept; harmless, because "
      "the engine drops walkovers.\n")
    if len(shadows):
        w(_md_table(shadows[["match_id", "tourney_id", "tourney_name", "tourney_date", "round", "match_num",
                             "winner_name", "loser_name", "score"]]) + "\n")

    exc = audit.get("excluded_tourneys", pd.DataFrame(columns=["tourney_name", "tourney_date"]))
    w(f"### 4.4 Excluded tourneys (D-022): {len(exc)} rows\n")
    w(f"Reason: `tourney_name` exactly in `data.excluded_tourneys` = {data_cfg.get('excluded_tourneys', [])} "
      "(exhibition, not an official competitive match). Removed from matches.parquet entirely.\n")
    if len(exc):
        per = (exc.assign(year=exc["tourney_date"].dt.year)
               .groupby(["tourney_name", "year"]).agg(tourney_id=("tourney_id", "first"), rows=("match_id", "size"))
               .reset_index())
        w(_md_table(per.assign(year=per["year"].astype(str))) + "\n")

    # ---- level overrides
    w("### 4.5 Level overrides (D-022), results untouched\n")
    w("`tourney_level` is set to the override level where `tourney_name` fully matches `name_regex` "
      "(`data.level_overrides`); `is_training_level` is computed after the override.\n")
    hits = audit.get("level_override_hits")
    if hits is not None and len(hits):
        w(_md_table(hits) + "\n")
    lc = audit.get("level_changes")
    w(f"Every change (year, tourney, old level -> new level): {0 if lc is None else int(lc['rows'].sum())} rows.\n")
    if lc is not None and len(lc):
        w(_md_table(lc.assign(year=lc["year"].astype(str))) + "\n")
    rxs = [str(o["name_regex"]) for o in data_cfg.get("level_overrides", [])]
    names = pd.Series(m["tourney_name"].unique())
    looks = names[names.str.contains(r"(?i)olympic|tour finals|masters cup|atp finals")
                  & ~names.str.contains(r"(?i)challenger|davis cup")]
    missed = sorted(n for n in looks if not any(re.fullmatch(rx, n) for rx in rxs))
    if missed:
        lv = m[m["tourney_name"].isin(missed)].assign(year=year).groupby("tourney_name").agg(
            levels=("tourney_level", lambda s: ", ".join(sorted(s.unique()))),
            years=("year", lambda s: ", ".join(str(v) for v in sorted(s.unique()))),
            rows=("match_id", "size")).reset_index()
        w(f"**Name variants not matched by any override** (Olympics / Tour Finals look-alikes; regexes were not "
          f"widened, pending an orchestrator decision): {len(missed)}\n")
        w(_md_table(lv) + "\n")

    # ---- 5. stats invalid
    inv = audit["stats_invalid_rows"]
    w(f"## 5. Impossible serve statistics (D-012): {len(inv)} rows\n")
    w("All 18 stat columns set to NaN and `stats_invalid=True`; the result is kept. Checked for either player: "
      "1stIn > svpt, 1stWon > 1stIn, 2ndWon > svpt - 1stIn, bpSaved > bpFaced, and any negative stat. "
      "(D-012 and source_check.md name one row, Roland Garros 2024; the check over every loaded file, "
      "including qual_chall, plus the negative-value check finds these.)\n")
    if len(inv):
        t = inv[_SHOW].copy()
        t["reason"] = [_stat_reason(r) for _, r in inv.iterrows()]
        t["in parquet"] = t["match_id"].isin(m["match_id"])
        w(_md_table(t) + "\n")

    # ---- 6. per year x level
    w("## 6. Rows per year x level (all rows in matches.parquet, incl. walkovers, retirements, qualifying)\n")
    lv = pd.crosstab(year, m["tourney_level"])
    std = ["G", "M", "A", "F", "D", "O", "C"]
    order = [c for c in std if c in lv.columns] + [c for c in lv.columns if c not in std]
    lv = lv[order]
    lv["total"] = lv[order].sum(axis=1)
    lv["of which qualifying"] = m.groupby(year)["is_qualifying"].sum()
    w(_md_table(lv, index=True) + "\n")
    q = m[m["is_qualifying"]]
    if len(q):
        firsts = q.groupby("tourney_level")["tourney_date"].min()
        w("Coverage of qualifying in the source is uneven: first qualifying row per level: "
          + ", ".join(f"{k} {v:%Y-%m-%d}" for k, v in firsts.items())
          + ". Before that, players' qualifying matches are simply absent from state.\n")

    # ---- 7. training per year
    w("## 7. Tour-level training rows per year and stats coverage\n")
    w(f"Training rows = `is_training_level` (level in {data_cfg['training_levels']}, not qualifying) and not "
      "walkover and not retirement: the rows the engine will emit (from `data.feature_first_year` on). "
      "Stats coverage = share of training rows with all 18 stat columns non-null.\n")
    tr = pd.DataFrame({
        "tour main-draw rows": m[m["is_training_level"]].groupby(year).size(),
        "walkovers": m[m["is_training_level"] & m["is_walkover"]].groupby(year).size(),
        "retirements": m[m["is_training_level"] & m["is_retirement"] & ~m["is_walkover"]].groupby(year).size(),
        "training rows": m[training].groupby(year).size(),
        "with stats": m[training & has_stats].groupby(year).size(),
    }).fillna(0).astype(int)
    tr["stats coverage"] = [f"{a / b:.1%}" if b else "" for a, b in zip(tr["with stats"], tr["training rows"])]
    allcov = m[~m["is_walkover"]].assign(_s=has_stats).groupby(year)["_s"].mean()
    tr["coverage, all non-WO rows"] = allcov.reindex(tr.index).map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
    w(_md_table(tr, index=True) + "\n")
    recent = tr.loc[(tr.index >= 2000) & (tr.index <= 2025) & (tr.index != 2020), "training rows"]
    ok = bool(((recent >= 2500) & (recent <= 4500)).all())
    get = lambda y: int(tr.loc[y, "training rows"]) if y in tr.index else 0  # noqa: E731
    w(f"**Acceptance** (2,500-4,500 training rows per year, 2000-2025 except 2020): **{'PASS' if ok else 'FAIL'}** "
      f"(min {recent.min():,}, max {recent.max():,}). 2020: {get(2020):,} (COVID season). "
      f"2026: {get(2026):,} (partial season, to Roland Garros).\n")

    # ---- 8. qualifying labelling
    w("## 8. How qualifying is labelled\n")
    in_qc = src.str.contains("qual_chall")
    ql = pd.crosstab([pd.Series(np.where(in_qc, "qual_chall", "tour"), index=m.index, name="file family"),
                      m["tourney_level"]],
                     pd.Series(np.where(m["is_qualifying"], "Q1-Q4", "main draw"), index=m.index, name="rounds"))
    w(_md_table(ql, index=True) + "\n")
    tq = m[in_qc & m["is_qualifying"] & m["tourney_level"].isin(["A", "G", "M"])]
    main = m[~in_qc].groupby("tourney_id")[["tourney_date", "tourney_name"]].first()
    tqe = tq.groupby("tourney_id")[["tourney_date", "tourney_name"]].first()
    common = tqe.index.intersection(main.index)
    same = int(((tqe.loc[common, "tourney_date"] == main.loc[common, "tourney_date"])
                & (tqe.loc[common, "tourney_name"] == main.loc[common, "tourney_name"])).sum())
    w(f"- Tour-event qualifying is in the qual_chall files, labelled with the **main event's level (A/G/M)** and "
      f"the **main draw's tourney_id**, rounds Q1-Q3: {len(tq):,} rows over {len(tqe):,} events. "
      f"{len(common):,} of those tourney_ids also appear in the tour files; {same:,} of them with the same "
      "tourney_date and tourney_name.")
    for oid in tqe.index.difference(main.index):
        r = tqe.loc[oid]
        w(f"- Exception: `{oid}` '{r['tourney_name']}' ({r['tourney_date']:%Y-%m-%d}) has qualifying rows only: "
          "the 2021 Australian Open qualifying, played in Doha under its own tourney_id. Its rows are "
          "qualifying, so they update state but are never training rows.")
    w("- Challenger qualifying is level `C` with Q rounds under the Challenger's tourney_id.")
    w("- The tour files contain no Q-round rows.")
    w(f"- Qualifying rows with `is_training_level=True`: "
      f"**{int((m['is_qualifying'] & m['is_training_level']).sum())}** (must be 0).")
    w("- Q rounds have the lowest round_rank, so within an event qualifying sorts before the main draw "
      "(same tourney_id and tourney_date).\n")

    w("### Match order and `order_key`\n")
    w("matches.parquet is stored sorted by (tourney_date, tourney_id, round_rank, match_num, match_id), and "
      "`order_key` is the 0-based int64 position in that order (CONTRACT section 1).")
    sorted_ok = bool((m["order_key"].to_numpy() == np.arange(len(m))).all()
                     and m[MATCH_ORDER].equals(sort_matches(m)[MATCH_ORDER].reset_index(drop=True)))
    ties = order_ties(m)
    w(f"- Stored order equals the sort and order_key equals the position: **{sorted_ok}**.")
    w(f"- Rows that tie on (tourney_date, tourney_id, round_rank, match_num), i.e. that need match_id to be "
      f"ordered: **{len(ties)}**.")
    if len(ties):
        w("\n" + _md_table(ties[["order_key", "match_id", "tourney_id", "tourney_name", "tourney_date", "round",
                                 "match_num", "winner_name", "loser_name", "score"]]))
    w("")

    # ---- 9. anomalies
    w("## 9. Other anomalies (data unchanged)\n")
    wo_stats = m[(cat == "walkover") & evidence]
    w(f"### 9.1 Explicit walkovers whose stats/minutes show play: {len(wo_stats)}\n")
    w("Kept as walkovers because the score says W/O. Listed for review (open question).\n")
    if len(wo_stats):
        w(_md_table(wo_stats[_SHOW + ["w_svpt", "l_svpt", "minutes"]]) + "\n")
    mtb = m["score"].fillna("").str.extract(r"\[(\d+)-(\d+)\]\s*$").astype(float)
    inverted = m[(mtb[0] < mtb[1]) & ~m["is_retirement"]]
    w(f"### 9.2 Completed matches whose final match tie-break is written as won by the loser: {len(inverted)}\n")
    w("Probably score-entry errors (or the tie-break written loser-first). The result is not changed.\n")
    if len(inverted):
        w(_md_table(inverted[_SHOW]) + "\n")
    partial = m[m[STAT_COLS].notna().any(axis=1) & m[STAT_COLS].isna().any(axis=1)]
    w(f"### 9.3 Rows with partial stats (some of the 18 stat columns null): {len(partial)}\n")
    w("Not flagged invalid. The engine's stats buffer needs all nine serve fields for both players, so these "
      "rows are skipped by the buffers when a needed field is missing.\n")
    if len(partial):
        t = partial[_SHOW].copy()
        t["null stat columns"] = partial[STAT_COLS].isna().apply(
            lambda r: ", ".join(c for c, v in r.items() if v), axis=1)
        w(_md_table(t) + "\n")
    file_year = src.str.extract(r"(\d{4})\.csv$")[0].astype(float)
    w(f"### 9.4 tourney_date year differs from file year: {int((file_year != year).sum()):,} rows\n")
    w("Events starting in late December are filed under the next season. Expected; ordering uses tourney_date.\n")
    w("### 9.5 Whitespace\n")
    raw_sc = audit["raw_scores"].dropna()
    n_ws = int((raw_sc != raw_sc.str.strip()).sum())
    w(f"{n_ws} raw scores have leading/trailing spaces (e.g. ' W/O'); all string columns are stripped when "
      "typed.\n")

    # ---- 10. nulls
    w("## 10. Nulls per column (matches.parquet)\n")
    nulls = m.isna().sum()
    nt = pd.DataFrame({"nulls": nulls, "share": [f"{v / len(m):.2%}" for v in nulls]}).rename_axis("column")
    w(_md_table(nt[nt["nulls"] > 0], index=True) + "\n")
    w("Every other column has no nulls. Stat-column nulls include the stats_invalid rows.\n")

    # ---- 11. merge
    w("## 11. Merge and player map\n")
    w(f"Source priority {data_cfg['source_priority']}; de-duplication on match_id, then on the D-009 key.\n")
    w(_md_table(contributions) + "\n")
    w(f"`data/interim/player_map.csv`: {len(player_map):,} Sackmann ids, each mapped to itself (method "
      "`identity`). Every winner_id/loser_id exists in `atp_players.csv` (an unknown id raises "
      "`UnmatchedPlayerError`).\n")
    return "\n".join(out) + "\n"
