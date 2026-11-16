"""Betting odds (tennis-data.co.uk yearly ATP files) -> `data/interim/odds.parquet`, matched to Sackmann `match_id`.

Used only for the bookmaker baseline (PLAN section 7: implied probability with the margin removed). Odds are never
model inputs.

Pipeline (all pure functions except `main`):
1. `read_odds_file` + `parse_odds`: one yearly Excel file -> typed frame (text stripped, ranks/odds numeric, comment
   normalised). Rows dated after `odds.max_date` are dropped (DECISIONS D-017: no June-November 2026 data).
2. `choose_book` + `implied_prob`: the first book in `odds.book_preference` with both odds valid (> 1) gives
   p = (1/oW) / (1/oW + 1/oL) for the odds-file winner (normalisation margin removal). Raw odds columns are kept.
3. `match_odds`: group odds rows into events; each event is matched to the Sackmann main-draw event in the date window
   [tourney_date - pre_days, tourney_date + window_days] (tennis-data `Date` is the MATCH date, Sackmann `tourney_date`
   the event START date) that matches the most player pairs (tie-break: tournament/location name similarity). Names
   ("Sinner J.") are resolved with `player_map.match_names` restricted to the event's players, then a "Last F." variant
   for multi-initial names ("Struff J.L."), then a unique-surname fallback within the event. Each row is matched by the
   unordered player pair (round as tie-break).
4. The Sackmann result is never changed: if the odds file names the other player as winner, `winner_disagrees` is set
   and `p_winner_book` is the probability for the SACKMANN winner.

Nothing is downloaded here. Files must already be under `paths.odds_dir` (see reports/data/odds_ingest.md).
"""

from __future__ import annotations

import difflib
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from unidecode import unidecode

from tennis.config import load_config, repo_path
from tennis.ingest.player_map import UnmatchedPlayerError, load_overrides, match_names, normalise_name
from tennis.ingest.sackmann import _md_table, load_players

SOURCE = "tennis-data"  # `source` value for config/player_overrides.csv rows

# Bookmaker column prefixes seen in tennis-data files (each has <prefix>W / <prefix>L). Kept raw when present.
BOOK_PREFIXES = ["PS", "B365", "Avg", "Max", "BFE", "EX", "LB", "SJ", "CB", "GB", "IW", "SB", "UB"]

# tennis-data column -> our column (text / numeric). Missing columns become null.
TEXT_COLUMNS = {"Location": "td_location", "Tournament": "td_tournament", "Series": "td_series", "Court": "td_court",
                "Surface": "td_surface", "Round": "td_round", "Winner": "td_winner", "Loser": "td_loser",
                "Comment": "comment"}
NUM_COLUMNS = {"ATP": "td_atp", "Best of": "td_best_of", "WRank": "WRank", "LRank": "LRank", "WPts": "WPts",
               "LPts": "LPts"}

# Absolute rounds only; "1st Round".."4th Round" depend on draw size and are not mapped (used only as tie-break).
ROUND_MAP = {"quarterfinals": "QF", "semifinals": "SF", "the final": "F", "round robin": "RR"}

COMMENTS = {"completed": "Completed", "retired": "Retired", "walkover": "Walkover", "disqualified": "Disqualified",
            "awarded": "Awarded", "sched": "Sched"}

STATUS_MATCHED = "matched"
MATCH_COLUMNS = ["match_id", "tourney_id", "tourney_name", "tourney_level", "tourney_date", "round", "winner_id",
                 "loser_id", "is_qualifying", "is_training_level", "is_walkover", "is_retirement"]


# --------------------------------------------------------------------------------------------------------------
# Reading and parsing
# --------------------------------------------------------------------------------------------------------------
def odds_file_paths(odds_dir: str | Path, first_year: int, last_year: int) -> dict[int, Path]:
    """{year: path} for `<year>.xlsx` / `<year>.xls` files present under `odds_dir` (missing years are absent)."""
    out: dict[int, Path] = {}
    for year in range(first_year, last_year + 1):
        for ext in (".xlsx", ".xls"):
            p = Path(odds_dir) / f"{year}{ext}"
            if p.is_file():
                out[year] = p
                break
    return out


def read_odds_file(path: str | Path) -> pd.DataFrame:
    """First sheet of one tennis-data Excel file, as read (openpyxl for .xlsx, xlrd for .xls)."""
    return pd.read_excel(path, sheet_name=0)


def _clean_text(s: pd.Series) -> pd.Series:
    """Strip whitespace; empty strings -> NA."""
    out = s.astype("string").str.strip()
    return out.mask(out == "")


def normalise_comment(c: Any) -> Any:
    """'completed ' -> 'Completed'; unknown values are kept (stripped) so they show up in the report."""
    if c is None or (not isinstance(c, str) and pd.isna(c)):
        return pd.NA
    s = str(c).strip()
    return COMMENTS.get(s.lower(), s)


def parse_odds(raw: pd.DataFrame, file_year: int) -> pd.DataFrame:
    """Type one raw tennis-data sheet. Text stripped, ranks/points/odds coerced to float (non-numeric such as 'NR' or
    'N/A' -> NaN), Date -> datetime64. Rows with no Winner or no Loser are dropped (empty spreadsheet rows; counted in
    the report). Odds values are not altered; validity (> 1) is checked only when computing probabilities."""
    raw = raw.rename(columns=lambda c: str(c).strip())
    out = pd.DataFrame(index=raw.index)
    out["file_year"] = file_year
    out["odds_date"] = pd.to_datetime(raw["Date"], errors="coerce") if "Date" in raw else pd.NaT
    for src, dst in TEXT_COLUMNS.items():
        out[dst] = _clean_text(raw[src]) if src in raw else pd.Series(pd.NA, index=raw.index, dtype="string")
    out["comment"] = out["comment"].map(normalise_comment).astype("string")
    for src, dst in NUM_COLUMNS.items():
        out[dst] = pd.to_numeric(raw[src], errors="coerce") if src in raw else np.nan
    for c in odds_columns(raw.columns):
        out[c] = pd.to_numeric(raw[c], errors="coerce").astype(float)
    keep = out["td_winner"].notna() & out["td_loser"].notna()
    return out[keep].reset_index(drop=True)


def odds_columns(columns: Iterable[Any]) -> list[str]:
    """Bookmaker odds columns present (<prefix>W and <prefix>L for every known prefix with both present)."""
    cols = {str(c).strip() for c in columns}
    return [f"{p}{s}" for p in BOOK_PREFIXES if f"{p}W" in cols and f"{p}L" in cols for s in ("W", "L")]


# --------------------------------------------------------------------------------------------------------------
# Probabilities
# --------------------------------------------------------------------------------------------------------------
def implied_prob(odds_w: Any, odds_l: Any) -> Any:
    """Margin-free probability that the 'W' side wins: (1/oW) / (1/oW + 1/oL). NaN unless both odds are finite and
    > 1 (decimal odds of 1 or less carry no usable price). Works on scalars or aligned Series/arrays."""
    ow = np.asarray(odds_w, dtype=float)
    ol = np.asarray(odds_l, dtype=float)
    valid = np.isfinite(ow) & np.isfinite(ol) & (ow > 1.0) & (ol > 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(valid, (1.0 / ow) / (1.0 / ow + 1.0 / ol), np.nan)
    if isinstance(odds_w, pd.Series):
        return pd.Series(p, index=odds_w.index)
    return float(p) if p.ndim == 0 else p


def choose_book(df: pd.DataFrame, preference: list[str]) -> pd.DataFrame:
    """Per row, the first book in `preference` (column prefixes, e.g. ['PS', 'Avg', 'B365']) with both odds valid.
    Returns columns `book` (NA if none), `book_odds_w`, `book_odds_l`, `p_odds_winner` (prob. for the ODDS-FILE
    winner, margin removed)."""
    out = pd.DataFrame({"book": pd.Series(pd.NA, index=df.index, dtype="string"),
                        "book_odds_w": np.nan, "book_odds_l": np.nan, "p_odds_winner": np.nan}, index=df.index)
    for book in preference:
        cw, cl = f"{book}W", f"{book}L"
        if cw not in df or cl not in df:
            continue
        p = implied_prob(df[cw], df[cl])
        take = out["book"].isna() & p.notna()
        out.loc[take, "book"] = book
        out.loc[take, "book_odds_w"] = df.loc[take, cw]
        out.loc[take, "book_odds_l"] = df.loc[take, cl]
        out.loc[take, "p_odds_winner"] = p[take]
    return out


# --------------------------------------------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------------------------------------------
def _is_initial(tok: str) -> bool:
    """'J.' 'J.L.' 'Zh.' 'J' -> True (an abbreviated first name); 'De' 'Minaur' -> False."""
    letters = re.sub(r"[^A-Za-z]", "", unidecode(tok))
    return bool(letters) and (("." in tok and len(letters) <= 3) or (len(letters) == 1 and tok.isalpha()))


def split_td_name(name: Any) -> tuple[str, str | None]:
    """'De Minaur A.' -> ('De Minaur', 'a'); 'Struff J.L.' -> ('Struff', 'j'); 'Zhang Zh.' -> ('Zhang', 'z');
    'Nadal' -> ('Nadal', None). Surname = tokens before the first initial token; initial = its first letter."""
    toks = str(name).strip().split()
    for i, t in enumerate(toks):
        if i > 0 and _is_initial(t):
            first = normalise_name(t)[:1] or None
            return " ".join(toks[:i]), first
    return " ".join(toks), None


def name_variants(name: Any) -> list[str]:
    """Raw name, then 'Surname F.' when the raw name has a longer or multi-letter initial ('Struff J.L.')."""
    raw = str(name).strip()
    surname, initial = split_td_name(raw)
    out = [raw]
    if initial:
        v = f"{surname} {initial.upper()}."
        if v != raw:
            out.append(v)
    return out


def _override_map(overrides: pd.DataFrame | None) -> dict[str, str]:
    """{raw and normalised name: sackmann id} for `source == tennis-data` rows of player_overrides.csv."""
    if overrides is None or overrides.empty:
        return {}
    o = overrides[overrides["source"].astype(str).str.strip() == SOURCE]
    out: dict[str, str] = {}
    for raw, pid in zip(o["source_id_or_name"], o["sackmann_player_id"]):
        out[str(raw).strip()] = str(pid).strip()
        out[normalise_name(raw)] = str(pid).strip()
    return out


def _surname_fallback(name: str, event_players: pd.DataFrame) -> str | None:
    """Unique event player whose normalised last name equals the odds surname (space-free), or whose last-name
    tokens are a prefix of it / it of them ('Ramos-Vinolas' vs 'Ramos', 'Mpetshi' vs 'Mpetshi Perricard').
    If the odds name has an initial, the player's first name must start with it."""
    surname, initial = split_td_name(name)
    s_toks = normalise_name(surname).split()
    if not s_toks:
        return None
    hits = set()
    for pid, first, last in zip(event_players["player_id"], event_players["name_first"], event_players["name_last"]):
        l_toks = normalise_name(last).split()
        if not l_toks:
            continue
        n = min(len(l_toks), len(s_toks))
        same = "".join(l_toks) == "".join(s_toks) or l_toks[:n] == s_toks[:n]
        if same and (initial is None or normalise_name(first).startswith(initial)):
            hits.add(str(pid))
    return next(iter(hits)) if len(hits) == 1 else None


def resolve_names(
    names: Iterable[str], event_players: pd.DataFrame, override_map: Mapping[str, str]
) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """Resolve odds names against ONE event's players. Returns ({name: (sackmann_id, method)}, unresolved names).

    Order: override -> `match_names` on the raw name (event players only) -> 'Surname F.' variant -> unique surname
    within the event. Overrides are applied here (not passed to `match_names`, which would reject override ids that
    are not in the event subset)."""
    resolved: dict[str, tuple[str, str]] = {}
    todo = []
    for name in dict.fromkeys(str(n).strip() for n in names):
        pid = override_map.get(name) or override_map.get(normalise_name(name))
        if pid:
            resolved[name] = (pid, "override")
        else:
            todo.append(name)
    if not todo:
        return resolved, []
    try:  # fast path: everything resolves in one call
        m = match_names(todo, event_players, source=SOURCE)
        for raw, pid, method in zip(m["source_id_or_name"], m["sackmann_player_id"], m["method"]):
            resolved[raw] = (pid, method)
        return resolved, []
    except UnmatchedPlayerError:
        pass
    unresolved = []
    for name in todo:
        for i, v in enumerate(name_variants(name)):
            try:
                m = match_names([v], event_players, source=SOURCE)
                resolved[name] = (m["sackmann_player_id"].item(), m["method"].item() + ("" if i == 0 else "_variant"))
                break
            except UnmatchedPlayerError:
                continue
        else:
            pid = _surname_fallback(name, event_players)
            if pid:
                resolved[name] = (pid, "event_surname")
            else:
                unresolved.append(name)
    return resolved, unresolved


# --------------------------------------------------------------------------------------------------------------
# Event and row matching
# --------------------------------------------------------------------------------------------------------------
def in_event_window(odds_date: pd.Timestamp, tourney_date: pd.Timestamp, pre_days: int, window_days: int) -> bool:
    """True if a match dated `odds_date` can belong to an event starting `tourney_date`:
    tourney_date - pre_days <= odds_date <= tourney_date + window_days. (Slams and some Masters start on the Sunday
    before Sackmann's Monday tourney_date, hence `pre_days`.)"""
    return tourney_date - pd.Timedelta(days=pre_days) <= odds_date <= tourney_date + pd.Timedelta(days=window_days)


def name_similarity(td_names: Iterable[Any], tourney_name: str) -> float:
    """Best similarity in [0, 1] between any tennis-data Location/Tournament string and a Sackmann tourney_name.
    1.0 if one normalised string's tokens are contained in the other's."""
    t = normalise_name(tourney_name)
    best = 0.0
    for s in td_names:
        n = normalise_name(s) if s is not None and not pd.isna(s) else ""
        if not n or not t:
            continue
        if set(n.split()) <= set(t.split()) or set(t.split()) <= set(n.split()):
            return 1.0
        best = max(best, difflib.SequenceMatcher(None, n, t).ratio())
    return best


def odds_event_key(df: pd.DataFrame) -> pd.Series:
    """Event key of each odds row: file year + ATP tournament number (falls back to location) + tournament name."""
    atp = df["td_atp"].astype("Float64").astype("string").fillna(df["td_location"].astype("string"))
    return df["file_year"].astype(str) + "|" + atp.fillna("") + "|" + df["td_tournament"].fillna("").astype(str)


def _main_draw(matches: pd.DataFrame, levels: list[str]) -> pd.DataFrame:
    """Sackmann main-draw rows of the levels tennis-data covers."""
    m = matches[(~matches["is_qualifying"]) & matches["tourney_level"].isin(levels)]
    return m[MATCH_COLUMNS].copy()


def _pair_index(ev_rows: pd.DataFrame) -> dict[frozenset, list[int]]:
    idx: dict[frozenset, list[int]] = {}
    for i, w, l in zip(ev_rows.index, ev_rows["winner_id"], ev_rows["loser_id"]):
        idx.setdefault(frozenset((str(w), str(l))), []).append(i)
    return idx


def _match_rows_in_event(
    od: pd.DataFrame, ev_rows: pd.DataFrame, resolved: Mapping[str, tuple[str, str]]
) -> pd.DataFrame:
    """For odds rows `od` of one event: winner/loser ids, Sackmann row index and status."""
    pairs = _pair_index(ev_rows)
    res = pd.DataFrame(index=od.index, columns=["td_winner_id", "td_loser_id", "sk_index", "status"], dtype=object)
    for i, wn, ln, rnd in zip(od.index, od["td_winner"], od["td_loser"], od["td_round"]):
        w, l = resolved.get(str(wn).strip()), resolved.get(str(ln).strip())
        res.at[i, "td_winner_id"] = w[0] if w else None
        res.at[i, "td_loser_id"] = l[0] if l else None
        if not (w and l):
            res.at[i, "status"] = "player_unresolved"
            continue
        hits = pairs.get(frozenset((w[0], l[0])), [])
        if len(hits) > 1:  # same pair twice in one event (e.g. Tour Finals RR and final): use the round
            want = ROUND_MAP.get(str(rnd).strip().lower()) if rnd is not None and not pd.isna(rnd) else None
            hits = [h for h in hits if want is not None and ev_rows.at[h, "round"] == want]
        if len(hits) == 1:
            res.at[i, "sk_index"] = hits[0]
            res.at[i, "status"] = STATUS_MATCHED
        else:
            res.at[i, "status"] = "pair_not_in_event" if not pairs.get(frozenset((w[0], l[0]))) else "ambiguous_pair"
    return res


def match_odds(
    odds: pd.DataFrame,
    matches: pd.DataFrame,
    players: pd.DataFrame,
    params: Mapping[str, Any],
    overrides: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Match parsed odds rows to Sackmann matches. Returns (every odds row with match columns and `status`, audit).

    `params`: `levels` (Sackmann levels to search), `pre_days`, `window_days`, `book_preference`.
    Status values: matched, event_not_found, player_unresolved, pair_not_in_event, ambiguous_pair, duplicate_match_id.
    """
    md = _main_draw(matches, list(params["levels"]))
    events = md.groupby("tourney_id").agg(tourney_date=("tourney_date", "first"), tourney_name=("tourney_name", "first"))
    players = players.assign(player_id=players["player_id"].astype(str))
    ev_players: dict[str, pd.DataFrame] = {}
    over = _override_map(overrides)

    def players_of(tid: str) -> pd.DataFrame:
        if tid not in ev_players:
            r = md[md["tourney_id"] == tid]
            ids = set(r["winner_id"].astype(str)) | set(r["loser_id"].astype(str))
            ev_players[tid] = players[players["player_id"].isin(ids)]
        return ev_players[tid]

    odds = odds.copy()
    odds["event_key"] = odds_event_key(odds)
    pieces, event_audit, unresolved_names = [], [], []
    pre, win = int(params["pre_days"]), int(params["window_days"])
    for key, od in odds.groupby("event_key", sort=False):
        d0, d1 = od["odds_date"].min(), od["odds_date"].max()
        if pd.isna(d0):
            cands = events.iloc[0:0]
        else:
            cands = events[(events["tourney_date"] - pd.Timedelta(days=pre) <= d0)
                           & (d1 <= events["tourney_date"] + pd.Timedelta(days=win))]
        td_names = [od["td_location"].iloc[0], od["td_tournament"].iloc[0]]
        best = None  # (n_matched, similarity, tid, rows_result, unresolved)
        for tid, ev in cands.iterrows():
            ep = players_of(tid)
            resolved, unres = resolve_names(pd.concat([od["td_winner"], od["td_loser"]]), ep, over)
            rr = _match_rows_in_event(od, md[md["tourney_id"] == tid], resolved)
            score = (int((rr["status"] == STATUS_MATCHED).sum()), name_similarity(td_names, ev["tourney_name"]))
            if best is None or score > best[:2]:
                best = (*score, tid, rr, unres)
        if best is None or best[0] == 0:
            rr = pd.DataFrame({"td_winner_id": None, "td_loser_id": None, "sk_index": None,
                               "status": "event_not_found"}, index=od.index)
            event_audit.append({"event_key": key, "td_tournament": td_names[1], "td_location": td_names[0],
                                "first_date": d0, "rows": len(od), "tourney_id": None, "tourney_name": None,
                                "similarity": np.nan, "matched": 0, "candidates": len(cands)})
        else:
            n, sim, tid, rr, unres = best
            unresolved_names += [{"name": u, "event_key": key, "tourney_id": tid,
                                  "rows": int(((od["td_winner"] == u) | (od["td_loser"] == u)).sum())} for u in unres]
            rr["tourney_id_matched"] = tid
            event_audit.append({"event_key": key, "td_tournament": td_names[1], "td_location": td_names[0],
                                "first_date": d0, "rows": len(od), "tourney_id": tid,
                                "tourney_name": events.at[tid, "tourney_name"], "similarity": sim, "matched": n,
                                "candidates": len(cands)})
        pieces.append(rr)
    res = pd.concat(pieces) if pieces else pd.DataFrame(columns=["td_winner_id", "td_loser_id", "sk_index", "status"])
    out = odds.join(res.drop(columns=["tourney_id_matched"], errors="ignore"))

    # Attach Sackmann columns for matched rows.
    sk = md.loc[out["sk_index"].dropna().astype(int)].reset_index(drop=True)
    matched_idx = out.index[out["sk_index"].notna()]
    for c in MATCH_COLUMNS:
        out[c] = pd.Series(sk[c].to_numpy(), index=matched_idx) if len(sk) else pd.NA
    out = out.drop(columns=["sk_index"])

    # One odds row per match_id: later duplicates are flagged and excluded from the output.
    dup = out["match_id"].notna() & out.duplicated("match_id", keep="first")
    out.loc[dup, "status"] = "duplicate_match_id"

    # Winner disagreement (flag only; the Sackmann result is authoritative and never changed).
    ok = out["status"] == STATUS_MATCHED
    out["winner_disagrees"] = ok & (out["td_winner_id"].astype(object) != out["winner_id"].astype(object))
    books = choose_book(out, list(params["book_preference"]))
    out = out.join(books)
    out["p_winner_book"] = np.where(out["winner_disagrees"], 1.0 - out["p_odds_winner"], out["p_odds_winner"])
    out.loc[~ok, "p_winner_book"] = np.nan

    audit = {"events": pd.DataFrame(event_audit), "unresolved": pd.DataFrame(
        unresolved_names, columns=["name", "event_key", "tourney_id", "rows"])}
    return out, audit


# --------------------------------------------------------------------------------------------------------------
# Output and report
# --------------------------------------------------------------------------------------------------------------
OUTPUT_FRONT = ["match_id", "odds_date", "file_year", "tourney_id", "tourney_name", "tourney_level", "tourney_date",
                "round", "winner_id", "loser_id", "book", "p_winner_book", "winner_disagrees", "comment",
                "td_tournament", "td_location", "td_series", "td_round", "td_winner", "td_loser", "td_winner_id",
                "td_loser_id", "WRank", "LRank", "book_odds_w", "book_odds_l", "p_odds_winner"]


def odds_output(matched: pd.DataFrame) -> pd.DataFrame:
    """Rows written to odds.parquet: matched rows only (one per match_id). Raw odds columns keep the ODDS-FILE
    orientation (<book>W = odds-file winner); `p_winner_book` is for the Sackmann winner."""
    m = matched[matched["status"] == STATUS_MATCHED]
    raw = [c for c in odds_columns(m.columns)]
    cols = [c for c in OUTPUT_FRONT if c in m.columns] + raw
    out = m[cols].reset_index(drop=True)
    for c in ["match_id", "tourney_id", "winner_id", "loser_id", "td_winner_id", "td_loser_id"]:
        out[c] = out[c].astype("string")
    out["winner_disagrees"] = out["winner_disagrees"].astype(bool)
    return out


def unmatched_player_policy(
    matched: pd.DataFrame, audit: Mapping[str, Any], max_unmatched: int
) -> tuple[bool, list[str]]:
    """(ok, offenders). Unresolved players are an error, except when there are at most `max_unmatched` of them and
    every affected odds row is a non-training row in the odds file (comment != 'Completed'): then they are reported
    and excluded with status `player_unresolved`."""
    un = audit["unresolved"]
    if un.empty:
        return True, []
    names = sorted(set(un["name"]))
    bad = matched[(matched["status"] == "player_unresolved")
                  & (matched["td_winner"].isin(names) | matched["td_loser"].isin(names))]
    tiny_nontraining = len(names) <= max_unmatched and not (bad["comment"] == "Completed").any()
    return tiny_nontraining, names


def render_report(files: pd.DataFrame, parsed: pd.DataFrame, matched: pd.DataFrame, audit: Mapping[str, Any],
                  dropped: pd.DataFrame, params: Mapping[str, Any]) -> str:
    """Markdown ingestion report."""
    L = ["# Ingestion report: betting odds (tennis-data.co.uk) -> `data/interim/odds.parquet`", "",
         f"Generated by `.venv/Scripts/python -I -m tennis.ingest.odds` on {pd.Timestamp.now():%Y-%m-%d %H:%M}. "
         f"Book preference {list(params['book_preference'])}; event window [tourney_date - {params['pre_days']}d, "
         f"tourney_date + {params['window_days']}d]; max odds date {params['max_date']} (D-017).", "",
         "## 1. Files", "", _md_table(files), "", "## 2. Dropped rows", "", _md_table(dropped), ""]
    m = matched.assign(year=matched["file_year"], series=matched["td_series"].fillna("<null>"),
                       ok=matched["status"] == STATUS_MATCHED)
    rate = m.groupby(["year", "series"]).agg(rows=("ok", "size"), matched=("ok", "sum"))
    rate["rate"] = (rate["matched"] / rate["rows"]).map(lambda x: f"{x:.1%}")
    per_year = m.groupby("year").agg(rows=("ok", "size"), matched=("ok", "sum"))
    per_year["rate"] = (per_year["matched"] / per_year["rows"]).map(lambda x: f"{x:.1%}")
    nwo = m[m["comment"] != "Walkover"].groupby("year")["ok"].mean().map(lambda x: f"{x:.1%}")
    per_year["rate excl. walkovers"] = nwo
    lvl = (m[m["ok"]].groupby(["year", "tourney_level"]).size().unstack(fill_value=0) if m["ok"].any()
           else pd.DataFrame({"note": ["no matched rows"]}))
    overall = m["ok"].mean() if len(m) else float("nan")
    L += ["## 3. Match rate", "",
          f"Overall: {int(m['ok'].sum()):,} / {len(m):,} = {overall:.2%} (target >= {params['min_match_rate']:.0%}: "
          f"**{'PASS' if overall >= params['min_match_rate'] else 'FAIL'}**). tennis-data covers main-draw tour-level "
          "matches only, so every row counts.", "",
          "### Per year", "", _md_table(per_year, index=True), "",
          "### Per year x tennis-data series", "", _md_table(rate, index=True), "",
          "### Matched rows per year x Sackmann level", "", _md_table(lvl, index=True), "",
          "### Status counts", "", _md_table(m["status"].value_counts().rename_axis("status").to_frame("rows"),
                                              index=True), ""]
    un = audit["unresolved"]
    L += ["## 4. Unresolved players (error unless tiny and non-training)", ""]
    L += [_md_table(un) if not un.empty else "None.", ""]
    ev = audit["events"]
    nf = ev[ev["tourney_id"].isna()] if not ev.empty else ev
    L += ["## 5. Events", "", f"{len(ev):,} odds events; {len(nf):,} not matched to a Sackmann event.", ""]
    if len(nf):
        L += [_md_table(nf), ""]
    low = ev[(ev["similarity"] < 0.6) & ev["tourney_id"].notna()] if not ev.empty else ev
    L += ["Matched events with low name similarity (< 0.6; matched on players, check by eye):", "",
          _md_table(low) if len(low) else "None.", ""]
    bad = m[~m["ok"] & (m["status"] != "event_not_found")]
    L += ["## 6. Unmatched rows (event found)", "",
          _md_table(bad[["odds_date", "td_tournament", "td_round", "td_winner", "td_loser", "comment", "status"]])
          if len(bad) else "None.", ""]
    dis = m[m["winner_disagrees"]]
    L += ["## 7. Winner disagreements (flagged, Sackmann result kept)", "",
          _md_table(dis[["match_id", "odds_date", "td_tournament", "td_winner", "td_loser", "comment", "winner_id",
                         "loser_id"]]) if len(dis) else "None.", ""]
    ok = m[m["ok"]]
    miss = ok.groupby("year").agg(rows=("ok", "size"), no_book=("book", lambda s: int(s.isna().sum())))
    for c in [f"{b}W" for b in params["book_preference"]]:
        if c in ok:
            miss[f"{c} missing"] = ok[c].isna().groupby(ok["year"]).sum()
    L += ["## 8. Missing odds (matched rows)", "", _md_table(miss, index=True), "",
          "Book used:", "", _md_table(ok["book"].fillna("<none>").value_counts().rename_axis("book").to_frame("rows"),
                                      index=True), "",
          "## 9. Comment vocabulary", "",
          _md_table(m["comment"].fillna("<null>").value_counts().rename_axis("comment").to_frame("rows"), index=True),
          "", "## 10. Nulls per column (parsed rows)", "",
          _md_table(parsed.isna().sum().rename_axis("column").to_frame("nulls").query("nulls > 0"), index=True), ""]
    return "\n".join(L)


def odds_params(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The `odds:` config block (must exist; all tunables live in config.yaml)."""
    if "odds" not in cfg:
        raise KeyError("config.yaml has no `odds:` block (book_preference, pre_days, window_days, levels, max_date, "
                       "first_year, last_year, min_match_rate, max_unmatched_players)")
    return dict(cfg["odds"])


def main() -> None:
    """Parse cached files under paths.odds_dir, match to matches.parquet, write odds.parquet + the report.
    Never downloads anything."""
    cfg = load_config()
    params = odds_params(cfg)
    files = odds_file_paths(repo_path(cfg["paths"]["odds_dir"]), int(params["first_year"]), int(params["last_year"]))
    if not files:
        raise FileNotFoundError(f"no <year>.xls[x] odds files under {cfg['paths']['odds_dir']}; see "
                                "reports/data/odds_ingest.md (download blocked by the site's policy)")
    max_date = pd.Timestamp(params["max_date"])
    frames, file_rows, drops = [], [], []
    for year, path in files.items():
        p = parse_odds(read_odds_file(path), year)
        n_raw = len(p)
        late = p["odds_date"] > max_date
        nat = p["odds_date"].isna()
        drops += [{"year": year, "reason": f"Date after {max_date:%Y-%m-%d} (D-017)", "rows": int(late.sum())},
                  {"year": year, "reason": "unparseable Date", "rows": int(nat.sum())}]
        p = p[~late & ~nat]
        frames.append(p)
        file_rows.append({"file": path.name, "bytes": path.stat().st_size, "rows parsed": n_raw, "rows kept": len(p),
                          "odds columns": " ".join(odds_columns(p.columns))})
    parsed = pd.concat(frames, ignore_index=True)
    matches = pd.read_parquet(repo_path(cfg["paths"]["matches"]))
    players = load_players(cfg)
    overrides = load_overrides(repo_path(cfg["paths"]["player_overrides"]))
    matched, audit = match_odds(parsed, matches, players, params, overrides)
    dropped = pd.DataFrame(drops).query("rows > 0") if drops else pd.DataFrame(columns=["year", "reason", "rows"])
    report = render_report(pd.DataFrame(file_rows), parsed, matched, audit, dropped, params)
    rpath = repo_path(cfg["paths"]["reports"]) / "data" / "odds_ingest.md"
    rpath.write_text(report, encoding="utf-8")
    ok, names = unmatched_player_policy(matched, audit, int(params["max_unmatched_players"]))
    if not ok:
        raise UnmatchedPlayerError(f"{len(names)} unresolved tennis-data player name(s); add rows to "
                                   "config/player_overrides.csv (source=tennis-data)", names)
    out = odds_output(matched)
    out.to_parquet(repo_path(cfg["paths"]["odds"]), index=False)
    print(f"wrote {len(out):,} rows to {cfg['paths']['odds']}; report {rpath}", file=sys.stderr)


if __name__ == "__main__":
    main()
