"""Player mapping to Sackmann `player_id`.

- Sackmann: every id maps to itself (method `identity`); ids missing from atp_players.csv raise.
- Other sources (TML, odds, scraped): `match_names` maps raw names to Sackmann ids using a normalised name index,
  honouring `config/player_overrides.csv`. Unmatched or ambiguous names raise `UnmatchedPlayerError` listing all
  offenders. Unmatched players are an error, never a warning (PLAN section 4.4).
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from unidecode import unidecode

PLAYER_MAP_COLUMNS = ["source", "source_id_or_name", "sackmann_player_id", "method"]
OVERRIDE_COLUMNS = ["source", "source_id_or_name", "sackmann_player_id", "note"]

_APOSTROPHES = re.compile(r"['’‘`´]")
_SEPARATORS = re.compile(r"[-‐‑‒–—._,]")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]")
_SPACES = re.compile(r"\s+")


class UnmatchedPlayerError(ValueError):
    """Raised when players cannot be mapped to a unique Sackmann player_id. `offenders` lists them."""

    def __init__(self, message: str, offenders: list[Any]):
        super().__init__(f"{message}: {offenders}")
        self.offenders = offenders


# --------------------------------------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------------------------------------
def normalise_name(name: Any) -> str:
    """Lower-case ASCII name: accents removed (unidecode), apostrophes dropped ("O'Connell" -> "oconnell"),
    hyphens/dots/underscores/commas -> space, other punctuation removed, whitespace collapsed."""
    if name is None or (not isinstance(name, str) and pd.isna(name)):
        return ""
    s = unidecode(str(name)).lower()
    s = _APOSTROPHES.sub("", s)
    s = _SEPARATORS.sub(" ", s)
    s = _NON_ALNUM.sub("", s)
    return _SPACES.sub(" ", s).strip()


def token_key(name: Any) -> str:
    """Order-independent key: sorted normalised tokens ("Nadal Rafael" == "Rafael Nadal")."""
    return " ".join(sorted(normalise_name(name).split()))


def squashed_keys(first: Any, last: Any) -> set[str]:
    """Space-free keys for both name orders (first "Alex", last "de Minaur" -> "alexdeminaur", "deminauralex")."""
    f = normalise_name(first).replace(" ", "")
    l = normalise_name(last).replace(" ", "")
    return {f + l, l + f} - {""}


# --------------------------------------------------------------------------------------------------------------
# Index
# --------------------------------------------------------------------------------------------------------------
class PlayerIndex:
    """Lookup tables over atp_players.csv (or a candidate subset of it)."""

    def __init__(self, players: pd.DataFrame):
        self.ids: set[str] = set(players["player_id"].astype(str))
        self.by_token: dict[str, set[str]] = defaultdict(set)
        self.by_squash: dict[str, set[str]] = defaultdict(set)
        self.by_last_initial: dict[tuple[str, str], set[str]] = defaultdict(set)
        for pid, first, last in zip(players["player_id"].astype(str), players["name_first"], players["name_last"]):
            full = f"{'' if pd.isna(first) else first} {'' if pd.isna(last) else last}"
            key = token_key(full)
            if not key:
                continue
            self.by_token[key].add(pid)
            for k in squashed_keys(first, last):
                self.by_squash[k].add(pid)
            nf, nl = normalise_name(first), normalise_name(last)
            if nf and nl:
                self.by_last_initial[(nl, nf[0])].add(pid)

    def lookup(self, name: str) -> tuple[set[str], str]:
        """Candidate ids for a raw name and the method that found them (empty set if none)."""
        key = token_key(name)
        if not key:
            return set(), "none"
        if key in self.by_token:
            return set(self.by_token[key]), "normalised"
        squash = normalise_name(name).replace(" ", "")
        if squash in self.by_squash:
            return set(self.by_squash[squash]), "squashed"
        # "Last F." / "F. Last" (odds files): exactly one single-letter token is the first initial.
        toks = normalise_name(name).split()
        initials = [t for t in toks if len(t) == 1]
        if len(initials) == 1 and len(toks) >= 2:
            last = " ".join(t for t in toks if len(t) > 1)
            hits = self.by_last_initial.get((last, initials[0]), set())
            if hits:
                return set(hits), "last_initial"
        return set(), "none"


# --------------------------------------------------------------------------------------------------------------
# Overrides
# --------------------------------------------------------------------------------------------------------------
def load_overrides(path: str | Path) -> pd.DataFrame:
    """Read config/player_overrides.csv (read-only). Columns: source, source_id_or_name, sackmann_player_id, note."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    missing = [c for c in OVERRIDE_COLUMNS[:3] if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df


def _override_lookup(overrides: pd.DataFrame | None, source: str) -> dict[str, str]:
    """{raw name/id and its normalised form: sackmann id} for one source."""
    if overrides is None or overrides.empty:
        return {}
    o = overrides[overrides["source"].str.strip() == source]
    out: dict[str, str] = {}
    for raw, pid in zip(o["source_id_or_name"], o["sackmann_player_id"]):
        raw, pid = str(raw).strip(), str(pid).strip()
        out[raw] = pid
        out[normalise_name(raw)] = pid
    return out


# --------------------------------------------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------------------------------------------
def match_names(
    names: Iterable[str],
    players: pd.DataFrame,
    source: str,
    overrides: pd.DataFrame | None = None,
    candidates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Map raw player names from `source` to Sackmann ids. Returns a PLAYER_MAP_COLUMNS frame.

    Order: override (exact or normalised raw string) -> normalised token match (either name order) ->
    space-free match -> "Last F." initial match. `candidates` (a subset of `players`, e.g. players active in the
    period) is searched first; if it gives no unique hit, the full table is tried. Unmatched or ambiguous names
    raise UnmatchedPlayerError with every offender; an override pointing at an unknown id also raises.
    """
    all_ids = set(players["player_id"].astype(str))
    over = _override_lookup(overrides, source)
    bad_over = sorted({pid for pid in over.values() if pid not in all_ids})
    if bad_over:
        raise UnmatchedPlayerError("player_overrides.csv points at unknown Sackmann ids", bad_over)

    indexes = []
    if candidates is not None:
        indexes.append(PlayerIndex(candidates))
    indexes.append(PlayerIndex(players))

    rows, offenders = [], []
    for name in pd.unique(pd.Series(list(names), dtype=object)):
        raw = "" if pd.isna(name) else str(name).strip()
        pid = over.get(raw) or over.get(normalise_name(raw))
        if pid:
            rows.append((source, raw, pid, "override"))
            continue
        found, method, ambiguous = None, "none", None
        for idx in indexes:
            hits, method = idx.lookup(raw)
            if len(hits) == 1:
                found = next(iter(hits))
                break
            if len(hits) > 1 and ambiguous is None:
                ambiguous = sorted(hits)
        if found is None:
            offenders.append(f"{raw!r} (ambiguous: {ambiguous})" if ambiguous else repr(raw))
        else:
            rows.append((source, raw, found, method))
    if offenders:
        raise UnmatchedPlayerError(f"{len(offenders)} unmatched {source} player name(s)", offenders)
    return pd.DataFrame(rows, columns=PLAYER_MAP_COLUMNS)


def sackmann_player_map(matches: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Identity map for every winner/loser id in a Sackmann-sourced frame. Ids not in atp_players.csv raise."""
    ids = pd.unique(pd.concat([matches["winner_id"], matches["loser_id"]]).astype(str))
    known = set(players["player_id"].astype(str))
    missing = sorted(i for i in ids if i not in known)
    if missing:
        raise UnmatchedPlayerError("Sackmann player ids not in atp_players.csv", missing)
    ids = sorted(ids, key=lambda x: (len(x), x))
    return pd.DataFrame(
        {"source": "sackmann", "source_id_or_name": ids, "sackmann_player_id": ids, "method": "identity"},
        columns=PLAYER_MAP_COLUMNS,
    )
