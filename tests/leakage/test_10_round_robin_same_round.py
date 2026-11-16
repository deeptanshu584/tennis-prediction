"""Test 10 (Phase 2 audit): round-robin rounds must not leak same-round results.

Inside one round-robin (RR) round of a tour event the authoritative order falls back to ``match_num``, but
``match_num`` is NOT chronological there in Sackmann's data. Example, Tour Finals 2019 group (match_num order):
286 Berrettini d. Thiem (played Thu 14 Nov), 287 Federer d. Berrettini (Tue), 288 Thiem d. Federer (Sun 10 Nov),
289 Djokovic d. Berrettini (Sun), 290 Thiem d. Djokovic (Tue), 291 Federer d. Djokovic (Thu). Processing 286 first
puts Thursday's result into the state that Sunday's row 288 is built from. In 84 of the 88 four-player RR groups of
the Tour Finals / Masters Cup / NextGen Finals, consecutive match_num pairs share a player, so the order can't be the
real one (each match day of a 4-player group has two disjoint matches).

Because the real order inside an RR round is unknown, a row of an RR round may only use state from BEFORE that
round: its pre-match features must be identical whatever the results of the other matches in the same RR round of
the same event. Davis Cup (level D) is excluded: each tie is its own tourney_id and its rubber numbers (1, 2 on day
one; 4, 5 on the last day) are chronological.

The check: swap winner/loser (with all winner_/loser_ attributes and stats) of every RR match of one event, rebuild,
and compare each player's pre-match features on that event's RR rows (keyed by (match_id, player_id), so the swap of
the row's own result doesn't matter: a pre-match feature can't depend on it).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.leakage.helpers import build, get_builder, swap_winner_loser

REAL_WINDOW = ("2019-01-01", "2020-02-29")  # Tour Finals 2019, NextGen 2019, Laver Cup 2019, ATP Cup 2020


def _rr_events(matches: pd.DataFrame) -> list[str]:
    """Tour-level (non-Davis-Cup) tourney_ids with an RR round in which some player plays more than once."""
    rr = matches[(matches["round"] == "RR") & (matches["tourney_level"] != "D") & matches["is_training_level"]
                 & ~matches["is_walkover"].astype(bool)]
    out = []
    for tid, g in rr.groupby("tourney_id"):
        counts = pd.concat([g["winner_id"], g["loser_id"]]).value_counts()
        if (counts > 1).any():
            out.append(str(tid))
    return sorted(out)


def _player_view(features: pd.DataFrame, match_ids: set[str]) -> pd.DataFrame:
    """Per-player pre-match features of the given rows, indexed by (match_id, player_id)."""
    pf = [c[2:] for c in features.columns
          if c.startswith("a_") and f"b_{c[2:]}" in features.columns and c not in ("a_id", "a_name")]
    sub = features[features["match_id"].isin(match_ids)]
    parts = []
    for side in ("a", "b"):
        part = sub[["match_id", f"{side}_id"] + [f"{side}_{f}" for f in pf]].copy()
        part.columns = ["match_id", "player_id"] + pf
        parts.append(part)
    return pd.concat(parts).set_index(["match_id", "player_id"]).sort_index()


def _check_event(matches: pd.DataFrame, cfg: dict[str, Any], base: pd.DataFrame, tid: str) -> list[str]:
    rr_mask = (matches["tourney_id"] == tid) & (matches["round"] == "RR")
    rr_ids = set(matches.loc[rr_mask, "match_id"].astype(str))
    swapped = build(swap_winner_loser(matches, rr_mask), cfg)
    a, b = _player_view(base, rr_ids), _player_view(swapped, rr_ids)
    assert a.index.equals(b.index), f"{tid}: emitted RR rows differ after swapping RR results"
    same = (a == b) | (a.isna() & b.isna())
    bad_rows = ~same.all(axis=1)
    if not bad_rows.any():
        return []
    name = str(matches.loc[rr_mask, "tourney_name"].iloc[0])
    ex = a.index[np.flatnonzero(bad_rows.to_numpy())[0]]
    cols = list(same.columns[~same.loc[ex]])
    return [f"{name} ({tid}): {int(bad_rows.sum())}/{len(a)} player-rows change, e.g. {ex} in {cols[:6]}"]


def _run(matches: pd.DataFrame, cfg: dict[str, Any]) -> None:
    events = _rr_events(matches)
    if not events:
        pytest.skip("no tour-level round-robin event with repeat players in this universe")
    base = build(matches, cfg)
    problems: list[str] = []
    for tid in events:
        problems += _check_event(matches, cfg, base, tid)
    assert not problems, (
        "RR rows read results of other matches in the same round-robin round, whose real order is unknown "
        "(match_num is not chronological inside RR). Snapshot every RR row of an event from the state before the "
        "RR round and apply the RR results only after the whole round:\n  " + "\n  ".join(problems))


def test_round_robin_same_round_synthetic(synthetic_matches: pd.DataFrame, cfg: dict) -> None:
    get_builder()
    _run(synthetic_matches, cfg)


@pytest.mark.realdata
@pytest.mark.slow
def test_round_robin_same_round_real(real_matches: pd.DataFrame, cfg: dict) -> None:
    get_builder()
    lo, hi = (pd.Timestamp(x) for x in REAL_WINDOW)
    window = real_matches[(real_matches["tourney_date"] >= lo) & (real_matches["tourney_date"] <= hi)]
    _run(window.reset_index(drop=True), cfg)
