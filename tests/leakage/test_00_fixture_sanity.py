"""Test 0: the synthetic fixture (and, when present, the real match table) is a valid unified-schema input.

Catches: a fixture too easy to expose leaks (missing walkovers, retirements, qualifying, Challengers, missing
stats, same-date tournaments, out-of-order rows), and a match table whose order_key / match_id / round_rank
disagree with CONTRACT.md section 1 (which would silently change the chronological order of the pass).
Authoritative order: (tourney_date, tourney_id, round_rank, match_num, match_id); order_key (int64) is the
0-based position in that sort.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.synthetic import STAT_FIELDS, UNIFIED_COLUMNS, make_synthetic_matches

SORT4 = ["tourney_date", "tourney_id", "round_rank", "match_num"]
SORT5 = SORT4 + ["match_id"]


def _check_schema_and_keys(df: pd.DataFrame, cfg: dict) -> None:
    missing = [c for c in UNIFIED_COLUMNS if c not in df.columns]
    assert not missing, f"missing unified columns: {missing}"
    assert pd.api.types.is_datetime64_any_dtype(df["tourney_date"])
    for c in ("is_walkover", "is_retirement", "is_qualifying", "is_training_level", "surface_imputed",
              "stats_invalid"):
        assert pd.api.types.is_bool_dtype(df[c]), c
    stat_cols = [f"{p}_{s}" for p in ("w", "l") for s in STAT_FIELDS]
    assert df.loc[df["stats_invalid"], stat_cols].isna().all().all(), "stats_invalid rows must have NaN stats"
    assert df["match_id"].is_unique, "duplicate match_id"
    round_order = cfg["data"]["round_order"]
    assert set(df["round"]) <= set(round_order), set(df["round"]) - set(round_order)
    rr = df["round"].map({r: i for i, r in enumerate(round_order)})
    assert (rr == df["round_rank"]).all(), "round_rank != index of round in config round_order"
    assert set(df["surface"]) <= {"Hard", "Clay", "Grass"}
    quali = set(cfg["data"]["qualifying_rounds"])
    assert (df["is_qualifying"] == df["round"].isin(quali)).all()
    expect_tl = df["tourney_level"].isin(cfg["data"]["training_levels"]) & ~df["is_qualifying"]
    assert (df["is_training_level"] == expect_tl).all()
    # order_key (int64) is exactly the 0-based position in the authoritative 5-column sort
    assert pd.api.types.is_integer_dtype(df["order_key"]), f"order_key dtype {df['order_key'].dtype}, expected int64"
    in_order = df.sort_values(SORT5, kind="mergesort")["order_key"].to_numpy()
    assert (in_order == np.arange(len(df))).all(), (
        "order_key != position in the sort by (tourney_date, tourney_id, round_rank, match_num, match_id)")


def test_schema_and_keys(synthetic_matches: pd.DataFrame, cfg: dict) -> None:
    df = synthetic_matches
    _check_schema_and_keys(df, cfg)
    assert df["source"].eq("sackmann").all()
    for c in ("draw_size", "best_of", "match_num", "round_rank"):
        assert pd.api.types.is_integer_dtype(df[c]), c
    # match_id follows the CONTRACT formula
    for r in df.sample(50, random_state=0).itertuples():
        lo, hi = sorted([r.winner_id, r.loser_id])
        mid = hashlib.sha1(f"{r.tourney_id}|{r.match_num}|{lo}|{hi}".encode()).hexdigest()[:16]
        assert r.match_id == mid
    # no exact ties in the match order
    assert not df.duplicated(SORT4).any()


def test_rows_are_not_in_chronological_order(synthetic_matches: pd.DataFrame) -> None:
    """The builder must sort; the fixture is shuffled so a builder that trusts row order fails."""
    assert not synthetic_matches["order_key"].is_monotonic_increasing


def test_ordering_traps_present(synthetic_matches: pd.DataFrame) -> None:
    df = synthetic_matches
    # qualifying match_num > main-draw match_num in the same event (round_rank must beat match_num)
    q = df[df["is_qualifying"]].groupby("tourney_id")["match_num"].min()
    m = df[~df["is_qualifying"]].groupby("tourney_id")["match_num"].max()
    both = q.index.intersection(m.index)
    assert len(both) > 0 and (q[both] > m[both]).all()
    # an event numbered backwards (final has the smallest match_num)
    finals = df[df["round"] == "F"].set_index("tourney_id")["match_num"]
    firsts = df[~df["is_qualifying"]].groupby("tourney_id")["match_num"].min()
    assert (finals == firsts.reindex(finals.index)).any()
    # several tournaments start on the same date
    per_date = df.groupby("tourney_date")["tourney_id"].nunique()
    assert (per_date >= 3).any()
    # some player plays two different tournaments with the same start date
    long = pd.concat([
        df[["tourney_date", "tourney_id", "winner_id"]].rename(columns={"winner_id": "pid"}),
        df[["tourney_date", "tourney_id", "loser_id"]].rename(columns={"loser_id": "pid"}),
    ])
    n_events = long.drop_duplicates().groupby(["pid", "tourney_date"])["tourney_id"].nunique()
    assert (n_events >= 2).any()


def test_edge_cases_present(synthetic_matches: pd.DataFrame) -> None:
    df = synthetic_matches
    assert df["tourney_date"].dt.year.nunique() == 3
    players = set(df["winner_id"]) | set(df["loser_id"])
    assert 30 <= len(players) <= 50
    assert {"A", "G", "M", "F", "D", "O", "C"} <= set(df["tourney_level"])
    assert {"Q1", "Q2", "Q3", "RR", "BR", "R32", "R16", "QF", "SF", "F"} <= set(df["round"])
    assert set(df["surface"]) == {"Hard", "Clay", "Grass"}
    assert set(df["best_of"]) == {3, 5}
    assert (df["score"] == "W/O").sum() >= 3 and df["is_walkover"].sum() == (df["score"] == "W/O").sum()
    assert df["score"].str.endswith("RET").sum() >= 3
    assert df["score"].str.endswith("DEF").sum() >= 1
    assert (df["is_qualifying"] & df["is_walkover"]).any()
    stat_cols = [f"{p}_{s}" for p in ("w", "l") for s in STAT_FIELDS]
    n_missing = df[stat_cols].isna().sum(axis=1)
    assert (n_missing == len(stat_cols))[~df["is_walkover"]].sum() >= 20, "matches with all stats missing"
    assert ((n_missing > 0) & (n_missing < len(stat_cols))).any(), "a match with only some stats missing"
    assert ((df["l_svpt"] == 0) & df["is_retirement"]).any(), "retirement before the loser served"
    assert df["winner_rank"].isna().any() or df["loser_rank"].isna().any()
    assert df["winner_ht"].isna().any() or df["loser_ht"].isna().any()
    hands = set(df["winner_hand"].dropna()) | set(df["loser_hand"].dropna())
    assert {"R", "L", "U"} <= hands
    assert df["winner_hand"].isna().any() or df["loser_hand"].isna().any()
    assert df["tourney_name"].str.contains("Carpet").any()
    # a non-exempt round-robin round where players meet several opponents (block snapshotting matters there)
    rr = df[(df["round"] == "RR") & (df["tourney_level"] == "F")]
    assert not rr.empty
    per_player = pd.concat([rr[["tourney_id", "winner_id"]].set_axis(["t", "p"], axis=1),
                            rr[["tourney_id", "loser_id"]].set_axis(["t", "p"], axis=1)]).value_counts()
    assert (per_player >= 3).all()
    assert df["surface_imputed"].any() and not df["surface_imputed"].all()
    stat_all_nan = df[stat_cols].isna().all(axis=1)
    assert df["stats_invalid"].any() and stat_all_nan[df["stats_invalid"]].all()
    assert (df["winner_entry"] == "Q").any() or (df["loser_entry"] == "Q").any()
    # debutants in the 2nd season and a > 100 day layoff
    first_year = pd.concat([
        df[["winner_id", "tourney_date"]].rename(columns={"winner_id": "pid"}),
        df[["loser_id", "tourney_date"]].rename(columns={"loser_id": "pid"}),
    ]).groupby("pid")["tourney_date"].min().dt.year
    assert (first_year > first_year.min()).any()


def test_serve_counts_are_consistent(synthetic_matches: pd.DataFrame) -> None:
    df = synthetic_matches
    for p in ("w", "l"):
        s = {k: df[f"{p}_{k}"] for k in STAT_FIELDS}
        full = pd.concat(s.values(), axis=1).notna().all(axis=1)
        d = {k: v[full] for k, v in s.items()}
        assert (d["svpt"] >= d["1stIn"]).all()
        assert (d["1stIn"] >= d["1stWon"]).all()
        assert (d["1stWon"] >= d["ace"]).all()
        assert (d["2ndWon"] + d["df"] <= d["svpt"] - d["1stIn"]).all()
        assert (d["bpSaved"] <= d["bpFaced"]).all()
        assert (pd.concat(d.values(), axis=1) >= 0).all().all()
    played = ~df["is_walkover"] & df["w_svpt"].notna()
    assert (df.loc[played & ~df["is_retirement"], "w_svpt"] > 0).all()


def test_draws_progress(synthetic_matches: pd.DataFrame, cfg: dict) -> None:
    """In knock-out draws every player in round r won their match in the previous round."""
    df = synthetic_matches
    ko = df[~df["round"].isin(["RR", "BR"])]
    checked = 0
    for _, ev in ko.groupby("tourney_id"):
        for stage in (ev[ev["is_qualifying"]], ev[~ev["is_qualifying"]]):
            rounds = sorted(stage["round_rank"].unique())
            for prev, cur in zip(rounds, rounds[1:]):
                winners = set(stage.loc[stage["round_rank"] == prev, "winner_id"])
                cur_rows = stage[stage["round_rank"] == cur]
                players = set(cur_rows["winner_id"]) | set(cur_rows["loser_id"])
                assert players <= winners, (ev["tourney_id"].iat[0], cur)
                checked += 1
    assert checked > 100
    # qualifiers who entered a main draw won their last qualifying match
    for tid, ev in df.groupby("tourney_id"):
        q = ev[ev["is_qualifying"]]
        if q.empty:
            continue
        last_q = q[q["round_rank"] == q["round_rank"].max()]
        main = ev[~ev["is_qualifying"]]
        main_players = set(main["winner_id"]) | set(main["loser_id"])
        assert set(last_q["winner_id"]) <= main_players, tid


def test_deterministic() -> None:
    a = make_synthetic_matches(seed=0)
    b = make_synthetic_matches(seed=0)
    pd.testing.assert_frame_equal(a, b)
    c = make_synthetic_matches(seed=1)
    assert set(a["match_id"]) != set(c["match_id"])


@pytest.mark.realdata
@pytest.mark.slow
def test_real_matches_schema_and_order(real_matches: pd.DataFrame, cfg: dict) -> None:
    """The real unified table obeys the same CONTRACT invariants (order_key, round_rank, flags, ids)."""
    _check_schema_and_keys(real_matches, cfg)
    ties = real_matches.duplicated(SORT4, keep=False)
    assert not ties.any(), (
        f"{int(ties.sum())} rows tie on (tourney_date, tourney_id, round_rank, match_num); only the match_id "
        f"tiebreaker orders them (likely duplicated or mis-numbered matches). Examples:\n"
        f"{real_matches.loc[ties, SORT5 + ['winner_id', 'loser_id']].head(6).to_string()}"
    )
