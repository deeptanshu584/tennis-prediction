"""Tests for src/tennis/ingest/odds.py (small in-test frames, no network, no files)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tennis.ingest.odds import (
    STATUS_MATCHED,
    choose_book,
    implied_prob,
    in_event_window,
    match_odds,
    name_similarity,
    name_variants,
    odds_columns,
    odds_output,
    parse_odds,
    resolve_names,
    split_td_name,
    unmatched_player_policy,
)

PARAMS = {"levels": ["G", "M", "A", "F", "O"], "pre_days": 3, "window_days": 21,
          "book_preference": ["PS", "Avg", "B365"]}


# ---------------------------------------------------------------------------------------------------- fixtures
@pytest.fixture
def players() -> pd.DataFrame:
    rows = [
        ("1", "Jannik", "Sinner"), ("2", "Felix", "Auger Aliassime"), ("3", "Alex", "De Minaur"),
        ("4", "Jan Lennard", "Struff"), ("5", "Gaël", "Monfils"), ("6", "Albert", "Ramos"),
        ("7", "Alexander", "Zverev"), ("8", "Mischa", "Zverev"), ("9", "Zhizhen", "Zhang"),
        ("10", "John", "Smith"), ("11", "James", "Smith"),  # two "Smith J." players overall
        ("12", "Carlos", "Alcaraz"), ("13", "Casper", "Ruud"),
    ]
    return pd.DataFrame(rows, columns=["player_id", "name_first", "name_last"])


def _m(mid, tid, name, date, rnd, w, l, level="A", **kw):
    r = dict(match_id=mid, tourney_id=tid, tourney_name=name, tourney_level=level,
             tourney_date=pd.Timestamp(date), round=rnd, winner_id=w, loser_id=l, is_qualifying=False,
             is_training_level=True, is_walkover=False, is_retirement=False)
    r.update(kw)
    return r


@pytest.fixture
def matches() -> pd.DataFrame:
    rows = [
        # Two concurrent ATP 250s in the same week (start Mon 2024-02-05) and a Challenger with the same players.
        _m("m1", "2024-A", "Dallas", "2024-02-05", "QF", "1", "2"),
        _m("m2", "2024-A", "Dallas", "2024-02-05", "SF", "1", "3"),
        _m("m3", "2024-B", "Marseille", "2024-02-05", "QF", "4", "5"),
        _m("m4", "2024-B", "Marseille", "2024-02-05", "SF", "6", "10"),
        _m("c1", "2024-C", "Somewhere CH", "2024-02-05", "QF", "1", "2", level="C"),
        # A later event with Zverev brothers and a pair met twice (RR and F).
        _m("m5", "2024-D", "Tour Finals", "2024-11-10", "RR", "7", "8", level="F"),
        _m("m6", "2024-D", "Tour Finals", "2024-11-10", "RR", "9", "12", level="F"),
        _m("m7", "2024-D", "Tour Finals", "2024-11-10", "F", "8", "7", level="F"),
        _m("q1", "2024-D", "Tour Finals", "2024-11-10", "Q1", "13", "12", level="F", is_qualifying=True),
    ]
    return pd.DataFrame(rows)


def _odds(rows):
    cols = ["file_year", "odds_date", "td_atp", "td_location", "td_tournament", "td_series", "td_round",
            "td_winner", "td_loser", "comment", "WRank", "LRank", "PSW", "PSL", "AvgW", "AvgL", "B365W", "B365L"]
    df = pd.DataFrame(rows, columns=cols)
    df["odds_date"] = pd.to_datetime(df["odds_date"])
    return df


@pytest.fixture
def odds() -> pd.DataFrame:
    return _odds([
        (2024, "2024-02-08", 5, "Dallas", "Dallas Open", "ATP250", "Quarterfinals", "Sinner J.", "Auger-Aliassime F.",
         "Completed", 4, 30, 1.25, 4.2, 1.24, 4.0, 1.22, 4.0),
        # odds file names the other player as winner -> flag, never change
        (2024, "2024-02-10", 5, "Dallas", "Dallas Open", "ATP250", "Semifinals", "De Minaur A.", "Sinner J.",
         "Completed", 10, 4, 3.0, 1.4, np.nan, np.nan, 2.9, 1.4),
        (2024, "2024-02-08", 6, "Marseille", "Open 13", "ATP250", "Quarterfinals", "Struff J.L.", "Monfils G.",
         "Completed", 25, 40, np.nan, np.nan, 1.9, 1.9, 1.8, 2.0),
        (2024, "2024-02-09", 6, "Marseille", "Open 13", "ATP250", "Semifinals", "Ramos-Vinolas A.", "Smith J.",
         "Retired", 60, 200, np.nan, np.nan, np.nan, np.nan, 1.5, 2.6),
        (2024, "2024-11-11", 70, "Turin", "Masters Cup", "Masters Cup", "Round Robin", "Zverev A.", "Zverev M.",
         "Completed", 2, 100, 1.1, 7.0, 1.1, 6.5, 1.1, 6.0),
        (2024, "2024-11-12", 70, "Turin", "Masters Cup", "Masters Cup", "Round Robin", "Zhang Zh.", "Alcaraz C.",
         "Completed", 40, 3, 5.0, 1.18, 4.8, 1.18, 4.5, 1.2),
        (2024, "2024-11-17", 70, "Turin", "Masters Cup", "Masters Cup", "The Final", "Zverev M.", "Zverev A.",
         "Completed", 100, 2, 6.0, 1.12, 6.0, 1.12, 5.5, 1.13),
        # an event with no Sackmann counterpart in the window
        (2024, "2024-07-01", 40, "Nowhere", "Nowhere Open", "ATP250", "1st Round", "Sinner J.", "Ruud C.",
         "Completed", 1, 8, 1.3, 3.5, 1.3, 3.4, 1.3, 3.4),
    ])


# ---------------------------------------------------------------------------------------------------- margin removal
def test_implied_prob_removes_margin():
    p = implied_prob(1.5, 2.5)
    assert p == pytest.approx((1 / 1.5) / (1 / 1.5 + 1 / 2.5)) == pytest.approx(0.625)
    assert implied_prob(1.5, 2.5) + implied_prob(2.5, 1.5) == pytest.approx(1.0)
    assert implied_prob(1.9, 1.9) == pytest.approx(0.5)  # 5.3% overround removed


@pytest.mark.parametrize("ow, ol", [(np.nan, 2.0), (1.0, 5.0), (0.0, 2.0), (2.0, -1.0), (np.inf, 2.0)])
def test_implied_prob_invalid_odds_is_nan(ow, ol):
    assert np.isnan(implied_prob(ow, ol))


def test_implied_prob_series_keeps_index():
    s = implied_prob(pd.Series([2.0, np.nan], index=[7, 9]), pd.Series([2.0, 2.0], index=[7, 9]))
    assert list(s.index) == [7, 9] and s[7] == pytest.approx(0.5) and np.isnan(s[9])


def test_choose_book_follows_preference():
    df = pd.DataFrame({"PSW": [1.5, np.nan, np.nan, np.nan], "PSL": [2.5, 2.0, np.nan, np.nan],
                       "AvgW": [1.4, 1.8, np.nan, np.nan], "AvgL": [2.6, 2.0, np.nan, np.nan],
                       "B365W": [1.4, 1.7, 2.0, np.nan], "B365L": [2.5, 2.1, 1.8, np.nan]})
    b = choose_book(df, ["PS", "Avg", "B365"])
    assert b["book"].tolist()[:3] == ["PS", "Avg", "B365"] and pd.isna(b["book"].iloc[3])
    assert b["p_odds_winner"].iloc[0] == pytest.approx(0.625)
    assert b["book_odds_w"].iloc[1] == 1.8 and np.isnan(b["p_odds_winner"].iloc[3])
    # a book absent from the frame is skipped
    assert choose_book(df, ["EX", "B365"])["book"].iloc[0] == "B365"


# ---------------------------------------------------------------------------------------------------- parsing
def test_parse_odds_types_and_cleans():
    raw = pd.DataFrame({
        "ATP": [1, 1, None], "Location": [" Brisbane ", "Brisbane", None], "Tournament": ["Brisbane International"] * 3,
        "Date ": ["2024-01-02", "not a date", None], "Series": ["ATP250"] * 3, "Round": ["1st Round"] * 3,
        "Winner": ["Sinner J. ", "Struff J.L.", None], "Loser": ["Ruud C.", "Monfils G.", None],
        "WRank": [4, "NR", None], "LRank": [8, 30, None], "Comment": ["completed ", "Retired", None],
        "PSW": [1.3, "N/A", None], "PSL": [3.5, 2.0, None], "B365W": [1.3, 1.9, None], "B365L": [3.4, 1.9, None],
        "MaxW": [1.4, 2.0, None],  # no MaxL -> not an odds pair
    })
    p = parse_odds(raw, 2024)
    assert len(p) == 2  # empty spreadsheet row dropped
    assert p["td_winner"].tolist() == ["Sinner J.", "Struff J.L."] and p["td_location"].iloc[0] == "Brisbane"
    assert p["comment"].tolist() == ["Completed", "Retired"]
    assert p["odds_date"].iloc[0] == pd.Timestamp("2024-01-02") and pd.isna(p["odds_date"].iloc[1])
    assert np.isnan(p["WRank"].iloc[1]) and np.isnan(p["PSW"].iloc[1]) and p["PSL"].iloc[1] == 2.0
    assert odds_columns(raw.columns) == ["PSW", "PSL", "B365W", "B365L"]
    assert "MaxW" not in p.columns and (p["file_year"] == 2024).all()


# ---------------------------------------------------------------------------------------------------- names
@pytest.mark.parametrize("raw, surname, initial", [
    ("De Minaur A.", "De Minaur", "a"), ("Struff J.L.", "Struff", "j"), ("Zhang Zh.", "Zhang", "z"),
    ("Auger-Aliassime F.", "Auger-Aliassime", "f"), ("Nadal", "Nadal", None), ("  Sinner J. ", "Sinner", "j"),
    ("Édouard R.", "Édouard", "r"),
])
def test_split_td_name(raw, surname, initial):
    assert split_td_name(raw) == (surname, initial)


def test_name_variants():
    assert name_variants("Struff J.L.") == ["Struff J.L.", "Struff J."]
    assert name_variants("Sinner J.") == ["Sinner J."]


def test_resolve_names_last_initial_accents_hyphens(players):
    names = ["Sinner J.", "Auger-Aliassime F.", "De Minaur A.", "Struff J.L.", "Monfils G.", "Ramos-Vinolas A.",
             "Zverev A.", "Zverev M.", "Zhang Zh.", "Monfils Gael", "Mönfils G."]
    resolved, unresolved = resolve_names(names, players, {})
    assert unresolved == []
    got = {k: v[0] for k, v in resolved.items()}
    assert got == {"Sinner J.": "1", "Auger-Aliassime F.": "2", "De Minaur A.": "3", "Struff J.L.": "4",
                   "Monfils G.": "5", "Ramos-Vinolas A.": "6", "Zverev A.": "7", "Zverev M.": "8", "Zhang Zh.": "9",
                   "Monfils Gael": "5", "Mönfils G.": "5"}
    assert resolved["Struff J.L."][1].endswith("_variant")
    assert resolved["Ramos-Vinolas A."][1] == "event_surname"


def test_resolve_names_candidates_disambiguate_and_report_unresolved(players):
    # "Smith J." is ambiguous among all players but unique among this event's players.
    _, unresolved = resolve_names(["Smith J."], players, {})
    assert unresolved == ["Smith J."]
    event = players[players["player_id"].isin(["10", "1"])]
    resolved, unresolved = resolve_names(["Smith J.", "Nobody X."], event, {})
    assert resolved["Smith J."][0] == "10" and unresolved == ["Nobody X."]
    # overrides win (raw or normalised key)
    resolved, unresolved = resolve_names(["Nobody X."], event, {"nobody x": "11"})
    assert resolved["Nobody X."] == ("11", "override") and unresolved == []


def test_name_similarity():
    assert name_similarity(["Dallas", "Dallas Open"], "Dallas") == 1.0
    assert name_similarity(["Indian Wells", "BNP Paribas Open"], "Indian Wells Masters") == 1.0
    assert name_similarity(["Marseille", "Open 13"], "Dallas") < 0.6


# ---------------------------------------------------------------------------------------------------- date window
@pytest.mark.parametrize("odds_date, ok", [
    ("2024-01-14", True),   # Sunday start before Monday tourney_date (Slams)
    ("2024-01-12", True),   # 3 days before
    ("2024-01-11", False),  # 4 days before
    ("2024-02-05", True),   # tourney_date + 21
    ("2024-02-06", False),  # + 22
])
def test_in_event_window(odds_date, ok):
    assert in_event_window(pd.Timestamp(odds_date), pd.Timestamp("2024-01-15"), 3, 21) is ok


# ---------------------------------------------------------------------------------------------------- matching
def test_match_odds_end_to_end(odds, matches, players):
    out, audit = match_odds(odds, matches, players, PARAMS)
    st = dict(zip(out["td_winner"] + "|" + out["td_loser"], out["status"]))
    assert st["Sinner J.|Ruud C."] == "event_not_found"
    by = out.set_index(out["td_winner"] + "|" + out["td_loser"])
    # concurrent events: each row lands in its own event; the Challenger (level C) is never used
    assert by.at["Sinner J.|Auger-Aliassime F.", "match_id"] == "m1"
    assert by.at["Struff J.L.|Monfils G.", "match_id"] == "m3"
    assert by.at["Ramos-Vinolas A.|Smith J.", "match_id"] == "m4"  # "Smith J." resolved via event candidates
    # same pair twice in one event (RR and final): round decides
    assert by.at["Zverev A.|Zverev M.", "match_id"] == "m5"
    assert by.at["Zverev M.|Zverev A.", "match_id"] == "m7"
    assert by.at["Zhang Zh.|Alcaraz C.", "match_id"] == "m6"
    assert (out["status"] == STATUS_MATCHED).sum() == 7 and audit["unresolved"].empty
    ev = audit["events"].set_index("td_location")
    assert ev.at["Dallas", "tourney_id"] == "2024-A" and ev.at["Marseille", "tourney_id"] == "2024-B"
    assert pd.isna(ev.at["Nowhere", "tourney_id"])


def test_winner_disagreement_is_flagged_not_changed(odds, matches, players):
    out, _ = match_odds(odds, matches, players, PARAMS)
    row = out[out["td_winner"] == "De Minaur A."].iloc[0]
    assert row["match_id"] == "m2" and bool(row["winner_disagrees"])
    assert row["winner_id"] == "1" and row["loser_id"] == "3"  # Sackmann result untouched
    p_td_winner = implied_prob(3.0, 1.4)
    assert row["book"] == "PS" and row["p_odds_winner"] == pytest.approx(p_td_winner)
    assert row["p_winner_book"] == pytest.approx(1 - p_td_winner)  # probability for the Sackmann winner
    agree = out[out["td_winner"] == "Sinner J."].iloc[0]
    assert not agree["winner_disagrees"] and agree["p_winner_book"] == pytest.approx(implied_prob(1.25, 4.2))
    assert out["winner_disagrees"].sum() == 1


def test_book_fallback_and_output(odds, matches, players):
    out, _ = match_odds(odds, matches, players, PARAMS)
    by = out.set_index("td_winner")
    assert by.at["Struff J.L.", "book"] == "Avg" and by.at["Ramos-Vinolas A.", "book"] == "B365"
    o = odds_output(out)
    assert len(o) == 7 and o["match_id"].is_unique
    assert {"match_id", "odds_date", "book", "p_winner_book", "winner_disagrees", "comment", "PSW", "PSL", "AvgW",
            "AvgL", "B365W", "B365L"} <= set(o.columns)
    assert o["p_winner_book"].between(0, 1).all()


def test_unresolved_player_and_policy(odds, matches, players):
    bad = odds.copy()
    bad.loc[0, "td_loser"] = "Unknownson Q."
    out, audit = match_odds(bad, matches, players, PARAMS)
    assert out.loc[0, "status"] == "player_unresolved"
    assert audit["unresolved"]["name"].tolist() == ["Unknownson Q."]
    ok, names = unmatched_player_policy(out, audit, max_unmatched=5)
    assert not ok and names == ["Unknownson Q."]  # a Completed (training) row -> error
    bad.loc[0, "comment"] = "Walkover"
    out, audit = match_odds(bad, matches, players, PARAMS)
    assert unmatched_player_policy(out, audit, max_unmatched=5)[0]  # tiny and non-training -> report and exclude
    assert not unmatched_player_policy(out, audit, max_unmatched=0)[0]


def test_duplicate_odds_rows_keep_first(odds, matches, players):
    dup = pd.concat([odds.iloc[[0]], odds], ignore_index=True)
    out, _ = match_odds(dup, matches, players, PARAMS)
    assert (out["status"] == "duplicate_match_id").sum() == 1
    assert odds_output(out)["match_id"].is_unique
