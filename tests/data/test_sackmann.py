"""Tests for src/tennis/ingest/sackmann.py: typing, score vocabulary, surface, round order, ids and cleaning."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from tennis.ingest import sackmann as sk
from tennis.ingest.sackmann import (
    STAT_COLS,
    UNIFIED_COLUMNS,
    classify_score,
    make_match_id,
    normalise_surface,
    round_rank,
    unify,
)
from tests.data.helpers import FIXTURE_DIR, GOOD_W, raw_row, typed_frame


# ---------------------------------------------------------------------------------------------------- reading/typing
def test_read_raw_csv_keeps_strings_and_na_text():
    df = sk.read_raw_csv(FIXTURE_DIR / "atp_matches_2023.csv")
    assert list(df.columns) == sk.SACKMANN_COLUMNS
    assert all(df[c].dtype == object or str(df[c].dtype).startswith("str") for c in df.columns)
    # 'NA' (Namibia's IOC code) must stay a string, not become NaN
    assert (df["loser_ioc"] == "NA").sum() == 1
    assert df.loc[0, "winner_id"] == "100001"


def test_read_raw_csv_rejects_wrong_columns(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"tourney_id": ["x"], "foo": ["y"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="unexpected columns"):
        sk.read_raw_csv(p)


def test_type_columns_types_and_strips():
    t = typed_frame([raw_row(score=" W/O ", stats=False, minutes=None)])
    assert t["tourney_date"].dtype == "datetime64[ns]"
    assert t["match_num"].dtype == "int64" and t["best_of"].dtype == "int64" and t["draw_size"].dtype == "int64"
    assert t["w_svpt"].dtype == "float64" and np.isnan(t.loc[0, "w_svpt"])
    assert isinstance(t.loc[0, "winner_id"], str) and t.loc[0, "winner_id"] == "100001"
    assert t.loc[0, "score"] == "W/O"


def test_type_columns_raises_on_non_numeric():
    with pytest.raises((ValueError, TypeError)):
        typed_frame([raw_row(w_ace="five")])


def test_type_columns_raises_on_bad_date():
    with pytest.raises(ValueError):
        typed_frame([raw_row(tourney_date="20231345")])


def test_type_columns_raises_on_missing_match_num():
    with pytest.raises(ValueError, match="match_num"):
        typed_frame([raw_row(match_num=None)])


# ---------------------------------------------------------------------------------------------------- score vocab
@pytest.mark.parametrize(
    "score, category, games",
    [
        ("6-4 6-4", "normal", 20),
        ("7-6(5) 3-6 6-4", "normal", 32),
        ("6-4 6-2 6-4?", "normal", 28),
        ("5-7 6-3 [10-7]", "normal", 21),
        ("W/O", "walkover", 0),
        (" W/O", "walkover", 0),
        ("0-0 W/O", "walkover", 0),
        ("Walkover", "walkover", 0),
        ("walkover", "walkover", 0),
        ("6-3 2-1 RET", "retirement", 12),
        ("0-0 RET", "retirement", 0),
        ("RET", "retirement", 0),
        ("5-1 RE", "retirement", 6),
        ("6-3 0-0 Ret'd", "retirement", 9),
        ("6-4 1-0 DEF", "default", 11),
        ("6-0 4-1 Def.", "default", 11),
        ("Def.", "default", 0),
        ("4-6 7-6(4) 1-0 Default", "default", 24),
        ("2-2 ABD", "abandoned", 4),
        ("6-1 5-7 3-2 Played and unfinished", "abandoned", 24),
        ("0-3 Played and abandoned", "abandoned", 3),
        (">", "no_score", 0),
        (None, "null", 0),
        (np.nan, "null", 0),
        ("", "null", 0),
        ("   ", "null", 0),
    ],
)
def test_classify_score(score, category, games):
    info = classify_score(score)
    assert info.category == category
    assert info.games == games


def test_classify_score_unknown_token_raises():
    with pytest.raises(ValueError, match="unknown score token"):
        classify_score("6-4 6-4 SUSP")
    with pytest.raises(ValueError):
        classify_score("6-4 and")  # filler outside the abandoned phrase


def _flags(rows):
    t = typed_frame(rows)
    return sk.score_flags(t)


def test_score_flags_walkover_retirement_normal():
    f = _flags([
        raw_row(score="6-4 6-4"),
        raw_row(score="W/O", stats=False, minutes=None),
        raw_row(score="6-3 2-1 RET"),
        raw_row(score="6-4 1-0 DEF"),
    ])
    assert f["is_walkover"].tolist() == [False, True, False, False]
    assert f["is_retirement"].tolist() == [False, False, True, True]


def test_null_score_without_evidence_is_walkover():
    f = _flags([raw_row(score=None, stats=False, minutes=None)])
    assert bool(f.loc[0, "is_walkover"]) and not bool(f.loc[0, "is_retirement"])


def test_null_score_with_stats_is_not_walkover():
    f = _flags([raw_row(score=None)])  # stats show 60/65 serve points
    assert not bool(f.loc[0, "is_walkover"])


def test_zero_game_retirement_with_points_is_retirement_only():
    f = _flags([raw_row(score="0-0 RET", stats=True, w={**GOOD_W, "svpt": 0, "1stIn": 0, "1stWon": 0,
                                                        "2ndWon": 0, "bpSaved": 0, "bpFaced": 0})])
    # loser served 65 points -> points were played -> retirement, not walkover
    assert not bool(f.loc[0, "is_walkover"]) and bool(f.loc[0, "is_retirement"])


def test_zero_game_retirement_with_minutes_only_is_retirement():
    f = _flags([raw_row(score="0-0 RET", stats=False, minutes="7")])
    assert not bool(f.loc[0, "is_walkover"]) and bool(f.loc[0, "is_retirement"])


def test_bare_ret_without_evidence_is_walkover_and_retirement():
    f = _flags([raw_row(score="RET", stats=False, minutes=None)])
    assert bool(f.loc[0, "is_walkover"]) and bool(f.loc[0, "is_retirement"])


def test_explicit_walkover_wins_over_stats():
    f = _flags([raw_row(score="W/O")])  # stats present, score says W/O -> still walkover (reported)
    assert bool(f.loc[0, "is_walkover"])


# ---------------------------------------------------------------------------------------------------- surface
def test_surface_mapping_and_imputation(cfg):
    s = pd.Series(["Hard", "Clay", "Grass", "Carpet", np.nan], dtype=object)
    out, imputed = normalise_surface(s, cfg["data"]["surface_map"])
    assert out.tolist() == ["Hard", "Clay", "Grass", "Hard", "Hard"]
    assert imputed.tolist() == [False, False, False, False, True]


def test_unknown_surface_raises(cfg):
    with pytest.raises(ValueError, match="unknown surface"):
        normalise_surface(pd.Series(["Ice"]), cfg["data"]["surface_map"])


# ---------------------------------------------------------------------------------------------------- round order
def test_round_rank_from_config(cfg):
    order = cfg["data"]["round_order"]
    rr = round_rank(pd.Series(["Q1", "R128", "ER", "RR", "F"]), order)
    assert rr.tolist() == [order.index(r) for r in ["Q1", "R128", "ER", "RR", "F"]]
    assert order.index("ER") < order.index("RR")
    assert max(order.index(q) for q in cfg["data"]["qualifying_rounds"]) < order.index("R128")


def test_round_rank_unknown_raises(cfg):
    with pytest.raises(ValueError, match="unknown round"):
        round_rank(pd.Series(["R32", "XX"]), cfg["data"]["round_order"])


def test_order_key_sorts_matches(cfg):
    rows = [
        raw_row(tourney_id="2023-0002", tourney_date="20230102", round="R32", match_num="1",
                winner_id="1", loser_id="2"),
        raw_row(tourney_id="2023-0001", tourney_date="20230102", round="F", match_num="9",
                winner_id="3", loser_id="4"),
        raw_row(tourney_id="2023-0001", tourney_date="20230102", round="R32", match_num="300",
                winner_id="5", loser_id="6"),
        raw_row(tourney_id="2023-0001", tourney_date="20230102", round="Q1", match_num="900",
                winner_id="7", loser_id="8"),
        raw_row(tourney_id="2007-0100", tourney_date="20070101", round="RR", match_num="1",
                winner_id="9", loser_id="10"),
        raw_row(tourney_id="2007-0100", tourney_date="20070101", round="ER", match_num="50",
                winner_id="11", loser_id="12"),
        raw_row(tourney_id="2022-9999", tourney_date="20221226", round="R32", match_num="1",
                winner_id="13", loser_id="14"),
        # prefix tourney_ids on the same date: "2023-1" < "2023-10"
        raw_row(tourney_id="2023-10", tourney_date="20230109", round="Q1", match_num="1",
                winner_id="15", loser_id="16"),
        raw_row(tourney_id="2023-1", tourney_date="20230109", round="F", match_num="99",
                winner_id="17", loser_id="18"),
    ]
    u, _ = unify(typed_frame(rows), cfg)
    assert u["order_key"].dtype == "int64"
    assert u["order_key"].tolist() == list(range(len(u)))
    seq = list(zip(u["tourney_id"], u["round"]))
    assert seq == [
        ("2007-0100", "ER"), ("2007-0100", "RR"),
        ("2022-9999", "R32"),
        ("2023-0001", "Q1"), ("2023-0001", "R32"), ("2023-0001", "F"),
        ("2023-0002", "R32"),
        ("2023-1", "F"), ("2023-10", "Q1"),
    ]
    resorted = u.sort_values(sk.MATCH_ORDER).reset_index(drop=True)
    assert resorted["match_id"].tolist() == u["match_id"].tolist()


def test_match_id_breaks_order_ties(cfg):
    rows = [
        raw_row(match_num="5", winner_id="200", loser_id="300"),
        raw_row(match_num="5", winner_id="100", loser_id="400"),
    ]
    u, audit = unify(typed_frame(rows), cfg)
    assert u["match_id"].tolist() == sorted(u["match_id"])
    assert len(audit["order_ties"]) == 2


# ---------------------------------------------------------------------------------------------------- match_id
def test_match_id_formula_and_stability():
    expected = hashlib.sha1("2024-0339|300|105777|208029".encode()).hexdigest()[:16]
    assert make_match_id("2024-0339", 300, "105777", "208029") == expected
    assert make_match_id("2024-0339", 300, "105777", "208029") == expected  # deterministic


def test_match_id_symmetric_in_players():
    assert make_match_id("2024-0339", 300, "105777", "208029") == make_match_id("2024-0339", 300, "208029", "105777")


def test_match_id_depends_on_match_num_and_tourney():
    base = make_match_id("2024-0339", 300, "1", "2")
    assert base != make_match_id("2024-0339", 301, "1", "2")
    assert base != make_match_id("2024-0340", 300, "1", "2")


# ---------------------------------------------------------------------------------------------------- cleaning rules
def test_d009_exact_duplicates_dropped_keep_first(cfg):
    rows = [
        raw_row(round="F", match_num="385", winner_id="1", loser_id="2", score="7-5 6-4"),
        raw_row(round="F", match_num="384", winner_id="1", loser_id="2", score="7-5 6-4"),
        # same pair and score in a different round is a real repeat (Tour Finals RR + F)
        raw_row(round="RR", match_num="10", winner_id="1", loser_id="2", score="7-5 6-4"),
    ]
    u, audit = unify(typed_frame(rows), cfg)
    assert len(u) == 2
    assert sorted(u["round"]) == ["F", "RR"]
    assert u.loc[u["round"] == "F", "match_num"].item() == 384  # first in match order kept
    assert audit["duplicates"]["match_num"].tolist() == [385]
    assert audit["duplicates"]["duplicate_of"].item() == u.loc[u["round"] == "F", "match_id"].item()


def test_d009_null_scores_compare_equal(cfg):
    rows = [raw_row(match_num="1", score=None), raw_row(match_num="2", score=None)]
    u, audit = unify(typed_frame(rows), cfg)
    assert len(u) == 1 and len(audit["duplicates"]) == 1


def _conflict_rows():
    return [
        raw_row(round="Q2", match_num="710", winner_id="1", loser_id="2", score="3-6 7-5 6-3"),
        raw_row(round="Q2", match_num="711", winner_id="2", loser_id="1", score="2-6 6-1 6-1"),
        raw_row(round="Q2", match_num="712", winner_id="3", loser_id="4", score="6-3 6-2"),
        # walkover + played row for the same meeting: not a conflict (walkovers are dropped by the engine)
        raw_row(round="F", match_num="270", winner_id="5", loser_id="6", score=None, stats=False, minutes=None),
        raw_row(round="F", match_num="301", winner_id="5", loser_id="6", score="6-3 3-6 6-4"),
    ]


def test_conflicting_results_flagged_and_kept_by_default(cfg):
    cfg["cleaning"].pop("drop_conflicting_results", None)
    u, audit = unify(typed_frame(_conflict_rows()), cfg)
    assert len(u) == 5
    assert sorted(audit["conflicts"]["match_num"]) == [710, 711]
    assert audit["conflicts_dropped"] is False
    assert audit["walkover_shadows"]["match_num"].tolist() == [270]
    # results are untouched
    assert u.loc[u["match_num"] == 711, "winner_id"].item() == "2"


def test_conflicting_results_dropped_when_configured(cfg):
    cfg["cleaning"]["drop_conflicting_results"] = True
    u, audit = unify(typed_frame(_conflict_rows()), cfg)
    assert sorted(u["match_num"]) == [270, 301, 712]
    assert audit["conflicts_dropped"] is True and len(audit["conflicts"]) == 2


def test_d010_self_matches_dropped(cfg):
    rows = [raw_row(match_num="1"), raw_row(match_num="2", winner_id="7", loser_id="7")]
    u, audit = unify(typed_frame(rows), cfg)
    assert len(u) == 1
    assert audit["self_matches"]["winner_id"].tolist() == ["7"]


def test_d011_null_surface_imputed(cfg):
    u, _ = unify(typed_frame([raw_row(surface=None),
                              raw_row(match_num="2", surface="Carpet", winner_id="5", loser_id="6")]), cfg)
    assert u["surface"].tolist() == ["Hard", "Hard"]
    assert u["surface_imputed"].tolist() == [True, False]


@pytest.mark.parametrize(
    "side, change",
    [
        ("w", {"1stIn": 70}),                       # 1stIn > svpt
        ("l", {"1stWon": 39}),                      # 1stWon > 1stIn (38)
        ("w", {"2ndWon": 21}),                      # 2ndWon > svpt - 1stIn (20)
        ("l", {"bpSaved": 8}),                      # bpSaved > bpFaced (7)
        ("w", {"ace": -1}),                         # negative
        ("l", {"bpSaved": -4, "bpFaced": 0}),      # negative (1991 Tokyo pattern)
    ],
)
def test_d012_impossible_stats(cfg, side, change):
    from tests.data.helpers import GOOD_L

    w = {**GOOD_W, **change} if side == "w" else GOOD_W
    l = {**GOOD_L, **change} if side == "l" else GOOD_L
    u, audit = unify(typed_frame([raw_row(w=w, l=l), raw_row(match_num="2", winner_id="5", loser_id="6")]), cfg)
    bad = u[u["match_num"] == 1].iloc[0]
    good = u[u["match_num"] == 2].iloc[0]
    assert bool(bad["stats_invalid"]) and not bool(good["stats_invalid"])
    assert bad[STAT_COLS].isna().all()
    assert good[STAT_COLS].notna().all()
    assert bad["winner_id"] == "100001" and bad["loser_id"] == "100002"  # result kept
    assert len(audit["stats_invalid_rows"]) == 1


def test_boundary_stats_are_valid(cfg):
    w = {**GOOD_W, "1stIn": 60, "1stWon": 60, "2ndWon": 0, "bpSaved": 3, "bpFaced": 3}
    u, _ = unify(typed_frame([raw_row(w=w)]), cfg)
    assert not bool(u.loc[0, "stats_invalid"])


def test_qualifying_and_training_level(cfg):
    rows = [
        raw_row(match_num="1", round="Q1", tourney_level="A"),
        raw_row(match_num="2", round="R32", tourney_level="A", winner_id="3", loser_id="4"),
        raw_row(match_num="3", round="R32", tourney_level="C", winner_id="5", loser_id="6"),
        raw_row(match_num="4", round="Q2", tourney_level="G", winner_id="7", loser_id="8"),
        raw_row(match_num="5", round="RR", tourney_level="D", winner_id="9", loser_id="10"),
    ]
    u, _ = unify(typed_frame(rows), cfg)
    got = dict(zip(u["match_num"], zip(u["is_qualifying"], u["is_training_level"])))
    assert got == {1: (True, False), 2: (False, True), 3: (False, False), 4: (True, False), 5: (False, True)}


def test_unify_schema(cfg):
    u, _ = unify(typed_frame([raw_row()]), cfg)
    assert list(u.columns) == UNIFIED_COLUMNS
    assert u.loc[0, "source"] == "sackmann"
    for c in sk.FLAG_COLS:
        assert u[c].dtype == bool
    assert u["round_rank"].dtype == "int64"


# ---------------------------------------------------------------------------------------------------- fixture build
def test_build_sackmann_on_fixture_files(fixture_cfg):
    u, audit = sk.build_sackmann(fixture_cfg)
    assert audit["rows_in"] == 13
    assert len(audit["duplicates"]) == 1 and len(audit["self_matches"]) == 1
    assert len(u) == 11
    assert u["match_id"].is_unique
    assert int(u["is_walkover"].sum()) == 2          # ' W/O' and the empty score without stats
    assert int(u["is_retirement"].sum()) == 1
    assert int(u["stats_invalid"].sum()) == 1
    assert int(u["surface_imputed"].sum()) == 1
    assert int(u["is_qualifying"].sum()) == 2
    assert audit["carpet_rows"] == 4
    assert set(u["surface"]) <= {"Hard", "Clay", "Grass"}
    training = u["is_training_level"] & ~u["is_walkover"] & ~u["is_retirement"]
    assert int(training.sum()) == 5
    # tour qualifying shares the main draw's tourney_id and sorts before it
    t1 = u[u["tourney_id"] == "2023-0001"]
    assert t1["round"].iloc[:2].tolist() == ["Q1", "Q2"]
    assert u.loc[u["loser_ioc"] == "NA"].shape[0] == 1


def test_render_report_on_fixture(fixture_cfg):
    from tennis.ingest.merge import merge_sources
    from tennis.ingest.player_map import sackmann_player_map

    u, audit = sk.build_sackmann(fixture_cfg)
    merged, contrib = merge_sources({"sackmann": u}, fixture_cfg)
    pmap = sackmann_player_map(merged, sk.load_players(fixture_cfg))
    text = sk.render_report(merged, audit, contrib, pmap, fixture_cfg)
    for heading in ("## 1. Summary", "## 3. Score vocabulary", "## 4. Dropped rows", "## 6. Rows per year",
                    "## 7. Tour-level training rows", "## 8. How qualifying", "## 10. Nulls per column"):
        assert heading in text
    assert "Self-matches (D-010): 1 rows" in text
    assert "Exact duplicates (D-009): 1 rows" in text


# ---------------------------------------------------------------------------------------------------- D-022
def _d022_cfg(cfg):
    cfg["data"]["excluded_tourneys"] = ["Laver Cup"]
    cfg["data"]["level_overrides"] = [
        {"name_regex": "^(Tour Finals|Masters Cup|ATP Finals)$", "level": "F"},
        {"name_regex": "^Olympics$", "level": "O"},
    ]
    return cfg


def test_excluded_tourneys_removed_exact_name(cfg):
    cfg = _d022_cfg(cfg)
    rows = [
        raw_row(tourney_id="2023-9210", tourney_name="Laver Cup", match_num="1", winner_id="1", loser_id="2"),
        raw_row(tourney_id="2023-9210", tourney_name="Laver Cup", match_num="2", winner_id="3", loser_id="4"),
        raw_row(tourney_id="2023-0001", tourney_name="Laver Cup Exhibition", match_num="1", winner_id="5",
                loser_id="6"),  # not an exact match -> kept
        raw_row(tourney_id="2023-0002", tourney_name="Testopen", match_num="1", winner_id="7", loser_id="8"),
    ]
    u, audit = unify(typed_frame(rows), cfg)
    assert sorted(u["tourney_name"]) == ["Laver Cup Exhibition", "Testopen"]
    assert audit["excluded_tourneys"]["match_num"].tolist() == [1, 2]


def test_level_overrides_full_match_and_training_level(cfg):
    cfg = _d022_cfg(cfg)
    rows = [
        raw_row(tourney_id="2023-0605", tourney_name="Tour Finals", tourney_level="A", round="RR",
                winner_id="1", loser_id="2"),
        raw_row(tourney_id="2005-0605", tourney_name="Masters Cup", tourney_level="F", round="F",
                tourney_date="20051113", winner_id="3", loser_id="4"),
        raw_row(tourney_id="2021-0096", tourney_name="Olympics", tourney_level="A", round="R64",
                tourney_date="20210724", winner_id="5", loser_id="6"),
        # partial matches are NOT overridden (fullmatch)
        raw_row(tourney_id="2021-0097", tourney_name="Tokyo Olympics", tourney_level="A", round="R64",
                tourney_date="20210724", winner_id="7", loser_id="8"),
        raw_row(tourney_id="2015-7000", tourney_name="Challenger Tour Finals", tourney_level="C", round="F",
                tourney_date="20151116", winner_id="9", loser_id="10"),
    ]
    u, audit = unify(typed_frame(rows), cfg)
    lvl = dict(zip(u["tourney_name"], u["tourney_level"]))
    assert lvl == {"Tour Finals": "F", "Masters Cup": "F", "Olympics": "O", "Tokyo Olympics": "A",
                   "Challenger Tour Finals": "C"}
    tl = dict(zip(u["tourney_name"], u["is_training_level"]))
    assert not tl["Challenger Tour Finals"]
    assert tl["Tour Finals"] and tl["Olympics"]
    ch = audit["level_changes"]
    assert ch[["year", "tourney_name", "old_level", "new_level", "rows"]].values.tolist() == [
        [2021, "Olympics", "A", "O", 1], [2023, "Tour Finals", "A", "F", 1]]
    hits = audit["level_override_hits"].set_index("level")
    assert hits.loc["F", "rows matched"] == 2 and hits.loc["F", "rows changed"] == 1
    # results untouched
    assert u.loc[u["tourney_name"] == "Tour Finals", "winner_id"].item() == "1"


def test_level_override_can_make_row_non_training(cfg):
    cfg["data"]["level_overrides"] = [{"name_regex": "Exhibition Cup", "level": "X"}]
    u, _ = unify(typed_frame([raw_row(tourney_name="Exhibition Cup", tourney_level="A")]), cfg)
    assert u.loc[0, "tourney_level"] == "X" and not bool(u.loc[0, "is_training_level"])


def test_conflicting_level_overrides_raise(cfg):
    cfg["data"]["level_overrides"] = [{"name_regex": "Tour Finals", "level": "F"},
                                      {"name_regex": "Tour.*", "level": "A"}]
    with pytest.raises(ValueError, match="conflict"):
        unify(typed_frame([raw_row(tourney_name="Tour Finals")]), cfg)


def test_d022_absent_config_is_noop(cfg):
    cfg["data"].pop("excluded_tourneys", None)
    cfg["data"].pop("level_overrides", None)
    u, audit = unify(typed_frame([raw_row(tourney_name="Laver Cup")]), cfg)
    assert len(u) == 1 and len(audit["level_changes"]) == 0
