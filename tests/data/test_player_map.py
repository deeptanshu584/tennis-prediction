"""Tests for src/tennis/ingest/player_map.py: name normalisation, matching, overrides, unmatched errors."""

from __future__ import annotations

import pandas as pd
import pytest

from tennis.ingest.player_map import (
    PLAYER_MAP_COLUMNS,
    UnmatchedPlayerError,
    load_overrides,
    match_names,
    normalise_name,
    sackmann_player_map,
    token_key,
)


@pytest.fixture
def players() -> pd.DataFrame:
    rows = [
        ("104745", "Rafael", "Nadal"),
        ("104925", "Novak", "Djokovic"),
        ("106421", "Felix", "Auger Aliassime"),
        ("200000", "Alex", "De Minaur"),
        ("105526", "Gael", "Monfils"),
        ("105357", "Christopher", "Oconnell"),
        ("104527", "Stan", "Wawrinka"),
        ("105223", "Juan Martin", "del Potro"),
        ("300001", "John", "Smith"),
        ("300002", "John", "Smith"),
        ("400001", None, "Unknown"),
    ]
    return pd.DataFrame(rows, columns=["player_id", "name_first", "name_last"])


# ---------------------------------------------------------------------------------------------------- normalisation
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Rafael Nadal", "rafael nadal"),
        ("  RAFAEL   Nadal ", "rafael nadal"),
        ("Gaël Monfils", "gael monfils"),
        ("Jo-Wilfried Tsonga", "jo wilfried tsonga"),
        ("Christopher O'Connell", "christopher oconnell"),
        ("Christopher O’Connell", "christopher oconnell"),
        ("Nadal R.", "nadal r"),
        ("Stanislas Wawrinka\t", "stanislas wawrinka"),
        (None, ""),
    ],
)
def test_normalise_name(raw, expected):
    assert normalise_name(raw) == expected


def test_token_key_ignores_order():
    assert token_key("Nadal Rafael") == token_key("Rafael Nadal")
    assert token_key("Auger-Aliassime Felix") == token_key("Félix Auger Aliassime")


# ---------------------------------------------------------------------------------------------------- matching
def test_match_names_variants(players):
    names = ["Rafael Nadal", "Nadal Rafael", "Félix Auger-Aliassime", "Auger-Aliassime Felix", "Alex DeMinaur",
             "Gaël MONFILS", "Christopher O'Connell", "del Potro Juan Martin", "Nadal R.", "Djokovic N."]
    out = match_names(names, players, source="tml")
    assert list(out.columns) == PLAYER_MAP_COLUMNS
    got = dict(zip(out["source_id_or_name"], out["sackmann_player_id"]))
    assert got == {
        "Rafael Nadal": "104745", "Nadal Rafael": "104745", "Félix Auger-Aliassime": "106421",
        "Auger-Aliassime Felix": "106421", "Alex DeMinaur": "200000", "Gaël MONFILS": "105526",
        "Christopher O'Connell": "105357", "del Potro Juan Martin": "105223", "Nadal R.": "104745",
        "Djokovic N.": "104925",
    }
    methods = dict(zip(out["source_id_or_name"], out["method"]))
    assert methods["Rafael Nadal"] == "normalised"
    assert methods["Alex DeMinaur"] == "squashed"
    assert methods["Nadal R."] == "last_initial"
    assert set(out["source"]) == {"tml"}


def test_unmatched_raises_with_all_offenders(players):
    with pytest.raises(UnmatchedPlayerError) as e:
        match_names(["Rafael Nadal", "Novak Đoković", "Nobody Atall"], players, source="tml")
    offenders = " ".join(e.value.offenders)
    assert "Đoković" in offenders and "Nobody Atall" in offenders
    assert "Nadal" not in offenders


def test_ambiguous_raises_and_candidates_resolve(players):
    with pytest.raises(UnmatchedPlayerError, match="ambiguous"):
        match_names(["John Smith"], players, source="odds")
    cands = players[players["player_id"] == "300002"]
    out = match_names(["John Smith"], players, source="odds", candidates=cands)
    assert out["sackmann_player_id"].item() == "300002"


def test_overrides_are_honoured_per_source(players, tmp_path):
    p = tmp_path / "player_overrides.csv"
    p.write_text(
        "source,source_id_or_name,sackmann_player_id,note\n"
        "tml,Novak Đoković,104925,diacritic D-stroke\n"
        "odds,Smith J.,300001,the older John Smith\n",
        encoding="utf-8",
    )
    ov = load_overrides(p)
    out = match_names(["Novak Đoković"], players, source="tml", overrides=ov)
    assert out["sackmann_player_id"].item() == "104925" and out["method"].item() == "override"
    # normalised form of the override key also matches
    out = match_names(["novak đoković "], players, source="tml", overrides=ov)
    assert out["sackmann_player_id"].item() == "104925"
    # an override for another source does not apply
    with pytest.raises(UnmatchedPlayerError):
        match_names(["Novak Đoković"], players, source="odds", overrides=ov)
    out = match_names(["Smith J."], players, source="odds", overrides=ov)
    assert out["sackmann_player_id"].item() == "300001"


def test_override_to_unknown_id_raises(players, tmp_path):
    p = tmp_path / "o.csv"
    p.write_text("source,source_id_or_name,sackmann_player_id,note\ntml,X Y,999999,bad\n", encoding="utf-8")
    with pytest.raises(UnmatchedPlayerError, match="unknown Sackmann ids"):
        match_names(["Rafael Nadal"], players, source="tml", overrides=load_overrides(p))


def test_repo_overrides_file_is_readable():
    from tennis.config import load_config, repo_path

    ov = load_overrides(repo_path(load_config()["paths"]["player_overrides"]))
    assert {"source", "source_id_or_name", "sackmann_player_id"} <= set(ov.columns)


# ---------------------------------------------------------------------------------------------------- sackmann ids
def test_sackmann_player_map_identity(players):
    m = pd.DataFrame({"winner_id": ["104745", "104925"], "loser_id": ["104925", "105526"]})
    pm = sackmann_player_map(m, players)
    assert list(pm.columns) == PLAYER_MAP_COLUMNS
    assert (pm["source_id_or_name"] == pm["sackmann_player_id"]).all()
    assert set(pm["sackmann_player_id"]) == {"104745", "104925", "105526"}
    assert set(pm["method"]) == {"identity"}


def test_sackmann_unknown_id_raises(players):
    m = pd.DataFrame({"winner_id": ["104745", "999998"], "loser_id": ["999999", "105526"]})
    with pytest.raises(UnmatchedPlayerError) as e:
        sackmann_player_map(m, players)
    assert e.value.offenders == ["999998", "999999"]
