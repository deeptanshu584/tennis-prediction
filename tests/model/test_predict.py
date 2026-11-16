"""tennis.model.predict: CLI, player lookup, production fit, draw mode, symmetry (synthetic data, tiny model)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tennis.config import config_hash, load_config
from tennis.engine.build_features import build_features
from tennis.engine.snapshot import build_state, load_or_build_state
from tennis.model import dataset
from tennis.model import predict as pr
from tests.fixtures.synthetic import make_match_row, make_synthetic_matches, rows_to_frame

N_TREES = 12


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Config pointing at a temp dir with synthetic matches + features, a cached state and a tiny production model."""
    d = tmp_path_factory.mktemp("predict")
    cfg = copy.deepcopy(load_config())
    cfg["paths"].update(matches=str(d / "matches.parquet"), features=str(d / "features.parquet"),
                        models=str(d / "models"))
    cfg["xgb"]["forward_best_params"]["n_estimators"] = N_TREES
    m = make_synthetic_matches(seed=0, cfg=cfg)
    m.to_parquet(cfg["paths"]["matches"], index=False)
    build_features(m, cfg).to_parquet(cfg["paths"]["features"], index=False)
    state = load_or_build_state(cfg, verbose=False)
    booster, meta = pr.load_or_fit_production(cfg, verbose=False)
    ids = sorted(state.players, key=lambda p: -state.players[p].matches_played)[:6]
    return {"cfg": cfg, "state": state, "booster": booster, "meta": meta, "ids": ids,
            "names": [state.names[i] for i in ids], "dir": d}


# --------------------------------------------------------------------------- player lookup


@pytest.fixture(scope="module")
def name_state():
    """A tiny state with accented, duplicated and walkover-only names."""
    cfg = load_config()

    def row(tid: str, date: str, w: tuple[str, str], l: tuple[str, str], score: str = "6-4 6-4") -> dict:
        r = make_match_row(tourney_id=tid, tourney_date=date, winner_id=w[0], loser_id=l[0], score=score, cfg=cfg)
        r["winner_name"], r["loser_name"] = w[1], l[1]
        return r

    sinner, alcaraz = ("1", "Jannik Sinner"), ("2", "Carlos Alcaraz")
    delpo, evans_a, evans_b = ("3", "Juan Martín del Potro"), ("4", "Daniel Evans"), ("5", "Daniel Evans")
    rows = [row("2024-1", "2024-01-08", sinner, alcaraz), row("2024-2", "2024-01-15", delpo, evans_a),
            row("2024-3", "2024-01-22", evans_b, sinner), row("2024-4", "2024-01-29", ("6", "Walk Over"), alcaraz,
                                                             score="W/O"),
            row("2024-5", "2024-02-05", ("7", "Carlos Alcaraz Garfia"), ("8", "Old Spelling"))]
    rows[-1]["loser_name"] = "Óld Spélling"
    return build_state(rows_to_frame(rows), cfg)


def test_normalize_name() -> None:
    assert pr.normalize_name("Juan Martín del Potro") == "juan martin del potro"
    assert pr.normalize_name("  FÉLIX  Auger-Aliassime ") == "felix auger aliassime"
    assert pr.normalize_name("Stan Wawrinka") == pr.normalize_name("stan  WAWRINKA")


def test_resolve_exact_case_and_accent_insensitive(name_state) -> None:
    assert pr.resolve_player("Jannik Sinner", name_state) == "1"
    assert pr.resolve_player("jannik SINNER", name_state) == "1"
    assert pr.resolve_player("Juan Martin del Potro", name_state) == "3"
    assert pr.resolve_player("old spelling", name_state) == "8"
    assert pr.resolve_player("2", name_state) == "2"  # an id is accepted as is


def test_resolve_errors(name_state) -> None:
    with pytest.raises(pr.PlayerLookupError, match="matches 2 players") as e:
        pr.resolve_player("Daniel Evans", name_state)
    assert "id 4" in str(e.value) and "id 5" in str(e.value) and "pass the id" in str(e.value)
    with pytest.raises(pr.PlayerLookupError, match="Jannik Sinner"):
        pr.resolve_player("Janik Siner", name_state)  # typo: candidates listed, never a silent default
    with pytest.raises(pr.PlayerLookupError, match="Carlos Alcaraz"):
        pr.resolve_player("Alcaraz", name_state)  # surname only: both Alcaraz candidates listed
    with pytest.raises(pr.PlayerLookupError, match="no player named"):
        pr.resolve_player("Roger Federer", name_state)
    with pytest.raises(pr.PlayerLookupError, match="walkovers only|no completed match"):
        pr.resolve_player("Walk Over", name_state)


# --------------------------------------------------------------------------- production model


def test_production_model_is_a_fixed_tree_fit_on_all_rows(env) -> None:
    cfg, booster, meta = env["cfg"], env["booster"], env["meta"]
    assert booster.num_boosted_rounds() == N_TREES
    assert meta["features"] == dataset.feature_columns(cfg) == list(booster.feature_names)
    assert meta["never_evaluated"] is True and meta["val_metrics"] == {} and meta["early_stopping"] is False
    feats = pd.read_parquet(cfg["paths"]["features"])
    assert meta["n_rows"] == len(feats)
    assert meta["train_window"][1] == str(pd.to_datetime(feats["date"]).max().date())
    path = pr.production_model_path(cfg)
    assert path.exists() and config_hash(cfg) in path.name
    assert {k: v for k, v in meta["params"].items()} == {k: v for k, v in cfg["xgb"]["forward_best_params"].items()
                                                         if k != "n_estimators"}


def test_fit_production_refuses_flipped_rows(env) -> None:
    feats = pd.read_parquet(env["cfg"]["paths"]["features"]).assign(is_flipped=1)
    with pytest.raises(ValueError):
        pr.fit_production(feats, env["cfg"])


# --------------------------------------------------------------------------- scoring and CLI


def test_predictions_in_unit_interval_and_symmetric(env) -> None:
    ids, cfg = env["ids"], env["cfg"]
    q = pd.DataFrame({"p1": ids[:3], "p2": ids[3:6], "surface": ["Hard", "Clay", "Grass"], "level": ["G", "M", "A"],
                      "round": ["F", "SF", "R32"], "best_of": [5, 3, 3]})
    ab = pr.predict_queries(env["state"], env["booster"], q, cfg)
    ba = pr.predict_queries(env["state"], env["booster"], q.rename(columns={"p1": "p2", "p2": "p1"}), cfg)
    for c in ("p_xgb", "p_xgb_ab", "p_xgb_ba", "p_elo", "p_elo_surface"):
        assert np.all((ab[c] > 0) & (ab[c] < 1)), c
    np.testing.assert_allclose(ab["p_xgb"] + ba["p_xgb"], 1.0, atol=1e-6)   # symmetrised: exact up to float32
    np.testing.assert_allclose(ab["p_elo"] + ba["p_elo"], 1.0, atol=1e-12)
    np.testing.assert_allclose(ab["p_elo_surface"] + ba["p_elo_surface"], 1.0, atol=1e-12)
    np.testing.assert_allclose(ab["p_xgb_ab"], 1 - ba["p_xgb_ba"], atol=1e-6)  # the same two model calls, swapped
    assert np.all(np.abs(ab["p_xgb_ab"] - ab["p_xgb_ba"]) < 0.25)            # raw orientations roughly agree


def test_cli_single_match(env, capsys) -> None:
    n = env["names"]
    out = pr.main(["--p1", n[0].upper(), "--p2", n[1], "--surface", "grass", "--level", "G", "--round", "F",
                   "--best-of", "5"], cfg=env["cfg"])
    text = capsys.readouterr().out
    assert len(out) == 1 and 0 < out["p_xgb"].iloc[0] < 1
    assert out["p1"].iloc[0] == env["ids"][0] and out["surface"].iloc[0] == "Grass"
    for s in ("XGBoost", "Elo (overall)", "Elo (surface)", f"Data cut-off: last tourney_date {env['state'].cutoff_date}",
              f"query date {env['state'].default_query_date}"):
        assert s in text


def test_cli_date_option_and_past_date_error(env, capsys) -> None:
    n, st = env["names"], env["state"]
    later = str(pd.Timestamp(st.cutoff_date) + pd.Timedelta(days=30))[:10]
    out = pr.main(["--p1", n[0], "--p2", n[1], "--surface", "Hard", "--level", "M", "--round", "R64",
                   "--best-of", "3", "--date", later], cfg=env["cfg"])
    assert out["date"].iloc[0] == later
    with pytest.raises(SystemExit) as e:
        pr.main(["--p1", n[0], "--p2", n[1], "--surface", "Hard", "--level", "M", "--round", "R64",
                 "--best-of", "3", "--date", "2019-01-01"], cfg=env["cfg"])
    assert e.value.code == 2 and "before the data cut-off" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [
    ["--p1", "A"],                                                     # single mode needs every match field
    ["--p1", "A", "--p2", "B", "--surface", "Hard", "--level", "G", "--round", "F", "--best-of", "4"],
    ["--draw", "x.csv", "--p1", "A"],                                  # draw and single are exclusive
])
def test_cli_parse_errors(argv, env) -> None:
    with pytest.raises(SystemExit) as e:
        pr.main(argv, cfg=env["cfg"])
    assert e.value.code == 2


def test_cli_unknown_player_and_bad_level(env, capsys) -> None:
    with pytest.raises(SystemExit) as e:
        pr.main(["--p1", "Nobody Atall", "--p2", env["names"][1], "--surface", "Hard", "--level", "C",
                 "--round", "F", "--best-of", "3"], cfg=env["cfg"])
    err = capsys.readouterr().err
    assert e.value.code == 2 and "no player named 'Nobody Atall'" in err and "unknown level 'C'" in err


def test_draw_mode(env, tmp_path, capsys) -> None:
    n = env["names"]
    draw = pd.DataFrame({"p1": [n[0], n[1], n[2].lower(), env["ids"][3]], "p2": [n[1], n[0], n[3], n[4]],
                         "surface": ["Grass"] * 4, "level": ["G"] * 4, "round": ["R16"] * 4,
                         "best_of": [5] * 4, "date": ["", "", "", str(env["state"].default_query_date)]})
    path = tmp_path / "draw.csv"
    draw.to_csv(path, index=False)
    out_path = tmp_path / "preds.csv"
    pr.main(["--draw", str(path), "--out", str(out_path)], cfg=env["cfg"])
    assert "wrote" in capsys.readouterr().out
    res = pd.read_csv(out_path, dtype={"p1": str, "p2": str})
    assert len(res) == 4 and res["p1"].tolist() == env["ids"][:4] and res["p2"].tolist() == [env["ids"][i] for i in (1, 0, 3, 4)]
    assert np.all((res["p_xgb"] > 0) & (res["p_xgb"] < 1))
    assert abs(res["p_xgb"].iloc[0] + res["p_xgb"].iloc[1] - 1) < 1e-6      # the swapped pair
    assert (res["data_cutoff"] == str(env["state"].cutoff_date)).all()
    # default output path next to the draw
    pr.main(["--draw", str(path)], cfg=env["cfg"])
    assert (tmp_path / "draw_predictions.csv").exists()


def test_draw_mode_lists_every_error(env, tmp_path, capsys) -> None:
    n = env["names"]
    draw = pd.DataFrame({"p1": [n[0], "Ghost One"], "p2": [n[1], "Ghost Two"], "surface": ["Ice", "Hard"],
                         "level": ["G", "G"], "round": ["F", "F"], "best_of": [5, 5]})
    path = tmp_path / "bad.csv"
    draw.to_csv(path, index=False)
    with pytest.raises(SystemExit):
        pr.main(["--draw", str(path)], cfg=env["cfg"])
    err = capsys.readouterr().err
    assert "row 1: unknown surface 'Ice'" in err and "row 2 p1" in err and "row 2 p2" in err


def test_draw_missing_columns(env, tmp_path, capsys) -> None:
    path = tmp_path / "cols.csv"
    pd.DataFrame({"p1": ["x"], "p2": ["y"]}).to_csv(path, index=False)
    with pytest.raises(SystemExit):
        pr.main(["--draw", str(path)], cfg=env["cfg"])
    assert "missing columns" in capsys.readouterr().err


def test_production_model_is_reused_not_refit(env, monkeypatch) -> None:
    monkeypatch.setattr(pr, "fit_production", lambda *a, **k: (_ for _ in ()).throw(AssertionError("refit")))
    booster, meta = pr.load_or_fit_production(env["cfg"], verbose=False)
    assert booster.num_boosted_rounds() == N_TREES
    meta_path = Path(env["cfg"]["paths"]["models"]) / f"{pr.PRODUCTION_NAME}_{config_hash(env['cfg'])}_meta.json"
    assert json.loads(meta_path.read_text(encoding="utf-8"))["purpose"] == "production"
