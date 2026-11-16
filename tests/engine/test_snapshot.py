"""tennis.engine.snapshot: final state + read-only query rows (PLAN section 6 Phase 8).

The key gate is CONSISTENCY: for real matches M, ``query_rows`` built from the state after every match strictly
before M must equal the rows ``build_features`` emits for M (same A/B orientation), for every feature column.
Targets are the first main-draw round of a tournament (no in-event state between those matches: each player plays
once) and whole round-robin blocks (all snapshotted from the pre-block state, D-019). Static attributes (rank,
rank points, age, height, hand) are passed as overrides from the real match row, because the state only knows the
LAST row's values; without overrides the intentional differences are rank / rank points (last known, not the
current ranking) and age (projected, within rounding).
"""

from __future__ import annotations

import copy
import pickle
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tennis.config import load_config, repo_path
from tennis.engine.build_features import build_features, output_columns, player_feature_names
from tennis.engine.ordering import sort_matches
from tennis.engine.snapshot import (
    build_state,
    load_or_build_state,
    query_rows,
    state_cache_path,
)
from tests.fixtures.synthetic import make_match_row, make_synthetic_matches, rows_to_frame

NON_FEATURES = {"match_id", "date", "tourney_id", "tourney_name", "a_id", "b_id", "a_name", "b_name", "y"}
STATIC = ("rank", "rank_points", "age", "height", "is_left")


def feature_cols(cfg: dict[str, Any]) -> list[str]:
    """Every feature column of the engine output (a superset of the model allow-list)."""
    return [c for c in output_columns(cfg) if c not in NON_FEATURES]


# --------------------------------------------------------------------------- consistency harness


def target_round(matches: pd.DataFrame, tourney_id: str, cfg: dict[str, Any]
                 ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """(prefix, target) for ``tourney_id``: the target is its first main-draw round, restricted to the matches
    whose players have not played earlier in that round (all of a knock-out first round; for an RR block every
    match, since the whole block reads the pre-block state). The prefix is every match strictly before the
    round's first match in the authoritative order. None if the tournament has no main-draw match."""
    ms = sort_matches(matches.loc[~matches["is_walkover"].astype(bool)])
    quali = set(cfg["data"]["qualifying_rounds"])
    ev = ms[(ms["tourney_id"] == tourney_id) & ~ms["round"].isin(quali)]
    if ev.empty:
        return None
    first_rank = ev["round_rank"].min()
    rnd = ev[ev["round_rank"] == first_rank]
    start_pos = int(rnd.index.min())
    is_block = (rnd["round"].iloc[0] == "RR"
                and rnd["tourney_level"].iloc[0] not in cfg["features"]["rr_block_exempt_levels"])
    keep, seen = [], set()
    for idx, w, l in zip(rnd.index, rnd["winner_id"], rnd["loser_id"]):
        if is_block or (w not in seen and l not in seen):
            keep.append(idx)
        seen.update((w, l))
    return ms.iloc[:start_pos], ms.loc[keep]


def queries_for(target: pd.DataFrame, feats: pd.DataFrame, *, overrides: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Queries in the A/B orientation of the emitted rows, and those emitted rows (aligned)."""
    rows = feats.set_index("match_id").loc[[m for m in target["match_id"] if m in set(feats["match_id"])]]
    t = target.set_index("match_id").loc[rows.index]
    a_is_w = (rows["a_id"].to_numpy() == t["winner_id"].astype(str).to_numpy())
    q = pd.DataFrame({"p1": rows["a_id"].to_numpy(), "p2": rows["b_id"].to_numpy(),
                      "surface": t["surface"].to_numpy(), "level": t["tourney_level"].to_numpy(),
                      "round": t["round"].to_numpy(), "best_of": t["best_of"].to_numpy(),
                      "date": t["tourney_date"].to_numpy()})
    if overrides:
        for p, side_if_a in (("p1", True), ("p2", False)):
            for f in ("rank", "rank_points", "age", "ht", "hand"):
                w, l = t[f"winner_{f}"].to_numpy(dtype=object), t[f"loser_{f}"].to_numpy(dtype=object)
                q[f"{p}_{f}"] = np.where(a_is_w == side_if_a, w, l)
    return q, rows.reset_index()


def assert_rows_equal(got: pd.DataFrame, exp: pd.DataFrame, cols: list[str], atol: float = 1e-9) -> None:
    assert len(got) == len(exp)
    assert got["a_id"].tolist() == exp["a_id"].tolist()
    assert got["b_id"].tolist() == exp["b_id"].tolist()
    for c in cols:
        np.testing.assert_allclose(got[c].to_numpy(dtype=float), exp[c].to_numpy(dtype=float), rtol=0, atol=atol,
                                   equal_nan=True, err_msg=c)


def check_consistency(matches: pd.DataFrame, feats: pd.DataFrame, tourney_ids: list[str], cfg: dict[str, Any]
                      ) -> int:
    """Run the consistency comparison for each tournament; returns the number of rows compared."""
    cols = feature_cols(cfg)
    no_static = [c for c in cols if not any(c.endswith(s) for s in STATIC)]
    n = 0
    for tid in tourney_ids:
        tr = target_round(matches, tid, cfg)
        if tr is None:
            continue
        prefix, target = tr
        if prefix.empty:
            continue
        state = build_state(prefix, cfg)
        q, exp = queries_for(target, feats, overrides=True)
        if q.empty:
            continue
        got = query_rows(state, q, cfg, allow_new_players=True)
        assert_rows_equal(got, exp, cols)
        # without overrides: everything except the static attributes is still identical, and the projected age is
        # within rounding of the recorded age (both rounded to 0.1 years in the data)
        q2, _ = queries_for(target, feats, overrides=False)
        got2 = query_rows(state, q2, cfg, allow_new_players=True)
        assert_rows_equal(got2, exp, no_static)
        for p in ("a", "b"):
            known = got2[f"{p}_id"].isin(list(state.info)).to_numpy().copy()  # debutants have no prior row: fills apply
            known &= exp[f"{p}_age"].to_numpy() != cfg["features"]["static_fill"]["age"]
            diff = np.abs(got2[f"{p}_age"].to_numpy() - exp[f"{p}_age"].to_numpy())[known]
            assert np.all(diff < 0.2), f"{tid}: projected age off by {diff.max()}"
            for f in ("height", "is_left"):
                np.testing.assert_array_equal(got2[f"{p}_{f}"].to_numpy()[known], exp[f"{p}_{f}"].to_numpy()[known])
        n += len(exp)
    return n


# --------------------------------------------------------------------------- synthetic fixtures


@pytest.fixture(scope="module")
def cfg() -> dict[str, Any]:
    return copy.deepcopy(load_config())


@pytest.fixture(scope="module")
def synth(cfg: dict[str, Any]) -> pd.DataFrame:
    return make_synthetic_matches(seed=0, cfg=cfg)


@pytest.fixture(scope="module")
def synth_feats(synth: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    return build_features(synth, cfg)


@pytest.fixture(scope="module")
def synth_state(synth: pd.DataFrame, cfg: dict[str, Any]):
    return build_state(synth, cfg)


def test_consistency_with_build_features_synthetic(synth, synth_feats, cfg) -> None:
    """query_rows from the state before each tournament's first round == the rows build_features emits."""
    ms = sort_matches(synth)
    train = ms[ms["is_training_level"].astype(bool)]
    last_season = train["tourney_date"].dt.year == train["tourney_date"].dt.year.max()
    special = train["tourney_level"].isin(["F", "D", "O"])  # RR blocks, exempt Davis Cup RR, bronze-medal event
    tids = list(dict.fromkeys(train.loc[last_season | special, "tourney_id"]))
    levels = set(train.loc[train["tourney_id"].isin(tids), "tourney_level"])
    assert {"G", "M", "A", "F", "D", "O"} <= levels
    n = check_consistency(synth, synth_feats, tids, cfg)
    assert n > 150


def test_whole_table_state_equals_state_after_last_match(synth, cfg, synth_state) -> None:
    """build_state over all matches gives the state after the last match: querying the final round of the
    last event would need in-event updates, so instead check the state's bookkeeping against build_features."""
    ms = sort_matches(synth.loc[~synth["is_walkover"].astype(bool)])
    assert synth_state.n_matches == len(ms)
    assert synth_state.cutoff_date == ms["tourney_date"].max().date()
    total = sum(st.matches_played for st in synth_state.players.values())
    assert total == 2 * len(ms)


def test_queries_do_not_mutate_state(synth_state, cfg) -> None:
    before = pickle.dumps(synth_state, protocol=pickle.HIGHEST_PROTOCOL)
    ids = sorted(synth_state.players)[:6]
    q = pd.DataFrame({"p1": ids[:3] + ["new_guy"], "p2": ids[3:6] + [ids[0]], "surface": ["Hard", "Clay", "Grass", "Hard"],
                      "level": ["A", "G", "M", "F"], "round": ["R32", "F", "QF", "RR"], "best_of": [3, 5, 3, 3]})
    query_rows(synth_state, q, cfg, allow_new_players=True)
    assert pickle.dumps(synth_state, protocol=pickle.HIGHEST_PROTOCOL) == before
    assert "new_guy" not in synth_state.players


def test_each_query_reads_the_same_state(synth_state, cfg) -> None:
    """A batch gives the same rows as each query alone (no query sees another), also with repeated pairs."""
    ids = sorted(synth_state.players)[:4]
    q = pd.DataFrame({"p1": [ids[0], ids[0], ids[2]], "p2": [ids[1], ids[1], ids[3]],
                      "surface": ["Hard", "Hard", "Clay"], "level": ["G", "G", "M"], "round": ["F", "F", "SF"],
                      "best_of": [5, 5, 3]})
    batch = query_rows(synth_state, q, cfg)
    cols = feature_cols(cfg)
    for i in range(len(q)):
        alone = query_rows(synth_state, q.iloc[[i]], cfg)
        assert_rows_equal(alone, batch.iloc[[i]].reset_index(drop=True), cols, atol=0)
    assert_rows_equal(batch.iloc[[0]].reset_index(drop=True), batch.iloc[[1]].reset_index(drop=True), cols, atol=0)


def test_swapping_players_is_antisymmetric(synth_state, cfg) -> None:
    ids = sorted(synth_state.players)
    q = pd.DataFrame({"p1": ids[:5], "p2": ids[5:10], "surface": ["Hard", "Clay", "Grass", "Hard", "Clay"],
                      "level": ["G", "M", "A", "F", "D"], "round": ["F", "SF", "R32", "RR", "R16"],
                      "best_of": [5, 3, 3, 3, 5]})
    ab = query_rows(synth_state, q, cfg)
    ba = query_rows(synth_state, q.rename(columns={"p1": "p2", "p2": "p1"}), cfg)
    for f in player_feature_names(cfg):
        np.testing.assert_array_equal(ab[f"a_{f}"], ba[f"b_{f}"])
        np.testing.assert_array_equal(ab[f"b_{f}"], ba[f"a_{f}"])
        np.testing.assert_allclose(ab[f"d_{f}"], -ba[f"d_{f}"], rtol=0, atol=1e-12, equal_nan=True)
    for c in ("elo_prob_a", "elo_surface_prob_a"):
        np.testing.assert_allclose(ab[c] + ba[c], 1.0, rtol=0, atol=1e-12)
    ctx = [c for c in feature_cols(cfg) if c.startswith(("surface_", "level_")) or c in ("round_rank", "best_of")]
    pd.testing.assert_frame_equal(ab[ctx], ba[ctx])


def test_default_date_and_day_features(cfg) -> None:
    """Default date = day after the last tourney_date; days features count from the query date."""
    rows = [make_match_row(tourney_id="2020-1", tourney_date="2020-01-06", winner_id="1", loser_id="2", cfg=cfg),
            make_match_row(tourney_id="2020-2", tourney_date="2020-01-13", winner_id="1", loser_id="3", cfg=cfg),
            make_match_row(tourney_id="2020-3", tourney_date="2020-01-20", winner_id="2", loser_id="3", cfg=cfg)]
    state = build_state(rows_to_frame(rows), cfg)
    assert str(state.cutoff_date) == "2020-01-20"
    assert str(state.default_query_date) == "2020-01-21"
    q = pd.DataFrame({"p1": ["1", "1"], "p2": ["2", "2"], "surface": ["Hard"] * 2, "level": ["A"] * 2,
                      "round": ["R32"] * 2, "best_of": [3, 3], "date": [None, "2020-02-10"]})
    r = query_rows(state, q, cfg)
    assert r["a_days_since_last_match"].tolist() == [8.0, 28.0]   # player 1 last played 2020-01-13
    assert r["b_days_since_last_match"].tolist() == [1.0, 21.0]
    assert r["a_matches_last_14_days"].tolist() == [1, 0]          # window: tourney_date >= date - 14 days
    assert r["a_h2h_wins"].tolist() == [1, 1] and r["b_h2h_wins"].tolist() == [0, 0]
    assert r["a_matches_played"].tolist() == [2, 2]


def test_query_validation_errors(synth_state, cfg) -> None:
    ids = sorted(synth_state.players)
    base = {"p1": ids[0], "p2": ids[1], "surface": "Hard", "level": "G", "round": "F", "best_of": 5}

    def run(**kw: Any) -> None:
        query_rows(synth_state, pd.DataFrame([{**base, **kw}]), cfg)

    run()
    for kw, exc in [({"p1": "nobody"}, KeyError), ({"p2": ids[0]}, ValueError), ({"surface": "Carpet"}, ValueError),
                    ({"level": "C"}, ValueError), ({"round": "R256"}, ValueError), ({"best_of": 4}, ValueError),
                    ({"date": "2019-01-01"}, ValueError)]:
        with pytest.raises(exc):
            run(**kw)
    with pytest.raises(ValueError):
        query_rows(synth_state, pd.DataFrame([{"p1": ids[0], "p2": ids[1]}]), cfg)


def test_state_cache_roundtrip(synth, cfg, tmp_path, monkeypatch) -> None:
    c = copy.deepcopy(cfg)
    c["paths"]["matches"] = str(tmp_path / "matches.parquet")
    c["paths"]["models"] = str(tmp_path / "models")
    synth.to_parquet(c["paths"]["matches"], index=False)
    s1 = load_or_build_state(c, verbose=False)
    assert state_cache_path(c).exists()
    import tennis.engine.snapshot as snap
    monkeypatch.setattr(snap, "build_state", lambda *a, **k: (_ for _ in ()).throw(AssertionError("rebuilt")))
    s2 = load_or_build_state(c, verbose=False)  # served from the cache
    assert s2.n_matches == s1.n_matches and s2.last_day == s1.last_day
    with pytest.raises(AssertionError, match="rebuilt"):
        load_or_build_state(c, rebuild=True, verbose=False)


# --------------------------------------------------------------------------- real data


@pytest.mark.realdata
@pytest.mark.slow
def test_consistency_with_build_features_realdata(cfg) -> None:
    """Real-data gate: the last Grand Slam's first round and the first rounds of the last few training-level
    events (state built from the leakage-suite universe, ``leakage_tests.real_start_year`` on)."""
    path = repo_path(cfg["paths"]["matches"])
    if not path.exists():
        pytest.skip(f"real data absent: {path}")
    m = pd.read_parquet(path)
    m["tourney_date"] = pd.to_datetime(m["tourney_date"])
    start = int((cfg.get("leakage_tests") or {}).get("real_start_year", 2018))
    m = m[m["tourney_date"].dt.year >= start].reset_index(drop=True)
    feats = build_features(m, cfg)
    ms = sort_matches(m)
    train = ms[ms["is_training_level"].astype(bool)]
    events = list(dict.fromkeys(train["tourney_id"]))
    last_slam = train.loc[train["tourney_level"] == "G", "tourney_id"].iloc[-1]
    tids = list(dict.fromkeys([last_slam] + events[-3:]))
    n = check_consistency(m, feats, tids, cfg)
    assert n >= 64
