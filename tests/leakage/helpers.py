"""Shared helpers for the leakage suite (not tests themselves)."""

from __future__ import annotations

import copy
import importlib
import importlib.util
from types import ModuleType
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.synthetic import STAT_FIELDS

ENGINE_MODULE = "tennis.engine.build_features"
ENGINE_SKIP = "engine not built yet (Phase 2): tennis.engine.build_features is missing"
ELO_MODULE = "tennis.engine.elo"
ELO_SKIP = "Elo module not built yet (Phase 1, feature-engineer): tennis.engine.elo is missing"
SPLITS_MODULE = "tennis.model.splits"
SPLITS_SKIP = "splits module not built yet (Phase 3): tennis.model.splits is missing"

BANNED_PREFIXES = ("w_", "l_")
BANNED_SUBSTRINGS = ("winner", "loser", "score", "minutes")


def import_or_skip(name: str, reason: str) -> ModuleType:
    """Skip the test if module `name` does not exist; otherwise import it and let real errors surface.

    (Plain `pytest.importorskip` would also skip when the module exists but one of ITS imports is missing,
    which would hide a broken engine behind a skip.)
    """
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        spec = None
    if spec is None:
        return pytest.importorskip(name, reason=reason)
    return importlib.import_module(name)


def get_builder() -> Callable[..., Any]:
    return import_or_skip(ENGINE_MODULE, ENGINE_SKIP).build_features


def build(matches: pd.DataFrame, cfg: dict[str, Any], record_k: bool = False) -> Any:
    """Call the contract builder on a private copy (so a mutating builder can't corrupt shared fixtures)."""
    fn = get_builder()
    if record_k:
        return fn(matches.copy(deep=True), cfg, record_k=True)
    return fn(matches.copy(deep=True), cfg)


def cfg_with(cfg: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Deep copy of cfg with dotted-path overrides, e.g. cfg_with(cfg, **{"elo.m_mode": "constant"})."""
    out = copy.deepcopy(cfg)
    for path, value in overrides.items():
        node = out
        keys = path.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return out


AUTHORITATIVE_ORDER = ["tourney_date", "tourney_id", "round_rank", "match_num", "match_id"]


def match_position(matches: pd.DataFrame) -> pd.Series:
    """0-based position of each row in the CONTRACT authoritative order, computed here (never trusts order_key)."""
    pos = matches.sort_values(AUTHORITATIVE_ORDER, kind="mergesort").index
    return pd.Series(np.arange(len(matches), dtype="int64"), index=pos).reindex(matches.index)


def canonical(df: pd.DataFrame, key: str = "match_id") -> pd.DataFrame:
    """Rows sorted by key, fresh index, columns sorted — for exact frame comparison."""
    return df.sort_values(key, kind="mergesort").reset_index(drop=True)[sorted(df.columns)]


def assert_frames_identical(a: pd.DataFrame, b: pd.DataFrame, what: str) -> None:
    """Exact (NaN-aware) equality of two feature tables after canonical ordering."""
    ca, cb = canonical(a), canonical(b)
    assert list(ca.columns) == list(cb.columns), f"{what}: column sets differ"
    assert len(ca) == len(cb), f"{what}: row counts differ ({len(ca)} vs {len(cb)})"
    assert ca["match_id"].tolist() == cb["match_id"].tolist(), f"{what}: different emitted match ids"
    bad = []
    for c in ca.columns:
        x, y = ca[c], cb[c]
        same = (x == y) | (x.isna() & y.isna())
        if not bool(same.all()):
            idx = np.flatnonzero(~same.to_numpy())[:3]
            bad.append(f"{c} (e.g. {[(ca['match_id'].iat[i], x.iat[i], y.iat[i]) for i in idx]})")
    assert not bad, f"{what}: {len(bad)} columns differ: " + "; ".join(bad[:15])


def swap_winner_loser(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Swap every winner_*/loser_* and w_*/l_* column on rows in `mask` (match_id is order-invariant)."""
    out = df.copy()
    pairs = [(c, "loser_" + c[len("winner_"):]) for c in df.columns if c.startswith("winner_")]
    pairs += [(c, "l_" + c[2:]) for c in df.columns if c.startswith("w_")]
    for a, b in pairs:
        if b not in df.columns:
            continue
        va = df.loc[mask, a].copy()
        out.loc[mask, a] = df.loc[mask, b].to_numpy()
        out.loc[mask, b] = va.to_numpy()
    return out


def scramble_future(matches: pd.DataFrame, future: pd.Series, seed: int = 123) -> pd.DataFrame:
    """Rewrite everything about the `future` matches: results, stats, score, minutes, ranks, flags.

    - winner and loser swapped (with all their attributes and stats);
    - serve stats permuted across future rows, 20% of future rows set to NaN stats;
    - score/minutes rewritten; ~10% retirement flags and ~5% walkover flags toggled.
    """
    rng = np.random.default_rng(seed)
    out = swap_winner_loser(matches, future)
    idx = np.flatnonzero(future.to_numpy())
    if len(idx) == 0:
        return out
    stat_cols = [f"{p}_{s}" for p in ("w", "l") for s in STAT_FIELDS]
    perm = rng.permutation(idx)
    out.iloc[idx, [out.columns.get_loc(c) for c in stat_cols]] = out.iloc[perm][stat_cols].to_numpy()
    nan_rows = idx[rng.random(len(idx)) < 0.2]
    out.iloc[nan_rows, [out.columns.get_loc(c) for c in stat_cols]] = np.nan
    out.iloc[idx, out.columns.get_loc("score")] = "6-0 6-0"
    out.iloc[idx, out.columns.get_loc("minutes")] = rng.integers(30, 300, len(idx)).astype(float)
    for c in ("winner_rank", "loser_rank"):
        out.iloc[idx, out.columns.get_loc(c)] = rng.integers(1, 500, len(idx)).astype(float)
    ret = idx[rng.random(len(idx)) < 0.10]
    out.iloc[ret, out.columns.get_loc("is_retirement")] = ~out["is_retirement"].to_numpy()[ret]
    wo = idx[rng.random(len(idx)) < 0.05]
    out.iloc[wo, out.columns.get_loc("is_walkover")] = ~out["is_walkover"].to_numpy()[wo]
    return out


def synthetic_split_cfg(cfg: dict[str, Any], matches: pd.DataFrame) -> dict[str, Any]:
    """cfg with `splits` rewritten onto the synthetic fixture's seasons (same structure as config.yaml)."""
    years = sorted(pd.to_datetime(matches["tourney_date"]).dt.year.unique())
    y0, y1, y2 = years[0], years[1], years[-1]
    grass = matches[(matches["tourney_level"] == "G") & (matches["surface"] == "Grass")
                    & (matches["tourney_date"].dt.year == y2)]
    slam_name = str(grass["tourney_name"].iloc[0])
    slam_date = pd.Timestamp(grass["tourney_date"].iloc[0])
    out = copy.deepcopy(cfg)
    out["splits"] = {
        "dev": {"train_start": f"{y0}-01-01", "train_end": f"{y0}-12-31",
                "val_start": f"{y1}-01-01", "val_end": f"{y1}-12-31"},
        "replication": {"train_start": f"{y0}-01-01", "train_end": f"{y1}-12-31",
                        "val_start": f"{y2}-01-01",
                        "val_end": (slam_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                        "test_tourney": slam_name, "test_year": int(y2)},
        "forward": {"train_start": f"{y0}-01-01", "train_end": f"{y1}-06-30",
                    "val_start": f"{y1}-07-01", "val_end": f"{y1}-12-31", "test_start": f"{y2}-01-01"},
    }
    return out


def model_features(cfg: dict[str, Any]) -> list[str]:
    feats = (cfg.get("model") or {}).get("features")
    if not feats:
        pytest.skip("cfg['model']['features'] is not defined yet (added by the orchestrator in Phase 2)")
    return list(feats)
