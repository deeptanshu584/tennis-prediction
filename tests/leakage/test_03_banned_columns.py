"""Test 3 (PLAN section 9): banned columns.

The model's feature allow-list (cfg["model"]["features"]) must contain no same-match columns: no `w_` / `l_`
prefixes and no `winner`, `loser`, `score`, `minutes` substrings; no non-feature column (ids, names, date, label);
every listed feature must exist in the built table. The built table itself must not carry raw same-match columns.

Catches: same-match statistics or the result leaking into the model inputs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.leakage.helpers import BANNED_PREFIXES, BANNED_SUBSTRINGS, model_features
from tests.leakage.reference import NON_FEATURE_COLUMNS


def _banned(name: str) -> bool:
    low = name.lower()
    return low.startswith(BANNED_PREFIXES) or any(s in low for s in BANNED_SUBSTRINGS)


def test_allow_list_has_no_banned_names(cfg: dict) -> None:
    feats = model_features(cfg)
    bad = [f for f in feats if _banned(f)]
    assert not bad, f"banned columns in cfg['model']['features']: {bad}"
    assert len(feats) == len(set(feats)), "duplicate entries in the feature allow-list"
    forbidden = set(NON_FEATURE_COLUMNS) | {"date", "order_key", "tourney_date", "a_is_winner"}
    assert not forbidden & set(feats), f"non-feature columns in the allow-list: {forbidden & set(feats)}"
    assert not any(f.endswith("_id") or f.endswith("_name") for f in feats)


def test_allow_list_exists_in_synthetic_build(cfg: dict, synthetic_build) -> None:
    feats = model_features(cfg)
    features, _ = synthetic_build
    missing = [f for f in feats if f not in features.columns]
    assert not missing, f"allow-listed features missing from the built table: {missing}"


def test_built_table_has_no_same_match_columns(synthetic_build) -> None:
    features, _ = synthetic_build
    bad = [c for c in features.columns if _banned(c)]
    assert not bad, f"same-match columns carried into the feature table: {bad}"


@pytest.mark.realdata
def test_allow_list_exists_in_features_parquet(cfg: dict, features_table: pd.DataFrame) -> None:
    feats = model_features(cfg)
    missing = [f for f in feats if f not in features_table.columns]
    assert not missing, f"allow-listed features missing from features.parquet: {missing}"
    bad = [c for c in features_table.columns if _banned(c)]
    assert not bad, f"same-match columns in features.parquet: {bad}"
