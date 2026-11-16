"""Test 18 (Phase 5 audit): the allow-list guard also bans same-match stats behind a player prefix.

PLAN section 2 rule 3 / test 3. `tennis.model.dataset.feature_columns` is the only way model code gets its inputs,
and it rejects banned names through `is_banned`. Same-match serve stats re-oriented to A/B would naturally be named
`a_w_ace`, `b_l_svpt`, `d_w_bpFaced` (player prefix + Sackmann w_/l_ name). Those must be rejected too: after
stripping one leading `a_` / `b_` / `d_`, the name must not start with `w_` or `l_`.

Catches: a same-match statistic slipping into the allow-list under a player prefix (the plain `w_`/`l_` prefix check
only looks at the first characters of the name).
"""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from tests.leakage.helpers import import_or_skip, model_features

PLAYER_PREFIXED_SAME_MATCH = ["a_w_ace", "b_l_svpt", "d_w_bpFaced", "a_l_1stWon", "b_w_SvGms"]


def _stripped(name: str) -> str:
    low = name.lower()
    return low[2:] if low[:2] in ("a_", "b_", "d_") else low


def test_allow_list_has_no_player_prefixed_same_match_names(cfg: dict) -> None:
    bad = [f for f in model_features(cfg) if _stripped(f).startswith(("w_", "l_"))]
    assert not bad, f"same-match stats behind a player prefix in the allow-list: {bad}"


@pytest.mark.parametrize("name", PLAYER_PREFIXED_SAME_MATCH)
def test_is_banned_rejects_player_prefixed_same_match(name: str) -> None:
    ds = import_or_skip("tennis.model.dataset", "dataset module not built yet")
    assert ds.is_banned(name), f"dataset.is_banned({name!r}) is False"


@pytest.mark.parametrize("name", PLAYER_PREFIXED_SAME_MATCH)
def test_feature_columns_rejects_player_prefixed_same_match(cfg: dict, name: str) -> None:
    ds = import_or_skip("tennis.model.dataset", "dataset module not built yet")
    c = copy.deepcopy(cfg)
    c["model"]["features"] = list(model_features(cfg)) + [name]
    # the table DOES carry the column, so only the banned-name check can reject it
    table = pd.DataFrame(columns=c["model"]["features"])
    with pytest.raises(ValueError, match="banned"):
        ds.feature_columns(c, table)


def test_is_banned_keeps_the_real_features(cfg: dict) -> None:
    ds = import_or_skip("tennis.model.dataset", "dataset module not built yet")
    assert not [f for f in model_features(cfg) if ds.is_banned(f)]
