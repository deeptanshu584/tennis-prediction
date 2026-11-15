"""Test 4 (PLAN section 9): orientation.

- A/B is the CONTRACT per-match coin flip: a_is_winner = sha256(f"{seed}:{match_id}")[:8] as int, even -> A won;
  y == a_is_winner, and a_id is the winner exactly when y == 1.
- Label mean in `leakage_tests.label_mean_band` (on small tables the band is widened to 4 binomial standard
  errors, otherwise a fair coin would fail by chance).
- Build with two seeds: where orientation differs, every a_/b_ pair swaps, every d_ negates, elo_prob_a and
  elo_surface_prob_a become 1 - p, y becomes 1 - y; context columns are unchanged. Where it agrees, rows are equal.

Catches: winner/loser layout leaking into the table (A always the winner, features computed for the winner and
then relabelled, d_ not antisymmetric, an asymmetric probability), a label that depends on row position.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.leakage import reference as ref
from tests.leakage.helpers import build, cfg_with, get_builder


def _check_coin_flip(features: pd.DataFrame, matches: pd.DataFrame, cfg: dict[str, Any]) -> None:
    seed = int(cfg["seed"])
    winners = matches.set_index("match_id")["winner_id"].astype(str)
    expected = features["match_id"].map(lambda m: int(ref.a_is_winner(seed, m)))
    assert (features["y"].astype(int) == expected).all(), "y does not follow the CONTRACT coin flip"
    a_won = features["a_id"].astype(str) == features["match_id"].map(winners)
    assert (a_won.astype(int) == features["y"].astype(int)).all(), "y != (a_id is the winner)"
    assert features["y"].nunique() == 2, "label is constant"


def _check_label_mean(features: pd.DataFrame, cfg: dict[str, Any]) -> None:
    lo, hi = cfg["leakage_tests"]["label_mean_band"]
    n = len(features)
    half = max((hi - lo) / 2, 4 * 0.5 / math.sqrt(n))
    mid = (hi + lo) / 2
    mean = float(features["y"].mean())
    assert mid - half <= mean <= mid + half, f"label mean {mean:.4f} outside [{mid - half:.4f}, {mid + half:.4f}]"


def _check_two_seeds(f1: pd.DataFrame, f2: pd.DataFrame, cfg: dict[str, Any]) -> None:
    a = f1.set_index("match_id").sort_index()
    b = f2.set_index("match_id").sort_index()
    assert a.index.equals(b.index), "different emitted rows for different seeds"
    flipped = a["a_id"].astype(str) != b["a_id"].astype(str)
    assert 0.3 < flipped.mean() < 0.7, f"seed change flipped {flipped.mean():.2%} of rows; expected ~50%"
    feats = ref.player_feature_names(cfg)
    problems = []

    def same(x: pd.Series, y: pd.Series) -> pd.Series:
        return (x == y) | (x.isna() & y.isna())

    def close(x: pd.Series, y: pd.Series) -> pd.Series:
        return (np.isclose(x.astype(float), y.astype(float), rtol=0, atol=1e-12)) | (x.isna() & y.isna())

    fl, st = flipped, ~flipped
    checks = [
        ("a_id<->b_id", same(a.loc[fl, "a_id"], b.loc[fl, "b_id"]) & same(a.loc[fl, "b_id"], b.loc[fl, "a_id"])),
        ("a_name<->b_name", same(a.loc[fl, "a_name"], b.loc[fl, "b_name"])),
        ("y -> 1-y", same(a.loc[fl, "y"].astype(int), 1 - b.loc[fl, "y"].astype(int))),
        ("elo_prob_a -> 1-p", close(a.loc[fl, "elo_prob_a"], 1 - b.loc[fl, "elo_prob_a"])),
        ("elo_surface_prob_a -> 1-p", close(a.loc[fl, "elo_surface_prob_a"], 1 - b.loc[fl, "elo_surface_prob_a"])),
    ]
    for f in feats:
        checks.append((f"a_{f}<->b_{f}", same(a.loc[fl, f"a_{f}"], b.loc[fl, f"b_{f}"])
                       & same(a.loc[fl, f"b_{f}"], b.loc[fl, f"a_{f}"])))
        checks.append((f"d_{f} negates", same(a.loc[fl, f"d_{f}"], -b.loc[fl, f"d_{f}"])))
    context = [c for c in ref.context_feature_names(cfg) if c not in ("elo_prob_a", "elo_surface_prob_a")]
    context += ["date", "tourney_id", "tourney_name"]
    for c in context:
        checks.append((f"{c} unchanged", same(a[c], b[c])))
    for name, ok in checks:
        if not bool(ok.all()):
            problems.append(f"{name}: {int((~ok).sum())} rows")
    # rows with the same orientation are identical
    for c in a.columns:
        ok = same(a.loc[st, c], b.loc[st, c])
        if not bool(ok.all()):
            problems.append(f"{c} differs on same-orientation rows")
    assert not problems, "orientation symmetry broken:\n" + "\n".join(problems[:30])
    # every d_ column in the table is antisymmetric, not only the CONTRACT ones
    extra_d = [c for c in a.columns if c.startswith("d_") and c[2:] not in feats]
    for c in extra_d:
        assert bool(same(a.loc[fl, c], -b.loc[fl, c]).all()), f"{c} does not negate"


def test_orientation_synthetic(synthetic_matches: pd.DataFrame, cfg: dict, synthetic_build) -> None:
    f1, _ = synthetic_build
    _check_coin_flip(f1, synthetic_matches, cfg)
    _check_label_mean(f1, cfg)
    cfg2 = cfg_with(cfg, seed=int(cfg["seed"]) + 1)
    f2 = build(synthetic_matches, cfg2)
    _check_coin_flip(f2, synthetic_matches, cfg2)
    _check_two_seeds(f1, f2, cfg)


def test_d_columns_are_a_minus_b(cfg: dict, synthetic_build) -> None:
    f, _ = synthetic_build
    for name in ref.player_feature_names(cfg):
        a, b, d = f[f"a_{name}"].astype(float), f[f"b_{name}"].astype(float), f[f"d_{name}"].astype(float)
        expect = a - b
        ok = (d == expect) | (d.isna() & expect.isna())
        assert bool(ok.all()), f"d_{name} != a_{name} - b_{name} (or NaN handling differs)"


@pytest.mark.realdata
@pytest.mark.slow
def test_orientation_real(real_universe: pd.DataFrame, cfg: dict, real_build) -> None:
    get_builder()
    f1, _ = real_build
    if len(f1) < 2000:
        pytest.skip(f"only {len(f1)} real rows; too few for the label-mean band")
    _check_coin_flip(f1, real_universe, cfg)
    _check_label_mean(f1, cfg)
    cfg2 = cfg_with(cfg, seed=int(cfg["seed"]) + 1)
    f2 = build(real_universe, cfg2)
    _check_two_seeds(f1, f2, cfg)


@pytest.mark.realdata
def test_label_mean_features_parquet(cfg: dict, features_table: pd.DataFrame) -> None:
    _check_label_mean(features_table, cfg)
    assert features_table["y"].nunique() == 2
