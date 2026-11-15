"""Test 2 (PLAN section 9): slow recompute.

For emitted matches, recompute every CONTRACT feature with the independent reference implementation in
tests/leakage/reference.py (Elo replayed with plain dicts over the strictly earlier matches; every other
feature by filtering the strictly earlier matches) and compare with the engine (tolerance 1e-9).
Also checks the emitted match set, the A/B orientation, and the logged K values.

Synthetic: EVERY emitted row is checked (a superset of the configured sample), under three configs.
Real: `leakage_tests.slow_recompute_samples` random emitted rows.

Catches: features computed after the state update (including the current match), off-by-one ordering, wrong
same-day / same-tournament order, wrong formulas (means of percentages instead of sums of counts), buffers that
include walkovers or out-of-policy retirements, emission of qualifying / retirement / walkover rows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.leakage import reference as ref
from tests.leakage.helpers import build, cfg_with, get_builder

TOL = 1e-9


def _variant(cfg: dict[str, Any], name: str, matches: pd.DataFrame) -> dict[str, Any]:
    if name == "default":
        return cfg
    if name == "flags_flipped":
        years = sorted(matches["tourney_date"].dt.year.unique())
        return cfg_with(cfg, **{
            "cleaning.retirement_updates_elo": not cfg["cleaning"]["retirement_updates_elo"],
            "cleaning.retirement_updates_stats": not cfg["cleaning"]["retirement_updates_stats"],
            "cleaning.retirement_updates_h2h": not cfg["cleaning"]["retirement_updates_h2h"],
            "cleaning.retirement_updates_form": not cfg["cleaning"]["retirement_updates_form"],
            "elo.overall.bonus_after_layoff": True,
            "elo.surface.bonus_after_layoff": True,
            "elo.surface.base_k": 8.0,
            "elo.overall.max_k": 40.0,
            "features.rolling_n": 7,
            "features.elo_grad_window": 5,
            "features.recent_days_window": 7,
            "data.feature_first_year": int(years[1]),
        })
    if name == "other_m_mode_fixed_surface_k":
        other = "constant" if cfg["elo"]["m_mode"] == "event_matches_so_far" else "event_matches_so_far"
        return cfg_with(cfg, **{"elo.m_mode": other, "elo.overall.m": 5.0, "elo.surface.div_number": 0.0})
    raise ValueError(name)


def _check(matches: pd.DataFrame, cfg: dict[str, Any], features: pd.DataFrame, k_log: pd.DataFrame,
           n_samples: int | None) -> None:
    ms = ref.chronological(matches)
    em = ref.emitted(ms, cfg)
    expected_ids = set(ms.loc[em, "match_id"])
    got_ids = set(features["match_id"].astype(str))
    assert got_ids == expected_ids, (
        f"emitted set differs: {len(got_ids - expected_ids)} unexpected rows "
        f"(e.g. {sorted(got_ids - expected_ids)[:3]}), {len(expected_ids - got_ids)} missing "
        f"(e.g. {sorted(expected_ids - got_ids)[:3]})"
    )
    assert features["match_id"].is_unique

    missing_cols = [c for c in ref.all_output_columns(cfg) if c not in features.columns]
    assert not missing_cols, f"CONTRACT columns missing from the engine output: {missing_cols}"

    ids = sorted(expected_ids)
    if n_samples is not None and n_samples < len(ids):
        rng = np.random.default_rng(int(cfg["seed"]))
        ids = list(rng.choice(ids, size=n_samples, replace=False))
    rows, k_ref = ref.reference_rows(matches, cfg, ids)
    engine_rows = features[features["match_id"].isin(ids)].copy()
    problems = ref.compare_rows(engine_rows, rows, ref.all_output_columns(cfg), tol=TOL)

    kl = k_log.copy()
    kl["match_id"] = kl["match_id"].astype(str)
    kl["player_id"] = kl["player_id"].astype(str)
    assert not kl.duplicated(["match_id", "player_id"]).any(), "k_log has duplicate (match_id, player_id)"
    k_ref = k_ref.assign(key=k_ref["match_id"] + "|" + k_ref["player_id"])
    kl = kl.assign(key=kl["match_id"] + "|" + kl["player_id"])
    problems += ref.compare_rows(kl[kl["key"].isin(k_ref["key"])], k_ref, ["k_overall", "k_surface"],
                                 key="key", tol=TOL)
    assert not problems, f"{len(problems)} mismatches vs the reference, first 25:\n" + "\n".join(problems[:25])


@pytest.mark.parametrize("variant", ["default", "flags_flipped", "other_m_mode_fixed_surface_k"])
def test_slow_recompute_synthetic(synthetic_matches: pd.DataFrame, cfg: dict, variant: str) -> None:
    get_builder()
    vcfg = _variant(cfg, variant, synthetic_matches)
    features, k_log = build(synthetic_matches, vcfg, record_k=True)
    _check(synthetic_matches, vcfg, features, k_log, n_samples=None)


def test_k_log_covers_every_processed_match(synthetic_matches: pd.DataFrame, synthetic_build) -> None:
    """CONTRACT: one k_log row per processed (non-walkover) match per player."""
    _, k_log = synthetic_build
    ms = ref.chronological(synthetic_matches)
    expected = set(zip(ms["match_id"], ms["winner_id"])) | set(zip(ms["match_id"], ms["loser_id"]))
    got = set(zip(k_log["match_id"].astype(str), k_log["player_id"].astype(str)))
    assert got == expected, f"{len(expected - got)} missing, {len(got - expected)} unexpected k_log rows"


@pytest.mark.realdata
@pytest.mark.slow
def test_slow_recompute_real(real_universe: pd.DataFrame, cfg: dict, real_build) -> None:
    features, k_log = real_build
    n = int(cfg["leakage_tests"]["slow_recompute_samples"])
    _check(real_universe, cfg, features, k_log, n_samples=n)
