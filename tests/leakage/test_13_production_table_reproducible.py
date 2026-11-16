"""Test 13 (Phase 2 audit): the production training table is exactly what the audited engine builds.

The engine-level leakage tests (1, 2, 4) run on a 2018+ universe, and tests 3, 4, 6, 9 read
data/processed/features.parquet without checking how it was made. A table left over from an older engine or
config would slip past all of them. Rebuild from data/interim/matches.parquet with the current config and require
an identical table (about 11 s).
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.leakage.helpers import assert_frames_identical, build


@pytest.mark.realdata
@pytest.mark.slow
def test_features_parquet_equals_fresh_build(real_matches: pd.DataFrame, features_table: pd.DataFrame,
                                             cfg: dict) -> None:
    fresh = build(real_matches, cfg)
    fresh["date"] = pd.to_datetime(fresh["date"])
    assert_frames_identical(fresh, features_table, "features.parquet vs a fresh build of matches.parquet")
