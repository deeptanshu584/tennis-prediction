"""Test 12 (Phase 2 audit): missingness of present-day player attributes must not be a model input.

``height``, ``is_left`` (from hand) and ``age`` (from birth date) come from Sackmann's player file as it stands
today, not as it stood at the match: 99.4% of the match-row heights equal atp_players.csv, and a player's height is
the same in every row of his career. Whether a height/hand/birth date is on file depends on the player's LATER
career -- players first seen on tour in 2000-2015 with no height on file played a median of 2 tour-level matches in
total, versus 19 for players with a height. So "height is NaN" partly encodes the future.

Evidence (features.parquet, 2000-2026): controlling for Elo, log matches played, log rank and a rank-NaN
indicator, a one-sided height-NaN indicator still has a logistic coefficient of about -0.77, and hand-NaN about
-0.49. In the 2023-24 validation window the NaN-height player loses 79% of the 81 one-sided rows, where Elo alone is
right 59% of the time. The effect is weaker in 2026 rows (60%), as expected of hindsight.

XGBoost learns from NaN directly, so an allow-listed snapshot attribute must have no NaN (neutralise missingness,
e.g. a constant from config such as height 185 / is_left 0, without a missing indicator), or not be allow-listed.
"""

from __future__ import annotations

import pandas as pd

from tests.leakage.helpers import model_features

SNAPSHOT_ATTRIBUTES = ("height", "is_left", "age")


def test_snapshot_attributes_have_no_missingness_signal(cfg: dict, features_table: pd.DataFrame) -> None:
    allowed = set(model_features(cfg))
    problems = []
    for f in SNAPSHOT_ATTRIBUTES:
        for col in (f"a_{f}", f"b_{f}", f"d_{f}"):
            if col in allowed and col in features_table.columns:
                n = int(features_table[col].isna().sum())
                if n:
                    problems.append(f"{col}: {n} NaN")
    assert not problems, ("allow-listed present-day player attributes carry a missingness signal that encodes the "
                          "player's later career: " + ", ".join(problems))
