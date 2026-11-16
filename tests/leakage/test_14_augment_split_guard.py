"""Test 14 (Phase 2 re-audit): the per-split flip-augmentation guard and the evaluation-side check.

With ``split_name=None`` the guard of ``maybe_augment_train`` can only use the LATEST train_end of all splits
(2025-12-31), so dev-validation rows (2023-24) would pass it. Model code must therefore pass ``split_name``; this test
pins down that, given the split, every split's own validation rows are refused, and that ``assert_not_augmented``
rejects any frame carrying flipped copies (CONTRACT section 4b).
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.leakage.helpers import import_or_skip


def _rows(date: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame({
        "match_id": ["m0", "m1"], "date": [date, date], "a_id": ["1", "1"], "b_id": ["2", "2"],
        "a_elo": [1600.0, 1600.0], "b_elo": [1500.0, 1500.0], "d_elo": [100.0, 100.0],
        "elo_prob_a": [0.64, 0.64], "y": [1, 0],
    })


@pytest.mark.parametrize("split_name", ["dev", "replication", "forward"])
def test_guard_refuses_each_splits_validation_rows(cfg: dict, split_name: str) -> None:
    ds = import_or_skip("tennis.model.dataset", "dataset module not built yet")
    cfg["features"]["flip_augment_train"] = True
    val_start = pd.Timestamp(cfg["splits"][split_name]["val_start"])
    with pytest.raises(ValueError):
        ds.maybe_augment_train(_rows(val_start), cfg, split_name=split_name)
    # the split's own last training day is still accepted
    out = ds.maybe_augment_train(_rows(pd.Timestamp(cfg["splits"][split_name]["train_end"])), cfg,
                                 split_name=split_name)
    assert len(out) == 4 and int(out["is_flipped"].sum()) == 2


def test_assert_not_augmented_rejects_flipped_rows(cfg: dict) -> None:
    ds = import_or_skip("tennis.model.dataset", "dataset module not built yet")
    cfg["features"]["flip_augment_train"] = True
    aug = ds.maybe_augment_train(_rows(pd.Timestamp(cfg["splits"]["dev"]["train_start"])), cfg, split_name="dev")
    with pytest.raises(ValueError):
        ds.assert_not_augmented(aug)
    ds.assert_not_augmented(aug[aug["is_flipped"] == 0])  # originals only: fine
