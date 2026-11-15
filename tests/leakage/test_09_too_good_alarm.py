"""Test 9 (PLAN section 9): too-good alarm.

Any validation accuracy above `evaluation.leakage_alarm_accuracy` (0.75) is treated as leakage:
- every validation accuracy logged in experiments/*.jsonl;
- a fixed, reasonable XGBoost trained on the dev-split training rows with the configured feature allow-list;
- the Elo-only baseline (elo_prob_a > 0.5) on the dev validation rows.

Catches: whatever slipped past tests 1-8 but still shows up as implausibly good predictions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pytest

from tennis.config import repo_path
from tests.leakage.helpers import SPLITS_MODULE, SPLITS_SKIP, import_or_skip, model_features

VAL_ACC = re.compile(r"(^|[_.])(val|valid|validation)[_.]?(acc|accuracy)$", re.IGNORECASE)


def _walk(obj: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{path}.{k}" if path else str(k))
    else:
        yield path, obj


def _is_val_accuracy(path: str) -> bool:
    low = path.lower()
    if "test" in low:
        return False
    if VAL_ACC.search(low):
        return True
    parts = low.split(".")
    return len(parts) >= 2 and parts[-1] in ("acc", "accuracy") and any(
        p in ("val", "valid", "validation") for p in parts[:-1])


def test_logged_validation_accuracies(cfg: dict) -> None:
    files = sorted(Path(repo_path(cfg["paths"]["experiments"])).glob("*.jsonl"))
    if not files:
        pytest.skip("no experiments/*.jsonl logged yet")
    alarm = float(cfg["evaluation"]["leakage_alarm_accuracy"])
    seen, bad = 0, []
    for f in files:
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            for path, v in _walk(rec):
                if _is_val_accuracy(path) and isinstance(v, (int, float)) and not isinstance(v, bool):
                    seen += 1
                    if v > alarm:
                        bad.append(f"{f.name}:{n} {path}={v:.4f}")
    if seen == 0:
        pytest.skip("experiments/*.jsonl contain no validation-accuracy fields")
    assert not bad, f"validation accuracy above {alarm} (suspected leakage): {bad[:20]}"


@pytest.mark.realdata
@pytest.mark.slow
def test_model_and_elo_validation_accuracy(features_table: pd.DataFrame, cfg: dict) -> None:
    feats = model_features(cfg)
    splits = import_or_skip(SPLITS_MODULE, SPLITS_SKIP)
    xgb = pytest.importorskip("xgboost")
    alarm = float(cfg["evaluation"]["leakage_alarm_accuracy"])
    parts = splits.time_split(features_table, "dev", cfg)
    train, val = parts["train"], parts["val"]
    yv = val["y"].astype(int).to_numpy()

    elo_acc = float(((val["elo_prob_a"].to_numpy() > 0.5).astype(int) == yv).mean())
    assert elo_acc <= alarm, f"Elo-only validation accuracy {elo_acc:.4f} > {alarm}: suspected leakage"

    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        random_state=int(cfg["seed"]), n_jobs=4,
    )
    model.fit(train[feats].astype(float), train["y"].astype(int))
    p = model.predict_proba(val[feats].astype(float))[:, 1]
    acc = float(((p > 0.5).astype(int) == yv).mean())
    assert acc <= alarm, f"XGBoost dev validation accuracy {acc:.4f} > {alarm}: suspected leakage"
    assert np.isfinite(p).all()
