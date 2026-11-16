"""Test 17 (Phase 5 audit): Elo tuning never saw anything after its scoring window (PLAN Phase 4, section 2 rule 6).

The Phase 4 harness walks every match 1990-2026 but scores only [elo_harness.tune_start, tune_end] (2015-2024).
- Logged runs: every line of experiments/elo_tuning.jsonl has window == the configured window, and only seasons
  inside it.
- Real data: the harness's scored set lies inside the window, and its scored probabilities are IDENTICAL when every
  match dated after tune_end is removed from the input. So no match after 2024 (in particular no test period:
  Wimbledon 2025, 2026-04-01..06-07) can have influenced the selection.
- The configured Elo settings are the best logged 2015-2024 log loss per flavour (selection used only that window).

Catches: a scoring window that reaches a test period; a harness whose 2015-2024 numbers depend on later matches
(e.g. ratings or K inputs computed over the whole table); config settings chosen on something other than the log.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tennis.config import repo_path
from tests.leakage.helpers import import_or_skip

HARNESS = "tennis.model.elo_harness"
SKIP = "Elo harness not built yet (Phase 4)"


def _records(cfg: dict) -> list[dict]:
    f = Path(repo_path(cfg["paths"]["experiments"])) / "elo_tuning.jsonl"
    if not f.exists():
        pytest.skip("experiments/elo_tuning.jsonl not logged yet")
    return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_logged_tuning_window(cfg: dict) -> None:
    h = cfg["elo_harness"]
    lo, hi = pd.Timestamp(h["tune_start"]), pd.Timestamp(h["tune_end"])
    for n, r in enumerate(_records(cfg), 1):
        assert r["params"]["window"] == [h["tune_start"], h["tune_end"]], f"line {n}: window {r['params']['window']}"
        for y in r["metrics"].get("seasons", {}):
            assert lo.year <= int(y) <= hi.year, f"line {n}: season {y} outside the tuning window"


def test_config_is_the_logged_best(cfg: dict) -> None:
    eh = import_or_skip(HARNESS, SKIP)
    recs = [r for r in _records(cfg) if not r["params"].get("probe")]
    for which in eh.WHICH:
        latest: dict[str, dict] = {}
        for r in recs:
            if r["params"]["which"] == which:
                latest[r["params"]["label"]] = r  # the last run of each setting
        best = min(latest.values(), key=lambda r: (r["metrics"]["val"]["log_loss"], -r["metrics"]["val"]["accuracy"],
                                                   r["params"]["label"]))
        cur = eh.current_setting(cfg, which)
        assert best["params"]["label"] == cur.label, (
            f"{which}: config.elo is {cur.label!r}, but the best logged 2015-2024 setting is "
            f"{best['params']['label']!r}")


@pytest.mark.realdata
@pytest.mark.slow
def test_scored_probabilities_blind_to_later_matches(real_matches: pd.DataFrame, cfg: dict) -> None:
    eh = import_or_skip(HARNESS, SKIP)
    tune_end = pd.Timestamp(cfg["elo_harness"]["tune_end"])
    full = eh.prepare(real_matches, cfg)
    cut = eh.prepare(real_matches[real_matches["tourney_date"] <= tune_end], cfg)
    n = len(cut.match_id)
    assert len(full.match_id) > n, "no matches after tune_end in the data: nothing to check"
    assert (full.match_id[:n] == cut.match_id).all()
    assert not full.score[n:].any(), "matches after tune_end are scored"
    assert (full.score[:n] == cut.score).all()
    dates = real_matches.set_index("match_id")["tourney_date"]
    scored = pd.to_datetime(dates.loc[full.match_id[full.score]].to_numpy())
    assert scored.max() <= tune_end and scored.min() >= pd.Timestamp(cfg["elo_harness"]["tune_start"])
    for which in eh.WHICH:
        s = eh.current_setting(cfg, which)
        p_full = eh.scored_win_prob(full, s)
        p_cut = eh.scored_win_prob(cut, s)
        assert np.array_equal(p_full, p_cut), f"{which}: scored Elo probabilities depend on matches after tune_end"
