"""Metric correctness, bootstrap determinism, the leakage alarm and the experiment logger (tennis.model.evaluate)."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from tennis.config import config_hash
from tennis.model import evaluate as ev

CFG = {
    "seed": 42,
    "evaluation": {"calibration_bins": 10, "bootstrap_samples": 500, "ci_level": 0.95,
                   "leakage_alarm_accuracy": 0.75},
    "data": {"surface_map": {"Hard": "Hard", "Clay": "Clay", "Grass": "Grass", "Carpet": "Hard"},
             "training_levels": ["A", "G", "M", "F", "D", "O"],
             "round_order": ["Q1", "Q2", "Q3", "Q4", "R128", "R64", "R32", "R16", "ER", "RR", "QF", "SF", "BR", "F"]},
    "paths": {"experiments": "experiments"},
}


# ------------------------------------------------------------------------------------------- point metrics
def test_accuracy_hand_computed_with_half_credit_for_no_pick() -> None:
    y = [1, 0, 1, 0, 1]
    p = [0.9, 0.2, 0.3, 0.6, 0.5]          # right, right, wrong, wrong, no pick
    assert ev.accuracy(y, p) == pytest.approx((1 + 1 + 0 + 0 + 0.5) / 5)
    assert ev.correct(y, p).tolist() == [1.0, 1.0, 0.0, 0.0, 0.5]


def test_log_loss_hand_computed() -> None:
    assert ev.log_loss([1, 0], [0.8, 0.4]) == pytest.approx(-(math.log(0.8) + math.log(0.6)) / 2)
    assert ev.log_loss([1, 0, 1], [0.5, 0.5, 0.5]) == pytest.approx(math.log(2))


def test_log_loss_is_clipped() -> None:
    v = ev.log_loss([1], [0.0])
    assert math.isfinite(v) and v == pytest.approx(-math.log(ev.LOG_LOSS_EPS))
    assert ev.log_loss([1], [1.0]) == pytest.approx(-math.log(1 - ev.LOG_LOSS_EPS), abs=1e-12)
    assert ev.log_loss([0], [1.0], eps=1e-3) == pytest.approx(-math.log(1e-3))


def test_brier_hand_computed() -> None:
    assert ev.brier([1, 0, 1], [0.8, 0.4, 0.5]) == pytest.approx((0.04 + 0.16 + 0.25) / 3)
    assert ev.brier([1, 0], [1.0, 0.0]) == 0.0


@pytest.mark.parametrize("y,p", [([1, 0], [0.5]), ([1, 2], [0.5, 0.5]), ([1], [1.2]), ([1], [float("nan")]),
                                 ([], [])])
def test_bad_inputs_raise(y, p) -> None:
    with pytest.raises(ValueError):
        ev.accuracy(y, p)


# ------------------------------------------------------------------------------------------ calibration
def test_calibration_table_bins() -> None:
    y = [0, 1, 1, 0, 1, 1]
    p = [0.05, 0.1, 0.15, 0.55, 0.95, 1.0]   # 0.1 sits on an edge -> bin 1 ([lo, hi)); 1.0 -> last bin
    t = ev.calibration_table(y, p, 10)
    assert len(t) == 10 and t["n"].sum() == 6
    assert t.loc[0, "n"] == 1 and t.loc[0, "frac_pos"] == 0.0
    assert t.loc[1, "n"] == 2 and t.loc[1, "mean_pred"] == pytest.approx(0.125) and t.loc[1, "frac_pos"] == 1.0
    assert t.loc[5, "n"] == 1
    assert t.loc[9, "n"] == 2 and t.loc[9, "mean_pred"] == pytest.approx(0.975)
    assert t.loc[3, "n"] == 0 and np.isnan(t.loc[3, "mean_pred"])


def test_expected_calibration_error_hand_computed() -> None:
    t = ev.calibration_table([0, 1, 1, 1], [0.15, 0.15, 0.85, 0.85], 10)
    # bin 1: pred 0.15, obs 0.5 (n 2); bin 8: pred 0.85, obs 1.0 (n 2)
    assert ev.expected_calibration_error(t) == pytest.approx((2 * 0.35 + 2 * 0.15) / 4)


def test_plot_calibration_writes_png(tmp_path) -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 300)
    y = (rng.uniform(0, 1, 300) < p).astype(int)
    out = ev.plot_calibration({"a": (y, p), "b": (y, np.clip(p * 0.8 + 0.1, 0, 1))}, tmp_path / "phase3_x.png",
                              histogram_of="a")
    assert out.exists() and out.stat().st_size > 1000


# -------------------------------------------------------------------------------------------- bootstrap
def _sample(n: int = 400, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(0, 1, n) < p).astype(int)
    return y, p


def test_bootstrap_is_deterministic_for_a_seed() -> None:
    y, p = _sample()
    a = ev.bootstrap_ci(y, p, cfg=CFG)
    b = ev.bootstrap_ci(y, p, cfg=CFG)
    assert a == b
    c = ev.bootstrap_ci(y, p, cfg={**CFG, "seed": 43})
    assert a != c


def test_bootstrap_independent_of_chunking_and_uses_cfg_count() -> None:
    y, p = _sample()
    a = ev.bootstrap_ci(y, p, ["brier"], cfg=CFG, chunk=500)
    b = ev.bootstrap_ci(y, p, ["brier"], cfg=CFG, chunk=37)
    # same RNG stream drawn in different chunk sizes gives the same resamples
    assert a == b


def test_bootstrap_ci_brackets_point_and_is_degenerate_when_constant() -> None:
    y, p = _sample()
    ci = ev.bootstrap_ci(y, p, cfg=CFG)
    for m, (lo, hi) in ci.items():
        assert lo <= ev.ROW_METRICS[m](y, p).mean() <= hi and lo < hi
    lo, hi = ev.bootstrap_ci([1, 0, 1], [0.9, 0.1, 0.8], ["accuracy"], cfg=CFG)["accuracy"]
    assert lo == hi == 1.0


def test_paired_diff_of_identical_predictions_is_zero() -> None:
    y, p = _sample()
    assert ev.paired_bootstrap_diff(y, p, p, "log_loss", cfg=CFG) == (0.0, 0.0, 0.0)
    d, lo, hi = ev.paired_bootstrap_diff(y, p, np.full_like(p, 0.5), "log_loss", cfg=CFG)
    assert d == pytest.approx(ev.log_loss(y, p) - math.log(2)) and lo <= d <= hi


def test_score_keys() -> None:
    y, p = _sample()
    s = ev.score(y, p, CFG)
    assert set(s) == {"n", "accuracy", "log_loss", "brier", "accuracy_ci", "log_loss_ci", "brier_ci"}
    assert set(ev.score(y, p, CFG, probabilistic=False)) == {"n", "accuracy", "accuracy_ci"}


# ------------------------------------------------------------------------------------ augmentation guard
def _frame(n: int = 60) -> pd.DataFrame:
    y, p = _sample(n, seed=5)
    surf = np.array(["Hard", "Clay", "Grass"])[np.arange(n) % 3]
    lvl = np.array(["A", "G", "M"])[np.arange(n) % 3]
    df = pd.DataFrame({"match_id": [f"m{i}" for i in range(n)], "y": y, "p": p,
                       "date": pd.Timestamp("2023-01-02") + pd.to_timedelta(np.arange(n) * 7, "D"),
                       "round_rank": np.where(np.arange(n) % 2 == 0, 4, 13)})
    for s in ("Hard", "Clay", "Grass"):
        df[f"surface_{s}"] = (surf == s).astype(int)
    for lv in CFG["data"]["training_levels"]:
        df[f"level_{lv}"] = (lvl == lv).astype(int)
    return df


def test_evaluate_frame_rejects_augmented_rows() -> None:
    df = _frame()
    ok = ev.evaluate_frame(df.assign(is_flipped=0), df["p"], CFG)
    assert ok["n"] == len(df)
    bad = df.assign(is_flipped=[0] * (len(df) - 1) + [1])
    with pytest.raises(ValueError, match="flipped"):
        ev.evaluate_frame(bad, df["p"], CFG)
    with pytest.raises(ValueError, match="flipped"):
        ev.breakdown(bad, df["p"], bad["y"], CFG)
    with pytest.raises(ValueError):
        ev.evaluate_frame(df, df["p"].iloc[:-1], CFG)


def test_context_labels_and_breakdown() -> None:
    df = _frame()
    ctx = ev.context_labels(df, CFG)
    assert ctx["surface"].tolist()[:3] == ["Hard", "Clay", "Grass"]
    assert ctx["level"].tolist()[:3] == ["A", "G", "M"]
    assert ctx["round"].tolist()[:2] == ["R128", "F"]
    assert (ctx["season"] >= 2023).all()
    t = ev.breakdown(df, df["p"], ctx["surface"], CFG, order=["Hard", "Clay", "Grass", "Carpet"])
    assert t["group"].tolist() == ["Hard", "Clay", "Grass"]
    hard = (ctx["surface"] == "Hard").to_numpy()
    assert t.loc[0, "n"] == hard.sum()
    assert t.loc[0, "accuracy"] == pytest.approx(ev.accuracy(df["y"][hard], df["p"][hard]))
    assert t.loc[0, "log_loss"] == pytest.approx(ev.log_loss(df["y"][hard], df["p"][hard]))


# ------------------------------------------------------------------------------------------- leakage alarm
def test_leakage_alarm_triggers_above_threshold() -> None:
    with pytest.raises(ev.LeakageAlarm, match="suspected leakage"):
        ev.leakage_alarm(0.7501, CFG)
    with pytest.raises(ev.LeakageAlarm):
        ev.leakage_alarm(0.9, CFG, what="xgb val accuracy")
    assert ev.leakage_alarm(0.75, CFG) is False        # strictly greater than the threshold
    assert ev.leakage_alarm(0.64, CFG) is False
    with pytest.warns(RuntimeWarning):
        assert ev.leakage_alarm(0.8, CFG, raise_on_alarm=False) is True


def test_leakage_alarm_reads_threshold_from_config() -> None:
    cfg = {**CFG, "evaluation": {**CFG["evaluation"], "leakage_alarm_accuracy": 0.6}}
    with pytest.raises(ev.LeakageAlarm):
        ev.leakage_alarm(0.65, cfg)


# ------------------------------------------------------------------------------------------ experiment log
def test_log_experiment_appends_json_lines(tmp_path) -> None:
    metrics = {"val": {"n": np.int64(10), "accuracy": np.float64(0.6), "accuracy_ci": (0.5, 0.7),
                       "log_loss": float("nan")}}
    r1 = ev.log_experiment("unit", split="dev", params={"k": np.int32(3), "flag": np.bool_(True)},
                           metrics=metrics, cfg=CFG, out_dir=tmp_path)
    ev.log_experiment("unit", split="dev", params={"k": 4}, metrics={"val": {"accuracy": 0.61}}, cfg=CFG,
                      out_dir=tmp_path)
    lines = (tmp_path / "unit.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert set(rec) == {"timestamp", "git_commit", "config_hash", "split", "params", "metrics"}
    assert rec["config_hash"] == config_hash(CFG) == r1["config_hash"]
    assert rec["split"] == "dev" and rec["params"] == {"k": 3, "flag": True}
    assert rec["metrics"]["val"] == {"n": 10, "accuracy": 0.6, "accuracy_ci": [0.5, 0.7], "log_loss": None}
    assert rec["git_commit"] is None or len(rec["git_commit"]) == 40


# ---------------------------------------------------------------------------------------------- baselines
def test_rank_rule() -> None:
    df = pd.DataFrame({"a_rank": [1, 50, np.nan, 10, np.nan, 7], "b_rank": [5, 2, 30, np.nan, np.nan, 7]})
    assert ev.rank_rule_prob(df).tolist() == [1.0, 0.0, 0.0, 1.0, 0.5, 0.5]


def test_rank_logistic_fit_on_train_only_and_matches_rule() -> None:
    rng = np.random.default_rng(0)
    n = 20000
    a = rng.integers(1, 300, n).astype(float)
    b = rng.integers(1, 300, n).astype(float)
    true_beta = 0.7
    p = 1 / (1 + np.exp(-true_beta * (np.log(b) - np.log(a))))
    train = pd.DataFrame({"a_rank": a, "b_rank": b, "y": (rng.uniform(0, 1, n) < p).astype(int)})
    train.loc[:9, "a_rank"] = np.nan
    m = ev.RankLogistic.fit(train)
    assert m.beta == pytest.approx(true_beta, abs=0.05)
    assert m.fill_rank == 300.0                                  # largest TRAINING rank (299) + 1
    val = pd.DataFrame({"a_rank": [1.0, 5000.0, np.nan, 3.0], "b_rank": [2.0, 3.0, 4.0, 3.0], "y": [1, 0, 0, 1]})
    pv = m.predict(val)
    assert pv[3] == pytest.approx(0.5)
    assert np.array_equal(ev.correct(val["y"], pv), ev.correct(val["y"], ev.rank_rule_prob(val)))
    with pytest.raises(ValueError, match="flipped"):
        ev.RankLogistic.fit(train.assign(is_flipped=1))
