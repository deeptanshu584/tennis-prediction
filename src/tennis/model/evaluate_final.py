"""Phase 6 (replication, Wimbledon 2025) and Phase 7 (forward, 2026-04-01 .. 2026-06-07) final evaluation.

Protocol (PLAN section 2 rules 5-6, DECISIONS D-028 retrain rule):

1. **Validation step** (always). Fit XGBoost with the chosen parameters on the split's TRAINING rows with early
   stopping on its VALIDATION rows (at most ``xgb.max_n_estimators`` rounds), and score the validation window
   against the Elo / surface Elo / blend / rank baselines. The tree count of the final model is this run's
   ``best_iteration + 1``.
   Without ``--score-test`` the feature table is cut at the split's ``val_end`` BEFORE it is split, so no
   test-period row is loaded at all, and the run stops here (validation report draft only).
2. **Final refit** (``--score-test`` only). :func:`tennis.model.train.refit_fixed` on train+val with that FIXED
   tree count and no early-stopping set. The test rows are only handed to its disjointness guards.
3. **Test scoring** (``--score-test`` only, once). Accuracy (half credit at p = 0.5), log loss and Brier with
   bootstrap CIs, paired bootstrap differences against the baselines, calibration, breakdowns. ONE line is
   appended to ``experiments/<split>.jsonl`` with the metrics under a ``test`` key and the run's config hash. A
   second scoring is refused if that file already holds a test record (leakage test 16 also checks this).

Parameters:
- replication: ``xgb.best_params`` (Phase 5 dev tuning, D-030). The tree count comes from the validation run on
  2025-01-01 .. 2025-06-29 (``xgb.replication_n_estimators`` once frozen in config; the dev count 689 is reported
  as an alternative).
- forward: re-tuned on the 2026 Q1 validation window (``--tune``: Optuna, ``xgb.n_trials_forward`` trials,
  trial log ``experiments/xgb_trials_forward.jsonl``), frozen in config as ``xgb.forward_best_params`` (with
  ``n_estimators`` = best trial's ``best_iteration + 1``). Before freezing, validation-only runs read the best
  trial from the log.
With ``--score-test`` the frozen config values are REQUIRED, and the validation step must reproduce the frozen tree
count exactly (deterministic fits: fixed seed, ``nthread`` pinned).

CLI::

    .venv/Scripts/python -m tennis.model.evaluate_final --split forward --tune [--trials N] [--workers W]
    .venv/Scripts/python -m tennis.model.evaluate_final --split replication|forward            # validation only
    .venv/Scripts/python -m tennis.model.evaluate_final --split replication|forward --score-test
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from tennis.config import config_hash, load_config, repo_path
from tennis.model import dataset
from tennis.model import evaluate as ev
from tennis.model import tune
from tennis.model.splits import time_split
from tennis.model.train import NTHREAD, fit_xgb, refit_fixed, save_model

FINAL_SPLITS: tuple[str, ...] = ("replication", "forward")
TRIALS_LOG: dict[str, str] = {"forward": "xgb_trials_forward"}
FIG_PREFIX: dict[str, str] = {"replication": "phase6", "forward": "phase7"}
REPORT_NAME: dict[str, str] = {"replication": "replication_wimbledon_2025.md", "forward": "forward_test_2026.md"}
REFS: tuple[str, ...] = ("elo", "surface_elo", "elo_blend_50_50", "rank_logistic")
METRICS: tuple[str, ...] = ("accuracy", "log_loss", "brier")
# Published reference points (PLAN section 0; D-029). Not computed on our rows: no CI, not like-for-like.
EXTERNAL_REFS: dict[str, list[tuple[str, float]]] = {
    "replication": [("Original author (published)", 0.663), ("IBM (published)", 0.638),
                    ("Bookmakers (quoted externally, D-029)", 0.72)],
    "forward": [],
}


def _check_split(split: str) -> None:
    if split not in FINAL_SPLITS:
        raise ValueError(f"split must be one of {FINAL_SPLITS}, got {split!r}")


# ------------------------------------------------------------------------------------------------ data
def pre_test_frame(df: pd.DataFrame, split: str, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Rows of ``df`` dated on or before the split's ``val_end`` (nothing from the test period or later)."""
    _check_split(split)
    end = pd.Timestamp(cfg["splits"][split]["val_end"])
    return df.loc[pd.to_datetime(df["date"]) <= end]


def partitions(cfg: Mapping[str, Any], split: str, *, df: pd.DataFrame | None = None,
               include_test: bool = False) -> dict[str, pd.DataFrame]:
    """``train`` / ``val`` (and ``test`` when ``include_test``) of a final split.

    Without ``include_test`` the table is cut at ``val_end`` first, so test-period rows are never loaded into a
    partition. With it, a non-empty ``test`` partition is required.
    """
    _check_split(split)
    df = dataset.load_features(cfg) if df is None else df
    if not include_test:
        parts = time_split(pre_test_frame(df, split, cfg), split, cfg)
        test = parts.pop("test", None)
        if test is not None and len(test):
            raise RuntimeError(f"{split}: {len(test)} test rows survived the val_end cut")
        return parts
    parts = time_split(df, split, cfg)
    if "test" not in parts or not len(parts["test"]):
        raise ValueError(f"{split}: no test rows")
    return parts


# ------------------------------------------------------------------------------------------------ tuning
def tune_forward(cfg: Mapping[str, Any], *, df: pd.DataFrame | None = None, n_trials: int | None = None,
                 workers: int | None = None, budget_min: float | None = None, log_name: str | None = None,
                 out_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Phase 7 re-tuning: Optuna over ``xgb.search_space`` on the FORWARD split's train / val windows only
    (early stopping and selection on 2026 Q1). Logs every trial to ``experiments/xgb_trials_forward.jsonl``
    (validation metrics only). The table is cut at ``val_end`` before splitting: test rows are never loaded."""
    df = dataset.load_features(cfg) if df is None else df
    data = tune.prepare_data(cfg, "forward", df=pre_test_frame(df, "forward", cfg))
    n = int(n_trials if n_trials is not None else cfg["xgb"]["n_trials_forward"])
    if n > int(cfg["xgb"]["n_trials_forward"]):
        raise ValueError(f"at most xgb.n_trials_forward = {cfg['xgb']['n_trials_forward']} trials")
    _, records = tune.run_study(cfg, data=data, n_trials=n, workers=workers, budget_min=budget_min,
                                log_name=log_name if log_name is not None else TRIALS_LOG["forward"],
                                out_dir=str(out_dir) if out_dir is not None else None)
    return records


def read_trials(cfg: Mapping[str, Any], split: str = "forward", out_dir: str | Path | None = None
                ) -> list[dict[str, Any]]:
    """Trial records of ``experiments/<TRIALS_LOG[split]>.jsonl`` in :func:`tennis.model.tune.run_study`'s
    record format (``number``, ``params``, ``best_iteration``, ``val``)."""
    d = Path(out_dir) if out_dir is not None else repo_path(cfg["paths"]["experiments"])
    path = d / f"{TRIALS_LOG[split]}.jsonl"
    if not path.exists():
        return []
    space = list(cfg["xgb"]["search_space"])
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("split") != split:
            continue
        out.append({"number": int(r["params"]["trial"]), "params": {k: r["params"][k] for k in space},
                    "best_iteration": int(r["metrics"]["best_iteration"]), "val": r["metrics"]["val"],
                    "config_hash": r.get("config_hash")})
    return out


# ------------------------------------------------------------------------------------------------ parameters
def model_params(cfg: Mapping[str, Any], split: str, *, require_frozen: bool = False,
                 trials_dir: str | Path | None = None) -> tuple[dict[str, Any], int | None, str]:
    """``(params, frozen tree count or None, source)`` for a final split.

    ``params`` never contains ``n_estimators`` (the validation run is capped by ``xgb.max_n_estimators`` only).
    - replication: ``xgb.best_params``; frozen tree count ``xgb.replication_n_estimators`` (if set).
    - forward: ``xgb.forward_best_params`` (with its ``n_estimators``) if set; otherwise, only when
      ``require_frozen`` is False, the best trial of ``experiments/xgb_trials_forward.jsonl``.
    ``require_frozen`` (test scoring) raises ``ValueError`` unless the parameters AND the tree count come from
    config, so the config hash of the test record covers every setting.
    """
    _check_split(split)
    x = cfg["xgb"]
    if split == "replication":
        params = {k: v for k, v in x["best_params"].items() if k != "n_estimators"}
        n = x.get("replication_n_estimators")
        if require_frozen and n is None:
            raise ValueError("xgb.replication_n_estimators is not set: freeze the validation tree count in "
                             "config before scoring the test")
        return params, (int(n) if n is not None else None), "xgb.best_params (Phase 5 dev tuning, D-030)"
    fp = x.get("forward_best_params")
    if fp:
        n = fp.get("n_estimators")
        if require_frozen and n is None:
            raise ValueError("xgb.forward_best_params has no n_estimators")
        return ({k: v for k, v in fp.items() if k != "n_estimators"}, (int(n) if n is not None else None),
                "xgb.forward_best_params (Phase 7 re-tuning on 2026 Q1)")
    if require_frozen:
        raise ValueError("xgb.forward_best_params is not set: run --tune and freeze the best trial in config "
                         "before scoring the test")
    recs = read_trials(cfg, "forward", trials_dir)
    if not recs:
        raise ValueError("no forward trials logged and no xgb.forward_best_params: run --tune first")
    best = tune.best_record(recs)
    return (dict(best["params"]), int(best["best_iteration"]) + 1,
            f"best of {len(recs)} logged forward trials (trial {best['number']}; not yet frozen in config)")


# ------------------------------------------------------------------------------------------------ scoring
def assess(frame: pd.DataFrame, p: np.ndarray, base: Mapping[str, np.ndarray], cfg: Mapping[str, Any], *,
           refs: tuple[str, ...] = REFS) -> dict[str, Any]:
    """Model metrics with bootstrap CIs, every baseline's metrics, and paired bootstrap differences
    ``model − baseline`` (same rows and resamples) for each baseline in ``refs``."""
    y = frame["y"].to_numpy()
    out: dict[str, Any] = {"model": ev.evaluate_frame(frame, p, cfg),
                           "baselines": {k: ev.evaluate_frame(frame, v, cfg, probabilistic=ev.PROBABILISTIC[k])
                                         for k, v in base.items()}}
    out["diffs"] = {k: {m: list(ev.paired_bootstrap_diff(y, p, base[k], m, cfg=cfg)) for m in METRICS}
                    for k in refs}
    return out


def calibration(y: np.ndarray, curves: Mapping[str, np.ndarray], n_bins: int) -> dict[str, Any]:
    """Per named prediction vector: ECE and the bin table (records)."""
    out = {}
    for name, p in curves.items():
        t = ev.calibration_table(y, p, n_bins)
        out[name] = {"ece": ev.expected_calibration_error(t), "table": t.to_dict(orient="records")}
    return out


def breakdowns(frame: pd.DataFrame, p: np.ndarray, base: Mapping[str, np.ndarray], cfg: Mapping[str, Any],
               dims: tuple[str, ...]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """``{dim: {"xgb"|"elo"|"blend": [row per group]}}`` with an accuracy CI per group."""
    ctx = ev.context_labels(frame, cfg)
    orders = {"round": list(cfg["data"]["round_order"]), "level": list(cfg["data"]["training_levels"]),
              "surface": sorted(set(cfg["data"]["surface_map"].values()))}
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for d in dims:
        out[d] = {k: ev.breakdown(frame, pr, ctx[d], cfg, order=orders.get(d)).to_dict(orient="records")
                  for k, pr in (("xgb", p), ("elo", base["elo"]), ("blend", base["elo_blend_50_50"]))}
    return out


def _window(df: pd.DataFrame) -> list[str]:
    d = pd.to_datetime(df["date"])
    return [str(d.min().date()), str(d.max().date())]


def validate(cfg: Mapping[str, Any], split: str, parts: Mapping[str, pd.DataFrame],
             params: Mapping[str, Any]) -> dict[str, Any]:
    """Validation step: fit on train with early stopping on val (cap ``xgb.max_n_estimators``), score val
    against the baselines (rank logistic fitted on train). ``n_trees`` = ``best_iteration + 1``."""
    train, val = parts["train"], parts["val"]
    spec = cfg["splits"][split]
    vd = pd.to_datetime(val["date"])
    if bool((vd < pd.Timestamp(spec["val_start"])).any() or (vd > pd.Timestamp(spec["val_end"])).any()):
        raise ValueError(f"{split}: validation rows outside [val_start, val_end]")
    if "test" in parts and set(val["match_id"]) & set(parts["test"]["match_id"]):
        raise ValueError(f"{split}: a test row is in the early-stopping set")
    vparams = {k: v for k, v in params.items() if k != "n_estimators"}
    fit = fit_xgb(train, val, vparams, cfg, split_name=split)
    ev.leakage_alarm(fit.val_metrics["accuracy"], cfg, what=f"{split} validation accuracy")
    base, rl = ev.baseline_predictions(train, val)
    res = assess(val, fit.val_pred, base, cfg)
    y = val["y"].to_numpy()
    res.update(n_trees=fit.best_iteration + 1, best_iteration=fit.best_iteration, rank_beta=rl.beta,
               calibration=calibration(y, {"xgb": fit.val_pred, "elo": base["elo"],
                                           "blend": base["elo_blend_50_50"]},
                                       int(cfg["evaluation"]["calibration_bins"])),
               _fit=fit, _base=base, _y=y)
    return res


def _public(d: Mapping[str, Any]) -> dict[str, Any]:
    """Drop the private (``_``-prefixed) objects before logging."""
    return {k: v for k, v in d.items() if not k.startswith("_")}


def already_scored(cfg: Mapping[str, Any], split: str, exp_dir: str | Path | None = None) -> bool:
    """True if ``experiments/<split>.jsonl`` already holds a record with test metrics."""
    d = Path(exp_dir) if exp_dir is not None else repo_path(cfg["paths"]["experiments"])
    path = d / f"{split}.jsonl"
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and "test" in (json.loads(line).get("metrics") or {}):
            return True
    return False


def score_test(cfg: Mapping[str, Any], split: str, parts: Mapping[str, pd.DataFrame], params: Mapping[str, Any],
               n_trees: int, *, model_dir: str | Path | None = None) -> dict[str, Any]:
    """Final refit on train+val with ``n_trees`` fixed trees (no early stopping), then ONE prediction of the
    test rows and every test metric. Does not log; :func:`run_final` logs the result once."""
    train, val, test = parts["train"], parts["val"], parts["test"]
    dataset.assert_not_augmented(test)
    refit_rows = pd.concat([train, val], ignore_index=True)
    fit = refit_fixed(refit_rows, params, cfg, n_trees=n_trees, split_name=split, test_df=test)
    p = fit.predict(test)
    base, rl = ev.baseline_predictions(refit_rows, test)
    out: dict[str, Any] = {"overall": assess(test, p, base, cfg)}
    acc = out["overall"]["model"]["accuracy"]
    out["leakage_alarm"] = ev.leakage_alarm(acc, cfg, what=f"{split} test accuracy", raise_on_alarm=False)
    lo, hi = out["overall"]["model"]["accuracy_ci"]
    out["accuracy_ci_width"] = hi - lo
    y = test["y"].to_numpy()
    out["calibration"] = calibration(y, {"xgb": p, "elo": base["elo"], "blend": base["elo_blend_50_50"]},
                                     int(cfg["evaluation"]["calibration_bins"]))
    dims = ("round",) if split == "replication" else ("round", "surface", "level")
    out["by"] = breakdowns(test, p, base, cfg, dims)
    out["tourneys"] = {}
    for name in cfg["splits"][split].get("report_tourneys", []) or []:
        m = (test["tourney_name"] == name).to_numpy()
        if m.any():
            sub = assess(test.loc[m], p[m], {k: v[m] for k, v in base.items()}, cfg)
            lo, hi = sub["model"]["accuracy_ci"]
            sub["accuracy_ci_width"] = hi - lo
            out["tourneys"][name] = sub
    out["rank_beta"] = rl.beta
    out["refit"] = dict(fit.extra)
    model_path = save_model(fit, cfg, f"xgb_{split}_final", out_dir=model_dir)
    pred = test[[c for c in ("match_id", "date", "tourney_name", "a_name", "b_name", "y") if c in test]].copy()
    pred["p_xgb"] = p
    for k, v in base.items():
        pred[f"p_{k}"] = v
    pred.to_csv(model_path.with_name(model_path.stem + "_test_predictions.csv"), index=False)
    out["model_path"] = model_path.name
    out["_p"], out["_base"], out["_y"] = p, base, y
    return out


def run_final(cfg: Mapping[str, Any], split: str, *, df: pd.DataFrame | None = None, score: bool = False,
              exp_dir: str | Path | None = None, model_dir: str | Path | None = None,
              report_dir: str | Path | None = None, fig_dir: str | Path | None = None,
              write: bool = True) -> dict[str, Any]:
    """Validation step, and with ``score=True`` the final refit + ONE test scoring logged to
    ``experiments/<split>.jsonl``. Writes the report (a validation-only draft without ``score``) when
    ``write``. Without ``score`` no test-period row is loaded."""
    _check_split(split)
    params, frozen_n, source = model_params(cfg, split, require_frozen=score, trials_dir=exp_dir)
    if score and already_scored(cfg, split, exp_dir):
        raise RuntimeError(f"the {split} test has already been scored (experiments/{split}.jsonl has a test "
                           "record); the test set is evaluated once")
    parts = partitions(cfg, split, df=df, include_test=score)
    v = validate(cfg, split, parts, params)
    if frozen_n is not None and frozen_n != v["n_trees"]:
        msg = (f"{split}: the validation run gives {v['n_trees']} trees but the frozen/selected count is "
               f"{frozen_n}")
        if score:
            raise RuntimeError(msg + " (fits are deterministic; check params, nthread and the data)")
        print("WARNING:", msg)
    n_trees = v["n_trees"]
    out: dict[str, Any] = {
        "split": split, "params": dict(params), "params_source": source, "n_trees": n_trees,
        "frozen_n_trees": frozen_n, "dev_n_estimators": int(cfg["xgb"]["best_params"].get("n_estimators", 0)),
        "windows": {"train_window": _window(parts["train"]), "val_window": _window(parts["val"])},
        "sizes": {"n_train": len(parts["train"]), "n_val": len(parts["val"])},
        "val": v, "test": None, "config_hash": config_hash(dict(cfg)), "record": None,
    }
    if score:
        t = score_test(cfg, split, parts, params, n_trees, model_dir=model_dir)
        out["test"] = t
        out["windows"]["test_window"] = _window(parts["test"])
        out["windows"]["refit_window"] = t["refit"]["refit_window"]
        out["sizes"]["n_test"] = len(parts["test"])
        out["record"] = ev.log_experiment(
            split, split=split, cfg=cfg, out_dir=exp_dir,
            params={"model_params": dict(params), "params_source": source, "n_trees": n_trees,
                    "tree_count_source": "validation run best_iteration + 1", "early_stopping_in_refit": False,
                    "nthread": NTHREAD, "model_path": t["model_path"], **out["windows"], **out["sizes"]},
            metrics={"val": _public(v), "test": _public(t)})
    if write:
        write_report(out, cfg, report_dir=report_dir, fig_dir=fig_dir, exp_dir=exp_dir)
    return out


# ------------------------------------------------------------------------------------------------ report
_TITLES = {"replication": "Phase 6 — Replication backtest: {test_tourney} {test_year}",
           "forward": "Phase 7 — Forward test: {test_start} → {test_end}"}
_ROW_LABELS = {"model": "**XGBoost (final)**", **ev.BASELINE_LABELS}


def _metric_rows(a: Mapping[str, Any]) -> list[list[str]]:
    rows = []
    for key, s in [("model", a["model"])] + list(a["baselines"].items()):
        rows.append([_ROW_LABELS[key], f"{s['n']:,}", ev._fmt_ci(s, "accuracy"), ev._fmt_ci(s, "log_loss"),
                     ev._fmt_ci(s, "brier")])
    return rows


def _diff_rows(a: Mapping[str, Any]) -> list[list[str]]:
    rows = []
    for k, d in a["diffs"].items():
        cells = [ev.BASELINE_LABELS[k]]
        for m in ("log_loss", "brier", "accuracy"):
            v, lo, hi = d[m]
            cells.append(f"{v:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        rows.append(cells)
    return rows


def _assessment_md(a: Mapping[str, Any], what: str) -> list[str]:
    md = ev._md_table
    return [md(["Model", "n", "Accuracy", "Log loss", "Brier"], _metric_rows(a)) + "\n",
            f"**Paired differences on {what}, XGBoost − baseline** (paired bootstrap; negative log loss / Brier and "
            "positive accuracy favour XGBoost):\n",
            md(["Baseline", "Δ log loss", "Δ Brier", "Δ accuracy"], _diff_rows(a)) + "\n"]


def _breakdown_md(by: Mapping[str, Any], dim: str) -> str:
    rows = []
    xs, es, bs = by[dim]["xgb"], by[dim]["elo"], by[dim]["blend"]
    for x_, e_, b_ in zip(xs, es, bs):
        lo, hi = x_["accuracy_ci"]
        rows.append([x_["group"], f"{int(x_['n']):,}", f"{x_['accuracy']:.3f} [{lo:.3f}, {hi:.3f}]",
                     f"{e_['accuracy']:.3f}", f"{b_['accuracy']:.3f}", f"{x_['log_loss']:.4f}",
                     f"{e_['log_loss']:.4f}", f"{b_['log_loss']:.4f}"])
    return ev._md_table([dim.capitalize(), "n", "XGB acc [95% CI]", "Elo acc", "Blend acc", "XGB log loss",
                         "Elo log loss", "Blend log loss"], rows) + "\n"


def _plot_trials(records: list[dict[str, Any]], refs: Mapping[str, float], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rs = sorted(records, key=lambda r: r["number"])
    x = [r["number"] for r in rs]
    y = [r["val"]["log_loss"] for r in rs]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.scatter(x, y, s=10, color=ev.SERIES_COLORS[0], alpha=0.45, label="trial")
    ax.plot(x, np.minimum.accumulate(y), color=ev.SERIES_COLORS[0], lw=2, label="best so far")
    for i, (name, v) in enumerate(refs.items()):
        ax.axhline(v, color=ev.SERIES_COLORS[1 + i], lw=1.5, ls="--", label=f"{name} ({v:.4f})")
    ax.set_ylim(min(min(y), *refs.values()) - 0.003, max(refs.values()) + 0.01)
    ax.set(xlabel="trial", ylabel="validation log loss (2026 Q1)")
    ax.set_title("Phase 7 re-tuning on the forward split (Optuna TPE)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color="#e6e6e3")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(out: Mapping[str, Any], cfg: Mapping[str, Any], *, report_dir: str | Path | None = None,
                 fig_dir: str | Path | None = None, exp_dir: str | Path | None = None) -> Path:
    """Write ``reports/replication_wimbledon_2025.md`` or ``reports/forward_test_2026.md`` (a validation-only
    draft until the test is scored) and the ``phase6_*`` / ``phase7_*`` figures."""
    split = out["split"]
    spec = cfg["splits"][split]
    prefix = FIG_PREFIX[split]
    figs = Path(fig_dir) if fig_dir is not None else repo_path(cfg["paths"]["figures"])
    rdir = Path(report_dir) if report_dir is not None else repo_path(cfg["paths"]["reports"])
    rdir.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(figs, rdir).replace("\\", "/")
    nb = int(cfg["evaluation"]["calibration_bins"])
    v, t = out["val"], out["test"]
    md = ev._md_table
    alarm = float(cfg["evaluation"]["leakage_alarm_accuracy"])

    vy, vb = v["_y"], v["_base"]
    ev.plot_calibration({"XGBoost": (vy, v["_fit"].val_pred), "Elo": (vy, vb["elo"]),
                         "50/50 Elo blend": (vy, vb["elo_blend_50_50"])}, figs / f"{prefix}_calibration_val.png",
                        n_bins=nb, histogram_of="XGBoost", title=f"{split} validation: calibration")

    L = [f"# {_TITLES[split].format(**spec)}\n",
         f"Generated by `python -m tennis.model.evaluate_final --split {split}"
         f"{' --score-test' if t is not None else ''}` · config hash `{out['config_hash']}` · git "
         f"`{ev.git_commit() or 'n/a'}`.\n"]
    if t is None:
        L.append("> **DRAFT — validation only. The test set has NOT been scored.** This run cut the feature table at "
                 f"`val_end` ({spec['val_end']}) before splitting, so no test-period row was loaded.\n")
    else:
        L.append(f"> **Single evaluation on the frozen config.** The test was scored exactly once "
                 f"(experiments/{split}.jsonl, one record, config hash `{out['config_hash']}`), after every setting "
                 "was frozen in config.yaml (D-032). Nothing was changed after the test results were seen.\n")

    w, s = out["windows"], out["sizes"]
    L.append("## Protocol\n")
    if split == "replication":
        test_desc = f"{spec['test_tourney']} {spec['test_year']} (main draw, tourney_date after {spec['val_end']})"
    else:
        test_desc = f"{spec['test_start']} → {spec['test_end']} (all training-level matches)"
    L.append(md(["Partition", "Window (tourney_date)", "Rows"], [
        ["Train", f"{w['train_window'][0]} → {w['train_window'][1]}", f"{s['n_train']:,}"],
        ["Validation", f"{w['val_window'][0]} → {w['val_window'][1]}", f"{s['n_val']:,}"],
        ["Test", test_desc + (f"; rows dated {w['test_window'][0]} → {w['test_window'][1]}" if t else ""),
         f"{s['n_test']:,}" if t else "not loaded"],
    ]) + "\n")
    L.append(f"- **Parameters:** {out['params_source']}.\n")
    L.append("```yaml\n" + "\n".join(f"{k}: {float(v_)!r}" if isinstance(v_, float) else f"{k}: {v_}"
                                     for k, v_ in out["params"].items()) + "\n```\n")
    L.append(f"- **Validation step:** fit on train with early stopping on validation "
             f"({cfg['xgb']['early_stopping_rounds']} rounds, ≤ {cfg['xgb']['max_n_estimators']} trees). "
             f"Best iteration {v['best_iteration']} → **{out['n_trees']} trees** for the final model.\n")
    if split == "replication":
        L.append(f"- **Tree count choice:** the validation run's `best_iteration + 1` on the replication validation "
                 f"window ({out['n_trees']}). Alternative: the dev early-stopping count "
                 f"`xgb.best_params.n_estimators = {out['dev_n_estimators']}` (D-030), fitted on 2000–2022 with "
                 "2023–24 for early stopping. We use the count measured closest to the test, on the window the "
                 "final model is refitted through.\n")
    else:
        L.append("- **Tree count:** the best re-tuning trial's `best_iteration + 1`, reproduced exactly by the "
                 "validation step above.\n")
    L.append("- **Final model:** `train.refit_fixed` on train + validation with that FIXED tree count and no "
             "early-stopping set (D-028 retrain rule). It asserts max(refit date) ≤ `val_end`, no `match_id` shared "
             "with the test, test rows dated after every refit row, `maybe_augment_train(…, split_name)` on the "
             "training rows and `assert_not_augmented` on the test. The test is never an early-stopping set. "
             f"`nthread` = {NTHREAD}, `tree_method` = {cfg['xgb'].get('tree_method')}, seed {cfg['seed']}.\n")
    L.append("- **Baselines:** Elo, surface Elo and their 50/50 blend come from the pre-match feature columns; the "
             "rank logistic (D-026) is fitted on the training rows of each step (train for validation, "
             "train + validation for the test). Metrics: accuracy with half credit at p = 0.5, log loss "
             f"(clip {cfg['evaluation']['log_loss_eps']}), Brier; {int(cfg['evaluation']['bootstrap_samples']):,} "
             f"bootstrap resamples, {int(float(cfg['evaluation']['ci_level']) * 100)}% percentile CIs, seed "
             f"{cfg['seed']}.\n")

    if split == "forward":
        recs = read_trials(cfg, "forward", exp_dir)
        L.append(f"## Re-tuning on the validation window {spec['val_start']} → {spec['val_end']}\n")
        if recs:
            best = tune.best_record(recs)
            lls = np.array([r["val"]["log_loss"] for r in recs])
            hashes = sorted({str(r.get("config_hash")) for r in recs})
            L.append(f"Optuna TPE (seed {cfg['seed']}) over `xgb.search_space`, {len(recs)} trials "
                     f"(≤ `xgb.n_trials_forward` = {cfg['xgb']['n_trials_forward']}), trial log "
                     f"`experiments/{TRIALS_LOG['forward']}.jsonl` (validation metrics only; config hash "
                     f"{', '.join(f'`{h}`' for h in hashes)}). Early stopping AND selection both use the 2026 Q1 "
                     f"window ({s['n_val']:,} rows), so the validation numbers are optimistic; the test is the "
                     f"honest check. Best trial **{best['number']}**: log loss {best['val']['log_loss']:.5f}, "
                     f"accuracy {best['val']['accuracy']:.4f}, {best['best_iteration'] + 1} trees. Trial log loss: "
                     f"best {lls.min():.5f}, median {np.median(lls):.5f}, worst {lls.max():.5f}.\n")
            _plot_trials(recs, {"Elo": v["baselines"]["elo"]["log_loss"],
                                "50/50 blend": v["baselines"]["elo_blend_50_50"]["log_loss"],
                                "rank logistic": v["baselines"]["rank_logistic"]["log_loss"]},
                         figs / "phase7_optuna_history.png")
            L.append(f"![history]({rel}/phase7_optuna_history.png)\n")
            space = list(cfg["xgb"]["search_space"])
            top = sorted(recs, key=lambda r: (r["val"]["log_loss"], -r["val"]["accuracy"], r["number"]))[:10]
            L.append(md(["Trial", "Log loss", "Accuracy", "Trees"] + space,
                        [[r["number"], f"{r['val']['log_loss']:.5f}", f"{r['val']['accuracy']:.4f}",
                          r["best_iteration"] + 1] + [f"{r['params'][k]:.3g}" for k in space] for r in top]) + "\n")
        else:
            L.append("No forward trials logged yet.\n")

    L.append(f"## Validation {w['val_window'][0]} → {w['val_window'][1]}\n")
    L.append("Caveat: the validation window is also the early-stopping set (and, for the forward split, the "
             "selection set), so these numbers are optimistic. The Elo settings (D-027) were tuned on 2015–2024, "
             "before either validation window.\n")
    L += _assessment_md(v, "validation")
    L.append(f"Validation accuracy {v['model']['accuracy']:.4f} ≤ {alarm} (leakage alarm checked). Calibration "
             f"(ECE, {nb} bins): " + ", ".join(f"{k} {c['ece']:.4f}" for k, c in v["calibration"].items()) + ".\n")
    L.append(f"![validation calibration]({rel}/{prefix}_calibration_val.png)\n")

    title = (f"{spec['test_tourney']} {spec['test_year']}" if split == "replication"
             else f"{spec['test_start']} → {spec['test_end']}")
    L.append(f"## Test: {title}\n")
    if t is None:
        L.append("**Not scored yet.** The test is evaluated once, with `--score-test`, after the settings are frozen "
                 "in config.\n")
    else:
        ty, tb, tp = t["_y"], t["_base"], t["_p"]
        ev.plot_calibration({"XGBoost": (ty, tp), "Elo": (ty, tb["elo"]), "50/50 Elo blend": (ty, tb["elo_blend_50_50"])},
                            figs / f"{prefix}_calibration_test.png", n_bins=nb, histogram_of="XGBoost",
                            title=f"{title}: calibration")
        o = t["overall"]
        L += _assessment_md(o, "the test")
        L.append(f"**Small sample.** n = {o['model']['n']}: the 95% CI of the XGBoost accuracy is "
                 f"[{o['model']['accuracy_ci'][0]:.3f}, {o['model']['accuracy_ci'][1]:.3f}], "
                 f"{t['accuracy_ci_width'] * 100:.1f} points wide (± {t['accuracy_ci_width'] * 50:.1f}). Compare models "
                 "with the paired differences above, not by eye from these marginal CIs.\n")
        if t["leakage_alarm"]:
            L.append(f"**Leakage alarm:** test accuracy {o['model']['accuracy']:.4f} > {alarm}. On this sample size "
                     "it can be chance, but it is treated as leakage until proven otherwise.\n")
        for name, sub in t["tourneys"].items():
            L.append(f"### {name} {pd.Timestamp(spec['test_start']).year if split == 'forward' else ''}\n")
            L += _assessment_md(sub, name)
            L.append(f"**Small sample.** n = {sub['model']['n']}: accuracy CI "
                     f"[{sub['model']['accuracy_ci'][0]:.3f}, {sub['model']['accuracy_ci'][1]:.3f}], "
                     f"{sub['accuracy_ci_width'] * 100:.1f} points wide.\n")
        refs = EXTERNAL_REFS[split]
        if refs:
            L.append("### Published reference points\n")
            rows = [["**This model (XGBoost)**", f"{o['model']['accuracy']:.3f} "
                     f"[{o['model']['accuracy_ci'][0]:.3f}, {o['model']['accuracy_ci'][1]:.3f}]", "our test rows"]]
            rows += [[n, f"{a:.3f}", "quoted, not computed on our rows; no CI"] for n, a in refs]
            L.append(md(["Model", "Accuracy", "Basis"], rows) + "\n")
            L.append("The published figures are on their authors' own match sets (which may include retirements or "
                     "count matches differently); they are not like-for-like comparisons. The bookmaker figure is "
                     "the PLAN.md quote; no odds data are used (D-029).\n")
        else:
            L.append("No bookmaker baseline: no odds data are used (D-029), and there is no published 2026 figure.\n")
        L.append("### Calibration\n")
        L.append("ECE (10 bins): " + ", ".join(f"{k} {c['ece']:.4f}" for k, c in t["calibration"].items())
                 + ". With this few rows per bin the curve is noisy.\n")
        L.append(f"![test calibration]({rel}/{prefix}_calibration_test.png)\n")
        for d in t["by"]:
            L.append(f"### Test by {d}\n")
            L.append(_breakdown_md(t["by"], d))

    L.append("## Disclosures\n")
    L.append("- **D-028 finding 2:** Elo-only accuracies for the FULL 2025 and 2026 seasons (63.71% / 64.91%, untuned "
             "default Elo) appeared earlier in `reports/audits/phase2.md` and in PROGRESS.md. Those seasons contain "
             "Wimbledon 2025 and the forward test window. Nothing was selected on them (Phase 4 tuned on 2015–2024 "
             "only, verified by leakage test 17), but the numbers were seen before this test.\n")
    L.append("- **Elo tuning window:** Elo settings were chosen on 2015–2024 (D-027), before both test periods; it "
             "overlaps the Phase 5 dev validation window (D-004), which affects Elo, the blend and XGBoost alike.\n")
    L.append("- **Selection optimism (D-031):** the Phase 5 audit estimated that about 0.002 of the dev-split "
             "log-loss gain over the blend (−0.012) was selection optimism (about −0.010 survives without early "
             "stopping on the selection window). A test gain over the blend somewhat smaller than the validation "
             "gain is therefore expected"
             + (". The forward parameters were re-selected on 2026 Q1, so the same kind of optimism applies to "
                "its validation numbers.\n" if split == "forward" else ".\n"))
    L.append("- **Retirements:** RET/DEF matches are never labels and walkovers are dropped (D-022), so accuracies "
             "are slightly optimistic relative to all matches played.\n")
    if split == "replication":
        L.append("- **Unassigned rows:** events starting after 2025-06-29 other than Wimbledon 2025 belong to no "
                 "partition of this split (D-026).\n")
        L.append("- **Parameters** were tuned on the dev split (train 2000–2022, validation 2023–24, D-030); the "
                 "replication test played no part in any choice.\n")
    else:
        L.append("- **Shortened forward test (D-017):** the archive ends with Roland Garros 2026; there is no "
                 "June–November 2026 data, so the test covers 2026-04-01 → 2026-06-07 only.\n")
    path = rdir / REPORT_NAME[split]
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def _main_tune(args: argparse.Namespace, cfg: Mapping[str, Any]) -> None:
    for var in tune.BLAS_ENV:
        os.environ.setdefault(var, "1")
    t0 = time.perf_counter()
    recs = tune_forward(cfg, n_trials=args.trials, workers=args.workers, budget_min=args.budget_min)
    best = tune.best_record(recs)
    print(f"{len(recs)} trials in {(time.perf_counter() - t0) / 60:.1f} min; best trial {best['number']}: "
          f"val log loss {best['val']['log_loss']:.5f}, acc {best['val']['accuracy']:.4f}, "
          f"trees {best['best_iteration'] + 1}")
    print(json.dumps(best["params"], indent=2))


def main(argv: list[str] | None = None) -> None:
    """CLI entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description="Phase 6/7 final evaluation")
    ap.add_argument("--split", required=True, choices=FINAL_SPLITS)
    ap.add_argument("--score-test", action="store_true", help="refit on train+val and score the test ONCE")
    ap.add_argument("--tune", action="store_true", help="forward only: re-tune on the 2026 Q1 validation window")
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--budget-min", type=float, default=None)
    args = ap.parse_args(argv)
    cfg = load_config()
    if args.tune:
        if args.split != "forward" or args.score_test:
            ap.error("--tune is for --split forward only, without --score-test")
        _main_tune(args, cfg)
        return
    out = run_final(cfg, args.split, score=args.score_test)
    v = out["val"]
    print(f"{args.split}: params from {out['params_source']}")
    print(f"validation n={v['model']['n']}: acc {v['model']['accuracy']:.4f} ll {v['model']['log_loss']:.5f} "
          f"brier {v['model']['brier']:.5f}; trees {out['n_trees']} (best_iteration {v['best_iteration']})")
    for k, s in v["baselines"].items():
        print(f"  {k:16s} acc {s['accuracy']:.4f}" + (f" ll {s['log_loss']:.5f}" if "log_loss" in s else ""))
    if out["test"] is not None:
        o = out["test"]["overall"]["model"]
        print(f"TEST n={o['n']}: acc {o['accuracy']:.4f} {o['accuracy_ci']} ll {o['log_loss']:.5f} "
              f"brier {o['brier']:.5f}")
    else:
        print("test NOT scored (validation only)")


if __name__ == "__main__":
    main()
