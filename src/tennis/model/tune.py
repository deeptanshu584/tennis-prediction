"""XGBoost hyper-parameter search on the dev split (PLAN section 6 Phase 5, sections 7-8).

- Optuna TPE sampler seeded with ``config.seed`` over ``config.xgb.search_space`` (learning_rate log-uniform,
  max_depth integer, the rest uniform floats); ``xgb.n_trials`` trials; early stopping on validation.
- Dev split ONLY (train 2000-2022, validation 2023-2024). There is no test partition in it, so Wimbledon 2025 and
  the 2026 test window are never loaded into the search.
- Selection: lowest validation log loss, then higher validation accuracy, then lower trial number.
- Every trial is one JSON line in ``experiments/xgb_trials.jsonl`` (params, best_iteration, validation metrics).
- Leakage alarm: a validation accuracy above ``evaluation.leakage_alarm_accuracy`` stops the search
  (:class:`tennis.model.evaluate.LeakageAlarm`).

Parallelism: ``workers`` processes, each training one booster with :data:`tennis.model.train.NTHREAD` threads.
Trials run asynchronously (a new trial is asked as soon as one finishes), so with ``workers > 1`` the TPE
proposals depend on completion order and the trial SEQUENCE is not bit-reproducible; each trial itself is
(fixed seed and thread count), and ``workers=1`` reproduces the whole study exactly.

CLI::

    .venv/Scripts/python -m tennis.model.tune [--trials N] [--workers W] [--budget-min M]
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import optuna
import pandas as pd

from tennis.config import load_config
from tennis.model import dataset
from tennis.model import evaluate as ev
from tennis.model.splits import time_split
from tennis.model.train import NTHREAD, matrix, predict, train_booster, val_metrics

BLAS_ENV: tuple[str, ...] = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
INT_PARAMS: frozenset[str] = frozenset({"max_depth"})
LOG_PARAMS: frozenset[str] = frozenset({"learning_rate"})


def suggest(trial: optuna.trial.BaseTrial, space: Mapping[str, Any]) -> dict[str, Any]:
    """Sample one parameter set from ``xgb.search_space`` (``[low, high]`` per name)."""
    out: dict[str, Any] = {}
    for name, (lo, hi) in space.items():
        if name in INT_PARAMS:
            out[name] = trial.suggest_int(name, int(lo), int(hi))
        else:
            out[name] = trial.suggest_float(name, float(lo), float(hi), log=name in LOG_PARAMS)
    return out


@dataclass
class DevData:
    """Prepared arrays of one split (training rows already passed through ``maybe_augment_train``)."""
    split: str
    features: list[str]
    X_tr: np.ndarray
    y_tr: np.ndarray
    X_va: np.ndarray
    y_va: np.ndarray
    train: pd.DataFrame
    val: pd.DataFrame


def prepare_data(cfg: Mapping[str, Any], split: str = "dev", df: pd.DataFrame | None = None) -> DevData:
    """Split ``df`` (default: features.parquet), guard augmentation, and build the allow-listed arrays."""
    df = dataset.load_features(cfg) if df is None else df
    parts = time_split(df, split, cfg)
    if split == "dev" and "test" in parts:
        raise RuntimeError("the dev split must not have a test partition")
    train, val = parts["train"], parts["val"]
    feats = dataset.feature_columns(cfg, train)
    dataset.assert_not_augmented(val)
    tr = dataset.maybe_augment_train(train, cfg, split)
    return DevData(split, feats, matrix(tr, feats), tr["y"].to_numpy(np.float32), matrix(val, feats),
                   val["y"].to_numpy(np.float32), train, val)


# ------------------------------------------------------------------------------------------ worker
_W: dict[str, Any] = {}


def _init_worker(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray, features: list[str],
                 cfg: Mapping[str, Any]) -> None:
    _W.update(X_tr=X_tr, y_tr=y_tr, X_va=X_va, y_va=y_va, features=features, cfg=cfg)


def _run_trial(number: int, params: dict[str, Any]) -> dict[str, Any]:
    """Train and score one parameter set on the worker's arrays."""
    t0 = time.perf_counter()
    b, best = train_booster(_W["X_tr"], _W["y_tr"], _W["X_va"], _W["y_va"], params, _W["cfg"],
                            feature_names=_W["features"])
    p = predict(b, _W["X_va"])
    return {"number": number, "params": params, "best_iteration": best,
            "val": val_metrics(_W["y_va"], p, _W["cfg"]), "seconds": time.perf_counter() - t0}


# ------------------------------------------------------------------------------------------ study
def best_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Lowest validation log loss, then higher accuracy, then lower trial number."""
    return min(records, key=lambda r: (r["val"]["log_loss"], -r["val"]["accuracy"], r["number"]))


def run_study(cfg: Mapping[str, Any], *, data: DevData | None = None, n_trials: int | None = None,
              workers: int | None = None, budget_min: float | None = None, log_name: str | None = "xgb_trials",
              out_dir: str | None = None) -> tuple[optuna.Study, list[dict[str, Any]]]:
    """Run the search; returns the study and one record per finished trial (in finishing order).

    ``budget_min``: stop ASKING new trials after this many minutes (running ones finish). ``log_name=None``
    disables the JSON log. Raises :class:`tennis.model.evaluate.LeakageAlarm` (after logging the offending trial)
    if a validation accuracy exceeds the alarm threshold.
    """
    data = data or prepare_data(cfg)
    n = int(n_trials if n_trials is not None else cfg["xgb"]["n_trials"])
    w = int(workers if workers is not None else max(1, (os.cpu_count() or 2) - 1))
    space = cfg["xgb"]["search_space"]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=int(cfg["seed"]), multivariate=True, constant_liar=w > 1)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    records: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    window = [str(data.val["date"].min().date()), str(data.val["date"].max().date())]

    def finish(trial: optuna.trial.Trial, rec: dict[str, Any]) -> None:
        study.tell(trial, rec["val"]["log_loss"])
        records.append(rec)
        if log_name:
            ev.log_experiment(log_name, split=data.split, out_dir=out_dir, cfg=cfg,
                              params={**rec["params"], "trial": rec["number"], "nthread": NTHREAD,
                                      "max_n_estimators": int(cfg["xgb"]["max_n_estimators"])},
                              metrics={"val": rec["val"], "best_iteration": rec["best_iteration"],
                                       "seconds": rec["seconds"], "val_window": window})
        ev.leakage_alarm(rec["val"]["accuracy"], cfg, what=f"xgb trial {rec['number']} validation accuracy")

    def out_of_budget() -> bool:
        return budget_min is not None and (time.perf_counter() - t_start) / 60.0 >= budget_min

    if w == 1:
        _init_worker(data.X_tr, data.y_tr, data.X_va, data.y_va, data.features, cfg)
        for _ in range(n):
            if out_of_budget():
                break
            trial = study.ask()
            finish(trial, _run_trial(trial.number, suggest(trial, space)))
        return study, records

    # Spawned workers each import numpy; without this every worker starts one OpenBLAS thread (with its own
    # buffers) per core, which exhausted memory with 11 workers on a 16 GB machine. XGBoost uses NTHREAD anyway.
    for var in BLAS_ENV:
        os.environ.setdefault(var, "1")
    with ProcessPoolExecutor(w, initializer=_init_worker,
                             initargs=(data.X_tr, data.y_tr, data.X_va, data.y_va, data.features, dict(cfg))) as ex:
        inflight: dict[Future, optuna.trial.Trial] = {}
        asked = 0
        try:
            while asked < n or inflight:
                while asked < n and len(inflight) < w and not out_of_budget():
                    trial = study.ask()
                    inflight[ex.submit(_run_trial, trial.number, suggest(trial, space))] = trial
                    asked += 1
                if not inflight:
                    break
                done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
                for fut in done:
                    finish(inflight.pop(fut), fut.result())
                if out_of_budget():
                    asked = n  # stop asking; drain the running trials
        except BaseException:
            for fut in inflight:
                fut.cancel()
            raise
    return study, records


# ------------------------------------------------------------------------------------------ report
def _yaml_params(best: Mapping[str, Any]) -> str:
    lines = ["  best_params:"]
    for k, v in best["params"].items():
        lines.append(f"    {k}: {int(v) if k in INT_PARAMS else repr(float(v))}")
    lines.append(f"    n_estimators: {best['best_iteration'] + 1}   # trees at the dev early-stopping point "
                 "(best_iteration + 1)")
    return "\n".join(lines)


def _plot_importance(imp: pd.DataFrame, path, highlight: set[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = imp.head(30).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 9))
    colors = [ev.SERIES_COLORS[1] if f in highlight else ev.SERIES_COLORS[0] for f in top["feature"]]
    ax.barh(top["feature"], top["gain"], color=colors, edgecolor="white", height=0.78)
    ax.set_xlabel("average gain per split")
    ax.set_title("XGBoost feature importance (gain), top 30" + (" — height features in orange" if
                 any(f in highlight for f in top["feature"]) else ""), loc="left", fontsize=10)
    ax.grid(axis="x", color="#e6e6e3")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8, colors="#6b6b6b")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_history(records: list[dict[str, Any]], refs: Mapping[str, float], path) -> None:
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
    lo = min(min(y), *refs.values())
    ax.set_ylim(lo - 0.002, max(refs.values()) + 0.01)
    ax.set(xlabel="trial", ylabel="validation log loss (2023–2024)")
    ax.set_title("Optuna TPE search on the dev split", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color="#e6e6e3")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_report(cfg: Mapping[str, Any], data: DevData, records: list[dict[str, Any]], *, runtime_s: float,
               workers: int, n_requested: int) -> dict[str, Any]:
    """Refit the best trial, evaluate it against the baselines, write figures, ``experiments/xgb_final.jsonl``,
    the model under ``models/`` and ``reports/xgb_tuning.md``. Returns the key numbers."""
    from tennis.config import config_hash, repo_path
    from tennis.model.train import fit_xgb, save_model

    best = best_record(records)
    t0 = time.perf_counter()
    res = fit_xgb(data.train, data.val, best["params"], cfg, split_name=data.split)
    refit_s = time.perf_counter() - t0
    reproduced = (res.best_iteration == best["best_iteration"]
                  and abs(res.val_metrics["log_loss"] - best["val"]["log_loss"]) < 1e-12)
    if not reproduced:
        raise RuntimeError(f"refit does not reproduce trial {best['number']}: {res.val_metrics} vs {best['val']}")
    model_path = save_model(res, cfg, f"xgb_{data.split}")
    val, p, y = data.val, res.val_pred, data.val["y"].to_numpy()
    ev.leakage_alarm(res.val_metrics["accuracy"], cfg, what="final XGBoost dev validation accuracy")

    xs = ev.evaluate_frame(val, p, cfg)
    base, rl = ev.baseline_predictions(data.train, val)
    bs = {k: ev.evaluate_frame(val, v, cfg, probabilistic=ev.PROBABILISTIC[k]) for k, v in base.items()}
    refs = ("elo", "surface_elo", "elo_blend_50_50", "rank_logistic")
    diffs = {(k, m): ev.paired_bootstrap_diff(y, p, base[k], m, cfg=cfg)
             for k in refs for m in ("log_loss", "accuracy", "brier")}

    ctx = ev.context_labels(val, cfg)
    dims = {"season": sorted(ctx["season"].unique()), "surface": sorted(ctx["surface"].unique()),
            "level": [lv for lv in cfg["data"]["training_levels"] if lv in set(ctx["level"])],
            "round": [r for r in cfg["data"]["round_order"] if r in set(ctx["round"])]}
    bd = {d: {k: ev.breakdown(val, pr, ctx[d], cfg, order=o)
              for k, pr in (("xgb", p), ("elo", base["elo"]), ("blend", base["elo_blend_50_50"]))}
          for d, o in dims.items()}
    for _, r in bd["season"]["xgb"].iterrows():
        ev.leakage_alarm(r["accuracy"], cfg, what=f"XGBoost season {r['group']} accuracy")

    gain = res.booster.get_score(importance_type="gain")
    total = res.booster.get_score(importance_type="total_gain")
    imp = pd.DataFrame({"feature": res.features})
    imp["gain"] = imp["feature"].map(gain).fillna(0.0)
    imp["total_gain"] = imp["feature"].map(total).fillna(0.0)
    imp = imp.sort_values(["gain", "feature"], ascending=[False, True]).reset_index(drop=True)
    imp["rank_gain"] = np.arange(1, len(imp) + 1)
    imp["rank_total_gain"] = imp["total_gain"].rank(ascending=False, method="min").astype(int)
    heights = ["a_height", "b_height", "d_height"]
    h_rows = imp[imp["feature"].isin(heights)]
    drop_height = bool((h_rows["rank_gain"] <= 15).any())

    figs = repo_path(cfg["paths"]["figures"])
    figs.mkdir(parents=True, exist_ok=True)
    nb = int(cfg["evaluation"]["calibration_bins"])
    ev.plot_calibration({"XGBoost": (y, p), "Elo": (y, base["elo"]), "50/50 Elo blend": (y, base["elo_blend_50_50"])},
                        figs / "phase5_calibration_xgb.png", n_bins=nb, histogram_of="XGBoost",
                        title="Dev validation 2023–2024: calibration")
    _plot_importance(imp, figs / "phase5_feature_importance.png", set(heights))
    _plot_history(records, {"Elo": bs["elo"]["log_loss"], "50/50 blend": bs["elo_blend_50_50"]["log_loss"],
                            "rank logistic": bs["rank_logistic"]["log_loss"]}, figs / "phase5_optuna_history.png")

    # shuffled-label sanity check: best params, permuted TRAINING labels
    rng = np.random.default_rng(int(cfg["seed"]))
    y_perm = rng.permutation(data.y_tr)
    b_sh, it_sh = train_booster(data.X_tr, y_perm, data.X_va, data.y_va, best["params"], cfg,
                                feature_names=data.features)
    sh = val_metrics(data.y_va, predict(b_sh, data.X_va), cfg)
    band = cfg["leakage_tests"]["shuffled_label_band"]

    ev.log_experiment("xgb_final", split=data.split, cfg=cfg,
                      params={**best["params"], "run": "best_refit", "trial": best["number"], "nthread": NTHREAD,
                              "model_path": str(model_path.name)},
                      metrics={"val": xs, "best_iteration": res.best_iteration,
                               "diffs_vs": {f"{k}.{m}": list(v) for (k, m), v in diffs.items()}})
    ev.log_experiment("xgb_final", split=data.split, cfg=cfg,
                      params={**best["params"], "run": "shuffled_train_labels", "nthread": NTHREAD},
                      metrics={"val": sh, "best_iteration": it_sh})

    # ---------------------------------------------------------------- markdown
    md = ev._md_table
    L = ["# Phase 5 — XGBoost on the dev split\n",
         f"Generated by `python -m tennis.model.tune` · config hash `{config_hash(dict(cfg))}` · git "
         f"`{ev.git_commit() or 'n/a'}`. Trials: `experiments/xgb_trials.jsonl`; refit and sanity runs: "
         f"`experiments/xgb_final.jsonl`; model: `models/{model_path.name}` (+ `_meta.json`).\n",
         f"**Split:** dev (D-004): train {data.train['date'].min().date()} → {data.train['date'].max().date()} "
         f"({len(data.train):,} rows{'' if len(data.X_tr) == len(data.train) else f', {len(data.X_tr):,} after flip augmentation'}), "
         f"validation {data.val['date'].min().date()} → {data.val['date'].max().date()} ({len(data.val):,} rows). "
         "No test partition exists in this split: Wimbledon 2025 and the 2026 test window were never loaded.\n",
         f"**Inputs:** the {len(data.features)} allow-listed columns of `config.model.features` "
         "(`dataset.feature_columns`); NaN left to XGBoost's native missing handling; no fitted preprocessing. "
         "`maybe_augment_train(train, cfg, 'dev')` and `assert_not_augmented(val)` on every fit (D-024 R2); "
         f"`features.flip_augment_train` = {cfg['features'].get('flip_augment_train', False)}.\n",
         f"**Search:** Optuna TPE (multivariate, seed {cfg['seed']}) over `xgb.search_space`, learning_rate "
         f"log-uniform; {len(records)} of {n_requested} requested trials finished in {runtime_s / 60:.1f} min on "
         f"{workers} processes × {NTHREAD} thread. `binary:logistic`, logloss, `tree_method=hist`, early stopping "
         f"{cfg['xgb']['early_stopping_rounds']} rounds on validation, ≤ {cfg['xgb']['max_n_estimators']} rounds. "
         "Selection: validation log loss, accuracy as tie-breaker. The best trial was refitted and reproduced "
         f"exactly (best iteration {res.best_iteration}, refit {refit_s:.0f}s).\n",
         "**Caveats.** (1) Early stopping AND selection both use the 2023–24 validation window, so the XGBoost "
         "validation numbers are optimistically biased (selection over hundreds of trials on one window); the "
         "honest numbers come from the Phase 6/7 test sets. (2) The Elo settings in the features (D-027) were tuned "
         "on 2015–2024, which includes this validation window (D-004) — this affects Elo, the blend and XGBoost "
         "alike. (3) With parallel asynchronous trials the trial sequence is not bit-reproducible; each trial is. "
         "(4) RET/DEF matches are not labels (D-022), so accuracies are slightly optimistic vs all matches.\n",
         "## Best parameters\n", f"Trial {best['number']}:\n",
         "```yaml\nxgb:\n" + _yaml_params(best) + "\n```\n"]

    L.append("## Validation 2023–2024 vs baselines\n")
    L.append(f"95% percentile bootstrap CIs ({int(cfg['evaluation']['bootstrap_samples']):,} resamples, seed "
             f"{cfg['seed']}).\n")
    rows = [["**XGBoost (tuned)**", f"{xs['n']:,}", ev._fmt_ci(xs, "accuracy"), ev._fmt_ci(xs, "log_loss"),
             ev._fmt_ci(xs, "brier")]]
    for k in ev.BASELINE_LABELS:
        s = bs[k]
        rows.append([ev.BASELINE_LABELS[k], f"{s['n']:,}", ev._fmt_ci(s, "accuracy"), ev._fmt_ci(s, "log_loss"),
                     ev._fmt_ci(s, "brier")])
    rows.append(["Bookmaker (implied, margin removed)", "—", "TODO (Phase 6/7; odds blocked, D-025)", "TODO", "TODO"])
    L.append(md(["Model", "n", "Accuracy", "Log loss", "Brier"], rows) + "\n")
    L.append("**Paired differences, XGBoost − baseline** (same rows, paired bootstrap; negative log loss / Brier and "
             "positive accuracy favour XGBoost):\n")
    rows = []
    for k in refs:
        cells = [ev.BASELINE_LABELS[k]]
        for m, d in (("log_loss", 4), ("brier", 4), ("accuracy", 4)):
            v, lo, hi = diffs[(k, m)]
            cells.append(f"{v:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]")
        rows.append(cells)
    L.append(md(["Baseline", "Δ log loss", "Δ Brier", "Δ accuracy"], rows) + "\n")

    for d, title in (("season", "season"), ("surface", "surface"), ("level", "level"), ("round", "round")):
        L.append(f"### Validation by {title}\n")
        rows = []
        for i, g in enumerate(bd[d]["xgb"]["group"]):
            x_, e_, b_ = bd[d]["xgb"].iloc[i], bd[d]["elo"].iloc[i], bd[d]["blend"].iloc[i]
            lo, hi = x_["accuracy_ci"]
            rows.append([g, f"{int(x_['n']):,}", f"{x_['accuracy']:.3f} [{lo:.3f}, {hi:.3f}]", f"{e_['accuracy']:.3f}",
                         f"{b_['accuracy']:.3f}", f"{x_['log_loss']:.4f}", f"{e_['log_loss']:.4f}",
                         f"{b_['log_loss']:.4f}"])
        L.append(md([title.capitalize(), "n", "XGB acc", "Elo acc", "Blend acc", "XGB log loss", "Elo log loss",
                     "Blend log loss"], rows) + "\n")
    alarm = float(cfg["evaluation"]["leakage_alarm_accuracy"])
    over = [f"{d} {r['group']} ({r['accuracy']:.3f}, n={int(r['n'])})" for d in ("surface", "level", "round")
            for _, r in bd[d]["xgb"].iterrows() if r["accuracy"] > alarm]
    L.append(f"Subgroups above {alarm}: {'; '.join(over) if over else 'none'}. Full-window and per-season "
             f"accuracies are all ≤ {alarm} (checked by `leakage_alarm`).\n")

    L.append("## Feature importance\n")
    L.append("![importance](figures/phase5_feature_importance.png)\n")
    rows = [[int(r.rank_gain), r.feature, f"{r.gain:.2f}", f"{r.total_gain:,.0f}", int(r.rank_total_gain)]
            for r in imp.head(30).itertuples()]
    L.append(md(["Rank (gain)", "Feature", "Gain (avg per split)", "Total gain", "Rank (total gain)"], rows) + "\n")
    unused = imp.loc[imp["gain"] == 0, "feature"].tolist()
    L.append(f"Features never used in a split: {', '.join(unused) if unused else 'none'}.\n")
    hr = "; ".join(f"`{r.feature}` rank {int(r.rank_gain)} by gain, {int(r.rank_total_gain)} by total gain"
                   for r in h_rows.itertuples())
    L.append(f"**Height (D-024 R1):** {hr} (of {len(imp)}). "
             + ("At least one is in the gain top 15: **recommend dropping a_height/b_height/d_height from the "
                "allow-list.**\n" if drop_height else "None is in the gain top 15: keep them (no action under R1).\n"))

    L.append("## Calibration\n")
    L.append("![calibration](figures/phase5_calibration_xgb.png)\n")
    eces = {k: ev.expected_calibration_error(ev.calibration_table(y, v, nb))
            for k, v in (("XGBoost", p), ("Elo", base["elo"]), ("50/50 blend", base["elo_blend_50_50"]))}
    L.append("Expected calibration error (10 bins): " + ", ".join(f"{k} {v:.4f}" for k, v in eces.items()) + ".\n")
    L.append("## Search history\n")
    L.append("![history](figures/phase5_optuna_history.png)\n")
    top10 = sorted(records, key=lambda r: (r["val"]["log_loss"], -r["val"]["accuracy"], r["number"]))[:10]
    space = list(cfg["xgb"]["search_space"])
    rows = [[r["number"], f"{r['val']['log_loss']:.5f}", f"{r['val']['accuracy']:.4f}", r["best_iteration"] + 1]
            + [f"{r['params'][k]:.3g}" for k in space] for r in top10]
    L.append(md(["Trial", "Log loss", "Accuracy", "Trees"] + space, rows) + "\n")
    lls = np.array([r["val"]["log_loss"] for r in records])
    L.append(f"Trial log loss: best {lls.min():.5f}, median {np.median(lls):.5f}, worst {lls.max():.5f}; "
             f"{int((lls < bs['elo']['log_loss']).sum())} of {len(lls)} trials beat Elo-only "
             f"({bs['elo']['log_loss']:.4f}), {int((lls < bs['elo_blend_50_50']['log_loss']).sum())} beat the blend "
             f"({bs['elo_blend_50_50']['log_loss']:.4f}). Mean trial time "
             f"{np.mean([r['seconds'] for r in records]):.1f}s.\n")

    L.append("## Shuffled-label sanity check\n")
    in_band = band[0] <= sh["accuracy"] <= band[1]
    L.append(f"Best parameters retrained on PERMUTED training labels (seed {cfg['seed']}), scored on the real "
             f"validation labels: accuracy **{sh['accuracy']:.4f}**, log loss {sh['log_loss']:.4f} (ln 2 = 0.6931), "
             f"{it_sh + 1} trees before early stopping. Expected band {band}: "
             f"{'PASS' if in_band else 'OUTSIDE BAND — investigate'}.\n")

    L.append("## Acceptance (Phase 5)\n")
    L.append(md(["Check", "Result"], [
        ["Beats Elo-only on val log loss", f"{xs['log_loss']:.4f} vs {bs['elo']['log_loss']:.4f}: "
         f"Δ {diffs[('elo', 'log_loss')][0]:+.4f} [{diffs[('elo', 'log_loss')][1]:+.4f}, "
         f"{diffs[('elo', 'log_loss')][2]:+.4f}]"],
        ["Beats the blend on val log loss", f"{xs['log_loss']:.4f} vs {bs['elo_blend_50_50']['log_loss']:.4f}: "
         f"Δ {diffs[('elo_blend_50_50', 'log_loss')][0]:+.4f} [{diffs[('elo_blend_50_50', 'log_loss')][1]:+.4f}, "
         f"{diffs[('elo_blend_50_50', 'log_loss')][2]:+.4f}]"],
        ["Val accuracy ≤ 0.75", f"{xs['accuracy']:.4f}"],
        ["Shuffled labels ≈ 0.5", f"{sh['accuracy']:.4f}"],
    ]) + "\n")
    (repo_path(cfg["paths"]["reports"]) / "xgb_tuning.md").write_text("\n".join(L), encoding="utf-8")
    return {"best": best, "val": xs, "baselines": bs, "diffs": diffs, "importance": imp, "heights": h_rows,
            "drop_height": drop_height, "shuffled": sh, "model_path": model_path, "yaml": _yaml_params(best)}


def main(argv: list[str] | None = None) -> None:
    """Run the dev-split search, refit the best trial and write the Phase 5 report."""
    ap = argparse.ArgumentParser(description="XGBoost tuning on the dev split")
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--budget-min", type=float, default=None)
    args = ap.parse_args(argv)
    cfg = load_config()
    n = int(args.trials if args.trials is not None else cfg["xgb"]["n_trials"])
    w = int(args.workers if args.workers is not None else max(1, (os.cpu_count() or 2) - 1))
    t0 = time.perf_counter()
    data = prepare_data(cfg)
    _, records = run_study(cfg, data=data, n_trials=n, workers=w, budget_min=args.budget_min)
    t1 = time.perf_counter()
    out = run_report(cfg, data, records, runtime_s=t1 - t0, workers=w, n_requested=n)
    print(f"{len(records)} trials in {(t1 - t0) / 60:.1f} min; best trial {out['best']['number']}")
    print(f"val: acc {out['val']['accuracy']:.4f} ll {out['val']['log_loss']:.5f} brier {out['val']['brier']:.5f}")
    print(f"shuffled-label acc {out['shuffled']['accuracy']:.4f}; drop height: {out['drop_height']}")
    print(out["yaml"])


if __name__ == "__main__":
    main()
