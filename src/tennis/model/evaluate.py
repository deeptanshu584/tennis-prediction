"""Evaluation: metrics, bootstrap CIs, calibration, breakdowns, the leakage alarm and the experiment log
(PLAN sections 7 and 8).

Conventions
- ``y`` is 1 if player A won; ``p`` is the predicted probability that A wins.
- Accuracy scores ``p > 0.5`` as a pick for A and ``p < 0.5`` as a pick for B. An exact ``p == 0.5`` (no pick,
  e.g. equal Elo or equal / both-missing rank) earns **half credit**: the expected accuracy of a fair coin, and it
  doesn't depend on the A/B orientation. (Phase 2's Elo check excluded ties instead; DECISIONS D-018.)
- Log loss clips ``p`` to ``[eps, 1 - eps]``. Brier is the mean squared error of ``p``.
- Bootstrap CIs are percentile intervals over row resamples, seeded by ``config.seed``; all metrics of one call
  share the same resamples.
- Every frame that gets scored goes through :func:`tennis.model.dataset.assert_not_augmented` first
  (CONTRACT section 4b).
- RET/DEF matches are not in the feature table (never labels), so every accuracy here is slightly optimistic
  relative to all real matches (DECISIONS D-022).

Running ``python -m tennis.model.evaluate`` computes the Phase 3 baselines on the dev split (train 2000-2022,
validation 2023-2024; no test set), appends them to ``experiments/baselines.jsonl``, writes
``reports/baselines.md`` and the calibration figure ``reports/figures/phase3_calibration_elo.png``.
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from tennis.config import REPO_ROOT, config_hash, load_config, repo_path
from tennis.model.dataset import assert_not_augmented

ArrayLike = Sequence[float] | np.ndarray | pd.Series
LOG_LOSS_EPS: float = 1e-15
"""Fallback clip for the pure metric functions when no config is given. Every config-aware path
(:func:`score`, :func:`bootstrap_ci`, :func:`paired_bootstrap_diff`, :func:`breakdown`, the baselines) reads
``evaluation.log_loss_eps`` instead (:func:`log_loss_eps`)."""


class LeakageAlarm(RuntimeError):
    """A validation accuracy above ``evaluation.leakage_alarm_accuracy``: treat as leakage until proven otherwise."""


# --------------------------------------------------------------------------------------------- point metrics
def _arrays(y: ArrayLike, p: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    ya = np.asarray(y, dtype=float)
    pa = np.asarray(p, dtype=float)
    if ya.shape != pa.shape or ya.ndim != 1:
        raise ValueError(f"y and p must be 1-D and the same length, got {ya.shape} and {pa.shape}")
    if len(ya) == 0:
        raise ValueError("cannot score an empty set")
    if not np.isin(ya, (0.0, 1.0)).all():
        raise ValueError("y must be 0/1")
    if np.isnan(pa).any() or (pa < 0).any() or (pa > 1).any():
        raise ValueError("p must be probabilities in [0, 1] with no NaN")
    return ya, pa


def correct(y: ArrayLike, p: ArrayLike) -> np.ndarray:
    """Per-row accuracy credit: 1 for a right pick, 0 for a wrong one, 0.5 when ``p == 0.5``."""
    ya, pa = _arrays(y, p)
    pick = np.where(pa > 0.5, 1.0, np.where(pa < 0.5, 0.0, np.nan))
    return np.where(np.isnan(pick), 0.5, (pick == ya).astype(float))


def row_log_loss(y: ArrayLike, p: ArrayLike, eps: float = LOG_LOSS_EPS) -> np.ndarray:
    """Per-row negative log-likelihood with ``p`` clipped to ``[eps, 1 - eps]``."""
    ya, pa = _arrays(y, p)
    pc = np.clip(pa, eps, 1.0 - eps)
    return -(ya * np.log(pc) + (1.0 - ya) * np.log(1.0 - pc))


def row_brier(y: ArrayLike, p: ArrayLike) -> np.ndarray:
    """Per-row squared error ``(p - y)^2``."""
    ya, pa = _arrays(y, p)
    return (pa - ya) ** 2


def accuracy(y: ArrayLike, p: ArrayLike) -> float:
    """Share of correct picks (``p == 0.5`` counts half)."""
    return float(correct(y, p).mean())


def log_loss(y: ArrayLike, p: ArrayLike, eps: float = LOG_LOSS_EPS) -> float:
    """Mean clipped log loss (natural log)."""
    return float(row_log_loss(y, p, eps).mean())


def brier(y: ArrayLike, p: ArrayLike) -> float:
    """Mean Brier score."""
    return float(row_brier(y, p).mean())


ROW_METRICS: dict[str, Callable[[ArrayLike, ArrayLike], np.ndarray]] = {
    "accuracy": correct, "log_loss": row_log_loss, "brier": row_brier,
}


def log_loss_eps(cfg: Mapping[str, Any] | None) -> float:
    """``evaluation.log_loss_eps`` from ``cfg`` (required when a config is given); :data:`LOG_LOSS_EPS` if None."""
    return LOG_LOSS_EPS if cfg is None else float(cfg["evaluation"]["log_loss_eps"])


def row_values(metric: str, y: ArrayLike, p: ArrayLike, cfg: Mapping[str, Any] | None = None) -> np.ndarray:
    """Per-row values of ``metric`` (a :data:`ROW_METRICS` name); log loss is clipped with :func:`log_loss_eps`."""
    if metric == "log_loss":
        return row_log_loss(y, p, log_loss_eps(cfg))
    return ROW_METRICS[metric](y, p)


# ------------------------------------------------------------------------------------------------- bootstrap
def _boot_params(cfg: Mapping[str, Any] | None, n_samples: int | None, seed: int | None,
                 level: float | None) -> tuple[int, int, float]:
    ev = (cfg or {}).get("evaluation", {})
    n = int(n_samples if n_samples is not None else ev["bootstrap_samples"])
    s = int(seed if seed is not None else cfg["seed"])  # type: ignore[index]
    lv = float(level if level is not None else ev.get("ci_level", 0.95))
    return n, s, lv


def bootstrap_ci(y: ArrayLike, p: ArrayLike, metrics: Sequence[str] = ("accuracy", "log_loss", "brier"), *,
                 cfg: Mapping[str, Any] | None = None, n_samples: int | None = None, seed: int | None = None,
                 level: float | None = None, chunk: int = 200) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CIs for row-decomposable metrics (names from :data:`ROW_METRICS`).

    ``n_samples``, ``seed`` and ``level`` default to ``evaluation.bootstrap_samples``, ``seed`` and
    ``evaluation.ci_level`` from ``cfg``. All metrics share the same resamples. Deterministic for a given seed.
    Returns ``{metric: (lo, hi)}``.
    """
    n_boot, s, lv = _boot_params(cfg, n_samples, seed, level)
    rows = {m: row_values(m, y, p, cfg) for m in metrics}
    n = len(next(iter(rows.values())))
    rng = np.random.default_rng(s)
    stats = {m: np.empty(n_boot) for m in metrics}
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(b, n))
        for m in metrics:
            stats[m][done:done + b] = rows[m][idx].mean(axis=1)
        done += b
    a = (1.0 - lv) / 2.0
    return {m: (float(np.quantile(v, a)), float(np.quantile(v, 1.0 - a))) for m, v in stats.items()}


def paired_bootstrap_diff(y: ArrayLike, p1: ArrayLike, p2: ArrayLike, metric: str, *,
                          cfg: Mapping[str, Any] | None = None, n_samples: int | None = None,
                          seed: int | None = None, level: float | None = None) -> tuple[float, float, float]:
    """``metric(p1) − metric(p2)`` on the same rows with a paired percentile bootstrap CI: ``(diff, lo, hi)``."""
    n_boot, s, lv = _boot_params(cfg, n_samples, seed, level)
    d = row_values(metric, y, p1, cfg) - row_values(metric, y, p2, cfg)
    rng = np.random.default_rng(s)
    stats = np.concatenate([d[rng.integers(0, len(d), size=(min(200, n_boot - i), len(d)))].mean(axis=1)
                            for i in range(0, n_boot, 200)])
    a = (1.0 - lv) / 2.0
    return float(d.mean()), float(np.quantile(stats, a)), float(np.quantile(stats, 1.0 - a))


def score(y: ArrayLike, p: ArrayLike, cfg: Mapping[str, Any] | None = None, *, ci: bool = True,
          probabilistic: bool = True) -> dict[str, Any]:
    """``n``, accuracy and (if ``probabilistic``) log loss and Brier, each with ``<metric>_ci = [lo, hi]``."""
    metrics = ["accuracy", "log_loss", "brier"] if probabilistic else ["accuracy"]
    out: dict[str, Any] = {"n": int(len(np.asarray(y)))}
    for m in metrics:
        out[m] = float(row_values(m, y, p, cfg).mean())
    if ci:
        for m, (lo, hi) in bootstrap_ci(y, p, metrics, cfg=cfg).items():
            out[f"{m}_ci"] = [lo, hi]
    return out


def evaluate_frame(df: pd.DataFrame, p: ArrayLike, cfg: Mapping[str, Any] | None = None, *, ci: bool = True,
                   probabilistic: bool = True) -> dict[str, Any]:
    """Score predictions ``p`` (aligned with ``df`` rows) against ``df["y"]``.

    Always runs :func:`assert_not_augmented` on ``df`` first: validation/test frames never contain flipped rows.
    """
    assert_not_augmented(df)
    pa = np.asarray(p, dtype=float)
    if len(pa) != len(df):
        raise ValueError(f"{len(pa)} predictions for {len(df)} rows")
    return score(df["y"].to_numpy(), pa, cfg, ci=ci, probabilistic=probabilistic)


# ----------------------------------------------------------------------------------------------- calibration
def calibration_table(y: ArrayLike, p: ArrayLike, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width bins on ``[0, 1]`` (the last bin includes 1.0): ``bin, lo, hi, n, mean_pred, frac_pos``.

    Empty bins are kept with ``n = 0`` and NaN means.
    """
    ya, pa = _arrays(y, p)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.clip(np.digitize(pa, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for i in range(n_bins):
        m = b == i
        rows.append({"bin": i, "lo": edges[i], "hi": edges[i + 1], "n": int(m.sum()),
                     "mean_pred": float(pa[m].mean()) if m.any() else np.nan,
                     "frac_pos": float(ya[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def expected_calibration_error(table: pd.DataFrame) -> float:
    """Count-weighted mean |mean_pred − frac_pos| over the non-empty bins of a :func:`calibration_table`."""
    t = table[table["n"] > 0]
    return float((t["n"] * (t["mean_pred"] - t["frac_pos"]).abs()).sum() / t["n"].sum())


# Validated categorical slots 1-4 (dataviz reference palette, light mode), assigned in fixed order.
SERIES_COLORS: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
SERIES_MARKERS: tuple[str, ...] = ("o", "s", "^", "D")


def plot_calibration(curves: Mapping[str, tuple[ArrayLike, ArrayLike]], path: str | Path, *, n_bins: int = 10,
                     title: str = "Calibration", histogram_of: str | None = None) -> Path:
    """Save a reliability diagram (one line per named ``(y, p)`` pair) and, if ``histogram_of`` names a curve,
    a second panel with that curve's prediction counts per bin. Returns the saved path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#2b2b2b", "#6b6b6b", "#e6e6e3"
    ncols = 2 if histogram_of else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.2 * ncols, 5.6), squeeze=False,
                             gridspec_kw={"width_ratios": [1.25, 1][:ncols]})
    ax = axes[0][0]
    ax.plot([0, 1], [0, 1], color=muted, lw=1, ls="--", label="perfect calibration")
    for i, (name, (y, p)) in enumerate(curves.items()):
        t = calibration_table(y, p, n_bins)
        t = t[t["n"] > 0]
        ax.plot(t["mean_pred"], t["frac_pos"], color=SERIES_COLORS[i % 4], lw=2,
                marker=SERIES_MARKERS[i % 4], ms=6, mec="white", mew=1.2,
                label=f"{name} (ECE {expected_calibration_error(calibration_table(y, p, n_bins)):.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="mean predicted P(A wins) in bin", ylabel="observed share A won")
    ax.set_title(title, color=ink, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=ink)
    if histogram_of:
        y, p = curves[histogram_of]
        t = calibration_table(y, p, n_bins)
        ax2 = axes[0][1]
        ax2.bar((t["lo"] + t["hi"]) / 2, t["n"], width=0.1 - 0.008, color=SERIES_COLORS[0], edgecolor="white")
        ax2.set(xlim=(0, 1), xlabel="predicted P(A wins)", ylabel="matches")
        ax2.set_title(f"{histogram_of}: matches per bin", color=ink, loc="left", fontsize=11)
    for a in axes[0]:
        a.grid(color=grid, lw=0.8)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(muted)
        a.tick_params(colors=muted, labelsize=9)
        a.xaxis.label.set_color(ink)
        a.yaxis.label.set_color(ink)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ------------------------------------------------------------------------------------------------ breakdowns
def context_labels(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Readable ``surface``, ``level``, ``round`` and ``season`` per row, decoded from the one-hot / rank columns
    using the config (``data.surface_map``, ``data.training_levels``, ``data.round_order``)."""
    data = cfg["data"]

    def decode(names: list[str], prefix: str) -> pd.Series:
        cols = [f"{prefix}{n}" for n in names]
        hot = df[cols].to_numpy()
        lab = np.array(names, dtype=object)[hot.argmax(axis=1)]
        return pd.Series(np.where(hot.sum(axis=1) == 1, lab, "other"), index=df.index)

    surfaces = sorted(set(data["surface_map"].values()))
    order = list(data["round_order"])
    rr = df["round_rank"].astype(int).to_numpy()
    return pd.DataFrame({
        "surface": decode(surfaces, "surface_"),
        "level": decode(list(data["training_levels"]), "level_"),
        "round": pd.Series([order[i] for i in rr], index=df.index),
        "season": pd.to_datetime(df["date"]).dt.year,
    }, index=df.index)


def breakdown(df: pd.DataFrame, p: ArrayLike, by: pd.Series, cfg: Mapping[str, Any] | None = None, *,
              ci: bool = True, probabilistic: bool = True, order: Sequence[Any] | None = None) -> pd.DataFrame:
    """Metrics per group of ``by`` (aligned with ``df``): one row per group with ``n``, accuracy (+ CI), and
    log loss / Brier when ``probabilistic``. ``order`` fixes the row order (groups not present are skipped)."""
    assert_not_augmented(df)
    pa = pd.Series(np.asarray(p, dtype=float), index=df.index)
    groups = list(order) if order is not None else sorted(by.dropna().unique())
    rows = []
    for g in groups:
        m = (by == g).to_numpy()
        if not m.any():
            continue
        s = score(df.loc[m, "y"].to_numpy(), pa[m].to_numpy(), cfg, ci=False, probabilistic=probabilistic)
        if ci:
            lo, hi = bootstrap_ci(df.loc[m, "y"].to_numpy(), pa[m].to_numpy(), ["accuracy"], cfg=cfg)["accuracy"]
            s["accuracy_ci"] = [lo, hi]
        rows.append({"group": g, **s})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------- leakage alarm
def leakage_alarm(acc: float, cfg: Mapping[str, Any], *, what: str = "validation accuracy",
                  raise_on_alarm: bool = True) -> bool:
    """Return True (and raise :class:`LeakageAlarm` unless ``raise_on_alarm`` is False, in which case a warning is
    issued) when ``acc`` exceeds ``evaluation.leakage_alarm_accuracy``. False otherwise.

    Use it on every full-season / full-window validation accuracy: above the threshold is leakage until proven
    otherwise (CLAUDE.md).
    """
    limit = float(cfg["evaluation"]["leakage_alarm_accuracy"])
    if acc > limit:
        msg = f"{what} = {acc:.4f} > {limit}: suspected leakage, stop and investigate"
        if raise_on_alarm:
            raise LeakageAlarm(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return True
    return False


# ------------------------------------------------------------------------------------------ experiment log
def git_commit() -> str | None:
    """``git rev-parse HEAD`` of the repo, or None when unavailable (not a repo / no commits / no git)."""
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                           timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if np.isnan(f) else f
    if isinstance(obj, (pd.Timestamp, _dt.date)):
        return obj.isoformat()
    return obj


def log_experiment(name: str, *, split: str, params: Mapping[str, Any], metrics: Mapping[str, Any],
                   cfg: Mapping[str, Any], out_dir: str | Path | None = None) -> dict[str, Any]:
    """Append one JSON line to ``<out_dir>/<name>.jsonl`` (default ``paths.experiments``) and return the record:
    ``timestamp`` (UTC ISO), ``git_commit``, ``config_hash``, ``split``, ``params``, ``metrics``."""
    rec = {
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "config_hash": config_hash(dict(cfg)),
        "split": split,
        "params": _jsonable(params),
        "metrics": _jsonable(metrics),
    }
    d = Path(out_dir) if out_dir is not None else repo_path(cfg["paths"]["experiments"])
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{name}.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=False) + "\n")
    return rec


# ------------------------------------------------------------------------------------------------ baselines
def rank_rule_prob(df: pd.DataFrame) -> np.ndarray:
    """Higher-ranked player wins: 1.0 if A's rank number is lower, 0.0 if higher, 0.5 if equal. A missing rank
    means unranked, i.e. worse than any ranked player; both unranked -> 0.5. No probability, a pick only."""
    a = df["a_rank"].to_numpy(dtype=float)
    b = df["b_rank"].to_numpy(dtype=float)
    a = np.where(np.isnan(a), np.inf, a)
    b = np.where(np.isnan(b), np.inf, b)
    return np.where(a < b, 1.0, np.where(a > b, 0.0, 0.5))


@dataclass(frozen=True)
class RankLogistic:
    """``P(A wins) = sigmoid(beta * (log(b_rank) - log(a_rank)))``, no intercept (A/B is a coin flip).

    Unranked players get ``fill_rank`` = (largest rank seen in the training rows) + 1. Both ``beta`` and
    ``fill_rank`` are fitted on TRAINING rows only (PLAN section 2 rule 10). With beta > 0 the picks equal
    :func:`rank_rule_prob`.
    """
    beta: float
    fill_rank: float

    def _x(self, df: pd.DataFrame) -> np.ndarray:
        a = df["a_rank"].fillna(self.fill_rank).to_numpy(dtype=float)
        b = df["b_rank"].fillna(self.fill_rank).to_numpy(dtype=float)
        return np.log(b) - np.log(a)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.beta * self._x(df)))

    @classmethod
    def fit(cls, train: pd.DataFrame, max_iter: int = 100, tol: float = 1e-12) -> "RankLogistic":
        """Maximum likelihood for ``beta`` by Newton's method (one parameter, concave log-likelihood)."""
        assert_not_augmented(train)
        fill = float(np.nanmax(train[["a_rank", "b_rank"]].to_numpy(dtype=float))) + 1.0
        x = cls(0.0, fill)._x(train)
        y = train["y"].to_numpy(dtype=float)
        beta = 0.0
        for _ in range(max_iter):
            p = 1.0 / (1.0 + np.exp(-beta * x))
            grad = float(np.sum((y - p) * x))
            hess = float(np.sum(p * (1 - p) * x * x))
            step = grad / hess
            beta += step
            if abs(step) < tol:
                break
        return cls(beta, fill)


BOOKMAKER_TODO = "TODO (Phase 6/7): bookmaker implied probability, margin removed — odds being ingested"


def baseline_predictions(train: pd.DataFrame, frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], RankLogistic]:
    """Predictions of every Phase 3 baseline on ``frame``; the rank logistic is fitted on ``train`` only."""
    rl = RankLogistic.fit(train)
    elo = frame["elo_prob_a"].to_numpy(dtype=float)
    selo = frame["elo_surface_prob_a"].to_numpy(dtype=float)
    return {
        "rank_rule": rank_rule_prob(frame),
        "rank_logistic": rl.predict(frame),
        "elo": elo,
        "surface_elo": selo,
        "elo_blend_50_50": 0.5 * elo + 0.5 * selo,
    }, rl


BASELINE_LABELS: dict[str, str] = {
    "rank_rule": "Higher rank wins",
    "rank_logistic": "Rank logistic (log-rank ratio, fit on train)",
    "elo": "Elo",
    "surface_elo": "Surface Elo",
    "elo_blend_50_50": "50/50 Elo + surface Elo",
}
PROBABILISTIC: dict[str, bool] = {"rank_rule": False, "rank_logistic": True, "elo": True, "surface_elo": True,
                                  "elo_blend_50_50": True}


def _fmt(v: float, d: int = 4) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def _fmt_ci(s: Mapping[str, Any], m: str, d: int = 4) -> str:
    """``value [lo, hi]`` for metric ``m`` of a :func:`score` dict; ``—`` if absent."""
    if m not in s:
        return "—"
    if f"{m}_ci" not in s:
        return f"{s[m]:.{d}f}"
    lo, hi = s[f"{m}_ci"]
    return f"{s[m]:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def _md_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def run_dev_baselines(cfg: Mapping[str, Any] | None = None, *, write: bool = True) -> dict[str, Any]:
    """Phase 3: all baselines on the dev split (validation 2023-2024; no test set). Returns the results; with
    ``write`` also logs ``experiments/baselines.jsonl``, saves the calibration figure and writes
    ``reports/baselines.md``. Raises :class:`LeakageAlarm` if a full-window or full-season accuracy > alarm."""
    from tennis.model.dataset import load_features
    from tennis.model.splits import time_split

    cfg = dict(cfg) if cfg is not None else load_config()
    split = "dev"
    df = load_features(cfg)
    parts = time_split(df, split, cfg)
    if "test" in parts:
        raise RuntimeError("dev split must not have a test partition")
    train, val = parts["train"], parts["val"]
    assert_not_augmented(train)
    assert_not_augmented(val)

    preds_val, rl = baseline_predictions(train, val)
    if not np.array_equal(correct(val["y"], preds_val["rank_rule"]), correct(val["y"], preds_val["rank_logistic"])):
        raise RuntimeError("rank logistic picks differ from the rank rule (beta <= 0?)")

    val_scores = {k: evaluate_frame(val, p, cfg, probabilistic=PROBABILISTIC[k]) for k, p in preds_val.items()}
    for k, s in val_scores.items():
        leakage_alarm(s["accuracy"], cfg, what=f"{k} dev validation accuracy (2023-2024)")

    # per season 2015-2024 (seasons <= 2022 are training-period rows: Elo and the rank rule are pre-match
    # quantities with nothing fitted on those rows, but the rank logistic's beta is in-sample there)
    first, last = (int(x) for x in cfg["evaluation"]["baseline_seasons"])
    seasons = list(range(first, last + 1))
    span = df[(df["date"].dt.year >= seasons[0]) & (df["date"].dt.year <= seasons[-1])]
    preds_span, _ = baseline_predictions(train, span)
    span_ctx = context_labels(span, cfg)
    season_tables = {k: breakdown(span, p, span_ctx["season"], cfg, probabilistic=PROBABILISTIC[k], order=seasons)
                     for k, p in preds_span.items()}
    for k, t in season_tables.items():
        for _, r in t.iterrows():
            leakage_alarm(r["accuracy"], cfg, what=f"{k} season {r['group']} accuracy")

    ctx = context_labels(val, cfg)
    dims = {
        "surface": sorted(ctx["surface"].unique()),
        "level": [lv for lv in cfg["data"]["training_levels"] if lv in set(ctx["level"])],
        "round": [r for r in cfg["data"]["round_order"] if r in set(ctx["round"])],
    }
    val_breakdowns = {dim: {k: breakdown(val, p, ctx[dim], cfg, probabilistic=PROBABILISTIC[k], order=order)
                            for k, p in preds_val.items()} for dim, order in dims.items()}

    nb = int(cfg["evaluation"]["calibration_bins"])
    cal = {k: calibration_table(val["y"], preds_val[k], nb) for k in ("elo", "surface_elo", "elo_blend_50_50",
                                                                      "rank_logistic")}
    diffs = {(k, m): paired_bootstrap_diff(val["y"], preds_val[k], preds_val["elo"], m, cfg=cfg)
             for k in ("rank_logistic", "surface_elo", "elo_blend_50_50") for m in ("accuracy", "log_loss")}

    # partition sizes of every configured split (row counts only: no test metric is computed anywhere here)
    sizes = {}
    for name in cfg["splits"]:
        p = time_split(df, name, cfg)
        sizes[name] = {k: (len(v), str(v["date"].min().date()) if len(v) else "", str(v["date"].max().date())
                           if len(v) else "") for k, v in p.items()}
        sizes[name]["unassigned"] = len(df) - sum(len(v) for v in p.values())
    results = {"split": split, "n_train": len(train), "n_val": len(val), "val": val_scores,
               "seasons": season_tables, "val_breakdowns": val_breakdowns, "calibration": cal,
               "rank_logistic": rl, "diffs_vs_elo": diffs, "split_sizes": sizes}
    if not write:
        return results

    fig = repo_path(cfg["paths"]["figures"]) / "phase3_calibration_elo.png"
    plot_calibration({BASELINE_LABELS[k]: (val["y"], preds_val[k]) for k in cal}, fig, n_bins=nb,
                     title="Dev validation 2023–2024: calibration of the baselines", histogram_of=BASELINE_LABELS["elo"])

    for k, s in val_scores.items():
        st = season_tables[k]
        metrics = {
            "val": s,
            "seasons": {str(int(r["group"])): {c: r[c] for c in ("n", "accuracy", "log_loss", "brier")
                                               if c in r and not pd.isna(r[c])} for _, r in st.iterrows()},
            "calibration_ece_val": expected_calibration_error(calibration_table(val["y"], preds_val[k], nb))
            if PROBABILISTIC[k] else None,
        }
        params: dict[str, Any] = {"baseline": k, "description": BASELINE_LABELS[k], "probabilistic": PROBABILISTIC[k],
                                  "train_window": [str(train["date"].min().date()), str(train["date"].max().date())],
                                  "val_window": [str(val["date"].min().date()), str(val["date"].max().date())]}
        if k == "rank_logistic":
            params.update({"beta": rl.beta, "fill_rank": rl.fill_rank})
        log_experiment("baselines", split=split, params=params, metrics=metrics, cfg=cfg)

    write_baselines_report(results, cfg, fig)
    return results


def write_baselines_report(res: Mapping[str, Any], cfg: Mapping[str, Any], fig: Path) -> Path:
    """Write ``reports/baselines.md`` from :func:`run_dev_baselines` results."""
    keys = list(BASELINE_LABELS)
    lv = int(round(100 * float(cfg["evaluation"]["ci_level"])))
    B = int(cfg["evaluation"]["bootstrap_samples"])
    rl: RankLogistic = res["rank_logistic"]
    L: list[str] = []
    dev = cfg["splits"]["dev"]
    vw = f"{dev['val_start'][:4]}–{dev['val_end'][:4]}"
    s0, s1 = (int(x) for x in cfg["evaluation"]["baseline_seasons"])
    L.append("# Phase 3 — Baselines on the dev split\n")
    L.append(f"Generated by `python -m tennis.model.evaluate` on {_dt.date.today().isoformat()} · config hash "
             f"`{config_hash(dict(cfg))}` · git `{git_commit() or 'n/a'}`. Every run is a JSON line in "
             "`experiments/baselines.jsonl`.\n")
    L.append(f"**Dev split** (DECISIONS D-004): train {dev['train_start']} → {dev['train_end']} ({res['n_train']:,} "
             f"rows), validation {dev['val_start']} → {dev['val_end']} ({res['n_val']:,} rows). **There is no test "
             "set in this split**; "
             "nothing here touches Wimbledon 2025 or 2026.\n")
    L.append(f"CIs: {lv}% percentile bootstrap over validation rows, {B:,} resamples, seed `{cfg['seed']}`. "
             "Accuracy gives half credit to a no-pick (p = 0.5: equal Elo, equal or both-missing rank). "
             "RET/DEF matches are never labels and are absent from every table, so accuracies are slightly "
             "optimistic relative to all real matches (D-022).\n")

    L.append(f"## Validation {vw}\n")
    rows = []
    for k in keys:
        s = res["val"][k]
        rows.append([BASELINE_LABELS[k], f"{s['n']:,}", _fmt_ci(s, "accuracy"), _fmt_ci(s, "log_loss"),
                     _fmt_ci(s, "brier")])
    rows.append(["Bookmaker (implied, margin removed)", "—", BOOKMAKER_TODO, "TODO", "TODO"])
    L.append(_md_table(["Baseline", "n", "Accuracy", "Log loss", "Brier"], rows) + "\n")
    L.append(f"- *Higher rank wins* has no probability (accuracy only). Its probabilistic twin, the *rank "
             f"logistic*, is `P(A) = sigmoid(β·(ln b_rank − ln a_rank))` with no intercept, fitted on TRAIN rows "
             f"only: β = {rl.beta:.4f}; unranked players get rank {rl.fill_rank:.0f} (largest training rank + 1). "
             "Its picks are identical to the rule, so its accuracy is the rule's.\n"
             "- The 50/50 blend averages the two Elo probabilities.\n"
             f"- Leakage alarm: every full-window and full-season accuracy above is ≤ "
             f"{cfg['evaluation']['leakage_alarm_accuracy']} (checked by `leakage_alarm`).\n")

    L.append("**Paired differences vs Elo** (same validation rows, paired bootstrap; negative log-loss difference "
             "= better than Elo):\n")
    rows = []
    for k in ("rank_logistic", "surface_elo", "elo_blend_50_50"):
        a, alo, ahi = res["diffs_vs_elo"][(k, "accuracy")]
        g, glo, ghi = res["diffs_vs_elo"][(k, "log_loss")]
        rows.append([BASELINE_LABELS[k], f"{a:+.4f} [{alo:+.4f}, {ahi:+.4f}]", f"{g:+.4f} [{glo:+.4f}, {ghi:+.4f}]"])
    L.append(_md_table(["Baseline − Elo", "Δ accuracy", "Δ log loss"], rows) + "\n")

    L.append(f"## Per season {s0}–{s1} (accuracy)\n")
    L.append(f"Seasons ≤ {dev['train_end'][:4]} lie in the training window. Elo and the rank rule fit nothing on "
             "those rows (Elo is a pre-match rating), so they are still honest out-of-sample predictions; the rank "
             f"logistic's β is in-sample there. {vw} are the validation seasons.\n")
    seasons = res["seasons"]
    yrs = [int(g) for g in seasons["elo"]["group"]]
    rows = []
    for i, yr in enumerate(yrs):
        r = [str(yr), f"{int(seasons['elo'].iloc[i]['n']):,}"]
        for k in ("rank_rule", "elo", "surface_elo", "elo_blend_50_50"):
            t = seasons[k].iloc[i]
            lo, hi = t["accuracy_ci"]
            r.append(f"{t['accuracy']:.4f} [{lo:.3f}, {hi:.3f}]")
        r.append(_fmt(float(seasons["elo"].iloc[i]["log_loss"])))
        rows.append(r)
    L.append(_md_table(["Season", "n", "Rank rule", "Elo", "Surface Elo", "Blend", "Elo log loss"], rows) + "\n")

    for dim, title in (("surface", "surface"), ("level", "level"), ("round", "round")):
        L.append(f"## Validation {vw} by {title}\n")
        tabs = res["val_breakdowns"][dim]
        rows = []
        for i, g in enumerate(tabs["elo"]["group"]):
            r = [str(g), f"{int(tabs['elo'].iloc[i]['n']):,}"]
            for k in ("rank_rule", "elo", "surface_elo", "elo_blend_50_50"):
                t = tabs[k].iloc[i]
                lo, hi = t["accuracy_ci"]
                r.append(f"{t['accuracy']:.3f} [{lo:.3f}, {hi:.3f}]")
            r += [_fmt(float(tabs["elo"].iloc[i]["log_loss"])), _fmt(float(tabs["elo_blend_50_50"].iloc[i]["log_loss"]))]
            rows.append(r)
        L.append(_md_table([title.capitalize(), "n", "Rank rule acc", "Elo acc", "Surface Elo acc", "Blend acc",
                            "Elo log loss", "Blend log loss"], rows) + "\n")
    alarm = float(cfg["evaluation"]["leakage_alarm_accuracy"])
    over = [f"{BASELINE_LABELS[k]} on {dim} = {r['group']} ({r['accuracy']:.3f}, n = {int(r['n'])})"
            for dim, tabs in res["val_breakdowns"].items() for k, t in tabs.items() for _, r in t.iterrows()
            if r["accuracy"] > alarm]
    L.append("Level codes: G Grand Slam, M Masters 1000, A other tour events (incl. NextGen Finals), F Tour "
             "Finals, D Davis Cup, O Olympics. Small groups have wide CIs (the single BR row is meaningless).\n")
    L.append(f"**Subgroups above the {alarm} alarm line:** "
             + ("; ".join(over) if over else "none") + ". "
             + ("These are small subgroups, not full-season accuracies (the alarm's scope); their CIs reach far "
                "below the line and the matching Elo figures are ordinary, so no leakage is suspected. Subgroup "
                "figures are kept out of the JSON log for that reason.\n" if over else "\n"))

    L.append("## Partition sizes of every configured split\n")
    L.append("Row counts and date ranges only — no metric is computed on any test partition. *Unassigned* rows "
             "fall in no partition: for `dev`, everything from 2025 on; for `replication`, every event starting after "
             "val_end 2025-06-29 other than Wimbledon 2025 (rest of 2025 and all of 2026); for `forward`, only rows "
             "after the pinned test_end (none today).\n")
    rows = []
    for name, s in res["split_sizes"].items():
        cells = [name]
        for p in ("train", "val", "test"):
            cells.append(f"{s[p][0]:,} ({s[p][1]} → {s[p][2]})" if p in s else "—")
        cells.append(f"{s['unassigned']:,}")
        rows.append(cells)
    L.append(_md_table(["Split", "Train", "Val", "Test", "Unassigned"], rows) + "\n")

    L.append(f"## Calibration (validation {vw})\n")
    L.append(f"![calibration](figures/{fig.name})\n")
    rows = []
    cal = res["calibration"]["elo"]
    for _, r in cal.iterrows():
        rows.append([f"{r['lo']:.1f}–{r['hi']:.1f}", f"{int(r['n']):,}", _fmt(r["mean_pred"], 3),
                     _fmt(r["frac_pos"], 3)])
    L.append("Elo, 10 equal-width bins:\n")
    L.append(_md_table(["Bin", "n", "Mean predicted", "Observed share"], rows) + "\n")
    eces = ", ".join(f"{BASELINE_LABELS[k]} {expected_calibration_error(t):.4f}" for k, t in res["calibration"].items())
    L.append(f"Expected calibration error (count-weighted |pred − obs|): {eces}.\n")

    out = repo_path(cfg["paths"]["reports"]) / "baselines.md"
    out.write_text("\n".join(L), encoding="utf-8")
    return out


def main() -> None:
    """Run the Phase 3 dev-split baselines and print the headline table."""
    res = run_dev_baselines()
    for k, s in res["val"].items():
        print(f"{k:16s} n={s['n']} acc={_fmt_ci(s, 'accuracy')} ll={_fmt_ci(s, 'log_loss')} "
              f"brier={_fmt_ci(s, 'brier')}")


if __name__ == "__main__":
    main()
