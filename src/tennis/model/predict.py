"""Win probabilities for upcoming matches from the state as of the latest data (PLAN section 6 Phase 8).

Single match::

    .venv/Scripts/python -m tennis.model.predict --p1 "Jannik Sinner" --p2 "Carlos Alcaraz" \\
        --surface Grass --level G --round F --best-of 5 [--date YYYY-MM-DD]

A whole round (CSV columns ``p1,p2,surface,level,round,best_of[,date]``)::

    .venv/Scripts/python -m tennis.model.predict --draw draw.csv [--out predictions.csv]

How it works:

- The engine state after the LAST match in ``paths.matches`` comes from
  :func:`tennis.engine.snapshot.load_or_build_state` (cached at ``models/state_<config hash>.pkl``). Every query is
  scored from that same final state and never updates it (:func:`tennis.engine.snapshot.query_rows`).
- ``--date`` is the tournament START date (the engine's ``tourney_date`` semantics); the default is the day after the
  last tourney_date in the data. Rank / rank points are each player's last known values (see snapshot docstring).
- The production model is XGBoost with ``xgb.forward_best_params`` and its fixed tree count, fitted on ALL feature
  rows (2000 to the end of the data), no evaluation set, allow-list features only: :func:`fit_production`. It is
  NEVER evaluated: every row it has seen is a training row, so no metric of it means anything. It is saved as
  ``models/xgb_production_<config hash>.json`` and fitted automatically when missing (or with ``--fit-production``).
- The XGBoost probability is symmetrised: ``(p(A, B) + 1 - p(B, A)) / 2``, so swapping p1 and p2 gives exactly
  ``1 - p``. The Elo and surface-Elo probabilities are symmetric by construction.
- Players are given by name (case- and accent-insensitive, most recent spelling in the data) or by Sackmann id.
  An unknown or ambiguous name is an error listing close candidates; nothing is ever defaulted.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import difflib  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import unicodedata  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

from tennis.config import config_hash, load_config, repo_path  # noqa: E402
from tennis.engine.build_features import surfaces  # noqa: E402
from tennis.engine.snapshot import EngineState, day_to_date, load_or_build_state, query_rows  # noqa: E402
from tennis.model import dataset  # noqa: E402
from tennis.model import train as tr  # noqa: E402

PRODUCTION_NAME: str = "xgb_production"
DRAW_COLUMNS: tuple[str, ...] = ("p1", "p2", "surface", "level", "round", "best_of")


class PlayerLookupError(ValueError):
    """A player name that matches no player, or more than one."""


# --------------------------------------------------------------------------- player lookup


def normalize_name(name: str) -> str:
    """Case- and accent-insensitive form: NFKD without combining marks, casefolded, punctuation -> spaces."""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).casefold()
    return " ".join(re.sub(r"[^0-9a-z]+", " ", s).split())


def _describe(state: EngineState, pid: str) -> str:
    """``Name (id 123, last seen 2026-05-25, rank 4)`` for error messages."""
    parts = [f"id {pid}"]
    day = state.name_day.get(pid)
    if day is not None:
        parts.append(f"last seen {day_to_date(day)}")
    inf = state.info.get(pid)
    if inf is not None and not np.isnan(inf.rank):
        parts.append(f"rank {int(inf.rank)}")
    return f"{state.names.get(pid, pid)} ({', '.join(parts)})"


def resolve_player(query: str, state: EngineState, *, max_candidates: int = 8) -> str:
    """The Sackmann player id for ``query`` (a name, matched on :func:`normalize_name`, or an id).

    Raises:
        PlayerLookupError: no player has that name (the message lists close candidates), several do (the message
            lists them with ids: pass the id instead), or the player has no completed match in the data.
    """
    q = str(query).strip()
    if q in state.players:
        return q
    if q in state.names:
        raise PlayerLookupError(f"player id {q} ({state.names[q]}) has no completed match in the data")
    key = normalize_name(q)
    by_name: dict[str, list[str]] = {}
    for pid, name in state.names.items():
        by_name.setdefault(normalize_name(name), []).append(pid)
    hits = by_name.get(key, [])
    playable = [pid for pid in hits if pid in state.players]
    if len(playable) == 1:
        return playable[0]
    if len(playable) > 1:
        playable.sort(key=lambda p: -state.name_day.get(p, 0))
        lines = "\n".join(f"  - {_describe(state, p)}" for p in playable)
        raise PlayerLookupError(f"{query!r} matches {len(playable)} players; pass the id instead:\n{lines}")
    if hits:
        raise PlayerLookupError(f"{query!r} ({', '.join(hits)}) has no completed match in the data (walkovers only)")
    # no exact match: close spellings, then names containing every word of the query (e.g. a surname)
    close = difflib.get_close_matches(key, list(by_name), n=max_candidates, cutoff=0.6)
    words = key.split()
    contains = [n for n in by_name if words and all(w in n.split() for w in words)]
    cands: list[str] = []
    for n in close + contains:
        cands += [p for p in by_name[n] if p in state.players and p not in cands]
    cands.sort(key=lambda p: -state.name_day.get(p, 0))
    msg = f"no player named {query!r} in the data"
    if cands:
        msg += "; did you mean:\n" + "\n".join(f"  - {_describe(state, p)}" for p in cands[:max_candidates])
    raise PlayerLookupError(msg)


# --------------------------------------------------------------------------- the production model


def production_model_path(cfg: Mapping[str, Any]) -> Path:
    """``<paths.models>/xgb_production_<config hash>.json``."""
    return repo_path(cfg["paths"]["models"]) / f"{PRODUCTION_NAME}_{config_hash(dict(cfg))}.json"


def fit_production(features: pd.DataFrame, cfg: Mapping[str, Any]) -> tr.FitResult:
    """Fit the PRODUCTION booster on every row of ``features`` (the whole feature table, 2000 to the end).

    ``xgb.forward_best_params`` (the Phase 7 frozen parameters) with exactly its ``n_estimators`` trees, no
    evaluation / early-stopping set, allow-list features only (:func:`tennis.model.dataset.feature_columns`). It
    has no held-out rows, so it is never evaluated: the result carries no metrics.

    Raises ``ValueError`` if ``features`` contains flipped rows or ``features.flip_augment_train`` is true (the
    frozen setting is false; flipped production training is not implemented).
    """
    if bool(cfg["features"].get("flip_augment_train", False)):
        raise ValueError("fit_production: features.flip_augment_train is true; production training follows the "
                         "frozen setting (false) only")
    if len(features) == 0:
        raise ValueError("fit_production got no rows")
    dataset.assert_not_augmented(features)
    params = dict(cfg["xgb"]["forward_best_params"])
    n_trees = int(params.pop("n_estimators"))
    feats = dataset.feature_columns(cfg, features)
    dtr = xgb.DMatrix(tr.matrix(features, feats), label=features["y"].to_numpy(dtype=np.float32), missing=np.nan,
                      feature_names=feats, nthread=tr.NTHREAD)
    booster = xgb.train(tr.native_params(params, cfg, tr.NTHREAD), dtr, num_boost_round=n_trees, evals=[],
                        verbose_eval=False)
    if booster.num_boosted_rounds() != n_trees:
        raise RuntimeError(f"production model has {booster.num_boosted_rounds()} trees, expected {n_trees}")
    dates = pd.to_datetime(features["date"])
    return tr.FitResult(booster, feats, params, n_trees - 1, np.empty(0), {},
                        {"purpose": "production", "never_evaluated": True, "params_key": "xgb.forward_best_params",
                         "fixed_trees": n_trees, "early_stopping": False, "n_rows": int(len(features)),
                         "train_window": [str(dates.min().date()), str(dates.max().date())]})


def load_or_fit_production(cfg: Mapping[str, Any], *, refit: bool = False, verbose: bool = True
                           ) -> tuple[xgb.Booster, dict[str, Any]]:
    """Load the production booster for this config hash, fitting it (from ``paths.features``) when missing,
    when its feature list is not the current allow-list, or with ``refit=True``."""
    path = production_model_path(cfg)
    if path.exists() and not refit:
        booster, meta = tr.load_model(path)
        if meta.get("features") == dataset.feature_columns(cfg):
            return booster, meta
    src = repo_path(cfg["paths"]["features"])
    if verbose:
        print(f"fitting the production model on {src} (fixed trees, no evaluation set) ...")
    features = dataset.load_features(cfg)
    res = fit_production(features, cfg)
    saved = tr.save_model(res, cfg, PRODUCTION_NAME, out_dir=path.parent)
    if verbose:
        print(f"saved {saved}")
    return tr.load_model(saved)


# --------------------------------------------------------------------------- scoring


def predict_queries(state: EngineState, booster: xgb.Booster, queries: pd.DataFrame, cfg: Mapping[str, Any]
                    ) -> pd.DataFrame:
    """Score queries (player IDS in ``p1``/``p2``) from the final state.

    Returns the queries plus ``p1_name, p2_name, date, p_xgb`` (symmetrised), ``p_xgb_ab`` / ``p_xgb_ba``
    (P(p1 wins) with p1 as A, and with p1 as B), ``p_elo``, ``p_elo_surface``.
    """
    swapped = queries.rename(columns={"p1": "p2", "p2": "p1"})
    swapped = swapped.rename(columns=lambda c: re.sub(r"^p([12])_", lambda m: f"p{3 - int(m.group(1))}_", c))
    ab = query_rows(state, queries.reset_index(drop=True), cfg)
    ba = query_rows(state, swapped.reset_index(drop=True), cfg)
    feats = list(booster.feature_names or [])
    if feats != dataset.feature_columns(cfg, ab):
        raise ValueError("the model's features are not the config allow-list; refit with --fit-production")
    p_ab = tr.predict(booster, tr.matrix(ab, feats))
    p_ba = 1.0 - tr.predict(booster, tr.matrix(ba, feats))
    out = queries.reset_index(drop=True).copy()
    out["p1_name"], out["p2_name"] = ab["a_name"].to_numpy(), ab["b_name"].to_numpy()
    out["date"] = ab["date"].dt.date.astype(str).to_numpy()
    out["p_xgb"] = 0.5 * (p_ab.astype(float) + p_ba.astype(float))
    out["p_xgb_ab"], out["p_xgb_ba"] = p_ab.astype(float), p_ba.astype(float)
    out["p_elo"] = ab["elo_prob_a"].to_numpy()
    out["p_elo_surface"] = ab["elo_surface_prob_a"].to_numpy()
    return out


# --------------------------------------------------------------------------- CLI


def _canonical(value: Any, allowed: Sequence[str], what: str) -> str:
    """``value`` matched case-insensitively against ``allowed``; ValueError listing the options otherwise."""
    m = {str(a).casefold(): str(a) for a in allowed}
    v = str(value).strip().casefold()
    if v not in m:
        raise ValueError(f"unknown {what} {value!r}; expected one of {list(allowed)}")
    return m[v]


def normalize_queries(raw: pd.DataFrame, state: EngineState, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Resolve names to ids and canonicalise surface / level / round / best_of. Collects every error and raises
    one ``ValueError`` (``PlayerLookupError`` when only names failed) listing them all."""
    missing = [c for c in DRAW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"draw is missing columns {missing}; expected {list(DRAW_COLUMNS)} [+ date]")
    errors: list[str] = []
    only_names = True
    rows: list[dict[str, Any]] = []
    for i, r in enumerate(raw.to_dict("records")):
        row = dict(r)
        for p in ("p1", "p2"):
            try:
                row[p] = resolve_player(str(r[p]), state)
            except PlayerLookupError as e:
                errors.append(f"row {i + 1} {p}: {e}")
        try:
            row["surface"] = _canonical(r["surface"], surfaces(cfg), "surface")
            row["level"] = _canonical(r["level"], cfg["data"]["training_levels"], "level")
            row["round"] = _canonical(r["round"], cfg["data"]["round_order"], "round")
            bo = int(r["best_of"])
            if bo not in (3, 5):
                raise ValueError(f"best_of must be 3 or 5, got {r['best_of']!r}")
            row["best_of"] = bo
        except (ValueError, TypeError) as e:
            only_names = False
            errors.append(f"row {i + 1}: {e}")
        rows.append(row)
    if errors:
        cls = PlayerLookupError if only_names else ValueError
        raise cls("\n".join(errors))
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface."""
    ap = argparse.ArgumentParser(prog="python -m tennis.model.predict",
                                 description="P(p1 wins) for ATP singles matches, from the state after the latest data.")
    ap.add_argument("--p1", help="player 1: name (case/accent-insensitive) or Sackmann id")
    ap.add_argument("--p2", help="player 2: name or id")
    ap.add_argument("--surface", help="Hard, Clay or Grass")
    ap.add_argument("--level", help="tourney level: one of data.training_levels (G, M, A, F, D, O)")
    ap.add_argument("--round", dest="round_", metavar="ROUND", help="round code, e.g. R128 ... QF, SF, F, RR")
    ap.add_argument("--best-of", type=int, choices=(3, 5), help="3 or 5 sets")
    ap.add_argument("--date", help="tournament start date YYYY-MM-DD (default: the day after the data cut-off)")
    ap.add_argument("--draw", help="CSV with columns p1,p2,surface,level,round,best_of[,date] (one match per row)")
    ap.add_argument("--out", help="draw mode: output CSV (default: <draw>_predictions.csv next to the draw)")
    ap.add_argument("--config", help="path to config.yaml (default: the repo config)")
    ap.add_argument("--rebuild-state", action="store_true", help="rebuild the cached engine state")
    ap.add_argument("--fit-production", action="store_true", help="refit the production model even if it exists")
    return ap


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.draw:
        single = [f"--{n}" for n, v in (("p1", a.p1), ("p2", a.p2)) if v]
        if single:
            ap.error(f"--draw can't be combined with {', '.join(single)}")
    else:
        need = [n for n, v in (("--p1", a.p1), ("--p2", a.p2), ("--surface", a.surface), ("--level", a.level),
                               ("--round", a.round_), ("--best-of", a.best_of)) if v is None]
        if need:
            ap.error(f"single-match mode needs {', '.join(need)} (or use --draw)")
    return a


def _print_single(r: Mapping[str, Any], state: EngineState) -> None:
    print(f"Data cut-off: last tourney_date {state.cutoff_date} ({state.n_matches:,} matches processed); "
          f"query date {r['date']}")
    print(f"{r['p1_name']} (id {r['p1']}) vs {r['p2_name']} (id {r['p2']}): {r['surface']}, level {r['level']}, "
          f"round {r['round']}, best of {r['best_of']}")
    print(f"  P({r['p1_name']} wins)")
    print(f"    XGBoost (production): {r['p_xgb']:.4f}")
    print(f"    Elo (overall):        {r['p_elo']:.4f}")
    print(f"    Elo (surface):        {r['p_elo_surface']:.4f}")


def main(argv: Sequence[str] | None = None, cfg: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Run the CLI; returns the predictions frame. Errors print to stderr and exit with status 2."""
    a = _parse(argv)
    cfg = dict(cfg) if cfg is not None else load_config(a.config)
    state = load_or_build_state(cfg, rebuild=a.rebuild_state)
    try:
        if a.draw:
            raw = pd.read_csv(a.draw, dtype=str, keep_default_na=False)
        else:
            raw = pd.DataFrame([{"p1": a.p1, "p2": a.p2, "surface": a.surface, "level": a.level,
                                 "round": a.round_, "best_of": a.best_of, "date": a.date or ""}])
        queries = normalize_queries(raw, state, cfg)
        booster, _ = load_or_fit_production(cfg, refit=a.fit_production)
        preds = predict_queries(state, booster, queries, cfg)
    except (ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(2) from e
    if a.draw:
        cols = ["p1_name", "p2_name", "surface", "level", "round", "best_of", "date", "p_xgb", "p_elo",
                "p_elo_surface"]
        print(f"Data cut-off: last tourney_date {state.cutoff_date}; {len(preds)} matches, P(p1 wins):")
        with pd.option_context("display.width", 200, "display.max_rows", 500):
            print(preds[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        out = Path(a.out) if a.out else Path(a.draw).with_name(Path(a.draw).stem + "_predictions.csv")
        keep = ["p1", "p2", "p1_name", "p2_name", "surface", "level", "round", "best_of", "date", "p_xgb",
                "p_xgb_ab", "p_xgb_ba", "p_elo", "p_elo_surface"]
        preds[keep].assign(data_cutoff=str(state.cutoff_date)).to_csv(out, index=False)
        print(f"wrote {out}")
    else:
        _print_single(preds.iloc[0], state)
    return preds


if __name__ == "__main__":
    main()
