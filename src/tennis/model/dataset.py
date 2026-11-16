"""Loading the feature table and choosing model inputs (PLAN 5.6, CONTRACT section 5).

- :func:`feature_columns` is the ONLY way model code gets its input columns: the explicit allow-list
  ``cfg["model"]["features"]``, checked against the banned same-match names.
- :func:`maybe_augment_train` adds A/B-flipped copies of TRAINING rows when ``features.flip_augment_train`` is
  true. It refuses rows dated after the split's ``train_end`` (D-021): each real match must be scored exactly once,
  in its coin-flip orientation. Flipped copies carry ``is_flipped = 1``; evaluation code calls
  :func:`assert_not_augmented` on validation/test frames.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from tennis.config import repo_path

BANNED_PREFIXES: tuple[str, ...] = ("w_", "l_")
BANNED_SUBSTRINGS: tuple[str, ...] = ("winner", "loser", "score", "minutes")
NON_FEATURES: frozenset[str] = frozenset({
    "match_id", "date", "tourney_id", "tourney_name", "a_id", "b_id", "a_name", "b_name", "y",
    "order_key", "tourney_date", "is_flipped",
})
PROB_COLUMNS: tuple[str, ...] = ("elo_prob_a", "elo_surface_prob_a")


PLAYER_PREFIXES: tuple[str, ...] = ("a_", "b_", "d_")


def is_banned(name: str) -> bool:
    """True for same-match columns (``w_``/``l_`` prefix, also behind one ``a_``/``b_``/``d_`` player prefix, e.g.
    ``a_w_ace``; winner/loser/score/minutes substrings) and non-feature columns.

    The ``w_``/``l_`` check is a prefix check on the name with one player prefix stripped. A plain substring check
    would wrongly ban e.g. ``level_A`` (audit phase5 finding 1).
    """
    low = name.lower()
    core = low[2:] if low.startswith(PLAYER_PREFIXES) else low
    return (low.startswith(BANNED_PREFIXES) or core.startswith(BANNED_PREFIXES)
            or any(s in low for s in BANNED_SUBSTRINGS)
            or name in NON_FEATURES or low.endswith("_id") or low.endswith("_name"))


def load_features(cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Read ``paths.features`` (data/processed/features.parquet) with ``date`` as datetime64."""
    df = pd.read_parquet(repo_path(cfg["paths"]["features"]))
    df["date"] = pd.to_datetime(df["date"])
    return df


def feature_columns(cfg: Mapping[str, Any], df: pd.DataFrame | None = None) -> list[str]:
    """The model input columns: ``cfg["model"]["features"]``, validated.

    Raises ``ValueError`` if a listed name is banned (same-match or non-feature column), duplicated, or missing
    from ``df`` (or, when ``df`` is None, from the engine's generated column list).
    """
    feats = list((cfg.get("model") or {}).get("features") or [])
    if not feats:
        raise ValueError("cfg['model']['features'] is empty or missing")
    bad = [f for f in feats if is_banned(f)]
    if bad:
        raise ValueError(f"banned columns in the feature allow-list: {bad}")
    if len(set(feats)) != len(feats):
        raise ValueError("duplicate names in the feature allow-list")
    if df is not None:
        available = set(df.columns)
    else:
        from tennis.engine.build_features import output_columns
        available = set(output_columns(cfg))
    missing = [f for f in feats if f not in available]
    if missing:
        raise ValueError(f"allow-listed features missing from the table: {missing}")
    return feats


def flip_orientation(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with players A and B swapped.

    ``a_X`` <-> ``b_X`` for every pair present (ids, names, player features), ``d_X`` -> ``-d_X``,
    ``elo_prob_a``/``elo_surface_prob_a`` -> ``1 - p``, ``y`` -> ``1 - y``. Context columns are unchanged.
    """
    out = df.copy()
    for c in df.columns:
        if c.startswith("a_") and ("b_" + c[2:]) in df.columns:
            b = "b_" + c[2:]
            out[c] = df[b].to_numpy()
            out[b] = df[c].to_numpy()
        elif c.startswith("d_"):
            out[c] = -df[c]
    for c in PROB_COLUMNS:
        if c in df.columns:
            out[c] = 1.0 - df[c]
    if "y" in df.columns:
        out["y"] = 1 - df["y"]
    return out


def _train_end(cfg: Mapping[str, Any], split_name: str | None) -> pd.Timestamp:
    """``cfg.splits[split_name].train_end``; with no split name, the latest train_end of every split."""
    splits = cfg["splits"]
    if split_name is not None:
        return pd.Timestamp(splits[split_name]["train_end"])  # KeyError for an unknown split
    ends = [pd.Timestamp(s["train_end"]) for s in splits.values() if "train_end" in s]
    if not ends:
        raise ValueError("no split in cfg['splits'] defines train_end")
    return max(ends)


def maybe_augment_train(train_df: pd.DataFrame, cfg: Mapping[str, Any], split_name: str | None = None
                        ) -> pd.DataFrame:
    """TRAINING rows only: append A/B-flipped copies when ``features.flip_augment_train`` is true.

    Guard (CONTRACT section 4b, DECISIONS D-021): raises ``ValueError`` if any row's ``date`` is after
    ``cfg["splits"][split_name]["train_end"]`` (with ``split_name=None``: after the latest ``train_end`` of every
    split), or if the frame already contains flipped rows. The guard runs whether or not the flag is set.

    Returns ``train_df`` itself when the flag is false. Otherwise a new frame: originals with ``is_flipped = 0``
    followed by flipped copies with ``is_flipped = 1`` (same ``match_id``). Never draw an early-stopping or CV
    holdout from the augmented frame: a flipped twin would sit on the other side of the cut.
    """
    end = _train_end(cfg, split_name)
    dates = pd.to_datetime(train_df["date"])
    late = dates > end
    if bool(late.any()):
        raise ValueError(f"maybe_augment_train got {int(late.sum())} rows dated after train_end {end.date()} "
                         f"(split {split_name!r}; latest row {dates.max().date()}): only training rows may be "
                         "augmented")
    assert_not_augmented(train_df)
    if not bool(cfg["features"].get("flip_augment_train", False)):
        return train_df
    orig = train_df.assign(is_flipped=0)
    flipped = flip_orientation(train_df).assign(is_flipped=1)
    return pd.concat([orig, flipped], ignore_index=True)


def assert_not_augmented(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if ``df`` contains flipped copies (``is_flipped`` present and not all 0).

    Evaluation code calls this on every validation / test frame before scoring (CONTRACT section 4b).
    """
    if "is_flipped" in df.columns and bool((df["is_flipped"] != 0).any()):
        raise ValueError(f"frame contains {int((df['is_flipped'] != 0).sum())} flipped (augmented) rows; "
                         "validation/test rows must never be augmented")
