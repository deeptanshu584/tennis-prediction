"""Time-based train / validation / test splits (PLAN section 2 rule 5, CONTRACT section 4).

:func:`time_split` is the only way model code partitions the feature table. It never shuffles: every partition is
a date window (or, for the replication test, one named tournament), and the result is checked to be disjoint and
time-ordered before it is returned.

Split types, chosen by the keys present in ``cfg["splits"][split_name]``:

- **dev** (no test keys): ``train`` and ``val`` by inclusive date windows on ``date``; there is no ``test`` key.
- **replication** (``test_tourney`` + ``test_year``): ``train`` / ``val`` by dates; ``test`` = rows with
  ``tourney_name == test_tourney`` and ``date.year == test_year``, which must all start after ``val_end``.
  Rows dated after ``val_end`` that are not the test tournament (for Wimbledon 2025: every other event starting
  2025-06-30 or later, about 1,100 rows) belong to NO partition. They are simply not returned.
- **forward** (``test_start``, optional ``test_end``): ``train`` / ``val`` by dates; ``test`` = ``date >=
  test_start`` and, when given, ``date <= test_end``.

``date`` is the tournament START date (``tourney_date``), so a whole event always falls in one partition.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

PARTITIONS: tuple[str, ...] = ("train", "val", "test")
_REQUIRED_KEYS: tuple[str, ...] = ("train_start", "train_end", "val_start", "val_end")


def split_kind(spec: Mapping[str, Any]) -> str:
    """``"replication"`` if the spec names a test tournament, ``"forward"`` if it has ``test_start``, else ``"dev"``."""
    if "test_tourney" in spec:
        return "replication"
    if "test_start" in spec:
        return "forward"
    return "dev"


def _ts(spec: Mapping[str, Any], key: str) -> pd.Timestamp:
    return pd.Timestamp(spec[key])


def _check_spec(spec: Mapping[str, Any], split_name: str) -> None:
    """Raise ``ValueError`` if the configured windows are missing, inverted or overlapping."""
    missing = [k for k in _REQUIRED_KEYS if k not in spec]
    if missing:
        raise ValueError(f"split {split_name!r} is missing keys {missing}")
    if "test_tourney" in spec and "test_start" in spec:
        raise ValueError(f"split {split_name!r} defines both test_tourney and test_start")
    if "test_tourney" in spec and "test_year" not in spec:
        raise ValueError(f"split {split_name!r} has test_tourney but no test_year")
    ts, te, vs, ve = (_ts(spec, k) for k in _REQUIRED_KEYS)
    if not (ts <= te < vs <= ve):
        raise ValueError(f"split {split_name!r}: need train_start <= train_end < val_start <= val_end, got "
                         f"{ts.date()} / {te.date()} / {vs.date()} / {ve.date()}")
    if "test_start" in spec:
        t0 = _ts(spec, "test_start")
        if not t0 > ve:
            raise ValueError(f"split {split_name!r}: test_start {t0.date()} must be after val_end {ve.date()}")
        if "test_end" in spec and _ts(spec, "test_end") < t0:
            raise ValueError(f"split {split_name!r}: test_end before test_start")


def _check_partitions(parts: dict[str, pd.DataFrame], split_name: str) -> None:
    """Raise ``ValueError`` unless the partitions are duplicate-free, disjoint and strictly time-ordered."""
    for name, part in parts.items():
        if part["match_id"].duplicated().any():
            raise ValueError(f"{split_name}.{name}: duplicate match_id")
    names = [p for p in PARTITIONS if p in parts]
    for i, p in enumerate(names):
        for q in names[i + 1:]:
            both = set(parts[p]["match_id"]) & set(parts[q]["match_id"])
            if both:
                raise ValueError(f"{split_name}: {len(both)} match_id in both {p} and {q}")
    present = [p for p in names if len(parts[p])]
    for p, q in zip(present, present[1:]):
        if not parts[p]["date"].max() < parts[q]["date"].min():
            raise ValueError(f"{split_name}: max({p}.date) {parts[p]['date'].max().date()} is not before "
                             f"min({q}.date) {parts[q]['date'].min().date()}")


def time_split(df: pd.DataFrame, split_name: str, cfg: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """Partition ``df`` by date for ``cfg["splits"][split_name]`` (CONTRACT section 4). Never shuffles.

    Args:
        df: feature table with at least ``match_id``, ``date`` (tournament start date) and, for a replication
            split, ``tourney_name``. Not modified.
        split_name: a key of ``cfg["splits"]`` (``dev``, ``replication``, ``forward``); ``KeyError`` otherwise.
        cfg: the loaded config.

    Returns:
        ``{"train": ..., "val": ...}`` plus ``"test"`` for replication / forward splits. Each partition is a new
        frame (a copy), ordered by ``date`` with a stable sort, so rows keep their input order within a date
        (the production table is already in authoritative match order). Window bounds are inclusive.
        Rows outside every window (e.g. events after ``val_end`` other than the replication test tournament)
        are dropped, not returned.

    Raises:
        ValueError: if ``df`` has duplicate ``match_id``, the configured windows are inverted / overlapping, the
            replication test tournament does not start after ``val_end``, or the resulting partitions are not
            disjoint and strictly time-ordered (max(train.date) < min(val.date) < min(test.date)).
    """
    spec = cfg["splits"][split_name]
    _check_spec(spec, split_name)
    if df["match_id"].duplicated().any():
        raise ValueError(f"input has {int(df['match_id'].duplicated().sum())} duplicate match_id rows")

    d = pd.to_datetime(df["date"])
    masks: dict[str, pd.Series] = {
        p: (d >= _ts(spec, f"{p}_start")) & (d <= _ts(spec, f"{p}_end")) for p in ("train", "val")
    }
    kind = split_kind(spec)
    if kind == "replication":
        masks["test"] = (df["tourney_name"] == spec["test_tourney"]) & (d.dt.year == int(spec["test_year"]))
        if masks["test"].any() and not d[masks["test"]].min() > _ts(spec, "val_end"):
            raise ValueError(f"{split_name}: test tourney {spec['test_tourney']!r} {spec['test_year']} starts "
                             f"{d[masks['test']].min().date()}, not after val_end {_ts(spec, 'val_end').date()}")
    elif kind == "forward":
        m = d >= _ts(spec, "test_start")
        if "test_end" in spec:
            m &= d <= _ts(spec, "test_end")
        masks["test"] = m

    parts: dict[str, pd.DataFrame] = {}
    for p, mask in masks.items():
        part = df.loc[mask].copy()
        part["date"] = pd.to_datetime(part["date"])
        parts[p] = part.sort_values("date", kind="mergesort")
    _check_partitions(parts, split_name)
    return parts


def unassigned(df: pd.DataFrame, split_name: str, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Rows of ``df`` that fall in no partition of the split (for reports; e.g. post-``val_end`` events that are
    not the replication test tournament, or rows outside every configured window)."""
    used: set[str] = set()
    for part in time_split(df, split_name, cfg).values():
        used |= set(part["match_id"])
    return df.loc[~df["match_id"].isin(used)]
