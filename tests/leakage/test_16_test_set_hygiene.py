"""Test 16 (Phase 5 audit): the test sets are evaluated once, and only by their own split (PLAN section 2 rule 6).

Scans every line of experiments/*.jsonl (the experiment log, PLAN section 8). Test periods come from config:
- replication: tourney dates (val_end, val_end + 14 days] (the Wimbledon 2025 fortnight; the event starts 2025-06-30);
- forward: [test_start, test_end] (2026-04-01 .. 2026-06-07).

Rules:
1. A record of any other split (dev, Elo tuning, ...) carries no metric whose key path mentions ``test``, no
   date window (a ``[start, end]`` pair of ISO dates anywhere in ``params``) that overlaps a test period, and no
   per-season metric (``...seasons.<year>``) for a year that contains a test period.
   (Windows are looked for in both ``params`` and ``metrics``.)
2. Tuning / trial logs (file name contains ``trial`` or ``tun``) carry no test metric at all, whatever the split.
3. For the replication and forward splits, all records carrying test metrics in one file share ONE config hash:
   the test is scored only after every setting is frozen, not once per candidate.
4. A replication/forward record's validation window does not overlap that split's test period.

The checker is self-tested on synthetic records so this test fails when the problem is present.

Catches: test windows used for selection or reported before Phases 6/7; tuning code that scores the test;
"evaluate the test, change something, evaluate again".
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import pytest

from tennis.config import repo_path

FINAL_SPLITS = ("replication", "forward")
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}")
YEAR = re.compile(r"^(19|20)\d{2}$")


def _walk(obj: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Any]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, path + (str(k),))
    elif isinstance(obj, list) and not _is_window(obj):
        for i, v in enumerate(obj):
            yield from _walk(v, path + (str(i),))
    else:
        yield path, obj


def _is_window(v: Any) -> bool:
    return (isinstance(v, list) and len(v) == 2 and all(isinstance(x, str) and ISO.match(x) for x in v))


def test_periods(cfg: dict[str, Any]) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    rep, fwd = cfg["splits"]["replication"], cfg["splits"]["forward"]
    ve = pd.Timestamp(rep["val_end"])
    return {"replication": (ve + pd.Timedelta(days=1), ve + pd.Timedelta(days=14)),
            "forward": (pd.Timestamp(fwd["test_start"]), pd.Timestamp(fwd["test_end"]))}


test_periods.__test__ = False  # a helper, not a pytest test


def _overlaps(win: list[str], period: tuple[pd.Timestamp, pd.Timestamp]) -> bool:
    a, b = pd.Timestamp(win[0][:10]), pd.Timestamp(win[1][:10])
    return a <= period[1] and b >= period[0]


def check_records(records: list[tuple[str, int, dict[str, Any]]], cfg: dict[str, Any]) -> list[str]:
    """Violations of rules 1-4 in ``(file name, line number, record)`` triples."""
    periods = test_periods(cfg)
    test_years = {y for lo, hi in periods.values() for y in range(lo.year, hi.year + 1)}
    bad: list[str] = []
    hashes: dict[tuple[str, str], set[str]] = {}
    for fname, n, rec in records:
        where = f"{fname}:{n}"
        split = str(rec.get("split", ""))
        metric_paths = [p for p, _ in _walk(rec.get("metrics") or {})]
        test_metric = [".".join(p) for p in metric_paths if any("test" in s.lower() for s in p)]
        is_trial = "trial" in fname.lower() or "tun" in fname.lower()
        if test_metric and is_trial:
            bad.append(f"{where}: tuning/trial log carries test metrics {test_metric[:3]}")
        windows = [(p, v) for p, v in _walk({"params": rec.get("params") or {}, "metrics": rec.get("metrics") or {}})
                   if _is_window(v)]
        if split in FINAL_SPLITS:
            if test_metric:
                hashes.setdefault((fname, split), set()).add(str(rec.get("config_hash")))
            for p, v in windows:
                if "val" in ".".join(p).lower() and _overlaps(v, periods[split]):
                    bad.append(f"{where}: {split} validation window {v} overlaps its test period")
            continue
        if test_metric:
            bad.append(f"{where}: split {split!r} carries test metrics {test_metric[:3]}")
        for p, v in windows:
            if _is_window(v):
                for name, period in periods.items():
                    if _overlaps(v, period):
                        bad.append(f"{where}: split {split!r} window {'.'.join(p)}={v} overlaps the {name} test "
                                   f"period {period[0].date()}..{period[1].date()}")
        for p in metric_paths:
            for i, s in enumerate(p[:-1]):
                if s.lower() == "seasons" and YEAR.match(p[i + 1]) and int(p[i + 1]) in test_years:
                    bad.append(f"{where}: split {split!r} reports season {p[i + 1]}, which contains a test period")
                    break
    for (fname, split), hs in hashes.items():
        if len(hs) > 1:
            bad.append(f"{fname}: {split} test metrics logged under {len(hs)} different config hashes {sorted(hs)} "
                       "(the test must be evaluated once, after all settings are frozen)")
    return sorted(set(bad))


def _load(cfg: dict[str, Any]) -> list[tuple[str, int, dict[str, Any]]]:
    out = []
    for f in sorted(Path(repo_path(cfg["paths"]["experiments"])).glob("*.jsonl")):
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                out.append((f.name, n, json.loads(line)))
    return out


def test_checker_flags_violations(cfg: dict) -> None:
    ok_dev = {"split": "dev", "config_hash": "h1", "params": {"val_window": ["2023-01-02", "2024-12-30"]},
              "metrics": {"val": {"accuracy": 0.65}, "seasons": {"2024": {"accuracy": 0.64}}}}
    ok_final = {"split": "replication", "config_hash": "h1",
                "params": {"val_window": ["2025-01-06", "2025-06-23"]}, "metrics": {"test": {"accuracy": 0.66}}}
    assert check_records([("baselines.jsonl", 1, ok_dev), ("final.jsonl", 1, ok_final),
                          ("final.jsonl", 2, dict(ok_final))], cfg) == []
    cases = [
        ("baselines.jsonl", {**ok_dev, "metrics": {"test": {"accuracy": 0.7}}}),
        ("baselines.jsonl", {**ok_dev, "metrics": {"seasons": {"2025": {"accuracy": 0.64}}}}),
        ("baselines.jsonl", {**ok_dev, "metrics": {"seasons": {"2026": {"accuracy": 0.64}}}}),
        ("elo_tuning.jsonl", {**ok_dev, "params": {"window": ["2015-01-01", "2026-06-30"]}}),
        ("elo_tuning.jsonl", {**ok_dev, "params": {"window": ["2025-06-30", "2025-06-30"]}}),
        ("xgb_trials.jsonl", {**ok_final, "split": "forward"}),
        ("final.jsonl", {**ok_final, "params": {"val_window": ["2025-01-06", "2025-07-01"]}}),
        ("xgb_trials.jsonl", {**ok_dev, "metrics": {"val": {"accuracy": 0.6},
                                                    "val_window": ["2026-01-04", "2026-05-25"]}}),
    ]
    for fname, rec in cases:
        assert check_records([(fname, 1, rec)], cfg), f"not flagged: {fname} {rec}"
    twice = [("final.jsonl", 1, ok_final), ("final.jsonl", 2, {**ok_final, "config_hash": "h2"})]
    assert check_records(twice, cfg), "two config hashes scoring the same test were not flagged"


def test_experiment_logs_respect_test_hygiene(cfg: dict) -> None:
    records = _load(cfg)
    if not records:
        pytest.skip("no experiments/*.jsonl logged yet")
    bad = check_records(records, cfg)
    assert not bad, f"{len(bad)} test-set hygiene violations, e.g.:\n" + "\n".join(bad[:20])
