"""Split invariants for tennis.model.splits.time_split (CONTRACT section 4) on a small synthetic frame."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from tennis.config import load_config, repo_path
from tennis.model.splits import split_kind, time_split, unassigned

# (tourney_name, start date, matches)
EVENTS = [
    ("Wimbledon", "2018-07-02", 3),     # an earlier Wimbledon: must stay in train
    ("Doha", "2018-12-31", 2),          # on train_end: train (inclusive)
    ("Doha", "2019-01-01", 2),          # on val_start: val (inclusive)
    ("Rome", "2019-05-13", 2),
    ("Eastbourne", "2019-06-23", 2),    # on val_end
    ("Wimbledon", "2019-07-01", 4),     # replication test
    ("Newport", "2019-07-01", 2),       # same start date as the test, other event: in no partition
    ("Hamburg", "2019-07-22", 2),       # after val_end, not the test: in no partition
    ("Basel", "2019-10-21", 2),
    ("Brisbane", "2020-01-06", 3),
    ("Miami", "2020-03-23", 2),
]


def _frame() -> pd.DataFrame:
    rows = []
    for name, start, n in EVENTS:
        for i in range(n):
            rows.append({"match_id": f"{name[:3]}{start}-{i}", "date": pd.Timestamp(start), "tourney_name": name,
                         "tourney_id": f"{start[:4]}-{name}", "y": i % 2})
    return pd.DataFrame(rows)


def _cfg() -> dict:
    return {"seed": 7, "splits": {
        "dev": {"train_start": "2018-01-01", "train_end": "2018-12-31",
                "val_start": "2019-01-01", "val_end": "2019-12-31"},
        "replication": {"train_start": "2018-01-01", "train_end": "2018-12-31",
                        "val_start": "2019-01-01", "val_end": "2019-06-30",
                        "test_tourney": "Wimbledon", "test_year": 2019},
        "forward": {"train_start": "2018-01-01", "train_end": "2018-12-31",
                    "val_start": "2019-01-01", "val_end": "2019-06-30",
                    "test_start": "2019-07-01", "test_end": "2019-12-31"},
    }}


def _ids(df: pd.DataFrame) -> set[str]:
    return set(df["match_id"])


def _names(df: pd.DataFrame) -> list[tuple[str, str]]:
    return sorted({(r.tourney_name, str(r.date.date())) for r in df.itertuples()})


def _assert_invariants(parts: dict[str, pd.DataFrame], df: pd.DataFrame) -> None:
    order = [p for p in ("train", "val", "test") if p in parts]
    for p in order:
        assert parts[p]["match_id"].is_unique
        assert _ids(parts[p]) <= _ids(df)
        assert parts[p]["date"].is_monotonic_increasing
    for i, p in enumerate(order):
        for q in order[i + 1:]:
            assert not (_ids(parts[p]) & _ids(parts[q]))
            if len(parts[p]) and len(parts[q]):
                assert parts[p]["date"].max() < parts[q]["date"].min()


def test_split_kind() -> None:
    s = _cfg()["splits"]
    assert [split_kind(s[k]) for k in ("dev", "replication", "forward")] == ["dev", "replication", "forward"]


def test_dev_split_inclusive_bounds_and_no_test() -> None:
    df = _frame()
    parts = time_split(df, "dev", _cfg())
    assert set(parts) == {"train", "val"}
    _assert_invariants(parts, df)
    assert _names(parts["train"]) == [("Doha", "2018-12-31"), ("Wimbledon", "2018-07-02")]
    assert parts["val"]["date"].min() == pd.Timestamp("2019-01-01")
    assert parts["val"]["date"].max() == pd.Timestamp("2019-10-21")
    assert len(parts["train"]) + len(parts["val"]) == (df["date"] < "2020-01-01").sum()


def test_replication_split_test_is_the_tourney_and_gap_rows_are_unassigned() -> None:
    df, cfg = _frame(), _cfg()
    parts = time_split(df, "replication", cfg)
    assert set(parts) == {"train", "val", "test"}
    _assert_invariants(parts, df)
    assert _names(parts["test"]) == [("Wimbledon", "2019-07-01")]
    assert len(parts["test"]) == 4
    assert ("Wimbledon", "2018-07-02") in _names(parts["train"])          # earlier Wimbledon stays in train
    assert parts["val"]["date"].max() == pd.Timestamp("2019-06-23")
    gap = unassigned(df, "replication", cfg)
    assert _names(gap) == [("Basel", "2019-10-21"), ("Brisbane", "2020-01-06"), ("Hamburg", "2019-07-22"),
                           ("Miami", "2020-03-23"), ("Newport", "2019-07-01")]
    assert len(gap) + sum(len(p) for p in parts.values()) == len(df)


def test_replication_test_must_start_after_val_end() -> None:
    cfg = _cfg()
    cfg["splits"]["replication"]["val_end"] = "2019-07-01"   # the test tourney's start date
    with pytest.raises(ValueError, match="not after val_end"):
        time_split(_frame(), "replication", cfg)


def test_forward_split_respects_test_end() -> None:
    df, cfg = _frame(), _cfg()
    parts = time_split(df, "forward", cfg)
    _assert_invariants(parts, df)
    assert parts["test"]["date"].min() == pd.Timestamp("2019-07-01")
    assert parts["test"]["date"].max() == pd.Timestamp("2019-10-21")
    assert _ids(unassigned(df, "forward", cfg)) == _ids(df[df["date"] >= "2020-01-01"])


def test_forward_split_without_test_end_runs_to_last_date() -> None:
    cfg = _cfg()
    del cfg["splits"]["forward"]["test_end"]
    parts = time_split(_frame(), "forward", cfg)
    assert parts["test"]["date"].max() == pd.Timestamp("2020-03-23")
    assert len(parts["test"]) == 4 + 2 + 2 + 2 + 3 + 2


@pytest.mark.parametrize("name", ["dev", "replication", "forward"])
def test_row_order_does_not_matter_and_input_untouched(name: str) -> None:
    df = _frame()
    shuffled = df.sample(frac=1.0, random_state=3).reset_index(drop=True)
    before = shuffled.copy(deep=True)
    a = time_split(df, name, _cfg())
    b = time_split(shuffled, name, _cfg())
    pd.testing.assert_frame_equal(shuffled, before)
    assert {k: _ids(v) for k, v in a.items()} == {k: _ids(v) for k, v in b.items()}
    _assert_invariants(b, shuffled)


def test_string_dates_are_accepted() -> None:
    df = _frame()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    parts = time_split(df, "dev", _cfg())
    assert pd.api.types.is_datetime64_any_dtype(parts["train"]["date"])
    assert len(parts["train"]) == 5


def test_duplicate_match_id_raises() -> None:
    df = _frame()
    df.loc[1, "match_id"] = df.loc[0, "match_id"]
    with pytest.raises(ValueError, match="duplicate"):
        time_split(df, "dev", _cfg())


@pytest.mark.parametrize("key,value", [("train_end", "2019-01-01"),   # overlaps val
                                       ("val_end", "2018-06-01"),     # inverted
                                       ("train_start", "2019-02-01")])
def test_bad_windows_raise(key: str, value: str) -> None:
    cfg = _cfg()
    cfg["splits"]["dev"][key] = value
    with pytest.raises(ValueError):
        time_split(_frame(), "dev", cfg)


def test_forward_test_start_must_follow_val_end() -> None:
    cfg = _cfg()
    cfg["splits"]["forward"]["test_start"] = "2019-06-30"
    with pytest.raises(ValueError, match="test_start"):
        time_split(_frame(), "forward", cfg)


def test_unknown_split_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        time_split(_frame(), "nope", _cfg())


def test_missing_keys_raise() -> None:
    cfg = copy.deepcopy(_cfg())
    del cfg["splits"]["dev"]["val_start"]
    with pytest.raises(ValueError, match="missing"):
        time_split(_frame(), "dev", cfg)


@pytest.mark.realdata
def test_real_feature_table_splits() -> None:
    cfg = load_config()
    path = repo_path(cfg["paths"]["features"])
    if not path.exists():
        pytest.skip(f"{path} not found")
    df = pd.read_parquet(path, columns=["match_id", "date", "tourney_name"])
    df["date"] = pd.to_datetime(df["date"])
    for name in cfg["splits"]:
        parts = time_split(df, name, cfg)
        _assert_invariants(parts, df)
        assert len(parts["train"]) > 1000 and len(parts["val"]) > 500
    rep = time_split(df, "replication", cfg)["test"]
    assert set(rep["tourney_name"]) == {cfg["splits"]["replication"]["test_tourney"]}
    assert set(rep["date"].dt.year) == {cfg["splits"]["replication"]["test_year"]}
    assert rep["date"].min() > pd.Timestamp(cfg["splits"]["replication"]["val_end"])
    assert "test" not in time_split(df, "dev", cfg)
