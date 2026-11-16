"""Unit tests for head-to-head counts (PLAN 5.1, CONTRACT section 2)."""

from __future__ import annotations

from tennis.engine.h2h import H2H


def test_empty_is_zero() -> None:
    h = H2H()
    assert h.wins("a", "b") == 0
    assert h.surface_wins("a", "b", "Clay") == 0


def test_record_and_query_overall_and_surface() -> None:
    h = H2H()
    h.record_win("a", "b", "Clay")
    h.record_win("a", "b", "Hard")
    h.record_win("a", "b", "Clay")
    h.record_win("b", "a", "Grass")
    assert h.wins("a", "b") == 3
    assert h.wins("b", "a") == 1
    assert h.surface_wins("a", "b", "Clay") == 2
    assert h.surface_wins("a", "b", "Hard") == 1
    assert h.surface_wins("a", "b", "Grass") == 0
    assert h.surface_wins("b", "a", "Grass") == 1
    # other pairs untouched
    assert h.wins("a", "c") == 0


def test_query_does_not_create_entries() -> None:
    h = H2H()
    h.wins("x", "y")
    h.surface_wins("x", "y", "Hard")
    assert len(h) == 0
