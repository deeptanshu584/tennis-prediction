"""Head-to-head win counts, overall and per surface (PLAN 5.1, CONTRACT section 2)."""

from __future__ import annotations


class H2H:
    """Prior wins of one player over another. Queries never create entries."""

    __slots__ = ("_overall", "_surface")

    def __init__(self) -> None:
        self._overall: dict[tuple[str, str], int] = {}
        self._surface: dict[tuple[str, str, str], int] = {}

    def __len__(self) -> int:
        return len(self._overall)

    def wins(self, player: str, opponent: str) -> int:
        """Prior wins of ``player`` over ``opponent`` on any surface."""
        return self._overall.get((player, opponent), 0)

    def surface_wins(self, player: str, opponent: str, surface: str) -> int:
        """Prior wins of ``player`` over ``opponent`` on ``surface``."""
        return self._surface.get((player, opponent, surface), 0)

    def record_win(self, winner: str, loser: str, surface: str) -> None:
        """Count one win of ``winner`` over ``loser`` on ``surface`` (call only after the row is written)."""
        k = (winner, loser)
        self._overall[k] = self._overall.get(k, 0) + 1
        ks = (winner, loser, surface)
        self._surface[ks] = self._surface.get(ks, 0) + 1
