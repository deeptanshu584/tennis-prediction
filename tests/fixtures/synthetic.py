"""Deterministic synthetic ATP-like match table in the unified schema (CONTRACT.md section 1).

`make_synthetic_matches()` simulates a few seasons of a small tour so that the leakage suite can run
against any feature builder without real data. It deliberately contains every awkward case the engine
must handle:

- tour events at levels A / G / M / F / D / O, Challengers (C), qualifying rounds Q1-Q3 feeding tour main draws;
- real draws: winners advance round by round (knock-out), a round robin (Tour Finals), a bronze match (Olympics);
- several tournaments starting on the same Monday, and players who play two tournaments with the same start date
  (tour-qualifying losers drop into the same-week Challenger);
- qualifying `match_num`s larger than the main-draw ones, and one event per season numbered "backwards"
  (final = match 1), so only the (tourney_date, tourney_id, round_rank, match_num) order is correct;
- all three surfaces, one indoor carpet event already normalised to Hard;
- best-of-3 and best-of-5, walkovers ('W/O'), retirements ('RET') and defaults ('DEF'), a retirement before the
  loser served a point (svpt = 0), matches with all stats missing and with one stat missing;
- realistic serve counts (svpt >= 1stIn >= 1stWon >= ace, 2ndWon + df <= svpt - 1stIn, bpSaved <= bpFaced);
- data-quality flags `surface_imputed` (one Davis Cup tie per season) and `stats_invalid` (stats blanked);
- rank / rank points (two never-ranked players), age, height (some missing), hand (R, L, U and missing);
- players who debut in the second season and one long injury layoff (> 100 days).

The rows are returned in a shuffled order (``shuffle=True``) so a builder that forgets to sort fails.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

STAT_FIELDS: tuple[str, ...] = (
    "ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced",
)
PLAYER_ATTRS: tuple[str, ...] = (
    "id", "seed", "entry", "name", "hand", "ht", "ioc", "age", "rank", "rank_points",
)

# Column order of the unified table (Sackmann order, then the CONTRACT extras).
SACKMANN_COLUMNS: list[str] = (
    ["tourney_id", "tourney_name", "surface", "draw_size", "tourney_level", "tourney_date", "match_num"]
    + [f"winner_{a}" for a in ("id", "seed", "entry", "name", "hand", "ht", "ioc", "age")]
    + [f"loser_{a}" for a in ("id", "seed", "entry", "name", "hand", "ht", "ioc", "age")]
    + ["score", "best_of", "round", "minutes"]
    + [f"w_{s}" for s in STAT_FIELDS]
    + [f"l_{s}" for s in STAT_FIELDS]
    + ["winner_rank", "winner_rank_points", "loser_rank", "loser_rank_points"]
)
EXTRA_COLUMNS: list[str] = [
    "match_id", "source", "round_rank", "order_key",
    "is_walkover", "is_retirement", "is_qualifying", "is_training_level",
    "surface_imputed", "stats_invalid",
]
UNIFIED_COLUMNS: list[str] = SACKMANN_COLUMNS + EXTRA_COLUMNS

QUALI_DRAW = 8
QUALI_MATCH_NUM_START = 300   # qualifying match numbers are LARGER than main-draw ones on purpose


def match_id_for(tourney_id: str, match_num: int, winner_id: str, loser_id: str) -> str:
    """CONTRACT section 1 match id: sha1 of tourney_id|match_num|min(id)|max(id), first 16 hex chars."""
    lo, hi = sorted([str(winner_id), str(loser_id)])
    return hashlib.sha1(f"{tourney_id}|{int(match_num)}|{lo}|{hi}".encode("utf-8")).hexdigest()[:16]


AUTHORITATIVE_ORDER: list[str] = ["tourney_date", "tourney_id", "round_rank", "match_num", "match_id"]


def assign_order_key(df: pd.DataFrame) -> pd.DataFrame:
    """CONTRACT section 1: order_key (int64) = 0-based position in the authoritative sort of the whole table."""
    out = df.copy()
    pos = out.sort_values(AUTHORITATIVE_ORDER, kind="mergesort").index
    out["order_key"] = pd.Series(range(len(out)), index=pos, dtype="int64").reindex(out.index)
    return out


@dataclass(frozen=True)
class _EventSpec:
    week: int
    level: str
    surface: str
    draw: int
    quali_rounds: int  # 0 = no qualifying; 2 -> Q1,Q2 (2 qualifiers); 3 -> Q1..Q3 (1 qualifier)
    best_of: int
    name: str
    legacy_numbering: bool = False  # final = match 1, then SF, QF ... (match_num decreasing with round)
    surface_imputed: bool = False   # source had no surface; ingestion imputed Hard


# One season. Weeks are counted from the first Monday of the year.
_SCHEDULE: tuple[_EventSpec, ...] = (
    _EventSpec(0, "A", "Hard", 16, 2, 3, "Harbour Open"),
    _EventSpec(0, "A", "Hard", 8, 0, 3, "Desert Classic", legacy_numbering=True),
    _EventSpec(0, "C", "Hard", 16, 0, 3, "Coastal Challenger"),
    _EventSpec(2, "G", "Hard", 32, 3, 5, "Southern Slam"),
    _EventSpec(2, "C", "Hard", 8, 0, 3, "Bay Challenger"),
    _EventSpec(5, "A", "Hard", 16, 2, 3, "Valley Open"),
    _EventSpec(5, "C", "Hard", 16, 0, 3, "Hill Challenger"),
    _EventSpec(6, "D", "Hard", 4, 0, 3, "Davis Cup Tie North", surface_imputed=True),
    _EventSpec(6, "D", "Clay", 4, 0, 3, "Davis Cup Tie South"),
    _EventSpec(8, "M", "Hard", 32, 2, 3, "Spring Masters"),
    _EventSpec(8, "C", "Hard", 8, 0, 3, "Canyon Challenger"),
    _EventSpec(10, "M", "Hard", 32, 2, 3, "Palm Masters"),
    _EventSpec(13, "A", "Clay", 16, 2, 3, "Red Earth Open"),
    _EventSpec(13, "A", "Clay", 8, 0, 3, "Riviera Cup"),
    _EventSpec(13, "C", "Clay", 8, 0, 3, "Terra Challenger"),
    _EventSpec(15, "M", "Clay", 32, 2, 3, "Clay Masters"),
    _EventSpec(17, "A", "Clay", 16, 2, 3, "Lake Open"),
    _EventSpec(17, "C", "Clay", 16, 0, 3, "Brick Challenger"),
    _EventSpec(19, "G", "Clay", 32, 3, 5, "Clay Slam"),
    _EventSpec(19, "C", "Clay", 8, 0, 3, "Dust Challenger"),
    _EventSpec(22, "A", "Grass", 16, 2, 3, "Lawn Open"),
    _EventSpec(22, "C", "Grass", 8, 0, 3, "Meadow Challenger"),
    _EventSpec(24, "G", "Grass", 32, 3, 5, "Grass Slam"),
    _EventSpec(27, "A", "Clay", 16, 2, 3, "Alpine Open"),
    _EventSpec(27, "C", "Clay", 16, 0, 3, "Summit Challenger"),
    _EventSpec(29, "O", "Hard", 16, 0, 3, "Olympic Tennis Event"),  # only in the middle season
    _EventSpec(31, "M", "Hard", 32, 2, 3, "Summer Masters"),
    _EventSpec(33, "A", "Hard", 16, 2, 3, "City Open"),
    _EventSpec(33, "C", "Hard", 16, 0, 3, "River Challenger"),
    _EventSpec(35, "G", "Hard", 32, 3, 5, "Northern Slam"),
    _EventSpec(38, "A", "Hard", 16, 2, 3, "Autumn Open"),
    _EventSpec(38, "A", "Hard", 8, 0, 3, "Indoor Carpet Trophy"),  # carpet, normalised to Hard
    _EventSpec(38, "C", "Hard", 8, 0, 3, "Forest Challenger"),
    _EventSpec(40, "M", "Hard", 32, 2, 3, "Indoor Masters"),
    _EventSpec(43, "F", "Hard", 8, 0, 3, "Tour Finals"),
    _EventSpec(43, "C", "Hard", 16, 0, 3, "Final Challenger"),
    _EventSpec(45, "D", "Hard", 4, 0, 3, "Davis Cup Finals Tie"),
)
_LEVEL_PRIORITY = {"F": 0, "G": 1, "M": 2, "O": 3, "A": 4, "D": 5, "C": 6}
_KO_ROUNDS = {32: ["R32", "R16", "QF", "SF", "F"], 16: ["R16", "QF", "SF", "F"], 8: ["QF", "SF", "F"],
              4: ["SF", "F"]}
_IOCS = ["ESP", "FRA", "USA", "SRB", "ITA", "GER", "ARG", "AUS", "GBR", "SUI", "RUS", "CRO"]


@dataclass
class _Player:
    pid: str
    name: str
    hand: str | None
    ht: float
    ioc: str
    birth: pd.Timestamp | None
    skill: float
    surf: dict[str, float]
    debut_year: int
    unranked: bool


@dataclass
class _Ctx:
    tourney_id: str
    tourney_name: str
    surface: str
    draw_size: int
    level: str
    date: pd.Timestamp
    best_of: int
    entries: dict[str, tuple[float, str | None]]  # pid -> (seed, entry)
    ranks: dict[str, tuple[float, float]]         # pid -> (rank, rank_points)
    surface_imputed: bool = False                 # surface was null in the source -> Hard


def _tourney_id(year: int, idx: int, level: str) -> str:
    # Ids look like Sackmann's: '2019-0105' (tour), '2019-C105' (Challenger), '2019-D106' (Davis Cup).
    if level == "C":
        return f"{year}-C{idx:03d}"
    if level == "D":
        return f"{year}-D{idx:03d}"
    return f"{year}-{idx:04d}"


def _make_players(rng: np.random.Generator, n: int, years: Sequence[int]) -> list[_Player]:
    players = []
    skills = np.sort(rng.normal(1700.0, 160.0, size=n))[::-1]
    for i in range(n):
        hand: str | None = "L" if i % 7 == 3 else "R"
        if i == n - 5:
            hand = "U"
        if i == n - 6:
            hand = None
        ht = float(rng.integers(170, 205)) if i % 11 != 5 else float("nan")
        birth = pd.Timestamp(int(rng.integers(1984, 2003)), int(rng.integers(1, 13)), int(rng.integers(1, 28)))
        players.append(_Player(
            pid=str(100001 + i),
            name=f"Synth Player{i + 1:02d}",
            hand=hand,
            ht=ht,
            ioc=_IOCS[i % len(_IOCS)],
            birth=None if i == n - 7 else birth,
            skill=float(skills[i]),
            surf={s: float(rng.normal(0.0, 60.0)) for s in ("Hard", "Clay", "Grass")},
            debut_year=years[1] if (len(years) > 1 and i >= n - 2) else years[0],
            unranked=i in (n - 3, n - 4),
        ))
    return players


def _first_monday(year: int) -> pd.Timestamp:
    d = pd.Timestamp(year, 1, 1)
    return d + pd.Timedelta(days=(7 - d.weekday()) % 7)


def _pow2_floor(n: int) -> int:
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


class _Season:
    """Simulates matches and accumulates unified-schema rows."""

    def __init__(self, rng: np.random.Generator, players: list[_Player], round_order: list[str],
                 training_levels: set[str], quali_rounds: set[str]) -> None:
        self.rng = rng
        self.players = {p.pid: p for p in players}
        self.round_rank = {r: i for i, r in enumerate(round_order)}
        self.training_levels = training_levels
        self.quali_rounds = quali_rounds
        self.rows: list[dict[str, Any]] = []
        self.forced: dict[tuple[str, str, int], str] = {}

    # ---- outcomes ---------------------------------------------------------------------------
    def _p_win(self, a: str, b: str, surface: str) -> float:
        pa, pb = self.players[a], self.players[b]
        ra, rb = pa.skill + pa.surf[surface], pb.skill + pb.surf[surface]
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def _outcome(self, ctx: _Ctx, rnd: str, pos: int, allow_wo: bool = True) -> str:
        forced = self.forced.get((ctx.tourney_id, rnd, pos))
        if forced:
            return forced
        u = self.rng.random()
        if allow_wo and u < 0.012:
            return "W/O"
        if u < 0.037:
            return "RET"
        if u < 0.041:
            return "DEF"
        return "normal"

    def play(self, ctx: _Ctx, rnd: str, match_num: int, a: str, b: str, pos: int = 0,
             allow_wo: bool = True) -> tuple[str, str, str]:
        """Play a match; append its row; return (winner, loser, outcome)."""
        w, l = (a, b) if self.rng.random() < self._p_win(a, b, ctx.surface) else (b, a)
        outcome = self._outcome(ctx, rnd, pos, allow_wo)
        self.rows.append(self._row(ctx, rnd, match_num, w, l, outcome))
        return w, l, outcome

    # ---- row construction ---------------------------------------------------------------------
    def _score(self, best_of: int, outcome: str) -> tuple[str, int]:
        """Score string and total games played."""
        if outcome == "W/O":
            return "W/O", 0
        if outcome == "RET0":
            return "0-0 RET", 0
        rng = self.rng
        need = (best_of + 1) // 2
        sets = ["W"] * (need - 1) + ["L"] * int(rng.integers(0, need))
        rng.shuffle(sets)
        sets.append("W")
        choices = [(6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (7, 5), (7, 6)]
        parts, games = [], 0
        for s in sets:
            hi, lo = choices[int(rng.integers(0, len(choices)))]
            games += hi + lo
            tb = f"({int(rng.integers(0, 10))})" if (hi, lo) == (7, 6) else ""
            parts.append(f"{hi}-{lo}{tb}" if s == "W" else f"{lo}-{hi}{tb}")
        if outcome in ("RET", "DEF"):
            keep = int(rng.integers(0, len(parts)))
            parts = parts[:keep]
            games = sum(int(p[0]) + int(p[2]) for p in parts)
            g1, g2 = int(rng.integers(0, 6)), int(rng.integers(0, 6))
            parts.append(f"{g1}-{g2}")
            games += g1 + g2
            parts.append(outcome)
        return " ".join(parts), games

    def _serve(self, games: int, quality: float) -> dict[str, float]:
        rng = self.rng
        svgms = max(1, int(round(games / 2 + rng.integers(-1, 2))))
        svpt = max(1, int(round(svgms * rng.uniform(5.2, 7.0))))
        first_in = int(rng.binomial(svpt, 0.6))
        first_won = int(rng.binomial(first_in, min(0.95, 0.64 + 0.12 * quality)))
        ace = int(rng.binomial(first_won, 0.15))
        second = svpt - first_in
        dfs = int(rng.binomial(second, 0.08))
        second_won = int(rng.binomial(second - dfs, min(0.95, 0.46 + 0.12 * quality)))
        bp_faced = int(rng.binomial(max(1, svpt // 6), 0.5))
        bp_saved = int(rng.binomial(bp_faced, 0.6))
        return {"ace": ace, "df": dfs, "svpt": svpt, "1stIn": first_in, "1stWon": first_won,
                "2ndWon": second_won, "SvGms": svgms, "bpSaved": bp_saved, "bpFaced": bp_faced}

    def _row(self, ctx: _Ctx, rnd: str, match_num: int, w: str, l: str, outcome: str) -> dict[str, Any]:
        rng = self.rng
        score, games = self._score(ctx.best_of, outcome)
        nan = float("nan")
        miss_p = {"D": 0.5, "C": 0.3}.get(ctx.level, 0.05)
        w_stats: dict[str, float] = {s: nan for s in STAT_FIELDS}
        l_stats: dict[str, float] = {s: nan for s in STAT_FIELDS}
        minutes = nan
        stats_invalid = False
        if outcome == "RET0":
            w_stats = {"ace": 0, "df": 0, "svpt": 4, "1stIn": 3, "1stWon": 2, "2ndWon": 1, "SvGms": 1,
                       "bpSaved": 0, "bpFaced": 0}
            l_stats = {s: 0 for s in STAT_FIELDS}
            minutes = 5.0
        elif outcome != "W/O" and rng.random() >= miss_p:
            w_stats = self._serve(games, 0.7)
            l_stats = self._serve(games, 0.3)
            minutes = float(15 + 4 * games + int(rng.integers(0, 15)))
            if outcome == "normal" and rng.random() < 0.015:   # one stat missing on one side
                side = w_stats if rng.random() < 0.5 else l_stats
                side[STAT_FIELDS[int(rng.integers(0, len(STAT_FIELDS)))]] = nan
            elif outcome == "normal" and rng.random() < 0.008:  # impossible counts, blanked by ingestion
                w_stats = {s: nan for s in STAT_FIELDS}
                l_stats = {s: nan for s in STAT_FIELDS}
                stats_invalid = True
        elif outcome != "W/O" and rng.random() < 0.5:
            minutes = float(15 + 4 * games)

        row: dict[str, Any] = {
            "tourney_id": ctx.tourney_id, "tourney_name": ctx.tourney_name, "surface": ctx.surface,
            "draw_size": ctx.draw_size, "tourney_level": ctx.level, "tourney_date": ctx.date,
            "match_num": int(match_num), "score": score, "best_of": ctx.best_of, "round": rnd,
            "minutes": minutes,
        }
        for side, pid in (("winner", w), ("loser", l)):
            p = self.players[pid]
            seed, entry = ctx.entries.get(pid, (nan, None))
            rank, pts = ctx.ranks.get(pid, (nan, nan))
            age = nan if p.birth is None else round((ctx.date - p.birth).days / 365.25, 1)
            row.update({
                f"{side}_id": pid, f"{side}_seed": seed, f"{side}_entry": entry, f"{side}_name": p.name,
                f"{side}_hand": p.hand, f"{side}_ht": p.ht, f"{side}_ioc": p.ioc, f"{side}_age": age,
                f"{side}_rank": rank, f"{side}_rank_points": pts,
            })
        for s in STAT_FIELDS:
            row[f"w_{s}"] = float(w_stats[s])
            row[f"l_{s}"] = float(l_stats[s])
        rr = self.round_rank[rnd]
        row["match_id"] = match_id_for(ctx.tourney_id, match_num, w, l)
        row["source"] = "sackmann"
        row["round_rank"] = rr
        row["order_key"] = -1  # assigned over the whole table in _finalise
        row["is_walkover"] = score == "W/O"
        row["is_retirement"] = score.endswith("RET") or score.endswith("DEF")
        row["is_qualifying"] = rnd in self.quali_rounds
        row["is_training_level"] = (ctx.level in self.training_levels) and rnd not in self.quali_rounds
        row["surface_imputed"] = ctx.surface_imputed
        row["stats_invalid"] = stats_invalid
        return row

    # ---- draws --------------------------------------------------------------------------------
    def knockout(self, ctx: _Ctx, entrants: list[str], rounds: list[str], match_nums: dict[str, list[int]],
                 ) -> tuple[list[str], list[str]]:
        """Play a knock-out draw. Returns (last-round winners, all losers)."""
        alive = list(entrants)
        losers: list[str] = []
        for rnd in rounds:
            nxt = []
            for j in range(len(alive) // 2):
                w, l, _ = self.play(ctx, rnd, match_nums[rnd][j], alive[2 * j], alive[2 * j + 1], pos=j + 1)
                nxt.append(w)
                losers.append(l)
            alive = nxt
        return alive, losers

    def main_draw_numbers(self, rounds: list[str], draw: int, legacy: bool) -> dict[str, list[int]]:
        counts = {}
        n = draw // 2
        for r in rounds:
            counts[r] = n
            n //= 2
        nums: dict[str, list[int]] = {}
        k = 1
        order = list(reversed(rounds)) if legacy else rounds
        for r in order:
            nums[r] = list(range(k, k + counts[r]))
            k += counts[r]
        return nums


def make_synthetic_matches(
    seed: int = 0,
    n_players: int = 44,
    years: Sequence[int] = (2019, 2020, 2021),
    cfg: dict[str, Any] | None = None,
    shuffle: bool = True,
) -> pd.DataFrame:
    """Simulate a small multi-season tour and return the unified (winner/loser oriented) match table.

    Deterministic for a given (seed, n_players, years, cfg["data"]). ``round_rank`` uses
    ``cfg["data"]["round_order"]``; ``is_training_level`` uses ``cfg["data"]["training_levels"]``.
    Rows are shuffled unless ``shuffle=False`` (then they are in chronological order).
    """
    if cfg is None:
        from tennis.config import load_config
        cfg = load_config()
    years = list(years)
    if n_players < 42:
        raise ValueError("n_players must be >= 42 (a Grand Slam main draw + qualifying needs 39 players)")
    rng = np.random.default_rng(seed)
    players = _make_players(rng, n_players, years)
    season = _Season(rng, players, list(cfg["data"]["round_order"]), set(cfg["data"]["training_levels"]),
                     set(cfg["data"]["qualifying_rounds"]))
    injured = players[5].pid  # long layoff in the first season (weeks 9..28 -> > 100 days)

    for yi, year in enumerate(years):
        specs = [(i, e) for i, e in enumerate(_SCHEDULE) if e.level != "O" or yi == len(years) // 2]
        # forced edge cases (deterministic), per season
        slams = [i for i, e in specs if e.level == "G"]
        masters = [i for i, e in specs if e.level == "M"]
        season.forced[(_tourney_id(year, 100 + slams[0], "G"), "R32", 2)] = "W/O"
        season.forced[(_tourney_id(year, 100 + slams[0], "G"), "R32", 4)] = "RET"
        season.forced[(_tourney_id(year, 100 + slams[0], "G"), "R32", 6)] = "DEF"
        season.forced[(_tourney_id(year, 100 + slams[1], "G"), "R32", 4)] = "RET0"
        season.forced[(_tourney_id(year, 100 + masters[0], "M"), "Q1", 2)] = "W/O"
        season.forced[(_tourney_id(year, 100 + masters[0], "M"), "R16", 1)] = "RET"

        monday = _first_monday(year)
        for week in sorted({e.week for _, e in specs}):
            date = monday + pd.Timedelta(days=7 * week)
            week_specs = sorted([(i, e) for i, e in specs if e.week == week],
                                key=lambda t: (_LEVEL_PRIORITY[t[1].level], t[0]))
            big_week = any(e.level in ("G", "M", "F") for _, e in week_specs)
            # weekly skill drift and ranking
            for p in players:
                p.skill += float(rng.normal(0.0, 12.0))
            active = [p for p in players if p.debut_year <= year]
            ranked = sorted([p for p in active if not p.unranked],
                            key=lambda p: -(p.skill + float(rng.normal(0.0, 40.0))))
            ranks = {p.pid: (float(r + 1), float(round(12000 / (r + 1) ** 0.9))) for r, p in enumerate(ranked)}
            available = []
            for p in active:
                if yi == 0 and p.pid == injured and 9 <= week <= 28:
                    continue
                if not big_week and rng.random() < 0.12:
                    continue
                available.append(p.pid)
            available.sort(key=lambda pid: (ranks.get(pid, (1e9,))[0], -season.players[pid].skill))
            pool = list(available)
            quali_losers: list[str] = []

            for idx, spec in week_specs:
                tid = _tourney_id(year, 100 + idx, spec.level)
                ctx = _Ctx(tid, spec.name, spec.surface, spec.draw, spec.level, date,
                           5 if (spec.level == "D" and yi == 0) else spec.best_of, {}, ranks,
                           spec.surface_imputed)
                if spec.level == "D":
                    if len(pool) < 4:
                        continue
                    pick = [pool[int(j)] for j in rng.choice(len(pool), size=4, replace=False)]
                    for pid in pick:
                        pool.remove(pid)
                    a1, a2, b1, b2 = pick
                    for k, (x, y) in enumerate([(a1, b1), (a2, b2), (a1, b2), (a2, b1)]):
                        season.play(ctx, "RR", k + 1, x, y, pos=k + 1, allow_wo=False)
                    continue
                if spec.level == "C":
                    cpool = sorted(set(pool) | set(quali_losers),
                                   key=lambda pid: (ranks.get(pid, (1e9,))[0], -season.players[pid].skill))
                    draw = min(spec.draw, _pow2_floor(len(cpool)))
                    if draw < 4:
                        continue
                    entrants = cpool[:draw]
                    for pid in entrants:
                        if pid in pool:
                            pool.remove(pid)
                    ctx.draw_size = draw
                    order = [entrants[int(j)] for j in rng.permutation(draw)]
                    rounds = _KO_ROUNDS[draw]
                    season.knockout(ctx, order, rounds, season.main_draw_numbers(rounds, draw, False))
                    continue
                if spec.level == "F":
                    _tour_finals(season, ctx, pool[:8])
                    del pool[:8]
                    continue

                draw, q_rounds = spec.draw, spec.quali_rounds
                n_qual = (QUALI_DRAW >> q_rounds) if q_rounds else 0
                if len(pool) < draw - n_qual + (QUALI_DRAW if q_rounds else 0):
                    q_rounds, n_qual = 0, 0
                    draw = min(draw, _pow2_floor(len(pool)))
                ctx.draw_size = draw
                direct = pool[: draw - n_qual]
                del pool[: draw - n_qual]
                qualifiers: list[str] = []
                if q_rounds:
                    q_entrants = pool[:QUALI_DRAW]
                    del pool[:QUALI_DRAW]
                    q_rnds = [f"Q{k}" for k in range(1, q_rounds + 1)]
                    nums, k = {}, QUALI_MATCH_NUM_START
                    n = QUALI_DRAW // 2
                    for r in q_rnds:
                        nums[r] = list(range(k, k + n))
                        k += n
                        n //= 2
                    order = [q_entrants[int(j)] for j in rng.permutation(len(q_entrants))]
                    qualifiers, losers = season.knockout(ctx, order, q_rnds, nums)
                    quali_losers.extend(losers)
                n_seeds = max(2, draw // 4)
                for s, pid in enumerate(direct[:n_seeds]):
                    ctx.entries[pid] = (float(s + 1), None)
                for pid in qualifiers:
                    ctx.entries[pid] = (float("nan"), "Q")
                unseeded = direct[n_seeds:]
                if unseeded and rng.random() < 0.5:
                    ctx.entries[unseeded[-1]] = (float("nan"), "WC")
                entrants = direct + qualifiers
                order = [entrants[int(j)] for j in rng.permutation(len(entrants))]
                rounds = list(_KO_ROUNDS[draw])
                if spec.level == "O":
                    _olympics(season, ctx, order, rounds)
                else:
                    season.knockout(ctx, order, rounds,
                                    season.main_draw_numbers(rounds, draw, spec.legacy_numbering))

    df = pd.DataFrame(season.rows)
    df = _finalise(df)
    if shuffle:
        df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    return df


def _tour_finals(season: _Season, ctx: _Ctx, eight: list[str]) -> None:
    """Round robin (2 groups of 4), semi-finals (group winner vs other runner-up), final."""
    for s, pid in enumerate(eight):
        ctx.entries[pid] = (float(s + 1), None)
    groups = [[eight[0], eight[3], eight[4], eight[7]], [eight[1], eight[2], eight[5], eight[6]]]
    num = 1
    wins: dict[str, int] = {pid: 0 for pid in eight}
    for g in groups:
        for i in range(4):
            for j in range(i + 1, 4):
                w, _, _ = season.play(ctx, "RR", num, g[i], g[j], pos=num, allow_wo=False)
                wins[w] += 1
                num += 1
    standings = [sorted(g, key=lambda pid: (-wins[pid], ctx.ranks.get(pid, (1e9,))[0])) for g in groups]
    f1, _, _ = season.play(ctx, "SF", num, standings[0][0], standings[1][1], pos=1, allow_wo=False)
    f2, _, _ = season.play(ctx, "SF", num + 1, standings[1][0], standings[0][1], pos=2, allow_wo=False)
    season.play(ctx, "F", num + 2, f1, f2, pos=1, allow_wo=False)


def _olympics(season: _Season, ctx: _Ctx, entrants: list[str], rounds: list[str]) -> None:
    """Knock-out to the semi-finals, then a bronze-medal match (BR) and the final."""
    pre_final = [r for r in rounds if r != "F"]
    nums = season.main_draw_numbers(rounds, len(entrants), False)
    alive = list(entrants)
    sf_losers: list[str] = []
    for rnd in pre_final:
        nxt = []
        for j in range(len(alive) // 2):
            w, l, _ = season.play(ctx, rnd, nums[rnd][j], alive[2 * j], alive[2 * j + 1], pos=j + 1,
                                  allow_wo=False)
            nxt.append(w)
            if rnd == "SF":
                sf_losers.append(l)
        alive = nxt
    final_num = nums["F"][0]
    season.play(ctx, "BR", final_num, sf_losers[0], sf_losers[1], pos=1, allow_wo=False)
    season.play(ctx, "F", final_num + 1, alive[0], alive[1], pos=1, allow_wo=False)


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the unified-schema dtypes and column order."""
    df = df[UNIFIED_COLUMNS].copy()
    df["tourney_date"] = pd.to_datetime(df["tourney_date"]).astype("datetime64[ns]")
    for c in ("draw_size", "best_of", "match_num", "round_rank", "order_key"):
        df[c] = df[c].astype("int64")
    float_cols = (["minutes", "winner_seed", "loser_seed", "winner_ht", "loser_ht", "winner_age", "loser_age",
                   "winner_rank", "winner_rank_points", "loser_rank", "loser_rank_points"]
                  + [f"{p}_{s}" for p in ("w", "l") for s in STAT_FIELDS])
    for c in float_cols:
        df[c] = df[c].astype("float64")
    for c in ("is_walkover", "is_retirement", "is_qualifying", "is_training_level", "surface_imputed",
              "stats_invalid"):
        df[c] = df[c].astype(bool)
    return assign_order_key(df.reset_index(drop=True))


def make_match_row(
    *,
    tourney_id: str,
    tourney_date: str | pd.Timestamp,
    winner_id: str,
    loser_id: str,
    round_: str = "R32",
    match_num: int = 1,
    level: str = "A",
    surface: str = "Hard",
    best_of: int = 3,
    score: str = "6-4 6-4",
    cfg: dict[str, Any] | None = None,
    stats: bool = False,
) -> dict[str, Any]:
    """Build one hand-written unified-schema row (for tiny scenario tests). Returns a dict."""
    if cfg is None:
        from tennis.config import load_config
        cfg = load_config()
    round_order = list(cfg["data"]["round_order"])
    date = pd.Timestamp(tourney_date)
    nan = float("nan")
    row: dict[str, Any] = {c: nan for c in UNIFIED_COLUMNS}
    rr = round_order.index(round_)
    row.update({
        "tourney_id": tourney_id, "tourney_name": f"Scenario {tourney_id}", "surface": surface, "draw_size": 32,
        "tourney_level": level, "tourney_date": date, "match_num": match_num, "score": score,
        "best_of": best_of, "round": round_, "winner_id": str(winner_id), "loser_id": str(loser_id),
        "winner_name": f"P{winner_id}", "loser_name": f"P{loser_id}", "winner_hand": "R", "loser_hand": "R",
        "winner_entry": None, "loser_entry": None, "winner_ioc": "ESP", "loser_ioc": "FRA",
        "match_id": match_id_for(tourney_id, match_num, str(winner_id), str(loser_id)),
        "source": "sackmann", "round_rank": rr, "order_key": -1,
        "is_walkover": score == "W/O",
        "is_retirement": score.endswith("RET") or score.endswith("DEF"),
        "is_qualifying": round_ in cfg["data"]["qualifying_rounds"],
        "is_training_level": (level in cfg["data"]["training_levels"])
        and round_ not in cfg["data"]["qualifying_rounds"],
        "surface_imputed": False, "stats_invalid": False,
    })
    if stats:
        vals = {"ace": 5, "df": 2, "svpt": 60, "1stIn": 36, "1stWon": 27, "2ndWon": 12, "SvGms": 10,
                "bpSaved": 2, "bpFaced": 3}
        for s, v in vals.items():
            row[f"w_{s}"] = float(v)
            row[f"l_{s}"] = float(v)
    return row


def rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Turn `make_match_row` dicts into a unified-schema DataFrame with the right dtypes."""
    return _finalise(pd.DataFrame(rows))
