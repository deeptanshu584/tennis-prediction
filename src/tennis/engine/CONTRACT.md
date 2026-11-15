# Engine contract (owned by the orchestrator)

Shared interface between ingestion, the feature engine, the model code and the leakage suite.
Only the orchestrator edits this file. Section numbers refer to PLAN.md.

## 1. Unified match table (`data/interim/matches.parquet`), winner/loser oriented
Columns (Sackmann names, §4.2), with these types/extras:
- `match_id` str — stable hash: `sha1(f"{tourney_id}|{match_num}|{min(winner_id,loser_id)}|{max(...)}")[:16]`
- `source` str in {sackmann, tml, scraped}
- `tourney_id` str, `tourney_name` str, `tourney_level` str, `tourney_date` datetime64[ns] (tournament START date)
- `surface` str in {Hard, Clay, Grass} (Carpet → Hard), `draw_size`, `best_of` int, `round` str, `match_num` int
- `round_rank` int = index of `round` in `config.data.round_order` (Q1=0 … R16=7, ER=8, RR=9 … F=13). Always look it
  up from config; never hard-code the integers.
- `order_key` int64 = 0-based position of the row after sorting the whole unified table by the authoritative
  match order (§2): `(tourney_date, tourney_id, round_rank, match_num, match_id)`. It is a convenience column and
  must equal that sort exactly. (A string key was dropped: when one tourney_id is a prefix of another, a string key
  sorts differently.)
- `winner_id`, `loser_id` str (Sackmann player_id as string)
- `winner_name/hand/ht/ioc/age/rank/rank_points/seed/entry`, `loser_*` likewise
- `score` str, `minutes` float, `w_ace … w_bpFaced`, `l_ace … l_bpFaced` float (NaN when missing)
- `is_walkover`, `is_retirement` (RET or DEF), `is_qualifying` (round in Q1–Q4), `is_training_level`
  (`tourney_level` in `config.data.training_levels` AND not qualifying) — bools
- Data-quality flags (bools; see reports/data/source_check.md and DECISIONS D-009…D-012):
  `surface_imputed` (surface was null → Hard), `stats_invalid` (impossible serve counts → all 18 stat columns set to
  NaN in the unified table, the result is kept). Exact duplicates (D-009) and self-matches (D-010) are removed from
  matches.parquet and listed in the ingestion report with their reason.

## 2. Feature builder
```python
from tennis.engine.build_features import build_features
features = build_features(matches: pd.DataFrame, cfg: dict)                     # -> pd.DataFrame
features, k_log = build_features(matches, cfg, record_k=True)                    # also K per player per match
```
- Must not mutate `matches`. Sorts internally by the authoritative match order
  `(tourney_date, tourney_id, round_rank, match_num, match_id)`; `match_id` is the final tiebreaker so the order is
  total. Never rely on input row order or on `order_key` alone.
- Drops walkovers before the pass. ALL other matches (tour, Challenger, qualifying, RET/DEF) update state.
- Emits a row only when `is_training_level & ~is_walkover & ~is_retirement` and the year ≥ `data.feature_first_year`.
- Per match: snapshot both players' state → emit row → THEN update state.
- `k_log` has one row per processed (non-walkover) match per player, RET/DEF included: `match_id, player_id,
  k_overall, k_surface`. For RET/DEF the K that would have applied is logged, even when Elo isn't updated.
- Names generated from config: `winrate_last_{k}` for k in `features.winrate_k`; `level_{L}` for L in
  `data.training_levels`; `surface_{S}` for S in the distinct values of `data.surface_map`. The names below are the
  current-config instances.

### Orientation
`a_is_winner = int(sha256(f"{seed}:{match_id}").hexdigest()[:8], 16) % 2 == 0` (per-match, independent of every
other row). `y = 1` if A won.

### Output columns
Non-features: `match_id, date (=tourney_date), tourney_id, tourney_name, a_id, b_id, a_name, b_name, y`.
Context features: `surface_Hard, surface_Clay, surface_Grass, level_A, level_G, level_M, level_F, level_D, level_O`
(0/1 ints), `round_rank`, `best_of`, `elo_prob_a`, `elo_surface_prob_a`.
For each player feature `f`: `a_f, b_f, d_f = a_f − b_f` (NaN if either is NaN).

### Player features `f` (snapshotted before the match)
| f | definition |
|---|---|
| elo, elo_surface | current ratings (start `elo.start_rating`) |
| elo_grad | OLS slope (np.polyfit deg 1, x = 0..n−1) of the last `features.elo_grad_window` values of the player's overall-Elo history; NaN if n < 2. History starts as [start_rating] at first appearance; a new value is appended after every Elo update |
| matches_played, matches_surface_played | count of prior non-walkover matches (incl. RET/DEF, qualifying, Challenger) overall / on this surface |
| winrate_last_5/10/50 | mean of the last k results (1 win, 0 loss) over min(k, available); results exclude RET/DEF when `cleaning.retirement_updates_form` is false; NaN if none |
| h2h_wins, h2h_surface_wins | prior wins vs this opponent (overall / on this surface); RET/DEF excluded when `cleaning.retirement_updates_h2h` is false |
| n_stat_matches | number of matches in the stats buffer |
| p_ace … p_totalPtsWon, dominance_ratio | §5.5 formulas: sums of numerators / sums of denominators over the buffer; NaN if the denominator is 0 or the buffer is empty. Serve-points-won rate for dominance = (1stWon+2ndWon)/svpt |
| rank, rank_points, age, height | from the match row (`winner_rank` etc. = values at tourney_date, known pre-match) |
| is_left | 1 if hand == 'L', 0 if 'R', NaN otherwise |
| days_since_last_match | (tourney_date − tourney_date of the player's last prior match) in days; NaN if none |
| matches_last_14_days | prior matches with tourney_date ≥ current tourney_date − `features.recent_days_window` days |

Stats buffer: deque(maxlen=`features.rolling_n`) of prior matches where all nine serve fields
(ace, df, svpt, 1stIn, 1stWon, 2ndWon, SvGms, bpSaved, bpFaced) are non-null for BOTH players and both svpt > 0.
Each entry holds the player's own nine counts and the opponent's nine counts. RET/DEF enter it iff
`cleaning.retirement_updates_stats`.

### Elo update (after the row is written)
- Skip for RET/DEF unless `cleaning.retirement_updates_elo`.
- Each player uses their own K. Overall: Ne = player's `matches_played` before this match. Surface: Ne = `matches_surface_played`.
- `m`: `elo.m_mode == "event_matches_so_far"` (the default, DECISIONS D-008) →
  `m = elo.<overall|surface>.m + (number of matches the player has ALREADY played in this tourney_id before this match)`
  (qualifying matches of the same tourney_id count; for surface Elo the count is the same event count). With `m = 1`
  this is "1 + matches already played in the event", so a debut (Ne = 0) gives K = div_number / 1.
  `"constant"` → `m = elo.<overall|surface>.m`. Never anything that depends on later matches.
- Layoff `played`: ONE entry per distinct tourney_id the player has played (prior non-walkover matches), holding that
  tournament's start date, in match order, plus the current tournament's start date if the current tourney_id isn't
  already present (e.g. it is, after a qualifying match in the same event). Two different tournaments starting on the
  same date are two entries (a 0-day gap).
- Configs must keep `elo.<which>.m ≥ 1`, so `Ne + m ≥ 1`; `elo.k_factor` raises on `Ne + m ≤ 0`, and it can't be
  reached with valid config.

## 3. Elo module (`tennis.engine.elo`)
```python
expected_score(r_a: float, r_b: float) -> float
update_rating(r: float, expected: float, score: float, k: float) -> float
k_factor(ne: int, m: float, *, k_factor: float, div_number: float, base_k: float, max_k: float,
         bonus_after_layoff: bool, played: Sequence[date-like] = (), layoff: dict | None = None) -> float
gaps_between_tourneys(played: Sequence[date-like]) -> list[int]      # days, MOST RECENT FIRST
k_bonus_after_layoff(played, threshold=100, min_gaps=15, start_bonus=1.5, step=0.05, floor=1.0) -> float
EloParams.from_config(cfg, which: "overall" | "surface") -> EloParams   # convenience dataclass
```

## 4. Splits (`tennis.model.splits`)
```python
time_split(df: pd.DataFrame, split_name: str, cfg: dict) -> dict[str, pd.DataFrame]   # keys: train, val[, test]
```
Uses `cfg["splits"][split_name]` `*_start`/`*_end` dates on the `date` column (inclusive). Per split type:
- `dev`: train and val by dates; no `test` key.
- `replication`: train and val by dates. `test` = rows with `tourney_name == test_tourney` and
  `date.year == test_year`, which must start after `val_end`.
- `forward`: train and val by dates. `test` = `date >= test_start` up to the last available date (`test_end` optional).
Partitions are disjoint and time-ordered: max(train.date) < min(val.date) < min(test.date).

## 5. Model feature allow-list
`cfg["model"]["features"]` (added by the orchestrator in Phase 2) lists the exact model input columns.
