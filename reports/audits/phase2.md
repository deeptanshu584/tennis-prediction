# Phase 2 audit: feature engine, plus a re-check of the Phase 1 unified table

**Verdict: FAIL.** There is one blocker: inside round-robin rounds, a row can read results of matches that were
really played after it (finding 1). It is a genuine leak but a small one. It touches 858 of 76,479 rows (1.1%) and
none of the replication or forward test rows. There are also two should-fix items: missingness of present-day
player attributes (finding 2), and an unguarded flip-augmentation helper (finding 3). Everything else checked out:
the chronological pass, snapshot-before-update, the orientation, the emitted set, walkovers/RET, K-factor `m`, the
unified table, and the Elo-only sanity numbers.

Auditor: fresh leakage-auditor. I wrote none of the code under review. Date: 2026-11-16.

## Leakage suite results
| Run | Result |
|---|---|
| `pytest tests/leakage -q` before the new tests (slow + realdata included) | **39 passed, 5 skipped** (96 s). Skips: 4 × `tennis.model.splits` missing (Phase 3); 1 × no experiments/*.jsonl yet |
| `TENNIS_LEAKAGE_REAL_START_YEAR=1990 pytest tests/leakage -k real` | **8 passed, 3 skipped** (109 s; same splits skips) |
| New tests added in this audit (tests 10–13, 6 tests) | **4 failed, 2 passed**. The failures reproduce findings 1–3; see the table below |
| Full suite with the new tests | **4 failed, 41 passed, 5 skipped** (106 s) |

### New tests (tests/leakage/)
| File | Test | Status now | Finding |
|---|---|---|---|
| `test_10_round_robin_same_round.py` | `test_round_robin_same_round_synthetic` | **FAIL** (3 synthetic Tour Finals, 16/24 player-rows change) | 1 |
| | `test_round_robin_same_round_real` (realdata, slow) | **FAIL** (Tour Finals 2019, NextGen 2019, Laver Cup 2019, ATP Cup 2020) | 1 |
| `test_11_augment_guard.py` | `test_flip_augmentation_refuses_rows_after_every_train_end` | **FAIL** (did not raise) | 3 |
| | `test_flip_augmentation_still_works_on_training_rows` | pass | 3 (guards the fix) |
| `test_12_snapshot_missingness.py` | `test_snapshot_attributes_have_no_missingness_signal` | **FAIL** (a_height 2,208 NaN, b_height 2,259; a_is_left 626, b_is_left 619; a_age 15, b_age 11) | 2 |
| `test_13_production_table_reproducible.py` | `test_features_parquet_equals_fresh_build` (realdata, slow) | pass (identical; 11 s rebuild) | gap closed: nothing checked how features.parquet was made |

Test 10 was validated against a scratch copy of the engine with the fix from finding 1 applied. The source was not
modified. With the fix, test 10 and all of test 01 pass (10/10).

## Findings

1. **BLOCKER: round-robin rows see results of same-round matches played later.**
   `src/tennis/engine/build_features.py:225-265`, with the order from `src/tennis/engine/ordering.py:14`.
   - **The cause.** Within one round of an event, the order falls back to `match_num`, and in Sackmann's RR rounds
     `match_num` is not chronological.
   - **Example: Tour Finals 2019, Thiem / Federer / Djokovic / Berrettini group.** In processing order:

     | match_num | Match | Played |
     |---|---|---|
     | 286 | Berrettini d. Thiem | Thu 14 Nov |
     | 287 | Federer d. Berrettini | Tue |
     | 288 | Thiem d. Federer | Sun 10 Nov |
     | 289 | Djokovic d. Berrettini | Sun |
     | 290 | Thiem d. Djokovic | Tue |
     | 291 | Federer d. Djokovic | Thu |

     The Sunday row 288 is built from a state that already holds Thursday's Berrettini d. Thiem.
   - **How widespread.** A 4-player group plays two disjoint matches per match day, so in a chronological order
     each consecutive pair of matches shares no player. That fails in **84 of 88** four-player groups of the Tour
     Finals, Masters Cup and NextGen Finals.
   - **Rows affected (excluding Davis Cup).** 858 emitted RR rows have at least one player with an earlier-processed
     match in the same RR round. A scratch fix changes exactly those 858 rows.
     - By event: Tour Finals/Masters Cup 265, Dusseldorf World Team Cup 253, ATP Cup 107, NextGen 79,
       United Cup 65, 2007 RR events 77, Laver Cup 22.
     - By window: 95 rows in the 2023–24 dev validation window, 24 in 2025 and 12 in 2026 (United Cup, Q1 2026,
       forward validation). No replication or forward **test** rows are affected.
   - **Size of the leak.** On the 858 rows, the leak lifts the "higher last-5 win rate wins" rule from 60.0% to
     62.0%. Elo is barely affected (top players have K ≈ 1–2), with accuracy 62.4% vs 62.0% for the scratch fix.
     Across the whole table the effect is negligible, but it is future information all the same.
   - **Davis Cup is not affected.** Each tie is its own tourney_id. A player's rubbers are (1, 4), (1, 5), (2, 4)
     or (2, 5) in 6,707 of 6,711 cases, which is chronological (day one, then the last day).
   - **Fix (feature-engineer, with an orchestrator CONTRACT change):**
     - For each block of consecutive rows with the same `tourney_id`, `round == "RR"` and `tourney_level != "D"`,
       compute every emitted row's snapshot from the state **before the block**. Then apply the block's updates in
       the current order.
     - For consistency, also take the block's K / `m` (event count) from before the block. This is not needed for
       leakage, because only later rounds read the post-block state.
     - Update `tests/leakage/reference.py` in the same change, or test 02 will fail.
     - Document the rule in CONTRACT §2 and DECISIONS.
     - Test 10 must pass, unchanged.

2. **SHOULD-FIX: missingness of present-day player attributes encodes the future (hindsight).**
   `build_features.py:127-128` and `:200-204` (height / age / hand taken from the match row), and the allow-list in
   `config.yaml:232-240` (`a_/b_/d_age`, `a_/b_/d_height`, `a_/b_/d_is_left`).
   - **Where the values come from.** Sackmann fills `winner_ht`, `winner_hand` and the birth date behind
     `winner_age` from today's player file:
     - 99.4% of match-row heights equal `atp_players.csv`.
     - A player's height is the same in every row of his career (only 92 of 9,934 players vary).
   - **Why that leaks.** Whether a height or hand is on file depends on the player's **later** career. Players first
     seen on tour in 2000–2015 with no height played a median of **2** tour-level matches in their whole career,
     against **19** for players with a height.
   - **Measured effect.**
     - Controlling for Elo, log matches played, log rank and rank-NaN, a one-sided height-NaN indicator keeps a
       logistic coefficient of −0.77, and hand-NaN −0.49. In-sample log loss improves by 0.0019.
     - In the 2023–24 validation window, the NaN-height player loses 79% of 81 one-sided rows, where Elo is right
       59% of the time. That bounds the gain at about 0.3 accuracy points.
     - The effect shrinks in 2026 rows (60%), as hindsight should, so it would also inflate validation relative to
       the forward test.
   - **Rank is clean.** Rank and rank-NaN are per-row and pre-event. For example, Čilić is rank 16 in every round
     of US Open 2014 and 9 at the next event.
   - **Fix (feature-engineer / orchestrator):** remove the missingness signal. Fill NaN height, is_left and age with
     constants from config (e.g. height 185, is_left 0, age = the training median written into config), with no
     missing indicator. Or drop these columns from the allow-list. Test 12 must pass.
   - **Rows involved.** About 2.9% of rows have a NaN height for a player (1.5% of rows in 2023–24), mostly Davis
     Cup (1,891 of 2,421 one-sided rows).

3. **SHOULD-FIX: flip augmentation is usable on validation/test rows without any error.**
   `src/tennis/model/dataset.py:88-95` (`maybe_augment_train`).
   - **The problem.** The function only documents "never call on validation or test". Called on later rows, it
     doubles them. Each real match would then be scored twice under the same `match_id`, in both orientations.
     Off by default (`config.yaml:76`), so there is no current damage.
   - **Fix (ml-engineer):**
     - Give the function a `split_name` argument and raise if any row's `date` is after
       `cfg["splits"][split_name]["train_end"]`. At minimum, raise for rows after the latest `train_end` of every
       split (what test 11 checks).
     - Add an `is_flipped` marker, and have evaluation assert that no scored row has it.
     - Phase 5: never draw an early-stopping or CV holdout from augmented rows. A flipped twin would sit on the
       other side of the cut.

4. **NOTE: other cross-event reorderings don't reach emitted rows.**
   - **Same start date.** 2,231 player-dates have one player in two events with the same tourney_date (mostly
     qualifying of one event plus a Challenger). Processing them by tourney_id can invert the real order. Only 5
     of them touch an emitted event: 4 before 2000, plus Nice qualifying 2010 → Dusseldorf 2010, which is
     chronologically fine.
   - **Overlapping events.** A player with ≥ 4 main-draw matches in event E1 and a next event within 1–4 days
     occurs 3 times; the only post-2000 case is Kitzbühel → London Olympics 2012, which is consistent.
   - No fix needed. Re-check after TML/scraped data is merged (Phase 3), since a mis-dated source event would make
     emitted rows read the future.

5. **NOTE: the qualifying order is correct.** All 2,978 qualifying tourney_ids share the main draw's tourney_id
   and tourney_date (0 mismatches). Q1–Q3 have round_rank 0–2, below R128 = 4. A qualifying result therefore can't
   reach an earlier main-draw row. The `m` event count includes the qualifying matches, as D-008 intends.

6. **NOTE: retirement exclusion biases evaluation slightly in the model's favour.** RET/DEF rows are never emitted
   (PLAN rule 7, verified: 0 emitted). In live use you can't know in advance that a match will end in a
   retirement, and those matches are noisier. Validation and test accuracy are therefore a bit optimistic. Say so
   in the Phase 6/7 reports. No code change.

7. **NOTE (data quality, not leakage): non-standard events are training rows, and some levels are mislabelled.**
   - Laver Cup (exhibition, 58 rows) and NextGen Finals (short sets, 122 rows) are emitted.
   - Tour Finals 2023 is labelled `A` (its 2023 `F` rows are the NextGen Finals).
   - The Olympics are `A` in 1992–2021 and `O` only in 2024.
   - Recommend the data-engineer review this: exclude exhibitions, and fix the level labels used by the one-hots.

8. **NOTE: gaps in the existing suite, now closed or documented.**
   - (a) No test checked that `features.parquet` is what the audited engine builds. Tests 1, 2 and 4 use a 2018+
     universe, and tests 3, 4, 6 and 9 only read the file. Closed by test 13; it passes now.
   - (b) No test exercised the within-RR order. Closed by test 10.
   - (c) `time_split` (test 5) and the fitted-preprocessing check (checklist item 6) can't be audited until
     Phase 3/5. `dataset.py` fits nothing. Re-run this audit's items 4 and 6 then.

## Checklist (role definition)
1. **Snapshot before update: OK for every feature** (`build_features.py:229-265`).
   - The K pair and `_snapshot` read state only. All `record_*`, `h2h.record_win` and `stats.push` calls come after
     the row.
   - Context features (surface/level one-hots, round_rank, best_of) come from the match row and are known before
     the match.
   - `elo_prob_a` and `elo_surface_prob_a` are recomputed from the a/b ratings (max difference 0). The one
     exception is the RR order (finding 1).
2. **Match order: OK, except RR (finding 1).** The order is `(tourney_date, tourney_id, round_rank, match_num,
   match_id)` from config `round_order`, and `check_round_rank` raises on disagreement. Input order and `order_key`
   are not trusted (test 01). Q < main draw, ER < RR < QF, and BR < F.
3. **Banned names: none** in `cfg["model"]["features"]` (test 03 passes). `dataset.is_banned` also rejects
   ids, names and non-feature columns.
4. **Splits: not testable yet** (no `tennis.model.splits`). In features.parquet, match_id is unique, there are 0
   same-meeting duplicates, and dates run 2000-01-03 → 2026-05-25.
5. **K-factor `m`: OK.** `m = base_m + event_matches[tourney_id]`, read before `record_match` increments it
   (`state.py:141-151`, `build_features.py:132-141`). Test 08 passes.
6. **Fitted preprocessing: none exists yet.** The augmentation guard is finding 3.
7. **Suite: see the results above.**
8. **Metrics: plausible, no red flag.** See below.

## Independent Elo-only numbers (from features.parquet)
- **2023–2024 (n = 5,937):**
  - Accuracy of `a_elo > b_elo` is **63.92%**, with the 7 exact ties counted as a "B" prediction. Excluding the
    ties gives the engine author's 63.96%, so the two agree.
  - `elo_prob_a` log loss is **0.6376**.
  - Surface Elo accuracy is 63.52%; the rank baseline is 63.76%.
- **Per season, 2015–2024** (accuracy / log loss):

  | Season | Accuracy | Log loss |
  |---|---|---|
  | 2015 | 68.02% | 0.5951 |
  | 2016 | 66.90% | 0.6010 |
  | 2017 | 64.85% | 0.6181 |
  | 2018 | 62.89% | 0.6333 |
  | 2019 | 62.21% | 0.6363 |
  | 2020 | 63.43% | 0.6351 |
  | 2021 | 61.17% | 0.6494 |
  | 2022 | 64.36% | 0.6301 |
  | 2023 | 63.85% | 0.6414 |
  | 2024 | 63.99% | 0.6340 |

- **Highest season over 2000–2026:** 2014, at 68.67%. 2012 is 68.51%, so the author's "max 68.6% (2012)" is
  essentially the same figure. No season is above 70%. That fits untuned default Elo (Phase 4 tunes it) and gives
  no sign of leakage.
- **Orientation:** the label mean is 0.4989, and `y` equals the sha256 parity for every row. `y` is independent of
  `winner_id < loser_id` (χ² p = 0.13), `match_num` parity (p = 0.50) and row parity (p = 0.70).
- **NaN symmetry:** the NaN rates of the a_ and b_ slots are equal for every feature (e.g. rank 1.36% / 1.34%,
  height 2.89% / 2.95%). Winner and loser missingness differs (rank NaN 0.74% vs 1.96%), but the coin flip
  spreads it evenly across the slots, so only the legitimate (or, for height and hand, hindsight; finding 2)
  information remains.

## Unified table re-check (`data/interim/matches.parquet`, 337,721 rows)
- **Keys and order.**
  - `match_id` is unique.
  - The stored order equals the authoritative 5-column sort, and int64 `order_key` equals each row's position.
  - No rows tie on the first 4 order columns.
  - There are 0 self-matches, 0 D-009 key duplicates, and 0 same-meeting duplicates among non-walkovers.
  - Each tourney_id has a single tourney_date.
- **Emitted set.** It equals `is_training_level & ~is_walkover & ~is_retirement & year >= 2000` exactly (76,479).
  It contains 0 walkovers, RET/DEF, qualifying or non-training rows.
- **Extras.** `a_rank`, `a_rank_points`, `a_age` and `a_height` equal the oriented `winner_*`/`loser_*` values
  exactly. The a_/b_ mapping comes from the hash, never from the winner/loser position.
- **Reproducibility.** `features.parquet` rebuilds identically from `matches.parquet` with the current config.

## Not checked / limits
- The real chronology inside non-RR rounds and the true match dates are unknowable from the data. tourney_date is
  the start date, so all checks are about consistency.
- The Phase 3+ split and model code doesn't exist yet.
