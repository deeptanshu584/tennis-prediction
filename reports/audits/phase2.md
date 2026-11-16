# Phase 2 audit: feature engine, plus a re-check of the Phase 1 unified table

> **Current verdict, after the re-audit (end of file): PASS.** The first audit below was FAIL; its findings 1–3
> and note 7 are fixed and verified.

**First-audit verdict: FAIL.** There is one blocker: inside round-robin rounds, a row can read results of matches that were
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

---
## Re-audit (2026-11-16): **PASS**
I re-audited the fixes for findings 1–3 and note 7, against DECISIONS D-019 to D-022 (including the D-022
amendment) and CONTRACT §2 ("Round-robin blocks", "Static player attributes") and §4b. I changed no source or
existing test. I added one new test file (test 14).

### (a) The fixes in the code
1. **Finding 1: RR blocks. Fixed.**
   - **The block.** `src/tennis/engine/build_features.py:222-224` flags rows with `round == "RR"` whose level is
     not in `features.rr_block_exempt_levels` (`[D]`, `config.yaml:82`). Lines `:253-260` extend a block over
     consecutive rows with the same tourney_id. In the authoritative sort these rows are always contiguous, since
     they share date, tourney_id and round_rank.
   - **Pass 1, reads only (`:263-285`).** For every match in the block, everything is read from the state before
     the block:
     - K via `_k_pair`, which also reads Ne, the event count `m` and the layoff list;
     - the Elo deltas via `state.elo_deltas` (`state.py:126-131`), whose expected score E (overall and surface)
       uses the pre-block ratings;
     - the emitted snapshot, including h2h, the stats buffer, form, recent days and `elo_prob_a`;
     - the `k_log` entries.

     Nothing in pass 1 writes state. `get()` only creates new players at the start rating.
   - **Pass 2, writes (`:288-303`).** It applies the stored deltas in authoritative order, adding them to the
     current ratings. The same `PlayerState` object accumulates correctly when a player appears twice. Stats,
     h2h, form and `record_match` updates all happen here. **No value read inside a block depends on another
     match of the same block.**
   - **Non-RR rounds.** They are blocks of one match, so their behaviour is unchanged.
   - **The reference.** `tests/leakage/reference.py:154-246` (`block_start`, `elo_replay`, and `:364-368` for the
     other features) implements the same rule on its own, from `ms.iloc[:block_start]`.
   - **The synthetic fixture.** `tests/fixtures/synthetic.py:528-552` now plays its RR day by day but numbers it
     latest day first, which is a real trap.
   - **No gap left in the data.** In the rebuilt data, no emitted round other than RR has a player twice
     (0 cases). The remaining repeats are Challenger or qualifying rows, which are never emitted.
2. **Finding 2: static attributes. Fixed.**
   - `build_features.py:122-152` fills height, is_left and age from `features.static_fill` (185 / 0 / 26,
     `config.yaml:83-86`), with no indicator. These are config constants, so nothing is fitted.
   - Rank and rank points keep NaN, which is correct: unranked is real pre-match information.
3. **Finding 3: augmentation guard. Fixed.**
   - `src/tennis/model/dataset.py:89-124`: `maybe_augment_train(train_df, cfg, split_name=None)` raises
     `ValueError` for rows after the train_end of `split_name` (or the latest train_end when it is None). The check
     runs whether or not the flag is set, and the function refuses frames that are already augmented. Flipped
     copies carry `is_flipped = 1`.
   - `:127-134`: `assert_not_augmented` exists for evaluation code to call.
4. **Note 7: level labels. Fixed.** `src/tennis/ingest/sackmann.py:358-398` and `:443-455`:
   - Laver Cup is removed entirely (0 rows remain).
   - Tour Finals and Masters Cup → F (540 rows).
   - Every "<City> Olympics", 1992–2024 → O (64 rows each; 63 for Barcelona).
   - NextGen Finals and Next Gen Finals → A (122 rows).
   - The override is applied before `is_training_level` is computed. Results are untouched.

### (b) Leakage suite (slow + realdata included)
- `pytest tests/leakage -q`: **45 passed, 5 skipped** (134 s).
  - Skips: 4 × `tennis.model.splits` missing (Phase 3); 1 × no experiments/*.jsonl yet.
  - My tests 10–13 now all **pass, unchanged**. Test 10 covers both synthetic and real data (Tour Finals 2019,
    NextGen 2019, ATP Cup 2020). Test 12 finds 0 NaN in a_/b_/d_ height, is_left and age. Test 13 confirms the
    rebuilt features.parquet equals a fresh build.
- **New test file:** `test_14_augment_split_guard.py`, 4 tests, **pass**. It checks that, given `split_name`, each
  split's own validation rows are refused while its last training day is accepted, and that
  `assert_not_augmented` rejects flipped rows.
- The data and output tables are still sound:
  - matches.parquet has 337,663 rows, a unique match_id, and `order_key` equal to each row's position.
  - features.parquet has 76,421 rows, exactly the expected emitted set. It contains 0 walkover, RET/DEF,
    qualifying or non-training rows.

### (c) Residual missingness signal after the static fill (empirical, new features.parquet)
- **No NaN left.** All 9 height, is_left and age columns have 0 NaN.
- **The fill value is well diluted.**
  - 2,208 filled a_height values sit among 11,701 genuine 185 cm values.
  - 15 filled ages sit among 829 genuine exact 26.0 values.
  - is_left 0 is indistinguishable from right-handed.
- **What remains, in-sample.** Controlling for Elo, log matches played, log rank and rank-NaN:
  - the indicator "height == 185" has a coefficient of −0.05 (missing-height had −0.77) and improves log loss by
    0.00007 (was 0.0019);
  - "age == 26.0" has a coefficient of −0.01.
  - The remainder is concentrated in Davis Cup rows: the interaction 185 × level_D has a coefficient of −0.52,
    against +0.01 outside Davis Cup. A filled 185 player has y = 0.34, against 0.51 for a genuine 185.
- **Out of sample there is nothing to exploit.** I trained a gradient-boosted tree on the dev split (2000–2022) with
  Elo, rank, experience and level features. Adding a_/b_/d_height changed 2023–24 validation log loss from 0.61519
  to 0.61522 (no gain) and accuracy from 64.91% to 64.95%.
- **Verdict on finding 2: resolved.** Note R1 below records the small in-sample remainder in Davis Cup.

### (d) Elo-only sanity check (new features.parquet)
- **2023–24 (n = 5,928):** `a_elo > b_elo` accuracy is **63.95%** (7 exact ties counted as B), and `elo_prob_a`
  log loss is **0.6377**. Surface Elo gets 63.53% and the rank baseline 63.78%.
- **Per season, 2015–2024** (accuracy / log loss):

  | Season | Accuracy | Log loss |
  |---|---|---|
  | 2015 | 68.02% | 0.5951 |
  | 2016 | 66.90% | 0.6009 |
  | 2017 | 64.85% | 0.6183 |
  | 2018 | 62.86% | 0.6333 |
  | 2019 | 62.39% | 0.6365 |
  | 2020 | 63.50% | 0.6352 |
  | 2021 | 61.12% | 0.6496 |
  | 2022 | 64.26% | 0.6295 |
  | 2023 | 63.85% | 0.6415 |
  | 2024 | 64.05% | 0.6341 |

- **2025 and 2026:** 63.71% and 64.91%.
- No season is above 70%, and the numbers are essentially unchanged from the first audit, as expected for a fix
  that touches about 1% of rows.
- **Orientation:** the label mean is 0.4988, and `elo_prob_a` equals the logistic of the a/b Elo values exactly.

### Remaining notes (no action needed for Phase 2)
- **R1 (note).** In-sample, a filled height of 185 still carries a small hindsight remainder in Davis Cup rows
  (above). It gives no out-of-sample gain. In Phase 5, check XGBoost feature importance and the SHAP values of
  `d_height`/`a_height`. If height ranks high, drop it from the allow-list (the D-020 alternative).
- **R2 (note).** With `split_name=None`, the augmentation guard only checks the LATEST train_end (2025-12-31), so
  dev-split validation rows from 2023–24 would pass it.
  - Phase 5 model code must always pass `split_name`. Test 14 pins down that behaviour.
  - Consider making `split_name` required once a caller exists.
  - Evaluation must call `assert_not_augmented` on every val/test frame.
- **R3 (note).** Re-run the checks on cross-event ordering and same-date events (first-audit note 4) after
  TML/scraped data is merged in Phase 3.
- **R4 (note).** The splits checks (checklist 4) and the fitted-preprocessing checks (checklist 6) remain
  untestable until Phase 3/5.
