# DECISIONS.md

Judgement calls made while executing PLAN.md. Format: decision · alternatives · why.

---

### D-001 (2026-11-16, Phase 0) — Python 3.13 in a local `.venv`
- **Decision:** use the system Python 3.13.5 to create `.venv/`, installing the project editable (`pip install -e .`).
- **Alternatives:** conda; uv; pin 3.11.
- **Why:** 3.13 satisfies the plan's ≥ 3.11, it's already installed, and every dependency ships 3.13 wheels.

### D-002 (2026-11-16, Phase 0) — Extra dependency `xlrd`
- **Decision:** add `xlrd` alongside `openpyxl`.
- **Alternatives:** convert the old `.xls` files by hand.
- **Why:** tennis-data.co.uk ships pre-2013 odds as `.xls`, which `openpyxl` can't read.

### D-003 (2026-11-16, Phase 0) — How subagents are invoked
- **Decision:** the role files live in `.claude/agents/` as specified. If the harness doesn't register them as agent types
  (custom agents are usually loaded only at session start), the orchestrator launches `general-purpose` agents whose
  prompt starts with the full text of the role file. The role, tool discipline and file ownership stay the same.
- **Alternatives:** restart the session; do the work inline.
- **Why:** keeps the plan's division of labour and parallelism without stopping for a routine choice.

### D-004 (2026-11-16, Phase 0) — Development split for Phases 3–5
- **Decision:** dev split = train 2000-01-01 → 2022-12-31, validate 2023-01-01 → 2024-12-31, with no test period.
- **Alternatives:** train ≤ 2021 / val 2022–23 / test 2024.
- **Why:** the plan's only real test sets are Wimbledon 2025 (split A) and 2026 H2 (split B), and Phase 4 already
  tunes Elo on 2015–2024. Calling 2024 a "test" would be false, because Elo selection has seen it. Phase 2 also
  checks Elo-only accuracy on 2023–2024, which matches this validation window.
  Caveat: Elo hyper-parameters are chosen on 2015–2024, which overlaps the XGBoost validation window. That is
  selection overlap on validation data only; no test period is involved.

### D-005 (2026-11-16, Phase 0) — Wimbledon 2025 boundary
- **Decision:** split A validation ends 2025-06-29. The Wimbledon 2025 main draw started Monday 2025-06-30, which
  should be its Sackmann/TML `tourney_date`. Phase 6 will check this against the data.
- **Why:** PLAN.md says "validate 2025-01-01 → day before Wimbledon 2025".

### D-006 (2026-11-16, Phase 0) — Retirements and H2H/form
- **Decision:** RET/DEF matches don't update H2H win counts or the last-k win-rate deque (config flags
  `retirement_updates_h2h: false`, `retirement_updates_form: false`). They don't move Elo (plan default) and their
  partial stats do enter the rolling buffers (plan default). Overall/surface match counts still increase for them.
- **Alternatives:** count a retirement as a win for the player who finished.
- **Why:** consistent with "retirements are never labels"; a retirement says little about relative strength.
  Revisit if needed (config flags).

### D-007 (2026-11-16, Phase 1) — Historical data source: archive mirror of Sackmann `tennis_atp`
- **Decision:** use the copy the human placed at `data/raw/tennis_atp/tennis-sackmann-archive-main/atp/` (the
  "tennis-sackmann-archive" mirror, also on Hugging Face as `Aneeshers/tennis-sackmann-archive`; ATP snapshot from
  upstream commits of June 2026; CC BY-NC-SA 4.0). `config.paths.sackmann_dir` points there and `data.last_sackmann_year`
  is 2026. `data/raw/` is not modified (no flattening of the nested folder).
- **Alternatives:** TML-Database as primary (tour level only, ATP IDs); unverified GitHub forks; scraping.
- **Why:** `github.com/JeffSackmann/tennis_atp` returns 404 as of 2026-11-16. The human supplied this copy, and it
  passed the source checks in `reports/data/source_check.md`: all 49 PLAN §4.2 columns, no read errors, no missing or
  unknown player IDs, no duplicate match IDs. Tour-level row counts for 2000–2025 also agree with the independent TML
  re-scrape to within 0–6 rows a year (Davis Cup excluded).
- **Consequence:** the archive also covers 2025 (complete) and 2026 up to Roland Garros (early June). So Sackmann
  data, not TML, will cover 2025 and 2026 H1. **No data exists in it after early June 2026**, which affects the Phase 7 test window.

### D-008 (2026-11-16, Phase 1) — K-factor `m` = 1 + matches already played in the current event (human decision)
- **Decision:** `elo.m_mode: event_matches_so_far`, with
  `m = elo.<overall|surface>.m + (matches the player has already played in this tourney_id before this match)`,
  and `elo.<which>.m = 1`. Qualifying matches of the same tourney_id count as matches already played in the event.
- **Alternatives:** constant m (plan default); the event's total match count (leaks how far the player goes; banned).
- **Why:** chosen by the human. It only uses matches before this one, so PLAN §2 rule 9 holds, and Ne + m ≥ 1 always
  (no division by zero on a debut, which the feature-engineer flagged for a bare count). Phase 4 tunes the base
  `m ∈ {1, 5, 10}`, i.e. the offset added to the in-event count.

### D-009 (2026-11-16, Phase 1) — Within-source duplicate matches
- **Decision:** after building `match_id`, also de-duplicate on (tourney_id, round, winner_id, loser_id, score), keeping
  the first row in match order. Every removed row is listed in the ingestion report.
- **Why:** the source check found 14 matches entered twice under different `match_num` (e.g. Viña del Mar 2007
  qualifying, Seoul CH 2024 final), which `match_id` doesn't catch; PLAN §2 rule 8 forbids duplicates. Including the
  round keeps real repeats in different rounds (Tour Finals 2024 Sinner d. Fritz 6-4 6-4 in RR and in F).

### D-010 (2026-11-16, Phase 1) — Self-matches
- **Decision:** the 4 Challenger rows where winner_id == loser_id are removed and listed in the report.
- **Why:** they can't be a real match and can't update two players' state. Not a "fix" of a result: no valid result exists.

### D-011 (2026-11-16, Phase 1) — Null surface
- **Decision:** the 53 rows (15 Davis Cup 2023 World Group I/II play-off ties) with null surface get `surface = Hard`
  and `surface_imputed = True`.
- **Alternatives:** an "Unknown" surface that skips surface Elo and emits no row.
- **Why:** it's 53 rows out of more than 330k; Hard is the most common surface. A flag keeps it auditable, and it avoids
  a fourth surface code path in the engine and the leakage reference implementation.

### D-012 (2026-11-16, Phase 1) — Impossible serve statistics
- **Decision:** rows whose serve counts are internally impossible (1stIn > svpt, 1stWon > 1stIn, 2ndWon > svpt − 1stIn,
  bpSaved > bpFaced, for either player) keep their result but get all 18 stat columns set to NaN and
  `stats_invalid = True`. That removes them from the stat buffers only. One such row exists (Roland Garros 2024 R64
  Machac d. Navone).
- **Why:** PLAN §4.4: missing stats stay NaN and the match is skipped for stat buffers only. The result isn't touched.

### D-013 (2026-11-16, Phase 1) — Round code `ER`
- **Decision:** add `ER` to `data.round_order` between R16 and RR.
- **Why:** 32 rows in four 2007 round-robin-experiment events (Adelaide, Buenos Aires, Delray Beach, Las Vegas) use
  `ER`, an elimination round whose winners went on to the RR groups (e.g. Adelaide 2007: Guccione ER → RR → QF → F).
  `match_num` isn't chronological in these events, so the round rank must place ER before RR.

### D-014 (2026-11-16, Phase 1) — Contradictory duplicate results
- **Decision:** when the same two players appear twice in the same round of the same tourney_id with OPPOSITE winners,
  drop both rows (`cleaning.drop_conflicting_results: true`) and list them in the ingestion report. 7 pairs, 14 rows:
  Kuala Lumpur CH 2007 R32, Burnie2 CH 2007 R32, Estoril 2010 Q3, Nice 2011 Q3, Newport 2011 Q2 (3 pairs).
- **Alternatives:** keep both (violates PLAN §2 rule 8 and the leakage duplicate test); keep one (would mean guessing
  the result).
- **Why:** a knock-out round can't contain the same meeting twice with opposite results, so no reliable result
  exists (same reasoning as D-010). Found by the data-engineer, confirmed independently by the leakage-auditor's
  test 5 and by the orchestrator's manual check. None of these rows is a training row.

### D-015 (2026-11-16, Phase 1) — Score vocabulary and walkover/retirement classification (data-engineer)
- W/O, Walkover → walkover. RET, RE, Ret'd, DEF, Def., Default → retirement. ABD and "Played and unfinished/abandoned"
  → retirement (an unfinished match is never a label). A trailing `?` on a set and `[a-b]` match tie-breaks are normal
  results. Unknown tokens raise.
- "No points played" = a score showing no game (null, `>`, `0-0 RET`, bare RET/DEF) with no evidence of play
  (`w_svpt>0`, `l_svpt>0` or `minutes>0`) → walkover. All 50 null scores are walkovers by this rule.
- An explicit W/O stays a walkover even when stats show play (3 rows; e.g. Hong Kong 1991 QF with 121 minutes). This
  is the conservative choice: a walkover is never a label. The rows are 1991, before any feature rows.
- 14 Davis Cup rows where the deciding match tie-break appears won by the loser (e.g. `1-6 6-4 [12-14]`) are kept
  unchanged and listed. Results are never altered (data-engineer rule), and the winner_id is the source's record.
- D-012 also treats negative stat values as impossible. That gives 5 `stats_invalid` rows, not 1 (adds Roland Garros
  2022 Q1, Roland Garros 2025 Q1, and Tokyo Outdoor 1991 SF and F with negative bpSaved).
- Ingestion report years are by `tourney_date` year, not file year.

### D-016 (2026-11-16, Phase 1) — Phase 1 audit
- **Decision:** no separate fresh audit after Phase 1. The leakage-auditor that wrote the suite also audited the real
  unified table (schema, int order_key vs the authoritative sort, no order ties, flags, duplicates). Its one FAIL
  (D-014) is fixed and test 5 now passes on real data. The Phase 2 audit, run by a fresh auditor, will re-check
  the unified table along with the engine.
- **Why:** Phase 1 produced no features, splits or models; a second auditor would rerun the same checks.

### D-017 (2026-11-16, Phase 7 plan) — Shortened forward test (human decision)
- **Decision:** split B = train 2000-01-01 → 2025-12-31 · validate 2026-01-01 → 2026-03-31 · test 2026-04-01 → end of the
  archive data (`test_end: 2026-06-07`; last tourney_date in the data is 2026-06-01, ending with Roland Garros 2026).
  Report test results overall and for Roland Garros 2026 separately, each with 95% bootstrap CIs. Flag the small sample.
- **Sizes (training-level, non-walkover, non-retirement rows, counted 2026-11-16):** validation 932 matches; test **473**
  matches, of which Roland Garros 2026 is 118. The human estimated about 600–800; the real test set is smaller, so CIs
  will be wide (roughly ±4.5 pts overall, ±9 pts for Roland Garros).
- **Constraints:** no scraping or downloading of June–November 2026 data. The human may ask for it later if the Phase 7
  results look good. `test_end` is pinned so added data can't silently enter the test set.
- **Alternatives:** the plan's 2026 H2 test (no data); scraping tennisabstract (not approved); dropping Phase 7.
- **Why:** the archive (D-007) ends in early June 2026. This keeps a true out-of-sample forward test while honouring the
  no-download instruction. Phase 7 re-tuning (≤ 300 trials) uses only the Q1-2026 validation window.

### D-018 (2026-11-16, Phase 2) — Feature-engine details (feature-engineer)
- The builder raises if `round_rank` disagrees with `data.round_order` or `match_id` isn't unique (fail loudly).
- `matches_last_14_days` is inclusive (`date >= tourney_date − 14`) and counts every prior non-walkover match,
  including qualifying in the same event. Because tourney_date is the start date, `days_since_last_match` is 0 for a
  main-draw match that follows qualifying in the same tourney_id. Both use only PRIOR matches.
- Elo history gets a new value only when Elo actually updates; with `retirement_updates_elo: false` a RET/DEF adds none.
- `elo_grad` is a closed-form OLS slope, equal to `np.polyfit` to 1e-9.
- The stats buffer keeps exact running integer sums (all serve counts in the data are integers).
- `is_left`: hand codes `U` (4,275 rows) and `A` (25 rows) → NaN.
- Flip augmentation keeps the same `match_id` on flipped copies, and the helper is documented as training-rows-only.
  Off by default (`features.flip_augment_train: false`).
- Elo-only accuracy excludes exact Elo ties (7 of 5,937 rows in 2023–2024).

### D-019 (2026-11-16, Phase 2 audit fix) — Round-robin blocks are snapshotted from pre-block state
- **Finding:** phase-2 audit finding 1 (blocker). Within RR rounds Sackmann's `match_num` isn't chronological (e.g.
  Tour Finals 2019 #286 played Thursday, processed before #288 played Sunday). 858 emitted rows (1.1%) could see
  same-round results that happened later. On those rows the last-5 win-rate heuristic rises from 60.0% to 62.0%.
- **Decision:** a block = consecutive matches with the same tourney_id, round RR, level not in
  `features.rr_block_exempt_levels` ([D]). Every match in a block takes its features, K/m and expected score from the
  pre-block state; updates are then applied in order. Spec in CONTRACT §2 "Round-robin blocks".
- **Alternatives:** reconstruct real match days (no date per match in the data); drop RR rows (loses Tour Finals etc.).
- **Why:** provably leak-free whatever the order inside the block. The cost is that a day-3 RR match can't use day-1
  results from the same group, a small loss of legitimate information. Davis Cup is exempt: each tie is its own tourney_id
  and the rubber order is chronological (6,707 of 6,711, per the audit). Knock-out rounds need nothing, since each
  player appears at most once per round.

### D-020 (2026-11-16, Phase 2 audit fix) — Fill missing static attributes with constants
- **Finding:** phase-2 audit finding 2. Height, hand and age come from today's `atp_players.csv`, so whether they are
  missing encodes the player's later career (players with no height: median 2 career tour matches vs 19). Controlling
  for Elo, rank and experience, missing height has a logit coefficient of −0.77.
- **Decision:** fill missing values with `features.static_fill` (height 185, is_left 0, age 26), with no missing
  indicator. These columns are never NaN.
- **Alternatives:** drop height, is_left and age from the allow-list (cleanest, loses real signal); leave NaN (leaks).
- **Why:** removes the missingness signal while keeping the real attribute for the ~97% of rows where it's known.
  Constants come from config, not fitted (PLAN §2 rule 10). Leftover risk: a tree could split on exactly 185.0, but
  so many real players are 185 cm that this is diluted. Revisit if feature importance highlights it.
- Rank is unaffected: the audit confirmed `winner_rank`/`loser_rank` are pre-event values, and rank missingness is
  real pre-match information (unranked).

### D-021 (2026-11-16, Phase 2 audit fix) — Hard guard on flip augmentation
- **Finding:** phase-2 audit finding 3. `maybe_augment_train` only warned in its docstring against val/test use.
- **Decision:** it takes `split_name` and raises if any row is dated after that split's `train_end`. Flipped copies
  carry `is_flipped = 1`, and evaluation asserts it is absent on val/test (CONTRACT §4b).

### D-022 (2026-11-16, Phase 2 audit notes) — Laver Cup excluded; level labels corrected
- **Laver Cup** (an exhibition, 58 rows) is removed from matches entirely via `data.excluded_tourneys`. It isn't
  an official ATP competitive match and shouldn't be a label or move ratings.
- **Level overrides** by exact tourney_name (`data.level_overrides`): Tour Finals / Masters Cup / ATP Finals → F (2023
  was labelled A); Olympics → O (1992–2021 were labelled A). This changes only the level one-hots and
  `is_training_level` (both already training levels); results are untouched. NextGen Finals stays a training row
  (official ATP event).
- **Retirements excluded from evaluation** (audit note 6): per the plan, RET/DEF are never labels, which makes
  reported accuracy slightly optimistic relative to every real match. The Phase 6/7 reports must say so.
- **Amendment (2026-11-16):** the first Olympics regex (`^Olympics$`) matched nothing, because the real names are
  "<City> Olympics". It is now `^[A-Za-z]+ Olympics$` → O, covering 1992–2024. NextGen Finals / Next Gen Finals were
  labelled F in 2017–19 and 2023–25 but A in 2021–22; they are now consistently **A**. It's a U21 event with a short-set
  format and shouldn't share `level_F` with the top-8 Tour Finals. Found by the feature-engineer.
