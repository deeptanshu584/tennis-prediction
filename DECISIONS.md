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

### D-023 (2026-11-16, Phase 3) — TML-Database is not ingested
- **Decision:** don't build `src/tennis/ingest/tml.py` or merge TML. Details in `reports/data/tml_coverage.md`.
- **Alternatives:** ingest TML 2025–Jan 2026 and de-duplicate against the archive.
- **Why:** the Sackmann archive (D-007) already covers 2025 completely and 2026 to 2026-06-01, including Challengers
  and qualifying, with native Sackmann ids. TML's GitHub copy is tour level only, has ATP codes and stops in
  mid-January 2026, and 2000–2025 per-year counts agree with the archive to within 0–6 rows. Post-June-2026 data is
  excluded by D-017. The generic `merge.py` and `match_names` stay ready if another source is approved.

### D-024 (2026-11-16, Phase 2 re-audit) — Follow-ups from the PASS re-audit
- **R1:** in Phase 5, check how much XGBoost relies on `height` (gain importance). If it ranks high, drop height from the
  allow-list. The residual signal from the 185-cm fill measured as negligible out of sample (log loss 0.61522 vs 0.61519).
- **R2:** `maybe_augment_train` keeps `split_name` optional, because test 11 calls it without one. With no split name
  it guards only against the latest train_end. **Rule:** all model code must pass `split_name` and call
  `assert_not_augmented` on every validation/test frame (test 14 covers the split-specific guard). The Phase 5 brief
  will require it.
- **R3:** cross-source ordering doesn't apply; no second source is merged (D-023).

### D-025 (2026-11-16, Phase 3) — Betting odds: blocked, waiting on the human (🛑)
- tennis-data.co.uk is reachable, but its robots.txt `Disallow: /` covers `Anthropic-AI`, `Claude-Web` and `ClaudeBot`,
  and the data page says the files are not for "automated bots/scrapers/AI". The odds agent requested no data file.
  The orchestrator will not fetch them under any User-Agent: that would get around the site owner's opt-out.
- Also, `2026.xlsx` is one season file updated weekly, so fetching it would bring June–November 2026 data (D-017).
- Built anyway: `src/tennis/ingest/odds.py` (parser and matcher, never downloads; 30 tests) and the `odds:` config block.
  The parser drops rows after `odds.max_date`.
- Only the bookmaker baseline in Phases 6–7 depends on odds. Phases 3–5 go ahead without them.

### D-026 (2026-11-16, Phase 3) — Evaluation conventions (ml-engineer)
- Accuracy gives half credit when p is exactly 0.5 (equal Elo, or equal/both-missing rank). That is the expected
  accuracy of a coin flip and doesn't depend on orientation. It supersedes D-018's "exclude Elo ties" in all reports.
- Rank rule: missing rank = unranked, worse than any ranked player. Rank logistic `P(A)=σ(β·(ln b_rank − ln a_rank))`,
  no intercept (A/B is a coin flip). β = 0.6238 and the unranked fill (2160) are fitted on TRAIN rows only.
- Bootstrap: percentile CIs over rows, with all metrics sharing the same seeded resamples. Model-vs-baseline
  comparisons use paired bootstrap differences.
- Calibration: 10 equal-width bins of [lo, hi); p = 1.0 goes in the last bin.
- The leakage alarm applies to full-window and per-season accuracies, never to tiny subgroups. Subgroups above 0.75
  are called out in reports (e.g. level F, n=29).
- Replication split: events starting after 2025-06-29 other than Wimbledon 2025 belong to no partition (1,110 rows
  in 2025, plus 2026).
- `evaluation.log_loss_eps` and `evaluation.baseline_seasons` moved into config (the code should read them; Phase 4
  brief).
- **Phase 5 bar:** XGBoost must beat the best baseline on validation log loss: currently the rank logistic (0.6297)
  and the Elo/surface blend (0.6336), not just Elo (0.6377).

### D-027 (2026-11-16, Phase 4) — Elo settings: tuned on 2015–2024 log loss, with the plan grid widened once
- **Plan grid result:** the best on both flavours was div 1200, m 1, base_k 16, max_k 64, bonus on (overall log loss
  0.60128 vs default 0.62690). It sat on the div_number and base_k edges, and edge probes one step out helped
  (−0.001 to −0.002).
- **Decision:** widen the grid ONCE to div_number {400…3200}, base_k {0…40} (the plan values are a subset; 650 settings
  × 2 flavours) and take the best:
  - **overall:** div 3200, m 1, base_k 16, max_k 100, bonus on. 2015–24 log loss **0.59865**, accuracy 0.6689.
  - **surface:** div 2400, m 10, base_k 24, max_k 64, bonus on. Log loss **0.60772**, accuracy 0.6641 (inside the grid).
- **No further widening:** overall still sits at the div edge, but the curve has flattened (2400 → 3200 is −0.0006), and
  m and max_k are barely identified (top settings differ by < 0.0001). Pushing further would mainly overfit the tuning window.
- **Why it helps:** with base_k = 16, experienced players keep K ≥ 16 instead of decaying to about 1.6, so ratings
  react faster. The layoff bonus gives a small but significant gain (≈ −0.001). Fixed K = 24 with the bonus is within
  noise of the plan-grid best.
- **Caveat (D-004):** the tuning window includes the dev validation years 2023–24, so Phase 5's Elo validation
  numbers are slightly optimistic. No test period (Wimbledon 2025; 2026-04-01 to 06-07) was used.
- **Harness details (ml-engineer):** it scores the winner's pre-match probability (equal to coin-flip-oriented
  scoring, since the metrics are symmetric). Edge probes are diagnostic only. Surface Elo is keyed by (player, surface),
  with Ne on that surface and the same in-event m and layoff list. The harness is bit-identical to the engine
  (tested on synthetic and real data).

### D-028 (2026-11-16, Phase 5 audit part 1) — Fixes and disclosures
- **Finding 1 (fixed):** `dataset.is_banned` now also catches `w_`/`l_` behind one `a_`/`b_`/`d_` player prefix
  (e.g. `a_w_ace`). It uses a prefix check, not a substring check, so `level_A` isn't banned. No allow-listed feature
  was affected. New auditor test 18 passes; tests 15–17 (test-window pinning, experiment-log hygiene, Elo tuning
  window) were added and pass.
- **Finding 2 (disclosure, rule 6):** Elo-only accuracy for the FULL 2025 and 2026 seasons (63.71% / 64.91%, untuned
  default Elo) appeared in reports/audits/phase2.md and in the "2000–2026 maximum" line of PROGRESS.md. Those seasons
  contain Wimbledon 2025 and the forward test window. Nothing was selected on them: Phase 4 used 2015–2024 only, as
  test 17 verifies. Phase 6/7 reports must disclose this. **Rule from now:** sanity checks before Phases 6/7 use
  seasons ≤ 2024 or the validation windows only.
- **Finding 3 (note):** re-selecting Elo on 2015–2022 only gives the same overall setting (no inflation of 2023–24) and
  a different surface setting (≈ 0.0012 log loss, 0.3 pt accuracy optimism on 2023–24). The bias hits all Elo-based
  baselines and XGBoost alike, and the tests are unaffected (tuning ends 2024-12-31).
- **Phase 6/7 retrain rule (auditor checklist):** refit on train+val with a fixed number of trees (`best_iteration + 1`
  from dev validation) and no early-stopping set. The test is never passed as `val_df`. Assert max(refit date) ≤
  `val_end` and no shared match_id with the test. Always pass `split_name`. Score the test once and log it with a
  single config hash under a `test` key.

### D-029 (2026-11-16, human decision) — Bookmaker baseline: quoted figure, no odds data
- **Decision (human, option b):** no betting-odds data is used. Phase 6/7 reports quote the published bookmaker
  figure (~72% on Wimbledon 2025, PLAN.md reference points) as an external reference only. It is not computed on our
  test rows, so it is not a like-for-like comparison and has no CI.
- **Why:** tennis-data.co.uk's robots.txt opts out of AI agents (D-025), and we don't bypass that. The human chose not
  to download the files by hand.
- **Alternatives:** (a) the human downloads 2015–2025 odds into `data/raw/odds/`; (c) another source.
- **Kept in place:** `src/tennis/ingest/odds.py` and its 30 tests. If odds files ever appear in `data/raw/odds/`,
  re-running the Phase 6/7 evaluation adds a computed bookmaker row. There is no 2026 forward-test bookmaker baseline
  until then.

### D-030 (2026-11-16, Phase 5) — XGBoost tuning choices
- **Best params:** trial 720 of 800 (Optuna TPE, seed 42), written to `xgb.best_params` in config.yaml with
  `n_estimators: 689` (best_iteration + 1 at the dev early-stopping point). Dev val 2023–24: accuracy 0.6672, log loss
  0.5936, against 0.6099 for Elo and 0.6056 for the blend. The paired-bootstrap CIs exclude 0 for both.
- **Early stopping and selection on the same window:** both use dev val 2023–24, so the dev metrics are slightly
  optimistic. The search surface is flat (0.5936–0.5971 across all 800 trials), so selection bias is small. Phases 6/7
  give the honest numbers.
- **Threads:** `nthread = 1` for every trial and refit (`xgb.nthread`), and `tree_method: hist` (`xgb.tree_method`).
  hist results depend on the thread count. Trials ran asynchronously on 8 processes, each with BLAS threads pinned to 1
  (11 workers ran out of memory). Each trial is bit-reproducible; the trial sequence is reproducible only with 1 worker.
- **Missing values:** NaN goes to XGBoost's native missing-value handling. Nothing learned outside the booster (rule 10).
- **Height (D-024 R1):** the height features rank 71–93 of 106 by gain, so they are kept.
- **Calibration:** ECE is 0.023 for XGBoost, 0.019 for Elo. No calibrator; revisit only if the Phase 7 validation
  shows drift.
- **Phase 6/7 tree count:** refit on train+val with a fixed number of trees and no early-stopping set (D-028 rule).
  - Phase 6 uses `n_estimators = 689` from dev.
  - Phase 7 re-tunes (≤ 300 trials) with early stopping on 2026 Q1, then refits on train+val with that run's
    best_iteration + 1.

### D-031 (2026-11-16, Phase 5 audit final pass) — Fixes; the gain over the blend is real
- **Verdict:** no leak found. Two FIX items were applied, after which test 19 passes.
- **F1 (fixed):** `xgb.best_params` had been written to 6 significant figures, which trains a different model
  (best_iteration 518, not 688). Config now holds the exact trial-720 values, and `tune._yaml_params` writes
  `repr(float(v))`. Forward params must be full precision too.
- **F2 (fixed):** `train.fit_xgb` raises if any early-stopping row lies outside `splits[split_name]` val_start..val_end.
  So the test can never be an early-stopping set.
- **F3 (note):** the report's shuffled-label run stopped after 1 tree. Test 19 re-does it with 689 fixed trees and no
  eval set: 0.5084, inside [0.47, 0.53].
- **F4 (fixed):** `native_params` raises if `xgb.nthread` != `train.NTHREAD`.
- **F5 (note):** the dev model carries the tuning-time config hash. Phase 6/7 models are saved under the frozen hash.
- **Gain over the blend (−0.0120 log loss, above the 0.01 second-look line):**
  - About −0.010 survives a protocol with no early stopping on 2023–24, so roughly 0.002 is selection optimism.
  - Serve/return stats add only about 0.0017. Most of the gain is Elo adjusted by format, level, experience, form and
    rest.
  - 35% of the gain is in best-of-5 matches. Unranked players (n = 92) win 19% against an Elo-implied 39%; this is
    real, since rank presence is fixed per player per tournament.
  - Recomputing 3,080 feature values for 55 validation matches from strictly earlier matches matched every one.

### D-032 (2026-11-16, Phases 6–7) — Frozen settings before the single test scoring
- **Replication tree count:** 405 = best_iteration + 1 of the replication validation run, using the full-precision
  `best_params` with early stopping on 2025-01-01..06-29. The final model is refit on train+val with 405 fixed trees
  and no eval set (D-028).
  - **Why:** this tree count comes from the training size closest to the refit, and it is the same rule as Phase 7.
  - **Alternative:** the dev count of 689, which is mentioned in the report and not used.
  - This supersedes the "Phase 6 uses 689" line in D-030.
- **Forward params:** trial 253 of 300 (Optuna TPE) on the 2026 Q1 validation window, 393 trees, frozen as
  `xgb.forward_best_params` at full precision. gamma (0.997) sits near the top of its [0, 1] range. We don't
  re-search: the dev surface was flat and the validation window is small (n = 932).
- **Validation-window rows in the refit are not flip-augmented:** the D-021 guard forbids it. No effect while
  `flip_augment_train` is false.
- **Freeze:** both keys were added together before either test was scored, so both test records carry one frozen
  config hash. Each test is scored once (`evaluate_final --score-test`), which refuses a second scoring.

### D-033 (2026-11-16, Phase 8) — Prediction tool
- **State:** `engine.snapshot.build_state` runs the same chronological pass as `build_features` (`run_pass`,
  refactored with bit-identical output, verified on the full table). `query_rows` builds rows from the final state
  without updating it, so every query in a draw sees the same state. Consistency gate (tests/engine/test_snapshot.py):
  on first-round matches, synthetic and real (Roland Garros, Geneva and Hamburg 2026), query rows equal the engine
  rows to 1e-9.
- **Query-time static attributes:** rank and rank points are the last known values; age is projected forward; players
  with no prior row get the D-020 fills; overrides are allowed per query. `--date` is the tournament start date,
  defaulting to the day after the cut-off. Earlier dates are refused.
- **Production model:** `forward_best_params`, 393 fixed trees, fitted on all 76,421 rows (2000 → 2026-05-25), allow-list
  only. It is never evaluated (meta `never_evaluated: true`). Its training data includes the forward test window,
  which is fine because no metric is computed with it.
- **Symmetrised output:** p = (p(A,B) + 1 − p(B,A)) / 2, so swapping players gives exactly 1 − p. Without flip
  augmentation, XGBoost isn't orientation-symmetric: for Sinner–Alcaraz on grass, the two orientations give 0.52 and
  0.70. The evaluated models scored one coin-flip orientation, so this output is a slightly different estimator. It
  is disclosed, and the raw values are kept in the draw CSV. Next idea: evaluate flip augmentation or symmetrised
  scoring on validation.
- **Names:** full name or id, ignoring case and accents, using the most recent spelling. A bare surname or an
  ambiguous name is an error that lists candidates, never a guess.
- **Cache:** `models/state_<config_hash>.pkl`. It is invalidated by the config hash, a format version, a hash of the
  engine source, and the matches file's size/mtime.

### D-034 (2026-11-16, Phase 7 final audit) — PASS; follow-ups to the notes
- **Verdict:** PASS, no leak (reports/audits/phase7.md). Leakage suite 96 passed; full suite 456 passed (clean
  worktree). New test 20 (12 tests) pins exactly one frozen test record per final split, and checks that the CSVs equal
  the test partitions, the metrics recompute, and a pre-test refit reproduces the scored models.
- **Note 1 (fixed):** `evaluate_final.run_final(score=True)` also refuses if a `xgb_<split>_final_*_test_predictions.csv`
  exists. Without this, a crash between saving the predictions and logging would allow a silent re-score. One unit test
  was added.
- **Note 3 (left as is):** the Parma CH 2020 F walkover/played twin survives D-009 de-duplication, because the key
  includes the score. It is harmless: walkovers are skipped everywhere, and it is a Challenger, never a feature row.
  Fixing it would change matches.parquet after the tests were scored, for no model effect.
- **Note 4:** the Phase 8 engine refactor is covered by leakage tests 01, 02 and 13 in the main-repo full run before
  the Phase 8 commit.
- **Predict settings stay in code, not config:** `DAYS_PER_YEAR` (a calendar constant), the thread default and
  "symmetrise" are fixed design choices of the tool (D-033), not tunable model settings. Adding them to config would
  change the config hash away from the one stamped on both test records (3c3f2599a311) for no model change.
