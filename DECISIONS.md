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
