# Phase 1 audit: leakage test suite written before the engine

**Verdict: FAIL.** One real-data finding needs a data fix (finding 1). The suite itself is complete and checked.

`pytest tests/leakage -q`: **11 passed, 1 failed, 32 skipped** (about 3 s).

## Findings
1. **Contradictory duplicate matches in `data/interim/matches.parquet`.** This breaks PLAN §2 rule 8.
   - Caught by `tests/leakage/test_05_splits.py::test_no_duplicates_real_matches` (line 104).
   - There are 7 pairs. Each pair has the same tourney_id, round and two players, a different `match_num`, and
     **opposite winners**. A knock-out draw can't produce that.
   - All 7 pairs come from Sackmann's own files, not from merging sources:

     | Tournament | tourney_id | Round | match_num | Players |
     |---|---|---|---|---|
     | Kuala Lumpur CH | 2007-3666 | R32 | 2 / 14 | 103921 v 104583 |
     | Burnie2 CH | 2007-3668 | R32 | 2 / 14 | 110685 v 103977 |
     | Estoril | 2010-468 | Q3 | 726 / 727 | — |
     | Nice | 2011-6120 | Q3 | 725 / 726 | — |
     | Newport | 2011-315 | Q2 | 710–715 (3 pairs) | — |

   - None of them is a training row. They only feed state: Elo, form, H2H and the stats buffers.
   - **Fix (data-engineer):** flag these pairs in ingestion and exclude them, like the D-009/D-010 removals. Keeping
     one of each pair would mean guessing the result, so drop both rows and list them in the ingestion report.
   - **Do not relax the test.**

## Real data that already passes (`matches.parquet`, 337,735 rows)
- Schema, int64 `order_key` equal to the position in the authoritative 5-column sort, round_rank, the flags, and no
  ties on the first 4 order columns.
- Against a mock builder on the 2018+ universe, tests 1, 2 and 4 pass in about 2.7 min. The real engine will be
  checked in Phase 2.

## What the suite covers (PLAN section 9)
| File | Test | Synthetic | Real |
|---|---|---|---|
| test_00 | Fixture sanity. Order is (tourney_date, tourney_id, round_rank, match_num, match_id) and int64 `order_key` must equal the position in that order | yes | schema, order_key, ties |
| test_01 | Future invariance: scramble everything after a cutoff and rebuild; earlier rows must be identical. Cutoffs: by date and mid-event. Also checks truncation, all non-default flags, no mutation of the input, that input row order is irrelevant, and that a stale `order_key` is irrelevant | yes | yes (2018+) |
| test_02 | Slow recompute against the independent `reference.py`. Covers every column, the emitted set, orientation and the K log (RET/DEF included) | every row, 3 configs | 200 samples |
| test_03 | Banned columns in the allow-list; every listed feature exists | yes | features.parquet |
| test_04 | Coin flip, label-mean band, two-seed symmetry, d = a − b | yes | yes |
| test_05 | Exact time_split membership per the rulings: dev has train/val only; replication test is by name and year; forward test is date ≥ test_start. Partitions are ordered and disjoint, with no duplicate matches | yes | yes |
| test_06 | Shuffled labels give dev val accuracy in [0.47, 0.53] | — | features.parquet |
| test_07 | 2063 beats 2120 at K=24 gives +13.95; vanilla K; the engine replays the formula after writing the row | yes | — |
| test_08 | Removing later rounds leaves the K log unchanged. Covers both m_modes, with and without the layoff bonus | yes | — |
| test_09 | Logged validation accuracies, XGBoost and Elo-only are all at most 0.75 | — | features.parquet |

## Validation of the suite itself
I used a throwaway mock builder and mock `time_split` in the scratchpad; nothing was added to the repo.

- **Clean mock:** passes all 30 synthetic engine-level tests. It agrees with `reference.py` to 1e-9 on every row.
- **Planted leaks:** each one was caught by the tests listed.

| Planted leak | Caught by |
|---|---|
| Post-update snapshot | 2, 7 |
| No sort | 1, 2, 8 |
| Sort without round_rank | 2 |
| Sort by `order_key` | 1 (stale order_key) |
| m from future event matches | 1, 2, 8 |
| A always the winner | 2, 4 |
| Qualifying rows emitted | 2 |
| Same-match stat as a feature | 2, 4 |
| Global normalisation | 1, 2, 4 |
| Overlapping split | 5 |

## Skipped now, and why
- Engine-level tests: `tennis.engine.build_features` does not exist yet (Phase 2).
- `features.parquet` is absent, and `cfg["model"]["features"]` is not defined yet (Phase 2).
- `tennis.model.splits` does not exist yet (Phase 3).
- No experiments/*.jsonl has been logged yet.

## Contract rulings the reference mirrors
- Layoff `played` has one entry per distinct tourney_id.
- match_id is the final tiebreaker.
- `k_log` includes RET/DEF matches.
- Dynamic K raises an error when ne + m ≤ 0.
- Column names come from config: `winrate_last_{k}`, `level_{L}`, `surface_{S}`.
- `m = elo.<which>.m + matches already played in this event`. Qualifying matches of the same tourney_id count.

---
## Orchestrator follow-up (2026-11-16)
The single FAIL finding, 7 contradictory duplicate pairs in `matches.parquet`, was fixed at ingestion by
DECISIONS D-014 (`cleaning.drop_conflicting_results: true`; both rows of each pair dropped and listed in
`reports/data/ingest_sackmann.md`). Matches went from 337,735 to 337,721. `test_05_splits.py::test_no_duplicates_real_matches`
now passes, unchanged. Full suite: 169 passed, 32 skipped (targets not yet built). A fresh auditor re-checks the
unified table in the Phase 2 audit (D-016).
