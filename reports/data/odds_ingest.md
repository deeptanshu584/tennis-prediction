# Ingestion report: betting odds (tennis-data.co.uk) -> `data/interim/odds.parquet`

**Status: BLOCKED. Nothing was downloaded.** `data/raw/odds/` was not created, and `data/interim/odds.parquet` was not
written. The parser and matcher (`src/tennis/ingest/odds.py`) and their tests (`tests/data/test_odds.py`, 30 passing,
no network) are ready. When the yearly files are in `data/raw/odds/`, `.venv/Scripts/python -I -m tennis.ingest.odds`
will regenerate this report with the real counts.

Written 2026-11-16 by the data-engineer subagent (Phase 3).

## 1. What was requested and what happened

| step | request (UA: generic research string, >= 3 s apart) | result |
|---|---|---|
| 1 | `http://www.tennis-data.co.uk/alldata.php` | 301 to `https://tennis-data.co.uk/data.php` (Cloudflare) |
| 2 | `https://tennis-data.co.uk/data.php` | 200, 71,865 bytes. Lists the yearly ATP files, e.g. `hrjk-85HytOjkhth76j_ygh4jf7/2025/2025.xlsx` (2013-2026 `.xlsx`, 2000-2012 `.xls`) |
| 3 | `https://tennis-data.co.uk/robots.txt` | 200. Disallows `/` for named AI agents (see below) |

No data file was requested. The two pages were saved only to the session scratchpad, not under `data/raw/`.

## 2. Why the download stopped

- **robots.txt** (under "# Block major AI training models") has `Disallow: /` for `GPTBot`, `ChatGPT-User`,
  `Google-Extended`, **`Anthropic-AI`, `Claude-Web`, `ClaudeBot`**, `PerplexityBot`, `Applebot-Extended`, and others.
- **The data page says:** "Tennis-Data maintains full copyright over the data files in their spreadsheet format. Its use
  is intended for private individuals only, not commerical or data training products using automated
  bots/scrapers/AI."

The download would be made by an AI agent through an automated client. Sending it with a generic User-Agent would get
around the operator's stated opt-out, so the agent didn't fetch the files and used no other source (as instructed:
"if the files can't be downloaded, STOP and report").

## 3. Second issue: the 2026 file (D-017)

Each `<year>.xlsx` is a whole-season file, updated weekly. On 2026-11-16, `2026.xlsx` holds matches through
mid-November 2026. Downloading it would fetch June-November 2026 match data, which DECISIONS D-017 forbids. If the human
approves a manual download, `parse` still drops every row dated after `odds.max_date` (proposed `2026-06-07`) before
matching, and this report counts those rows. But that filter doesn't change the fact that the file was fetched. The
human should decide this explicitly.

## 4. Options for the human (decision needed)

1. **Manual download by the human**, as a private individual, for this non-commercial project. Place
   `2015.xlsx` ... `2025.xlsx` (and `2026.xlsx` only if D-017 is relaxed for this purpose) in `data/raw/odds/`, then run
   `.venv/Scripts/python -I -m tennis.ingest.odds`. Ingestion is fully local, and the human should judge whether this
   fits the site's terms.
2. **Contact Tennis-Data** for permission (the page invites questions about use).
3. **Drop the bookmaker baseline** and report only the literature figure (~72% for bookmakers, PLAN section 1) in
   Phases 6-7.
4. Another odds source. Not explored: the brief forbids switching sources without approval.

## 5. What the code does once files exist

- Reads `<year>.xlsx|.xls` for `odds.first_year..last_year` from `paths.odds_dir` (missing years are skipped and listed).
- Parses Date, ATP no., Location, Tournament, Series, Court, Surface, Round, Best of, Winner, Loser, WRank/LRank,
  WPts/LPts and Comment (normalised: Completed / Retired / Walkover / Disqualified / Awarded / Sched; unknown values are
  kept and listed). It also reads every bookmaker pair present (`PS`, `B365`, `Avg`, `Max`, `BFE`, `EX`, `LB`, `SJ`, `CB`, `GB`,
  `IW`, `SB`, `UB`), keeping the raw values. Non-numeric odds or ranks ('NR', 'N/A') become NaN. Empty spreadsheet rows,
  rows with an unparseable Date and rows after `odds.max_date` are dropped and counted.
- Book probability: the first book in `odds.book_preference` with both odds > 1, `p = (1/oW)/(1/oW + 1/oL)`.
- Matching: odds rows are grouped into events (year + ATP no. + Tournament). Candidate Sackmann main-draw events
  (levels `odds.levels`) are those with `tourney_date - pre_days <= first odds Date` and
  `last odds Date <= tourney_date + window_days`. The candidate matching the most player pairs wins, with tournament
  name similarity as the tie-break; low-similarity matches are listed for review. Names are resolved inside the
  chosen event's players: override, then `match_names` (incl. "Last F."), then a 'Surname F.' variant for
  multi-initial names ("Struff J.L.", "Zhang Zh."), then a unique surname within the event
  ("Ramos-Vinolas A." -> Albert Ramos). Rows are matched by the unordered pair, with the round as tie-break (RR vs F).
- If the odds file's winner is Sackmann's loser, `winner_disagrees=True` and `p_winner_book = 1 - p`. The Sackmann
  result is never changed. Duplicate odds rows for one match_id: the first is kept, the rest get status `duplicate_match_id`.
- Unresolved players make the run fail with `UnmatchedPlayerError` (after the report is written). The exception is at
  most `odds.max_unmatched_players` names whose rows are all non-training in the odds file (comment != Completed):
  those rows are excluded and their status is listed.
- Report sections: files and sizes, dropped rows, match rate per year / per series / per Sackmann level (with
  PASS/FAIL against `odds.min_match_rate`), status counts, unresolved players, unmatched events, unmatched rows,
  winner disagreements, missing odds per book per year, book used, comment vocabulary, nulls per column.

## 6. Counts

None: no files were downloaded.
