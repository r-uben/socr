# D3 — Fed table-lane re-measure on main

2026-09-07. TICKET-D3, `docs/plans/defect-census-fixes/TICKETS.md`. Purpose: the
2026-09-06 Fed defect census (`docs/log/2026-09-06_fed-defect-census.md`) ran on
`socr@6fa89d9` (349 commits behind, judge degraded to heuristic on every document —
no vision judge, no real verification). The plan's claim to fix institution 1 (the
Fed) is unverified until re-measured on `main` with the ladder tickets landed. This
is that re-measurement. Measurement only — no source changes.

## Setup

- Checkout: `~/repos/.worktrees/socr-d3`, branch `docs/d3-fed-remeasure`, at
  `main@9367c83` (2026-09-07, includes A1a/A1b/A1c/A2/B1/C1/D1/E1/E2 — every plan
  ticket merged before D3 except D2/F1a/F1b/F2, still TODO).
- Pinned-tree protocol: `PYTHONPATH=~/repos/.worktrees/socr-d3/src`; confirmed
  `import socr` resolves to this worktree before any run. Every page sidecar across
  all 103 pages carries the identical `socr_source_digest` —
  `edb778a57b38167e7abbca4d930689b85a4bb98383376241272d9254fee61900` — one value,
  confirmed programmatically, no drift mid-run.
- `run_fingerprint` example: `fp:b6941442c9941af9ca48128fa8d5e85573a41728c637d542c5a62ca209c33fee`,
  `socr_version 2.5.0`.
- Runs driven by the team lead's `run.sh` (one `socr process … --write-manifest`
  per document, foreground, sequential; socr's resume gate skipped the first
  document since I had already run it myself before the run-ownership handoff).
  Mac, Ollama `qwen3-vl:30b-a3b-instruct`, default agentic profile, table-judge
  ladder on.
- Sample: 8 fed-01 documents, 103 pages, stratified 1960s–2020s (`sample.json`):
  5 scanned pre-1990s minutes (typewriter, `PRESENT:` attendee blocks + one
  November swap-line renewal table each on 1977/1982/1990) and 3 born-digital
  minutes (2008 with SEP projection charts/tables, 2019 and 2020 with dense
  multi-column FOMC tables).
- Comparator: the same 8 documents' prior run at
  `~/repos/research/central-bank-network/data/ocr-runs/fed-01/<stem>/`
  (`socr@6fa89d9`, heuristic judge — the census run).
- Scorer: `~/Data/socr/d3-fed-remeasure-2026-09-07/census_score.py`, extending the
  A1c live-verification scorer
  (`docs/log/2026-09-07_A1c-corroboration-surfacing.md`). Numeric-multiset recall
  of the shipped page body vs `pdftotext -layout` for that single page (disclosure
  note blocks `[page N: ...]` stripped before extraction, per plan wording), word
  recall, every cached candidate's own recall plus a winner flag, and the sidecar
  fields the plan tickets added: `table_corroboration` (bound/total/share),
  `<!-- row unverified -->` count, chart-mark presence, `table_not_scorable`,
  `landscape_page_refused`, `structure_floor_overrode_ladder`. The OLD run has no
  per-page sidecars (phase-major, pre-progressive-pages), so it is scored at
  document granularity (whole assembled `.md` vs whole-document `pdftotext`) plus
  page-scoped `audit_log.json`/`manifest.json` events. Full JSON:
  `~/Data/socr/d3-fed-remeasure-2026-09-07/score.json`.

## Run times (from `run.log`, HH:MM granularity; each line is the *start* of that
document, so a document's wall time is the gap to the next start)

| document | pages | start | wall time |
|---|---|---|---|
| 1968-10-29 | 5 | 19:27 (pre-run, not timed here) | ~1.5 min (my earlier foreground run) |
| 1970-12-15 | 5 | 19:27 | ~1 min |
| 1977-11-15 | 6 | 19:28 | ~3 min |
| 1982-11-16 | 6 | 19:31 | ~5 min |
| 1990-11-13 | 5 | 19:36 | ~4 min |
| 2008-10-29 | 20 | 19:40 | ~45 min |
| 2019-06-19 | 29 | 20:25 | ~45 min |
| 2020-06-10 | 27 | 21:10 | ~78 min |
| **total** | **103** | 19:27 | **~3h01m** (`DONE 22:28`) |

Cost is dominated by the three born-digital documents with dense multi-column
tables (76 of the 103 pages, across 2008+2019+2020), consistent with D1's
throughput-per-page finding; not table-page count alone but candidate retries
on hard grids.

## Per-document summary (numeric-multiset recall)

| document | pages | new (main) recall | old (`6fa89d9`) recall | new warn/error pages | chart marks new/old | `table_not_scorable` new/old | landscape-refusal events new/old |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1968-10-29 | 5 | 1.000 | 1.000 | 1 | 1 / 4 | 0 / 1 | 0 / 0 |
| 1970-12-15 | 5 | 1.000 | 1.000 | 1 | 0 / 4 | 1 / 1 | 0 / 0 |
| 1977-11-15 | 6 | 0.434 | 1.000 | 1 | 0 / 5 | 0 / 0 | 0 / 0 |
| 1982-11-16 | 6 | 0.608 | 1.000 | 1 | 1 / 3 | 0 / 0 | 0 / 0 |
| 1990-11-13 | 5 | 0.253 | 1.000 | 1 | 0 / 4 | 0 / 0 | 0 / 0 |
| 2008-10-29 | 20 | 0.548 | 0.949 | 4 | 0 / 0 | 2 / 5 | 0 / 0 |
| 2019-06-19 | 29 | 0.694 | 0.989 | 5 | 5 / 5 | 3 / 9 | 1 / 1 |
| 2020-06-10 | 27 | 0.434 | 0.941 | 7 | 1 / 1 | 9 / 9 | 3 / 3 |

**The new-run recall numbers are not directly comparable to the old ones as a
"worse" signal** — see per-class findings below. The old run's heuristic judge
accepted whatever the model produced with no real verification; several of main's
lower numbers are the ladder correctly refusing a candidate and shipping a scoped
fail-closed marker instead of silently-wrong or silently-partial content. Numeric
multiset recall cannot distinguish "correct values, wrong table binding" from
"values genuinely missing" — that limitation is inherent to this scorer, not new
to this run.

## Per-class findings

**#511 large half (scan ≠ chart, E1) — fixed, broadly.** On the five scanned
documents, `chart_asset_page` fired on 20 of 27 pages under the old run (every
page whose raster is a full-page scan). On main it fires on 2 of 27 — both
genuine scanned-figure pages, confirmed by hand (1968 p5, 1982 p5: real
photographed/hand-marked figures, not typewritten prose). On the three
born-digital documents chart marks are unchanged (0/0, 5/5, 1/1) — E1 targets the
scan-raster-coverage case specifically and correctly leaves real chart pages
alone. This class is fixed for this sample.

**E2 (`table_not_scorable` scope) — fixed on prose, unchanged on genuine tables.**
Total events: old 25, new 15. The reduction concentrates on documents with few or
no real tables (2008: 5→2, 2019: 9→3); 2020, which genuinely has 9 dense
FOMC-projection table pages in this window, holds at 9→9 in both runs — expected,
since E2's scope guard is `detected_table_count > 0`, not a suppression of real
findings.

**#263-class landscape refusal — absent as a failure mode; confirmed CLOSED.**
4 `landscape_page_refused` audit events fired (2019 p14, 2020 pp16/19/21), same
count old and new. Read the event detail on all four: `"native table
reconstruction refused (dominant text direction is rotated); prose retained, page
routed to OCR"` — this is an internal routing signal, not a page-level outcome.
2019 p14 shipped SUCCESS. The three 2020 pages failed/warned for unrelated
reasons (dense multi-lane FOMC tables — `structure_class_ladder_exhausted`,
`model_output_flagged`), not because of the landscape routing. No page in this
sample was refused outright. #263 stays closed; nothing here reopens it.

**#592 (column-wise attendee list) — NOT reliably fixed; this contradicts the
assumption in the dispatch message.** Checked the `PRESENT:` block, page 1, on
all five scanned documents:

| document | engine | shape |
|---|---|---|
| 1968-10-29 | native | **broken** — 11 consecutive bare `Mr.` lines, then all 12 names |
| 1970-12-15 | native | fixed — `Mr. Burns, Chairman`, one line per person |
| 1977-11-15 | native | **broken** — 12 consecutive bare `Mr.` lines, then names |
| 1982-11-16 | qwen | fixed — model (not native) read the block, no split |
| 1990-11-13 | native | **broken** — 10 consecutive bare `Mr.` lines, then names |

3 of 5 sampled pages still ship the exact defect the census reported (honorific
and name on separate lines, geometrically column-interleaved), all `SUCCESS`, all
`engine=native`. C1 (`#631`, "native geometry #592", DONE) evidently fixed some
column geometries and not others — 1970's typewriter layout binds correctly,
1968/1977/1990's do not, and I did not find an obvious layout difference between
them without a deeper geometry dump. This is the single largest disagreement
between what the plan believes is fixed and what this sample shows; it needs its
own look, out of D3's scope to diagnose further.

**November swap-line table (1977/1982/1990, page 3 in each) — still lossy, by a
different, more principled mechanism.** All three ship a scoped D3 marker
(`[page N failed: unverifiable table — see image]` plus the page image); the
surrounding prose on the same page is preserved (confirms B1's marker-scoping
fix — the census's original complaint was the marker replacing the *whole*
page). But the cache holds a candidate for all three pages that transcribes the
table **exactly** (100% numeric-multiset recall against `pdftotext`, including
every ditto-implied value) — and the ladder rejects it every time with
`judge_reason: "source_evidence_table: no local content evidence available for
scanned table"`. These are pure scans with no native text layer, so
`corroborate_rows` has nothing to score against and abstains per its documented
`total == 0 → abstain, never clears` rule (A1a) — a correct-but-unwitnessed
candidate cannot pass. Net effect vs the old run: the old heuristic-judge run
shipped this table's values (with the ditto/`&nbsp;` cosmetic defects the census
flagged, #625/#624) as `SUCCESS`; main now refuses to trust it at all and ships
nothing for the table. That is a deliberate, principled trade (never ship
unwitnessed content) but it is still a content-loss regression against the old
run's number, and F1a/F1b/F2 (ditto text, derived-cell provenance, `&nbsp;`
hierarchy — all TODO) do not touch this shape, because the table doesn't reach
the point where those normalizers would run. Scanned tables with zero native
text layer are the shape D3/A1a-A1b cannot corroborate by construction; that is
worth a named follow-up (a scan-specific corroboration signal, or a deliberate
policy choice to trust word-count-derived table candidates below some floor when
evidence is unavailable) rather than folding it into F1a/F1b/F2 as currently
scoped.

**Born-digital dense tables (2008 SEP projections, 2019/2020 FOMC minutes) — hard
shapes, correctly failing loud rather than silently wrong, still a net content
loss.** 16 of 76 born-digital pages across the three documents are
`warning`/`error` (2008: 4/20, 2019: 5/29, 2020: 7/27). Spot-checked 2008 p16
(33 native lanes vs a 3-column model read, `value_guard_row_count_warning` +
`native_table_verifier_warn` + `table_escalation_rejected 0%`) — `main` ships the
D3 marker rather than a 3-column mis-binding of a 33-lane table; the old run's
manifest shows this same page shipped `engine=qwen`, `failure_mode=none` (silent
`SUCCESS`) under the heuristic judge, with no way from the old artifacts alone to
confirm whether that silent output was actually right or wrong. I was not able to
recover per-page text from the old run's single assembled `.md` (no page
boundary markers in the phase-major output) to check directly; this is a real gap
in the old-run comparison, noted rather than papered over. Separately, several
low-recall "success" pages in 2008/2019 (pp14/15/18 across docs) are SEP
fan-chart pages losing only histogram axis-tick labels — the same
"drop-chart-axis-ticks-only" shape already recorded on the 2026-09-06 ECB sample,
not a table-lane defect and not new.

**Corroboration path (A1a/A1b) rarely exercised in this sample.**
`table_corroboration` populated on exactly 1 of 103 pages (2008 p12, `bound=16
total=16 share=1.0`, a clean 5-column variable table). `<!-- row unverified -->`
never fired; `structure_floor_overrode_ladder` never fired. Most table pages in
this sample either succeed without needing corroboration (native/qwen agree) or
fail before reaching it (no grid candidate at all, or no native-word evidence on
scans). This sample is too small and too skewed toward "clean success" or
"total failure" to exercise A1b's middle path (a candidate that clears
corroboration with one named-bad row); that path remains effectively unverified
at scale by this measurement.

## Verdict: is institution 1 (Fed) fixed?

**Partially, and unevenly across defect classes.** Fixed or closed in this
sample: #511 large half (scan≠chart), the E2 `table_not_scorable` prose
over-firing, and #263 (confirmed absent, was already believed closed). **Not
fixed**: #592 (column-wise attendee lists) persists on 3 of 5 sampled scanned
documents — the plan's belief that this class is resolved does not hold on this
sample and needs its own investigation. **Behavior-changed, not fixed**: the
hardest table shapes — scanned tables with no native text layer (November
swap-line renewal, 1977/1982/1990) and dense multi-lane born-digital tables
(2008/2019/2020) — now fail loudly with a scoped marker instead of shipping
silently-plausible-but-unverified content, which is the correct safety
direction, but real prose/table content is still not delivered on 16 of 76
born-digital pages and on all 3 sampled swap-line tables. F1a/F1b/F2 (ditto,
provenance, `&nbsp;`) remain TODO and, per the finding above, would not resolve
the swap-line table's loss even once implemented, because that table currently
never reaches the normalizer at all.

## Open items for the plan owner

1. Reopen investigation on #592 — 3/5 native-engine pages in this sample still
   split honorific from name; C1's fix is geometry-dependent, not general.
2. Scanned tables with zero native-text evidence (`source_evidence_table: no
   local content evidence available`) cannot pass A1a/A1b's corroboration gate
   even when the candidate is provably correct (measured: 3/3 such candidates in
   this sample were 100% right and still rejected) — needs its own ticket; not
   in scope for F1a/F1b/F2 as written.
3. D2 (route cost) is still TODO; the ~3h01m / 103-page wall time here (dominated
   by the three born-digital documents) is a useful reference point for that
   ticket.
