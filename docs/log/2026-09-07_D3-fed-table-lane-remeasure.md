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

## Page classification (table / figure / prose)

The per-document recall table above mixes three different page kinds under one
number, and that mixing is what makes 1977/1982/1990/2020 look uniformly bad.
Every page in the sample was reclassified by hand: for the three born-digital SEP
documents (2008, 2019, 2020), by reading the actual page title via `pdftotext
-layout -f N -l N` (`"Table N. ..."` vs `"Figure N.X. Distribution of
participants' ..."` vs narrative prose with no title); for the five scanned
documents, by `detected_table_count` from the sidecar (the swap-line renewal page
is the only true table in each). A **figure** page here is a SEP histogram/dot-plot
chart whose "numbers" are axis-tick labels (`-1.0, -0.8, ..., 8`, a `Number of
participants` axis), not table cells — the repo's figures policy treats these as
gist-only, so losing axis-tick text is not a table-lane defect. Classification and
full per-page detail: `~/Data/socr/d3-fed-remeasure-2026-09-07/page_classification.json`
(script: `classify_and_report.py` in the same directory).

| class | pages | source numbers | shipped | recall |
|---|---:|---:|---:|---:|
| prose | 76 | 1,008 | 1,003 | 99.5% |
| table (born-digital) | 5 | 572 | 364 | 63.6% |
| table_scanned (swap-line) | 3 | 194 | 33 | 17.0% |
| **table + table_scanned** | **8** | **766** | **397** | **51.8%** |
| figure (SEP charts, gist-only) | 19 | 2,084 | 745 | 35.7% |
| **all pages** | **103** | **3,858** | **2,145** | **55.6%** |

(This 3,858/2,145 whole-sample total is close to but not identical to the team
lead's independently-scored 3,775/1,843 — likely a difference in disclosure-marker
stripping or numeric-token thresholds between the two scorers; not reconciled
further here since the class split below is the number that matters for the
verdict, not the unscoped aggregate.)

**The table-lane verdict is about the 8 `table`/`table_scanned` rows only:
51.8% recall (397/766), not the 55.6% whole-sample or the lower figure-diluted
number.** Prose is essentially undamaged (99.5%). Figure pages lose axis-tick
labels as designed/expected and are excluded from the table verdict below.

## Per-class findings

**Class B — SEP figure pages, gist-only by policy, not a table-lane defect.**
19 pages: 2008 pp16–19 (Figure 2.A–D), 2019 pp15–16/18–22 (Figures 1–3.A–E, minus
p17 which is prose), 2020 pp17–19/21–24/26 (Figures 1–4). Spot-checked 2008 p16:
the qwen candidate has 272/273 "numbers" but they are histogram axis ticks
(`-1.0- -0.8- ... 5.1`), zero pipe rows — the floor correctly withholds it as a
table candidate. The old run shipped these as loose prose text (no structural
loss to lose, since it never tried to structure them); main's 35.7% figure
recall is axis-tick text the repo doesn't promise to preserve. Not counted
toward the table-lane verdict.

**Class A — scanned swap-line tables (1977 p3, 1982 p3, 1990 p3), zero-witness
design gap.** Old run: 100% recall on all three. Main: 10%, 39%, 2%. The cache
holds a candidate for all three that transcribes the table **exactly** (100%
numeric-multiset recall against `pdftotext`, `best_cand=1.0` in the scorer
output) — and the ladder rejects it every time with `judge_reason:
"source_evidence_table: no local content evidence available for scanned
table"`. Root cause, confirmed in code: `tables/source_evidence.py` builds
scanned-page evidence from classical OCR (`pytesseract`) when installed, and
explicitly excludes the page's native text layer when it is distrusted (GH-163,
line ~191/337) — on this machine `pytesseract`/`tesseract` is absent
(`which tesseract` → not found), so the evidence bundle is empty
(`"no local content evidence available for scanned table"`, line 282), and
`corroborate_rows` abstains per its documented `total == 0 → abstain, never
clears` rule (A1a) — a provably correct candidate cannot pass without a witness.
Filed as **#658** ("scanned table pages fail closed with 'no local content
evidence' when classical OCR is absent — perfect candidates discarded (Fed
swap-line tables, D3)") — cite #658 for this class, not a new ticket.
**The fix is that ticket, not this log.** B1's marker-scoping fix (`#651`) is confirmed
working here: all three pages ship the D3 marker (`[page N failed: unverifiable
table — see image]` + page image) scoped to the table region only — the
surrounding prose on the same page is preserved, checked directly in the
sidecar text. The census's original complaint was the marker replacing the
*whole* page; that shape is gone.

**Class C — real born-digital table pages: 4 of 5 fine, one regression worth its
own ticket.** 2008 p15 (Table 2), 2019 p14/p23 (Table 1/2), 2020 p20 (Table 2)
all ship at recall ≥ 95% in this sample (2020 p20: 73/73 = 100%). **2020 p16
(Table 1, "Economic projections... June 2020") is the exception: main ships
0/205 (0%)** — `status=error`, `failure_mode=structure_class_ladder_exhausted`,
`landscape_page_refused=true` — while the cached gemini candidate scores 85.9%
recall (176/205) and was never shipped. This is a genuine table-lane regression
against the old run (94.1% document-level recall on this page under the
heuristic judge), not a figure-classification artifact. `landscape_page_refused`
fired 4 times total in the whole 103-page sample (2019 p14 — shipped SUCCESS
unaffected; 2020 pp16/19/21 — p16 is this table regression, p19/p21 are figure
pages), not "4x in 2020" as a single cluster of table loss — of the 3 in 2020,
only p16 is a real table page. #393 (rejected `/curia` alternative for GH-367)
is the open, mechanically-specified fix for exactly this shape: rotated-page
coordinate-frame mismatch between native word geometry and the upright-rendered
table crop, which would explain why a landscape-refused page's row/structure
checks fail even when a high-recall candidate exists. Cite #393, not #263 (#263
is confirmed closed above — it is a routing signal, not a failure, on the other
3 firing sites).

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
column geometries and not others.

**Geometry dump, calling the real `socr.core.born_digital` functions directly
against each PDF's page 1** (not a reimplementation — `_median_word_space_width`,
`_line_word_extents`, `_cluster_two_bands`, `_try_aligned_run`,
`_find_aligned_runs`, imported and run in-process): on all three still-broken
pages, `_find_aligned_runs` returns zero runs, and the reason is **not** a failed
gap/width-ratio/fill-share check — it never reaches those. It fails the
bijection precondition (`len(left) != len(right)` inside `_try_aligned_run`) on
every extension it tries within the 4-block fail-streak window:

| doc | step (blocks walked from `PRESENT:`) | left (honorific) lines | right (name) lines | result |
|---|---|---:|---:|---|
| 1968 p1 | +1 block | 1 | 4 | bijection fails |
| 1968 p1 | +2 | 1 | 7 | bijection fails |
| 1968 p1 | +3 | 1 | 11 | bijection fails |
| 1968 p1 | +4 (streak limit) | 12 | 2 | bijection fails — gives up |
| 1977 p1 | +1 | 13 | 2 | bijection fails |
| 1977 p1 | +2 | 13 | 10 | bijection fails |
| 1977 p1 | +3 | 13 | 12 | bijection fails |
| 1977 p1 | +4 (streak limit) | 14 | 12 | bijection fails — gives up |
| 1990 p1 | +1 | 1 | 7 | bijection fails |
| 1990 p1 | +2 | 1 | 10 | bijection fails |
| 1990 p1 | +3 | 11 | 3 | bijection fails |
| 1990 p1 | +4 (streak limit) | 11 | 9 | bijection fails — gives up |

**Root cause: the source PDF stores the honorific column and the name column as
separate multi-line text blocks whose line counts never match at any point
within the fail-streak window.** 1968's "Mr." column is split across 3 PDF
blocks (4+3+4 = 11 lines); its name column is split across 3 different blocks
(2+8+2 = 12 lines) — the block boundaries don't align 1:1 with visual rows, so
`_find_aligned_runs`'s block-granularity walk (`block_lines[end]` added whole,
one block at a time, capped at 4 consecutive non-matches) exhausts its
fail-streak budget before a block-range boundary happens to land on equal
left/right counts. **This does not reopen #592 as "C1's three geometric
discriminants declined a list shape they should have caught"** — the gap
(word-space), left/right width-ratio, and right-block fill-share checks the
team lead asked about never execute on these pages; the precondition ahead of
them is what fails. It is a distinct, narrower shape: block segmentation from
the source PDF doesn't line up with C1's per-block walk, on documents where the
honorific and name columns happen to break across an unequal number of PDF
text blocks. Worth naming precisely in a follow-up ticket (walk at line
granularity across block boundaries, or raise the fail-streak budget) rather
than folding into a "loosen the geometric thresholds" fix, which would not
touch this cause. 1970 (fixed) was not checked at this depth — plausible it
simply has one PDF block per column with matching line counts, but not
confirmed.

This is the single largest disagreement between what the plan believes is
fixed and what this sample shows; it needs its
own look, out of D3's scope to diagnose further.

(The swap-line table and the born-digital table pages are covered above as
class A and class C, with the corrected per-page numbers — including that
2020 p20's dense multi-lane table actually ships at 100% recall, and that the
"born-digital dense tables are broadly lossy" framing from the first draft of
this log conflated real table pages with SEP figure pages; see the
classification table and the note above F1a/F1b/F2 do not touch class A's
shape, because that table never reaches the normalizer at all.)

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

**No — not on the table lane, and the failure mode is now well-characterized
instead of an unscoped aggregate.** The verdict is scoped to the 8 pages that
are actually tables (5 born-digital + 3 scanned); the 19 SEP figure pages are
excluded as gist-only by policy, and the 76 prose pages (99.5% recall) confirm
the loss is table-specific, not a general regression.

**Table-lane recall: 51.8% (397/766 numbers), against ~97% document-level
recall for the same page set under the old heuristic-judge run.** That gap
splits into two named, differently-actionable causes, not one:

- **Class A — scanned tables, structurally cannot pass (1977/1982/1990 p3,
  17.0% recall).** A provably correct cached candidate (100% recall on all
  three) is rejected every time because the evidence bundle is empty
  (`pytesseract` absent on this machine, native text layer excluded by design
  per GH-163). This is an environment + witness-design gap, not a table-ladder
  logic bug. **The fix is a ticket, not this log** — either install classical
  OCR in the runtime, or add a policy for scanned tables with zero available
  witnesses.
- **Class C — one born-digital table regression (2020 p16, 63.6% class
  recall driven down from what would otherwise be ~100%).** 2008 p15, 2019
  p14/p23, and 2020 p20 all ship at ≥95% recall — the born-digital table
  lane is essentially fixed. 2020 p16 alone ships 0/205 while an 85.9%-recall
  candidate sits unused in cache, gated out by
  `structure_class_ladder_exhausted` + `landscape_page_refused`. #393 (open,
  mechanically specified) is the right existing issue for this shape —
  rotated-page coordinate-frame mismatch between native word geometry and the
  upright-rendered crop.

**Also not fixed, unrelated to the table ladder:** #592 (column-wise attendee
lists) persists on 3 of 5 sampled scanned documents — the plan's belief that
C1 resolved this class does not hold on this sample.

**Confirmed fixed or closed:** #511 large half (scan≠chart, E1), the E2
`table_not_scorable` prose over-firing, and #263 (confirmed absent as a
failure mode — the landscape-refusal event is a routing signal, fires 4 times
total across the sample, and only correlates with the one class-C regression
above, not with the other 3 firing sites).

## Open items for the plan owner

1. Class A (scanned-table zero-witness rejection) is #658 — a scan-specific
   witness path, or an explicit policy for trusting a high-confidence
   candidate when no witness is available. Not in scope for F1a/F1b/F2 as
   currently written (that table never reaches the normalizer).
2. Class C's single regression (2020 p16) plus the 3 other landscape-refusal
   sites point at #393 (rotated-page coordinate-frame mismatch) as the
   relevant open issue — worth confirming #393 explains 2020 p16 specifically
   before scoping a fix.
3. #592 needs a follow-up, but narrower than "geometric thresholds too
   strict": on the 3 still-broken pages, `_find_aligned_runs`'s bijection
   precondition (`len(left) == len(right)`) never holds within the 4-block
   fail-streak window, because the source PDF splits the honorific and name
   columns across an unequal number of text blocks — the gap/width-ratio/
   fill-share checks never execute. A line-granularity walk across block
   boundaries (or a larger fail-streak budget) is the shape of fix this
   points at, not a threshold change.
4. D2 (route cost) is still TODO; the ~3h01m / 103-page wall time here
   (dominated by the three born-digital documents) is a useful reference point
   for that ticket.
