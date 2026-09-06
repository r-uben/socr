# Fed defect census — fed-01 corpus (institution 1 of 2)

2026-09-06. Corpus `central-bank-network/data/ocr-runs/fed-01`: 764 audited documents,
766 assembled, 5,507 pages (2,690 on 424 scanned docs, 1930s–1990s; 2,765 born-digital,
2000s–2020s). Run by another session on HPC (`vllm`, `Qwen3-VL-30B-A3B-Instruct`) at
`socr@6fa89d9` (merge of #388, 2026-08-31, 349 commits behind `main@eb14c82`), **judge degraded to
heuristic on every document** (`judge_degraded_to_heuristic`, 766/766: no vision judge on
the node). Source PDFs in `data/ocr-staging/fed-01/pdf/`.

Purpose: fix the order of the next plan by measured frequency × severity, and test which
defects are rules about documents rather than about the Fed. Second institution (ECB, 792
PDFs on `/Volumes/Main/Library/Databases/central_banks/ecb/`, no socr run yet) is the
next step; nothing below is ranked final until it is in.

Method: signature counts over all 766 assembled `.md` files plus all `audit_log.json`
events; page-level word recall against `pdftotext` on 40 random scanned docs (≤12 pages);
hand-reads of one page per class. Scripts were one-off, run inline; every number is
reproducible from the two globs above.

## Headline

| class | issue | pages | docs | silent? | loss? |
|---|---|---|---|---|---|
| two-column attendee list read column-wise (`Mr.`×12 then names) | #592 | 174 | 174 | yes, SUCCESS | binding loss (honorific ↔ name), order loss |
| phantom `![Chart page N]` on plain scanned text pages | #511 (large half) | 2,765 | 650 | yes | **no** — prose is transcribed alongside; asset is noise |
| hyphenated line breaks shipped verbatim (`finan-\ncial`) | none | 43,734 lines | 358 | yes | word-search loss, not number loss |
| D3 fail-closed marker replaces whole page | #591 | 6 | 4 | no (marker) | prose loss on those pages |
| ditto marks kept verbatim | #625 | 240 rows | 16 | yes | none (faithful) |
| `&nbsp;` sub-row indentation | #624 | 104 cells | 11 | yes | hierarchy, not values |
| landscape page refused | #263-class | 25 | 21 | no | page-level |

## Per class

**#592 — column-wise list.** Signature: ≥3 consecutive lines that are a bare honorific.
174 pages in 174 docs, every one the `PRESENT:` block; by decade 1960s 39, 1970s 93,
1980s 30, 1990s 12 — it stops when the typeset minutes begin. Hand-read 1969-05-27 p1
(`figures/chart_page_1.png` is the full page): twelve `Mr.` lines, then `Martin,
Chairman …`. Every affected page shipped SUCCESS. This is the highest-frequency silent
defect in the corpus and it is geometric: a tab-aligned two-column run inside prose. Any
typewritten institutional document has it (attendance, votes, distribution lists).
The alphabetical-order proxy (member names between Vice Chairman and the next blank line
out of order) fires on only 6 of 113 parseable blocks, so order is *usually* preserved
within each column; the loss is the honorific ↔ name binding and the column interleave.

**#511 (large half) — phantom chart marks.** 2,765 marks on 2,618 of 2,690 scanned pages
(97%) and 147 of 2,765 born-digital pages (5%); the run's own `audit.json` counts only
140 `chart_pages`, so the audit counter and the markdown disagree by a factor of 20.
On scanned pages the "chart" is the page raster itself: `chart_page_N.png` is a full-page
render of typewritten prose (1969-05-27 p1 above). **Prose is not lost**: on 134 marked
scanned pages, word recall against `pdftotext` has median 1.00, p10 0.95, none below
0.5; the 303 marked pages with <20 output words are blank in the source too (40/40
sampled). Cost is one PNG per page, a misleading asset link on every page of every
scanned archive, and any downstream reader that trusts `chart_pages`. Rule: a raster that
covers the page *and* carries a text layer is the scan, not a figure.

**Hyphenated line breaks (no issue yet).** 43,734 hyphen-terminated lines, 39,851 of
them in born-digital docs — the native lane ships the text layer's line structure, so
`finan-\ncial` and `frame-\nwork` are searchable as neither word. Not number loss, and
arguably out of socr's remit (it is faithful to the layer); recorded because it is the
single largest fidelity defect by line count and it is independent of the institution.
Decide explicitly whether socr dehyphenates; do not let it ride into the plan by default.

**#591 — D3 marker over the whole page.** 6 markers in 4 born-digital docs (2008-01,
2008-04, 2020-06, 2020-09 minutes) on this run; the 1989-11-14 fixture that raised the
issue was run on `main` with the ladder, not here. Low frequency at `6fa89d9` with a
heuristic judge; expect more under the ladder, where fail-closed fires more often. Severity
is the highest of any class (unrecoverable prose on a page that says so), which is why it
stays near the top despite the count.

**#625 / #624 — ditto marks and `&nbsp;`.** Both are one table: the November swap-line
renewal, 1977–1992, 16 docs. 240 ditto cells, 104 `&nbsp;` cells. Values verified faithful
on the issue. Frequency is bounded by that one recurring table; rank last unless ECB shows
the same shapes elsewhere.

**Table lane, for context (all events, docs):** `table_not_scorable` 400/68,
`dualpass_flagged` 368/51, `chart_table_arbitration` 337/58,
`structure_class_native_fallback` 188/50, `table_structure_failed` 150/53,
`table_header_unverifiable` 98/52, `native_table_verifier_hard_fail` 26/25,
`table_value_drift_unadjudicated` 21/21. 74 docs carry markdown tables (4,083 rows).
These are pre-verifier-independence numbers with a heuristic judge and are not
comparable to run 3; re-run a table-bearing sample on `main` before drawing on them.

## Provisional order (Fed only; ECB pending)

1. #592 — most frequent silent loss; geometric fix generalises to any typewritten list.
2. #591 — rarest but worst; scope the marker to the table region.
3. #511 large half — corpus-wide noise, no loss; cheap rule, unblocks trust in
   `chart_pages`.
4. hyphenation — owner decision needed on remit before it becomes a ticket.
5. #625 → #624 — one table, one institution so far.

Order changes vs the 2026-09-06 STATUS proposal: #592 moves above #591 on frequency;
#511 drops below both because it loses nothing.

---

# ECB sample — institution 2 of 2 (same day)

Run on `main@eb14c82` (pinned worktree `~/repos/.worktrees/socr-census`, `PYTHONPATH`
set, source digest `f3510cf2…` on every sidecar), Mac, Ollama `qwen3-vl:30b-a3b-instruct`,
default agentic profile with the table-judge ladder on. Sample: 10 three-page excerpts
cut at random from `/Volumes/Main/Library/Databases/central_banks/ecb/` (3 meetings, 3
reports, 2 speeches, 2 surveys; `sample.json` records source + page window), 30 pages.
Inputs, outputs, `run.log`, `discarded.html` (source ↔ shipped ↔ every cached candidate,
side by side) at `~/Data/socr/census-ecb-2026-09-06/`. Scoring: numeric multiset of the
shipped page vs `pdftotext -layout`; each cached model candidate scored the same way,
plus **row binding** — a candidate row counts as bound iff its ordered numbers occur as a
contiguous run on one source line.

## Headline

| outcome | pages | of which had a ≥98%-numbers, row-bound candidate in cache |
|---|---|---|
| success, native (prose) | 8 | — |
| success, chart_asset (native text kept, page raster attached) | 5 | — |
| success, qwen | 6 | — (3 drop chart axis ticks only) |
| **error, fail-closed marker** (`structure_class_ladder_exhausted`) | **9** | **8** (the 9th: 70/104, ticks missing, table rows 8/8 bound) |
| warning, qwen `model_output_flagged` | 1 | gemini 389/389, ladder-ACCEPTED — socr shipped qwen's output truncated mid-number (`7,05`), 34/389 right + 55 wrong |
| error, qwen `hallucination` + `fabricated_image_ref` | 1 | — (chart page, loud) |

**Every statistical table in the sample was lost, and in every case socr already held a
reading that reproduced the page.** Candidate scores on the 9 fail-closed pages: 495/497,
414/417, 176/178, 471/473, 582/582, 73/74, 51/51, 51/53, 70/104; row binding 40/40, 36/39,
15/15, 37/37, 38/38, 8/8, 11/11, 5/5, 8/8. Shipped: 1–4% of the page's numbers. The
mechanism is identical on all nine: `native_table_verifier_warn: ambiguous_lane_count_mismatch
(paired/spanning headers possible — deferring to VLM)` → `table_structure_failed:
header_unattributed` on the model candidate → structure-class floor → marker. On three of
them the judge ladder had **ACCEPTED** the discarded table (#589's shape). Time cost:
851 s mean per fail-closed page (route 5–7 min, extract 8–15 min), 169 min for 30 pages.

Two D3 endings, not one: on 8 of 9 the marker is prefixed and the native layer follows
shredded (words 0.65–0.93 recovered — footnotes and titles survive); on survey-2013 p1 the
`page_failed` path shipped the marker alone and the question prose above the table
(words 0.00) is gone — **that is #591, reproduced on main on an ECB page**, alongside the
bulletin pages where it does not fire.

## Per class, ECB vs Fed

| class | Fed (766 docs) | ECB (30 pp) | general? |
|---|---|---|---|
| complete candidate discarded by header guard / structure floor (#589, #215) | not measurable at `6fa89d9` (heuristic judge) | 9/9 statistical pages | yes — the layout is every central bank's statistical annex |
| truncated model output shipped over a complete one | — | 1 (as WARNING) | yes, mechanical (ends mid-token, rows ≪ native) |
| #591 marker drops page prose | 6 markers / 4 docs (old tree) | 1/9 markers | yes — path-dependent (`page_failed`) |
| #592 column-wise list | 174 pp | 0 (no typewritten era) | geometric; needs a scanned second witness (BoE 1997 MPC minutes are scans but single-column) |
| #511 phantom chart marks | 97% of scanned pp | 5/30 (text kept, all numbers kept) | yes; no loss either side |
| `table_not_scorable` false trust flag on prose pages | 400 ev / 68 docs | every prose excerpt (3/3 pp on a transcript) | yes; noise |
| chart axis ticks dropped on qwen-lane chart pages | — | 3 pp (speeches, survey) | gist-only figures; acceptable, but inconsistent with chart_asset lane which keeps them |
| ditto / `&nbsp;` | 16 docs, one table | 0 | Fed-only so far |
| hyphenated line breaks | 43,734 lines | present (native prose) | remit question |

## Owner ruling (2026-09-06, mid-run)

When the header-attribution guard abstains but the candidate's rows match the native text
layer in order, **ship the table flagged "header binding unverified" instead of failing
closed.** Applies only where a native layer exists; scanned pages keep fail-closed. The flag
surfaces at every level (page, document, metadata, CLI). Corollary: never ship a candidate
that ends mid-token or carries far fewer numeric rows than the native layer while a complete
candidate is cached.

## Final order

1. **Ship-flagged on native corroboration + truncation guard** — covers #589 (ruling (c),
   generalised) and the four failed header-attribution attempts by changing what happens on
   abstain, not the heuristic. Largest loss per page by two orders of magnitude.
2. **#591** — reproduced on main via the `page_failed` ending; scope the marker to the
   table region on both endings.
3. **#592** — top Fed silent defect; geometric rule; one commit.
4. **Throughput** — route 5–7 min/page and a CPU nougat rung; mechanical, separate ticket.
5. **#511 large half** — scan raster ≠ figure; noise, no loss.
6. **`table_not_scorable` on pages with no table** — false trust flag; noise.
7. #625 → #624 — one Fed table; last.
Hyphenation: owner remit decision pending; not a ticket until ruled.
