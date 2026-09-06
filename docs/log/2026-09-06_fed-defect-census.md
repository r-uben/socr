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
