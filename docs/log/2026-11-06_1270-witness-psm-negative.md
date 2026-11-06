# #1270: tesseract `--psm` and crop coverage for the old-scan table witness (negative result)

Date: 2026-11-06. Base: main 4e170962. No source change. Read this before trying psm, padding or DPI
on the scanned-table witness again.

## Question

On old Fed scans, the classical witness (`build_scanned_evidence`, `src/socr/tables/source_evidence.py`)
reads each `locate_tables` crop with tesseract (default psm 3, 150 DPI, 4 s per call), and
`verify_table_tokens` refuses the page when any qwen number is missing from the witness text (a
whole-page multiset). The census for #1261 found 106 old-scan pages refused this way, mostly because
tesseract misread the number. #1270 asked whether a different page-segmentation mode or a larger crop
raises correct passes without admitting a wrong table.

## Measurement

134 old-scan floor pages: the 106 refused, plus 28 other old-scan pages. The witness was replayed on
Rorqual with socr 4e170962 and tesseract 5.3.0, through `verify_scanned_table(native_trusted=False,
layer_is_evidence=False)`, the suspect-layer arm the orchestrator takes. Inputs were qwen's cached
reading from the Fed run. The 19 arms were psm {3, 4, 6, 11} x region {table box, box +5%, box +15%,
full-width band} at 150 DPI, full page with psm 3 and 6, and psm 6 at 300 DPI. Unions of psm modes on
the same region (per-token max of counts) were scored offline from the recorded texts. The offline
recompute reproduced every recorded single-arm verdict on numeric tables. The production arm
reproduced the run's recorded refusal reason, character for character, on the smoke page.

Scripts and data (not in git): `scratch/army/1270/step1/` (`arms.py`, `analyze.py`, `analysis.txt`,
`flips.json`, `arms.jsonl`, table tiles in `tables8/`, qwen readings in `qwen8/`).

| arm | refused pages that pass (of 106) | passing pages lost | page wall p50 / p99 (s) |
| --- | --- | --- | --- |
| production (psm 3, box, 150 DPI) | 0 | 0 | about 1.0 |
| psm 6, box | 4 | 1 | 1.00 / 2.96 |
| psm 4, box | 2 | 1 | not tabulated |
| psm 11, box | 1 | 3 | psm 11 hit the 4 s limit on 8 pages |
| psm 3+6 union, box | 4 | 0 | 1.94 / 5.70 (two calls) |
| psm 3+6 union, box +15% | 6 | 0 | 2.24 / 6.74 |
| best union (psm 3+6+11, box +15%) | 8 | 0 | 3.39 / 14.86 |

Among the arms tested here, at most 8 of the 106 refused pages are recovered; untested psm/crop/DPI settings are not bounded by this. Timings come from the shared login
node rorqual1 (load average about 7) and are noisy.

## Eye check: every gained page, whole table

A refused page that passes ships all its numbers as checked, not only the numbers that were in
dispute (Kimi's design review). Each of the 8 gained pages was therefore compared cell by cell with
its scan at 200 DPI.

| page | result |
| --- | --- |
| 2000-12-19 bluebook p19 | correct (180 cells) |
| 2001-01-31 greenbook part 2 p46 | correct (144 numbers, two tables) |
| 2002-08-13 greenbook part 2 p79 | correct (25) |
| 2016-12-14 presentation materials p23 | correct (48) |
| 2019-12-11 presentation materials p62 | correct (22) |
| 2001-03-20 greenbook supplement p9 | numbers correct; row labels wrong. The "New home sales" values carry the label "Existing home sales", and the real existing-home-sales row has no label |
| 2001-06-27 greenbook part 1 p40 | wrong. qwen dropped two printed rows (Industrial Countries GDP, Mexico GDP), and every label from Developing Countries to Brazil sits one row off in both sections. Korea's 1998 GDP reads -2.0; the page prints -5.2 |
| 2002-08-13 greenbook part 2 p44 | numbers correct; column-group header wrong. Every column is labelled "12-month percent change", but Jan/Apr are changes to July and June/July are monthly rates |

Every one of the 60 arms and unions that gains any page admits p9, p40 or p44 (`flips.json`). The
narrowest arm, psm 4 on the box, gains p19 and p9. No arm raises correct passes without a new wrong
pass, so nothing ships. Losses (pages that passed and would be refused) were not eye-checked: no arm
is adopted, so they decide nothing.

## What this shows

- The witness is a one-way coverage check: it asks only that every number the reading emits appears in
  the OCR evidence. Dropped rows remove numbers from the reading and shifted labels move none, so neither
  trips it, and p40 (a badly misbound table) passes as soon as tesseract reads every digit. Today it
  is refused only because tesseract misreads some of its numbers. Better OCR on this check turns
  luck into wrong passes.
- So the witness's failure to read numbers is the smaller problem on these pages. A whole-table
  pass/fail on number counts is the wrong unit: a refusal throws away correct tables, and a pass
  certifies misbound ones.
- This supports the owner-approved parallel design (`scratch/routing-baseline/census1261/parallel-design.pdf`, not in git; the owner approved the design in the 2026-11-05 session):
  structure is decided first by mechanical grid and row-label checks (cf. #1267), then each number is
  voted on separately. Under that design p40 and p9 fail on structure, and the 5 correct pages pass
  per cell. p44 (right numbers, wrong column-group header) is NOT caught by a row-label check; it needs
  column-header attribution, which stays unresolved here. That design needs the owner before it is built.
- Kimi's attack points apply to any future witness change. Padding and full-page reads take in
  titles, footnotes and page numbers as evidence. Padded boxes that overlap are added together, not
  maxed. psm 11 splits `12.4` into `1` and `24` and drops minus signs. A union across modes admits
  the misreads of every mode.
