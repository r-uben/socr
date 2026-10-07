# 2026-10-07 -- #988 M2: a navigation bar written as a table is not judged as one

Branch `fix/table-checks-nav-and-shortfall`, second of two commits. The first (row-shortfall term, #988
M1) is logged in `2026-10-07_988-shortfall-missing-content.md`. Issue r-uben/socr#988.

## Why

Coca-Cola's sustainability reports (2018-2021, cluster job 687398) print a website navigation bar
across the top of every page: a menu line, a section sub-menu line, and drawn rules. Gemini writes the
bar as a Markdown table. The gate then judged it as a table and refused complete readings of the real
table below:

- `table_content_empty`: the menu written as a header and delimiter with no body (20 pages). The real
  table on the same answer was complete on 18 of them.
- `grid_shape` / `table_width_mismatch`: menu and sub-menu in one ragged run (2021 p74: 12 + 8 cells).
- `header_unattributed`: `header_cut._header_band` takes the two nearest full-width rules above the
  table. These tables draw only per-column underlines, so the two rules it finds are the menu's
  (2021: y = 54 and 69). The sub-menu words between them are then owed to the table's header.

## Change

- New `socr.tables.furniture`. A word is furniture when its text is printed at the same position on 2 or
  more pages of the document. Positions are rounded to whole points. A drawn rule is furniture when it
  repeats on more than half the pages (and on at least 2).
  - Sub-menus repeat only within their section, hence the 2-page test for words.
  - Removing a rule changes the header cut of every table on the page, hence the majority test for
    rules. A continued table repeats its rules on a few pages; the page template repeats them on most.
- `strip_furniture_runs`: a pipe run whose every letter-and-digit chunk is a furniture word on this
  page is left out of what the gate checks. Chunks absorb "Portfolio/Reducing" written against
  "Portfolio/" + "Reducing" printed, and `[Home]` or `&`. When every run is furniture nothing is
  removed: the answer has no table of its own, and the gate judges what it wrote.
- `NativeTableVerifierJudge` scans the document once, cached by file name, because `get_fitz_page`
  reopens the PDF on each call. It drops furniture rules from `rules`, and `_apply_structural_gate`
  gates the stripped text (`table_output_defect` and `table_header_verdicts`). The shipped text is
  unchanged. A scan that raises leaves the gate as it was before (nothing stripped, every rule kept).
  The value guard still sees the full answer.
- `reconcile.table_content_defect`: a run with NO body row (header and delimiter only) is a defect
  only when no other run on the answer has body content. A placeholder body is still a defect wherever
  it is. This change also reaches the paths without a document (the manifest backstop, native-first),
  which would otherwise demote the answers the gate now accepts.

## Measured

Production gate replayed on the 73 cached answers (`NativeTableVerifierJudge`, inner judge stubbed to
accept, same probe for every tree, `socr.__file__` canary; audit scratch `probes/test_harness.py`):

| complete answers passing (of 56) | |
|---|---|
| origin/main | 1 |
| M1 | 15 |
| M1 + M2 (this branch) | 40 |

- No answer that main accepts is refused (4 answers on 3 pages).
- 6 answers that are not fully correct now pass. On main each was refused only by a check that
  misfired on it, so these are not new defects; they are defects that were caught by accident:
  - wrong: 2018 p60 (invented value; M1), 2019 p62 (invented 2015 column; M2) and 2020 p63
    (percent rows one year group left; M2);
  - mixed: 2019 p51 (M1), 2019 p59 (`42` bound to the wrong row; M2) and 2020 p76 (no data table,
    content correct; M2).
- Synthetic cuts: 40 complete answers pass on this branch, and 38 of them have a table of 4+ body rows.
  Cuts of that table refused on this branch:
  - tail cuts of 10/25/50%: 33/31/33 of 38;
  - middle cuts of 10/25/50%: 20/31/33 of 38.

  There is no fair main baseline, because main refuses the uncut answers too.
- The strip removed 53 runs on 34 of 72 answers. All 53 were printed and read, and every one is a menu
  or a section sub-menu.
- Scan cost: 0.6-1.0 s per report (72-86 pages). Not measured on long or vector-heavy documents.
- Still refused, 16 of the 56 complete answers:
  - `table_truncated` 7: 2018 p41; 2019 p49, p57; 2020 p14, p64, p66, p72;
  - `grid_shape` 6: 2018 p10, p53; 2019 p24; 2021 p66, p69, p72;
  - value guard 2: 2020 p73, 2021 p59;
  - `header_unattributed` 1: 2019 p48.

## Tests

`tests/tables/test_gh988_navigation_furniture.py`:

- The real 2021 p74 answer the cluster gated (`fixtures/gh988_coke_2021_p74/cluster_answer.txt`), with
  the words of p74 and p75 and the rules of all 86 pages. Each half of the fix is needed there: with
  only the run stripped the menu rules give `header_unattributed`, and with only the rules dropped the
  run gives `grid_shape`.
- A generated 3-page PDF: repeated words and majority rules found, page words not. The production
  gate accepts the menu-plus-table answer, and with the scan failing it refuses it with `grid_shape`.
  The table alone is accepted, and with the scan failing it is refused with `header_unattributed`.
- `table_content_defect` on header-only runs, plus a pinned known limit (below).

Mutants (external copy, `socr.__file__` canary, anchor counted exactly once), each killed:

| mutant | killed by |
|---|---|
| header-only run is empty again | header-only tests |
| gate does not strip | gate menu test |
| gate keeps every rule | both gate tests |
| `keep_rules` no-op | p74, rule tests, gate tests |
| strip even when every run is furniture | furniture-only test |
| words need 3 pages | p74, repeated-words, gate menu test |
| word position ignored | repeated-words test |
| no rule majority | rule-share test |
| whole words instead of chunks | p74, strip tests, gate menu test |

## Known limits

- A sub-menu whose current item is set bold shifts the words after it, so their positions do not repeat
  and the run is kept (2021 p59).
- A second data table written header-only beside a populated one is no longer refused as empty (pinned
  by `test_known_limit_…`). Its numbers are missing for the value guard and the row-shortfall term.
- Text written as its own run that the document prints identically at the same place on two pages is
  treated as furniture: for example, a continued table's repeated header.
- A table layout repeated on most pages loses its rules for the header cut, and the header term then
  abstains.
- Only the agentic table gate uses furniture. The value guard, the manifest backstop and native-first
  do not.
