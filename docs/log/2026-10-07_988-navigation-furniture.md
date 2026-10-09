# 2026-10-07 -- #988 M2: a navigation bar written as a table is not judged as one

Branch `fix/table-checks-nav-and-shortfall`, PR #1042. The row-shortfall term (#988 M1) is logged in
`2026-10-07_988-shortfall-missing-content.md`. Issue r-uben/socr#988. The first version of this change
(commit 20157757) was revised after the PR review; see "Review" below.

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
  more pages of the document. Positions are rounded to whole points. Sub-menus repeat only within their
  section, hence 2 pages and not a share of the document. Drawn rules are not compared (see Review).
- Tokens: text is split on whitespace, pipes and `*`. A numeric piece is compared whole; any other piece
  by its letter-and-digit chunks, which absorb "Portfolio/Reducing" written against "Portfolio/" +
  "Reducing" printed, and `[Home]` or `&`.
- `strip_furniture_runs`: a pipe run whose every token is a furniture word on this page is a furniture
  run. When every run on the answer is furniture nothing is removed: the answer has no table of its own,
  and the gate judges what it wrote.
- `header_cut.header_cut_verdict` takes an optional `is_menu_band` test
  (`DocumentFurniture.is_menu_band`). A header band (the words between the two rules above the anchor
  row) is not taken as the header when (a) every word in it is furniture, (b) a word of it with a letter
  or digit is also printed, at the same place, on a page with no data row (no row of 3 numbers, the
  least the header cut takes as table data), and (c) the answer's header row (`grid[0]`) carries none of
  its words; only tokens with a letter or digit count, so a shared `&` does not. With no other band the
  verdict is UNVERIFIABLE, as for any page without two rules: the term does not refuse, and records no
  event of its own (the separate `table_header_verdicts` abstain event is unaffected).
  `structure_check.table_output_defect` passes the test through.
- `NativeTableVerifierJudge` scans the document once, cached by file name, because `get_fitz_page`
  reopens the PDF on each call. `_apply_structural_gate` checks the text without furniture runs
  (`table_output_defect` with the word test, and `table_header_verdicts`). When the page is accepted,
  the furniture runs with no numeric cell are also removed from the text that ships, recorded as
  `table_furniture_removed` (`TABLE_FURNITURE_REMOVED_KIND`, `docs/OUTPUT.md`) with the removed runs in
  `data["runs"]`. A run with a cell that is a number stays in the text. The manifest backstop, which has
  no document, then sees the text the gate judged. A scan that raises leaves the gate as before (nothing
  removed, every header band owed). The value guard still sees the full answer.
- `table_furniture_removed` is replayed on resume (`orchestrator._RESUME_REPLAYED`): a terminal resumed
  page skips the gate, and the removed runs live only on the event.
- `reconcile.table_content_defect` is unchanged from main: a header-only run is a defect wherever it is.

## Review (PR #1042) and what changed

The first version dropped rules repeated on more than half the pages, and made a header-only run a
defect only when no other run on the answer had a body. The review found both too wide:

1. The header-only relaxation reached every caller of `table_content_defect`, including three with no
   document (manifest backstop `manifest.py:3990`, native-first `native_first.py:148`, born-digital
   `born_digital.py:3712`): a data table whose body the model dropped passed whenever another table on
   the page had a body. Confirmed on the first version. Now the content term is main's, and the gate
   removes the menu from the shipped text instead, so the backstop has nothing to excuse.
2. Dropping majority rules switched the header cut off on documents that print the same table layout on
   most pages (an appendix of booktabs tables). Confirmed on the first version: a generated appendix whose
   answer dropped a header word was accepted. Now no rule is dropped; only a band of furniture words is
   excused, which the same appendix does not have.
3. Chunks split `0.32` into `0` and `32`. Numbers are now compared whole.
4. Scan cost: measured below on 308-700 page PDFs.

Removing the menu from the shipped text interacts with `rejudge_candidate` (#1013), which treats a
rewrite by the judge chain as an error. That path re-judges a timed-out candidate on resume; a page whose
menu is removed there does not ship and is recorded `rejudge_error`, as with `table_header_repair`.

## Re-review (PR #1042, on 73fdafc3) and what changed

1. A header band repeated word for word on every page (an appendix under `(1) (2) (3)`, a continued
   table) was still excused, so a dropped header word passed. Now a furniture band is excused only when
   the answer's header row carries none of its words, as the reviewer proposed, with one change: only
   `grid[0]` is compared, not the second tier `_emitted_header_tokens` admits. On 2021 p74 that tier is a
   section label in the body whose "...emissions" shares a word with the sub-menu ("Greenhouse Gas
   Emissions & Waste"), and the page was refused again. The former known-limit test now asserts the
   refusal; what remains open is an answer that writes none of a repeated header (pinned).
2. Removing shipped text: the reviewer asked to remove only runs with no numeric token. Measured: 10 of
   the 38 removed runs print a year inside a menu item ("2020 Sustainability Goals"), and keeping them
   left a header-only table that the backstop refused on 5 complete answers (2018 p56, p64; 2019 p68;
   2020 p68, p70). A run is now kept when it has a cell that IS a number (`2019`, `0.32`, `**2030**`),
   which a data table has and none of the 38 menu runs has. A table with no numeric cell can still be
   removed when every token in it is printed at the same place on another page.
3. (cubic) `table_furniture_removed` was not replayed on resume, so a resumed run lost the record while
   the edited text stayed. Added to `_RESUME_REPLAYED`, with a test on `resume_restore_kinds()`.
4. (cubic) The scan runs inside the table gate, under the page judge's deadline (`_TimeoutJudge`, the
   longest of `DEFAULT_PROVIDER_TIMEOUTS`, 300 s). The longest scan measured is 3.7 s on 700 pages. Not
   changed.

## Review 6 (PR #1042, on 162657a2) and what changed

The known limit was wider than its test: the excuse fired whenever `grid[0]` shared no word with a
repeated header band, which includes the model dropping the header row and promoting the first body row
into it. On a continued table or an appendix that was HARD on main and became UNVERIFIABLE; through the
gate, with the inner judge accepting, it shipped SUCCESS (probe on the appendix fixture). No other check
refused it.

The second signal is the one the reviewer proposed: a menu is also printed on pages with no table, a
table's header only above its table. "No table" is a page with no data row, the header cut's own
definition (a row of `_MIN_DATA_NUMERIC_CELLS` = 3 numbers); on such a page the header cut never finds a
table. Not every band word has to qualify: a menu sets its current item apart, which moves that item's
words on its own pages. On 2021 p74 the whole 23-word band repeats only on p75, which carries a table
too; word by word, 20 of the 23 (17 of the 19 with a letter or digit) are also printed at the same
place on p65, the Data Appendix overview (prose, no data row). The 3 others are the current item's
"Greenhouse", "&" and "Waste".

Before writing it, every band the previous code excused on the 73 answers was checked (96 excusals, on
2019 p11, p55-p68 and 2021 p68-p78): each had 17-48 words with a letter or digit also printed, at the
same place, on a page with no data row. None had zero.

The `grid[0]` test stays as a second, independent condition.

## Measured

Production gate replayed on the 73 cached answers (`NativeTableVerifierJudge`, inner judge stubbed to
accept, same probe for every tree, `socr.__file__` canary; audit scratch `probes/test_harness.py`), then
the manifest backstop (`_apply_table_emission_guard`) on the text the gate let through:

| complete answers passing (of 56) | |
|---|---|
| origin/main | 1 |
| M1 | 15 |
| M1 + M2, first version (20157757) | 40 |
| M1 + M2, after the review (73fdafc3) | 38 |
| M1 + M2, after the re-review (162657a2) | 38 |
| M1 + M2, after review 6 (this commit) | 38 |

- The re-review commit changes no verdict against 73fdafc3, on the 73 answers or on the 300 cuts below.
  The review-6 commit changes none against 162657a2: gate verdict, backstop verdict and removed runs are
  identical on the 73 answers, and the 300 cut verdicts are identical.
- Every answer this version passes also passes the backstop on the shipped text.
- No answer that main accepts is refused (4 answers on 3 pages).
- The two answers the first version passed and this one refuses (`table_content_empty`) each carry an
  empty table that main also refuses, which only the removed relaxation excused:
  - 2019 p8: the menu written with `[Home]`, a word the page does not print (an icon), so the run is
    not furniture;
  - 2020 p9: the page number written as a one-cell table with no body.
- 6 answers that are not fully correct pass, the same 6 as the first version. On main each was refused
  only by a check that misfired on it, so these are not new defects; they are defects that were caught by
  accident:
  - wrong: 2018 p60 (invented value; M1), 2019 p62 (invented 2015 column; M2) and 2020 p63
    (percent rows one year group left; M2);
  - mixed: 2019 p51 (M1), 2019 p59 (`42` bound to the wrong row; M2) and 2020 p76 (no data table,
    content correct; M2).
- Shipped text: 38 runs removed on the 24 passing answers that carry one. All 38 were printed and read;
  every one is a menu or a section sub-menu.
- Scan cost (words only, once per document): 1.1-3.7 s on five PDFs of 308-700 pages (2.6-7.7 ms per
  page; two annual reports, a CDP climate response, two Joint Committee on Taxation publications), timed
  while the test suite ran on the same machine. 0.6-1.0 s on the Coca-Cola reports. With the data-row
  check of review 6: 1.3-4.5 s on the same five PDFs (3.0-8.6 ms per page), load average about 13.
- Synthetic cuts of the largest table, on the 36 passing complete answers that have one of 4+ body rows,
  refused (gate or backstop):
  - tail cuts of 10/25/50%: 31/29/31 of 36;
  - middle cuts of 10/25/50%: 20/31/31 of 36.

  On those 36 answers every cut gets the same verdict as on the first version (300 cuts compared; the
  only 4 that differ belong to the two answers above). There is no fair main baseline, because main
  refuses the uncut answers too.
- Still refused, 18 of the 56 complete answers:
  - `table_truncated` 7: 2018 p41; 2019 p49, p57; 2020 p14, p64, p66, p72;
  - `grid_shape` 6: 2018 p10, p53; 2019 p24; 2021 p66, p69, p72;
  - value guard 2: 2020 p73, 2021 p59;
  - `table_content_empty` 2: 2019 p8, 2020 p9;
  - `header_unattributed` 1: 2019 p48.

## Tests

`tests/tables/test_gh988_navigation_furniture.py`:

- The real 2021 p74 answer the cluster gated (`fixtures/gh988_coke_2021_p74/cluster_answer.txt`), with
  the words of p74, p75 and p65 (the Data Appendix overview, no table) and the rules of p74. Each half of
  the fix is needed there: with only the run removed the menu band gives `header_unattributed`, and with
  only the band excused the run gives `grid_shape`. Without p65 the band is owed again.
- A generated 3-page report with a menu: the gate accepts the menu-plus-table answer, ships it without
  the menu and records the event; the backstop passes the shipped text and demotes the text as written;
  with the scan failing the gate refuses (`grid_shape`, and `header_unattributed` for the table alone).
- A generated appendix with the same table layout on every page: an answer that drops a header word is
  refused, both when one header word differs per page and when the whole band repeats word for word. An
  answer that writes none of a repeated header (a blank header row, or the first body row promoted into
  it) is refused by the header cut and, for the promoted row, by the gate (review 6). With a notes page
  that prints the header with no table under it, the band passes for a menu: the dropped header word is
  still owed, a blank or promoted header is the pinned known limit (below), and a band with one word of
  the page's own is owed even to a blank header.
- `is_menu_band` on hand-made pages: a band only on table pages is not a menu, one also on a page with
  no data row is, and an `&` there does not count.
- A menu band sharing only `&` with the table header is still excused.
- A furniture run with a numeric cell stays in the shipped text, through the production gate; a number
  inside a text cell does not keep it.
- `table_furniture_removed` is in `UnifiedPipeline.resume_restore_kinds()`.
- Numbers compared whole; header-only runs still empty for the content term.

Run against the first version, the two review tests fail: the appendix answer that drops a header word
is accepted, and a header-only data table beside a populated one is not a content defect.

Mutants (external copy, `socr.__file__` canary, anchor counted exactly once), each killed:

| mutant | killed by |
|---|---|
| furniture band never excused | p74, gate menu, backstop, menu-band, ampersand, known-limit, numeric-run gate tests |
| band excused when ANY word is furniture | page-specific-word blank-header test |
| answer's header row not consulted | repeated-word-for-word header test |
| second header tier compared too | p74 test |
| punctuation counts as written | ampersand test |
| `table_output_defect` drops the word test | p74, gate menu, backstop, menu-band, ampersand, known-limit, numeric-run gate tests |
| gate keeps the menu in the shipped text | gate menu, backstop tests |
| gate does not strip before checking | gate menu, backstop, numeric-run gate tests |
| gate removes runs with a numeric cell | numeric-run gate test |
| a number inside a text cell keeps a run | numeric-cell test |
| numbers compared in chunks | numbers-whole test |
| header-only run excused again | still-empty, backstop tests and 7 GH-190 pins |
| strip even when every run is furniture | furniture-only test |
| words need 3 pages | p74, repeated-words, numbers, gate menu, backstop, menu-band, ampersand, numeric-run gate tests |
| word position ignored | repeated-words test |
| no `table_furniture_removed` event | gate menu test |
| event not replayed on resume | resume test |
| gate passes no word test | gate menu, backstop, menu-band, ampersand, numeric-run gate tests |
| trailing newline lost | menu-run and gate menu tests |

An unmutated copy passes the same 18 suite files (479 tests). The page-specific-word test was added after
the first run left the "ANY word" mutant alive; that mutant and the unmutated copy were then run again.

Review 6 mutants, same harness (`probes/lead2/mut_review3.sh`), each killed:

| mutant | killed by |
|---|---|
| no page-without-a-table condition | p74, `is_menu_band`, none-of-a-repeated-header tests |
| an `&` counts as printed without a table | `is_menu_band` test |
| every page counts as a page with no table | p74, `is_menu_band`, none-of-a-repeated-header tests |
| no page counts as a page with no table | p74, `is_menu_band`, gate menu, backstop, menu-band, ampersand, notes-page, known-limit, numeric-run gate tests |
| no page-without-a-table positions kept | same 9 tests |
| a data row needs 4 numbers | `is_menu_band`, none-of-a-repeated-header tests |
| band a menu when ANY word is furniture | page-specific-word blank-header test |
| answer's header row not consulted | notes-page partial-header test |
| header row the only condition | p74, menu-band, none-of-a-repeated-header, page-specific-word tests |
| gate passes no band test | gate menu, backstop, menu-band, ampersand, numeric-run gate tests |

An unmutated copy passes the same 18 suite files (481 tests).

## Known limits

- A sub-menu whose current item is set bold shifts the words after it, so their positions do not repeat
  and the run is kept (2021 p59). A menu item printed as an icon and written as a word (`[Home]`) also
  keeps the run.
- A header band whose every word is printed at the same place on another page, one word of which is
  also printed on a page with no data row (a notes page under the header), in an answer whose header row
  carries none of those words (a blank header, or the first body row promoted into it), is taken for a
  menu, and the header cut abstains on it without an event of its own (pinned by `test_known_limit_…`).
- "No data row" means no row of 3 numbers, so a page whose only table has two value columns counts as a
  page with no table (2021 p77). A header band printed both over such a table and over a wider one passes
  the second condition.
- A menu printed only on pages that carry a table (a short extract of a report) is owed, as on main.
- Text written as its own run that the document prints identically at the same place on two pages is
  treated as furniture and left out of the gate's checks. On an accepted page it is also removed from the
  shipped text unless it has a numeric cell: for example, a continued table's repeated text-only header
  written as a separate run, or a repeated data table whose cells all carry a unit (`12 t`) or a word
  (`Yes`), since a cell must BE a number to keep the run (review 6). The removal is recorded in the event.
- Only the agentic table gate uses furniture. The value guard, the manifest backstop and native-first
  do not.
