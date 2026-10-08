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
- `header_cut.header_cut_verdict` takes an optional `is_furniture` word test. A header band (the words
  between the two rules above the anchor row) is not taken as the header when every word in it is
  furniture AND the answer's header row (`grid[0]`) carries none of them; only tokens with a letter or
  digit count, so a shared `&` does not. With no other band the verdict is UNVERIFIABLE, as for any page
  without two rules: the term does not refuse, and records no event of its own (the separate
  `table_header_verdicts` abstain event is unaffected). `structure_check.table_output_defect` passes the
  test through.
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
| M1 + M2, after the re-review (this commit) | 38 |

- The re-review commit changes no verdict against 73fdafc3, on the 73 answers or on the 300 cuts below.
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
  while the test suite ran on the same machine. 0.6-1.0 s on the Coca-Cola reports.
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
  the words of p74 and p75 and the rules of p74. Each half of the fix is needed there: with only the run
  removed the menu band gives `header_unattributed`, and with only the band excused the run gives
  `grid_shape`.
- A generated 3-page report with a menu: the gate accepts the menu-plus-table answer, ships it without
  the menu and records the event; the backstop passes the shipped text and demotes the text as written;
  with the scan failing the gate refuses (`grid_shape`, and `header_unattributed` for the table alone).
- A generated appendix with the same table layout on every page: an answer that drops a header word is
  refused, both when one header word differs per page and when the whole band repeats word for word. A
  pinned known limit (below) for an answer that writes none of a repeated header. A band with one word
  of the page's own is owed even to a blank header.
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

## Known limits

- A sub-menu whose current item is set bold shifts the words after it, so their positions do not repeat
  and the run is kept (2021 p59). A menu item printed as an icon and written as a word (`[Home]`) also
  keeps the run.
- A header band whose every word is printed at the same place on another page, in an answer whose header
  row carries none of those words (a blank header), is taken for furniture, and the header cut abstains on
  it without an event of its own (pinned by `test_known_limit_…`).
- Text written as its own run that the document prints identically at the same place on two pages is
  treated as furniture and left out of the gate's checks. On an accepted page it is also removed from the
  shipped text unless it has a numeric cell: for example, a continued table's repeated text-only header
  written as a separate run. The removal is recorded in the event.
- Only the agentic table gate uses furniture. The value guard, the manifest backstop and native-first
  do not.
