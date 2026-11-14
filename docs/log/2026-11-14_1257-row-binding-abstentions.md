# 2026-11-14 B1: the row-binding check abstains less (plan B1, issue #1257)

Branch `fix/issue-1257-b1`, base `b2d6454e`. Brief: `scratch/army/phaseBC/BRIEF-B1.md`, rulings in
`DECISIONS.md` (target: at most 26 abstaining tables of the 39 on gold-clean pages), conditions from
`advice-astra.md` (B1). Code: `src/socr/tables/row_binding_check.py`.

## What the check did

A shipped row was BOUND only when its whole label was printed before its values on one text line
(#1271, rounds 4 to 8, kept strict on purpose). A label wrapped over two lines, a heading stacked over
two lines, a label spread over two cells, or a row of ranges could not satisfy that. The table then
came out UNVERIFIABLE and the page was marked, although nothing was wrong.

## What changed

Four additions. Each can only move an UNVERIFIABLE row (or heading) to BOUND. No conviction rule, no
`MIN_VALUES_TO_CONVICT`, no strict same-line rule and no page-wide withhold was touched. A row whose
verdict is a conviction is never a candidate. Every check is exact: footnote marks and symbols must be
equal, as on the same-line path.

| kind | record name | binds when |
| --- | --- | --- |
| E1 wrapped label | `wrapped_label_exact` | the shipped label is the exact join (strict gate) of the same-line fragment and valueless lines directly above and/or below. Fragments: consecutive (gap bounded by the region's own median line pitch plus the in-line jitter tolerance `LINE_CENTRE_TOL`, both derived), inside the label column (every word ends before the row's first value), none starting left of the label's first line by more than `DUP_OVERPRINT_TOL_HEIGHTS`, none the line of another shipped row, none a run of another row's label or cells (group labels and categorical rows included), exactly one split, `_cells_cover_line` still true. Group labels (valueless rows) bind the same way on two or more consecutive free lines. |
| E2 stacked heading | `stacked_heading_geometry` | every heading cell is exactly the words stacked over its data column. Columns are the x-extents of the j-th value of every body row that has one value per heading cell; a heading word belongs to the slot its centre lies in (slot borders halfway between columns; the label column ends where the printed labels end). Words of a slot read top to bottom, left to right equal the cell word for word, the stub words equal the label, no word is left over. A group label printed right under the heading is skipped when it is itself a shipped row. Not the flattened word sequence: the same words under the wrong column fail. |
| E3 whole line | `whole_line_exact` | the row's label cell and data cells are the printed line (or its value panel) word for word, none missing, none left over: a label over two cells (`1` / `LR`), an empty label cell, ranges and ellipses (`.0-3.0`, `...`; a run of dots compares as dots). A row with no numbers needs exactly one such line in the region, owned by no other row. |
| completeness | gate on all three | none of the above binds a table that left a row-shaped line of its region without a shipped row (a first number, then only numbers and symbols, numbers in at least `MIN_VALUES_TO_CONVICT` different value columns). Running heads with a date sit in one column and do not count. |

Not changed on purpose: the point-repair gate (`propose_point_repairs`) calls the check with
`extra_evidence=False`, so a repair is still licensed only by the same-line rule.

Record (sidecar `row_binding["pN-tK"]`): new `rows_bound`, `rows_total` and, when used, `evidence`
(rows per kind). `rows` still lists only rows that are not BOUND. No new audit event kind, so
`tests/test_resume_restore_kinds.py` needs no entry: the whole payload restores with the existing
`row_binding` restore, and a BOUND table has no event to restore.

## Measurement

Frozen replay, no OCR: `docs/log/2026-11-14_1257-replay.py`. Same cached page markdown
(`pages/NNNNN.md`), same PDF page, same `locate_tables` boxes for both versions (baseline file read
from git at `b2d6454e`), so every difference is the binding logic. Run `51270eb5`, 539 pages with a
row-binding record. (The brief's "re-run twice for model noise" is not needed: nothing is re-read.)

| denominator | before | after |
| --- | --- | --- |
| all tables, replay (539 pages; 580 tables: 537 UNVERIFIABLE, 43 BOUND; the 84 contradicted are replayed separately) | 537 UNVERIFIABLE | 491 UNVERIFIABLE (46 freed; the measured ceiling is 263) |
| gold-clean tables (33 pages, 39 tables, any seat marked the page clean) | 39 abstain | **26 abstain** (13 freed, target <= 26: met, exactly at the bar) |
| contradicted tables (84, from the withheld markdown kept in the sidecars) | 84 contradicted | 84 contradicted, the same convicted rows in all 84 |
| gold misalignment pages (31 in gold, 19 with a row-binding record: 9 abstained, 1 bound, the rest empty of tables here) | 9 abstained | the same 9 abstained, none bound, no table newly BOUND on any of them |

Newly BOUND tables: 46 (rows by kind: 20 tables by E2 alone, 11 E2 with E1, 8 E1 with E3, 5 E3 alone,
2 E2 with E3). The replay reproduces the sidecars' 539 pages (537 vs the 532 of the brief differ only in
live vs replayed boxes, as the brief noted).

### What the 26 that stay marked are

All exact reasons, none touched: footnote marks that differ by glyph (`workers1` printed, `workers¹`
shipped; round 6 ruling), ligature glyphs missing from the text layer (`infation`), soft hyphens
(`1974­` / `95`), values the layer does not carry, labels absent from the layer (genuine model
defects), triangular tables whose last column has no value, spanning headings over word cells.

## Cost: pages that lose the row-binding mark (read this)

44 replayed pages lose their row-binding mark, and on all 44 it was the page's only mark
(`row_binding_mark_only`), so they go WARNING to SUCCESS. By gold:

* 22 have no gold verdict; 10 are gold-clean on this exact markdown;
* 12 are gold-defective: 9 by page identity only (an auditor marked another reading of the page
  defective, the current reading was never judged defective: `lost_text`, `lost_qualifier`,
  `corrupt_text`) and 3 with a gold row on this exact markdown (two are carbon "worse" rows with no
  defect type, i.e. pages that only gained the mark; one, 2011-09-21 tealbook-partb p51, is `lost_text`
  on this reading). None of these defects is about row binding. The check never saw them; the
  abstention marked the page by accident. After the change those pages are silent.
* No newly BOUND table sits on a page whose gold row, on this exact markdown, carries a binding-type
  defect (`row_or_column_misalignment`, `wrong_number`, `sign_flip`, `decimal_drop`): 0.

The brief's guard "newly BOUND tables on gold-defective pages: 0" is therefore NOT met in the broad
reading (any defect type, 12 pages, one of them confirmed on this exact reading): the abstention was
masking `lost_text` and similar on those pages. It is met for binding-type defects. This is the risk the brief named ("freed labels mask a defect only the mark
catches"). The completeness gate removed the worst case (financial-stability-report p26: exact labels,
two printed rows missing from the shipped table, a v1 misalignment page; without the gate it was
newly BOUND, with it it stays marked). Closing the remaining 12 needs the text-layer words check
(#1294) to catch `lost_text` itself; B1 cannot.

## Tests

* `tests/test_gh1257_row_binding_evidence.py`: 107 check-level tests at rotation 0, 90, 270. Every
  one runs the SAME page and markdown with the evidence on and off (the evidence functions patched
  out): positives change from UNVERIFIABLE to BOUND with the named evidence kind; every adversary must
  give identical verdicts in both arms. Adversaries: values moved one row, label one fragment short or
  long, footnote and case differences, non-consecutive fragment, fragment another row owns (group
  label, categorical row), shared continuation, fragment inside the value columns, continuation left
  of the label, a printed word between the values, two equal splits, swapped stacked columns, words
  moved across the column border, spanning word (binds only in the slot its centre sits over), leftover
  heading word, missing column, header with fewer cells than data columns, wrong stub, non-consecutive
  stack, a table that lost a row, a running head with a date, duplicate rows for one printed line, a
  convicted table (stays convicted, same rows), the repair gate.
* `tests/test_gh1257_row_binding_pipeline.py`: `process()` with and without the evidence, both
  provider states (`primary_engine`, `local_engine`, `enabled_engines` pinned through `_run`,
  `_available_engines_for_agentic` patched, `_resolve_judge_model` patched to `""`). With providers:
  the table is BOUND with `rows_bound == rows_total` and the evidence in the sidecar, and only the
  control arm carries the mark, the audit event and `failure_mode == "table_unverified"`; the table's
  bytes are identical in both arms in the fragment and in the assembled document. A slid wrapped row and
  a contradicted table give the same record, bytes and status in both arms. Without providers the loop
  bails before routing: pinned as a difference of zero. The sidecar record restores through
  `_restore_terminal_page_state` with outcome, counts and evidence intact and no mark or event; a
  resumed run (document index deleted, page-level path) ships the same bytes and record.
* Round 4 of #1271 is kept as written for the same-line rule: `test_wrapped_two_line_label_is_never_convicted`
  and `test_a_header_row_binds_only_on_its_exact_printed_label` now pin the same-line rule with the
  evidence off, and the new behaviour with it on. `test_condition_6...` only gained `**kw` in its fake.
* The 13-round #1271 suite, the #1282 repair suite, the resume suites and the golden byte-identity tests
  run unchanged (811 passed in the row-binding, refinalize, repair, restore selection).

### Mutants (CLAUDE.md guard rule)

`docs/log/2026-11-14_1257-mutants.py`: copies `src`, `tests` and `pyproject.toml` outside the repo,
asserts `socr.__file__` is inside the copy, counts every anchor uncapped (must be 1), removes the
guard, runs the owning tests. The unmutated copy passes; 18 of 18 mutants are killed: E1 consecutive
baseline; E1 no other row claims the fragments; E1 non-exact join; E1 left edge; E1 label column; E1
line of another row; E1 ambiguous split; group label consecutive baseline; E3 length equality;
shared-line elimination; E2 leftover word; E2 consecutive stack; E2 membership by flattened words
(kills swapped columns, moved words and the spanning test); E2 heading cells = data columns;
completeness (both directions: binds a table that lost a row, and over-blocks a running head with a
date); repair gate; record. Two guards were redundant and removed instead of mutated: the E2
"exactly one N" (two stack heights cannot both fit with the leftover test) and the intersection test
between E1 fragments (every fragment is already claimed by the other row's label). Not mutated:
`_single_home` (several options in one stretch of a line) and the `v not in CONVICTING` guard on the
candidates, which no input reaches (a row whose label and cells are the line is not convicted); both
are defensive.

## Open points

1. The 12 gold-defective pages that go silent (above). Decision for the owner: accept (the mark was
   accidental, B1's bar is met), or hold B1 until the words check covers `lost_text` on them.
2. Exactly at the bar (26 of 26). Gains beyond it need footnote glyph tolerance (ruled out by round 6),
   ligature repair of the layer, or spanning-cell semantics.
3. E3 (whole line), the wrapped group label and the completeness gate go beyond the brief's E1/E2.
   Order of measurement on the gold-clean tables (replay, same frozen inputs): E1 and E2 alone 29;
   adding E3 and the group-label wrap 25; adding the completeness gate 26 (it re-marks one gold-clean
   table, financial-stability-report p26, a v1 misalignment page that lost two printed rows, and no
   other). E3 was needed to reach the bar; the owner may drop it and the bar then reads
   about 30.
4. `socr-refinalize-point-drops` rechecks tables with the new evidence too (it calls
   `check_page_row_binding`); old runs refinalized after this change may lose marks the same way.

## Release rule (2026-11-14, after C2 #1332 and C3 #1338 merged)

Coordinator ruling (`scratch/army/phaseBC/DECISIONS.md`, "B1 HOLD"): B1's evidence still moves a
table's record from UNVERIFIABLE to BOUND, but the page's row-binding mark goes only when the page has
independent page-scope evidence on the text that ships. The "Cost" section above is the reason: a table
bound by exact evidence says its rows sit on the layer's lines, nothing about the rest of the page.

Rule (`verdict.row_binding_release_evidence`, `page_row_binding_release`; applied in
`_run_row_binding_check`):

* model-read page: `independent_page_evidence(trace, text)` (C3): words check PASS at page scope,
  numbers PASS or not applicable, no FAIL or UNKNOWN entry, every PASS digest-verified on the text.
* native page (text = the layer, so the words check is circular): a digest-verified `native_coverage`
  (ink vs layer, #1050) PASS at page scope and no FAIL or UNKNOWN entry. The words and numbers checks
  are not consulted. Nothing emits a native_coverage PASS today, so no native page is released. The row
  check itself never runs on a native engine's output (`layer_can_judge_tables`), so this branch is
  reached only through the predicate.
* without evidence: the table keeps its BOUND record (with its `evidence`) and stays in
  `row_binding_unverifiable_tables`; the page keeps its WARNING and `table_unverified`. The mark is
  reworded to what is true: `> [table unverified: table N rows bound to the page's text layer; page text
  not independently confirmed; verify it against the page image before citing]`
  (`marks.row_binding_held_table_mark`; still matches `UNVERIFIED_TABLE_MARK_RE`). No UNVERIFIABLE audit
  event is written: the table is bound, so no new durable event kind and nothing for the resume restore
  test to list. The sidecar's `row_binding_unverifiable_tables` and the record restore as before; the
  mark is re-derived from them (record BOUND and listed = held wording). `socr-refinalize-point-drops`
  never lifts a mark, so a bound record it rewrites that stays listed ships the held wording.

Deviation from the ruling's wording: "its own clause scope" is not a new verdict `Check`. A new check
cascades into the failure-mode chooser, the mark parser and every consumer; the held table is already
surfaced by page status, failure mode, the table mark and the sidecar list. Owner call if a
verdict clause is wanted.

### Replay with the rule

`docs/log/2026-11-14_1257-release-replay.py` (run after the replay above). Same 44 pages B1 alone frees;
for each, the page analysis of its PDF supplies the layer, the cached text (banners and table marks
removed, the check ran before them) the shipped text, and the trace is recomputed (the run predates
C2). Provider state: with providers, as the run had.

| pages | freed by B1 alone | freed with the rule | kept marked |
| --- | --- | --- | --- |
| all | 44 | 34 | 10 |
| gold-clean (this exact markdown) | 10 | 10 | 0 |
| gold-defective (12) | 12 | **9** | 3 |
| no gold verdict (22) | 22 | 15 | 7 |

Gold-defective freed is 9, not 0. The rule keeps 3 of the 12; the other 9 pass the words and numbers
checks against their own layer, so the evidence the ruling names cannot see their defect (`lost_text`,
`lost_content`; two are on the exact markdown: tealbook-partb 2012-10-24 p55 and 2011-09-21 p51). The
words check compares the layer's words with the page; a defect that is not a missing layer word (the
auditors' `lost_text` here) passes it. The rule is implemented as ruled and does not meet its own target;
PR #1326 stays draft.

### Tests and mutants

`tests/test_gh1257_release_rule.py`: pipeline, both provider states, as a difference between two layers
of one page (clean layer, and a layer with a word the model did not transcribe): freed with evidence
(no mark, no list, words PASS in the sidecar trace), kept without (held mark, listed, page not SUCCESS,
no UNVERIFIABLE event, same table bytes), the held mark survives a resumed run; the predicate for both
page kinds (model: words and numbers, scope, digest, FAIL and UNKNOWN; native: coverage only, circular
words refused, scope, digest, FAIL and UNKNOWN). `docs/log/2026-11-14_1257-release-mutants.py`: 6 of 6
killed (release without evidence; native released on words alone; native released with a FAIL or
UNKNOWN; native coverage at any scope; native coverage not digest-verified; held table with the
unsettled wording). No existing test was changed.
