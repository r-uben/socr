# GH-932: `text_in_numeric_column` classifies numbers through the verifier's predicate (2026-10-01)

Branch `fix/932-tnc-numeric-classifier` from origin/main 0eb6121 (ancestry verified). Follow-up to
`2026-10-01_917-text-in-numeric-column.md`.

## Change

`src/socr/tables/ship_gate.py::_cell_kind`: a cell is a `number` if `is_numeric_token(cell)` (already
imported) OR the existing one-letter-marker regex matches. No new regex; the shared predicate is
called as is. Currency-prefixed (`$ € £ ¥`) and `∗`/`✱`-dressed cells now count, so a column of them
is a numeric column and a text footrow under it DEFERs (under-fire fix only: DEFER-only predicate).

## Deviation from the issue text: ★ and ⋆

`is_numeric_token("★0.05")` and `("0.05⋆")` are False: the verifier's `_PRESENTATION_MARKS` has
`∗ ⁎ ✱ † ‡ §` but not U+2605 / U+22C6. I tried adding both to the local marker class in
`_NUMBER_CELL_RE`; the ★ column still did not DEFER, because such a markdown cannot exact-pass (its
tokens are not in the numeric multiset), so there is no SHIP for the gate to override. I reverted that
edit. Pins use `∗` and `✱`. Widening `_PRESENTATION_MARKS` is a shared-predicate change touching every
verifier consumer; left for its own ticket if a corpus rate justifies it.

## Measurement (frozen sources: words + markdown captured once, then main's `ship_gate.py` and the branch's evaluated on identical inputs)

- 92 upright census pages (deduplicated by (doc,page)): old DEFER 47, new DEFER 47, **0 predicate-set changes**.
- 13 lift pages (all `2017__fama__ap.pdf`): old DEFER 4, new DEFER 4, **0 changes**.
- So no new false DEFER and no new catch on these 105 pages; the corpus rate is ~0.4% of numeric cells
  (issue comment), below what 105 pages can show.

## Tests (`tests/test_gh917_text_in_numeric_column.py::TestNumericClassifierIsTheVerifiers`)

Difference pins: `$ € £ ¥` column and `∗`/`✱` column, clean grid vs the same grid plus a text footrow
(quiet vs `{text_in_numeric_column}`); `_cell_kind` pins; must-not-fire controls on a `$` column (empty
tail, dash tail, one merged label-cell note).
Mutation (fix reverted in place, anchor count asserted == 1, restored after): dropping the canonical
call fails 3 pins (currency, star, `_cell_kind`); canonical-only (dropping the marker regex) fails the
existing `test_footnote_markers_and_stars_on_numbers`. The control test does not die under either
mutant by construction (a controls test stays quiet when the column is not numeric).
Full suite, default OLLAMA_HOST: 6060 passed, 2 skipped, 4 xfailed. `uvx ruff@0.16.0 format --check .` clean.

## Round 2: Astra's P1 (dilution) and P2 (mutation method), same day

**P1.** With the classifier widened, a table of disjoint panels (Alpha/Beta fill A-B with plain numbers,
Gamma/Delta fill C-D with `$31 $32`, a tail row says "see note" in A) made all four rows candidates; every
value column then held numbers in exactly half the data rows and `_numeric_columns` returned `[]`, so TNC
abstained. The same hole existed on main for plain-number panels. Two changes in `ship_gate.py`:

1. `_numeric_columns`: numbers must fill more than half of the data rows that have a NON-EMPTY cell in the
   column, and at least `_PLACEHOLDER_MIN_ROWS` (2: one filled cell is not a column; reused, same
   "a repeat is evidence" rule). An empty column is never numeric.
2. `_data_rows`: the per-row coverage rule (values in more than half of the numeric columns) still sees each
   disjoint-panel row at exactly half, so a candidate row also counts when its exact set of number columns is
   shared by another candidate (`_PLACEHOLDER_MIN_ROWS`); a one-off panel label with a year range is not.
   Placeholders: untouched; the existing placeholder tests pass.
The existing `test_numeric_column_needs_more_than_half_the_data_rows` pinned the old semantic (empty cells in
column 4 counted against it); it now fills the other rows with distinct text, which keeps the same "exactly
half is not most" intent.

Pins (`TestDisjointPanels`): Astra's `$` table and the plain-number version both abstain on main (verified by
running the whole file against main's `ship_gate.py`: mutant M6 fails both) and DEFER on the branch; empty
column and single-filled-cell boundary pins.

**Gate comparison, main vs branch, frozen sources (92 upright + 13 lift): 9 new predicate-set changes, all
`+text_in_numeric_column`, no removals.** Each checked against its render:

| page | what the page is | verdict |
|---|---|---|
| 2017__fama__ap p481 (lift) | rotated correlation matrix; the sub-header row "Correlations" sits in a numeric column | real fault; one of the 4 previously uncaught "wrong" pages, now stopped |
| 2025__gomez-cram...earnings_expectations p10 | a real table plus body prose; the grid pulls the notes and body paragraphs in as rows | real fault |
| 2003__woodford p786, p800, p802 | index pages, two-column prose, no table | real fault (grid of non-table); p802 also had data_row_missing |
| 2018__ljungvist_sargent__macro p7 | table of contents, prose | real fault (grid of non-table) |
| 2026__theodoridis p371, p545, p1203 | reference lists, no table | real fault (grid of non-table) |

Totals: lift DEFER 4 -> 5; upright DEFER 47 -> 53. No false DEFER found: a DEFER of a prose page costs one
model read and replaces a table grid of non-table text. Honest limit: 8 of the 9 are non-table or
prose-in-grid pages; the predicate is still not a clean "table with footnote" detector there.

**P2 (mutation method).** Redone in a copy outside the repo (`src` + `tests` + `pyproject.toml`), an added
canary test asserting `socr.__file__` is inside the copy (it passed in every run), uncapped
`orig.count(anchor) == 1` asserted before each edit, file restored after. Mutants vs the 90 tests in the
917/916 files:

| mutant | killed by |
|---|---|
| M1 no canonical call in `_cell_kind` | currency pin, star pin, `_cell_kind` pin, currency-panels pin |
| M2 canonical only (marker regex dropped) | `test_footnote_markers_and_stars_on_numbers` |
| M3 numeric column over ALL data rows | both disjoint-panel pins |
| M4 no minimum filled rows | `test_one_filled_cell_is_not_a_column` |
| M5 no shared-support clause | both disjoint-panel pins |
| M6 whole file = main | currency, star, `_cell_kind`, both disjoint-panel pins (5) |
| M7 `>=` for `>` | `test_numeric_column_needs_more_than_half_the_data_rows` |

The in-place mutation of round 1 is superseded by this table.

Full suite (default OLLAMA_HOST, nohup): 6064 passed, 2 skipped, 4 xfailed. `uvx ruff@0.16.0 format --check .` clean.

## Round 3: the predicate is monotone by construction (Astra rejected da3234e)

**Defect in round 2.** The shared-support clause admitted rows as data that main's rule excludes. Astra's
table: four 4-column data rows plus two note rows (`1968 | 2021 | see appendix`, `1970 | 2022 | sample
restriction`). Main: columns 1-4 numeric, each note covers 2 of 4 (not data), the text in column 3 fires.
Round 2: both notes share the support {1, 2}, become data rows, are skipped; the catch was lost.

**Fix.** `text_in_numeric_column_faults` now runs the block under BOTH rules and reports the union, de-duplicated
by row (main's fault wins on a shared row): `_rule_faults(..., panels=False)` is main's original column and
data-row rule (with the new canonical `_cell_kind`); `_rule_faults(..., panels=True)` is the filled-row column
rule plus shared-support rows. Every fault main's rule finds is therefore still found; the gate is DEFER-only,
so the union can add a DEFER and never lose one. `_data_rows` and `_numeric_columns` take a keyword-only
`panels` flag (no default, so no caller silently gets one rule).

**Pins** (`TestMonotone`): Astra's table fires on the branch exactly as on main (two faults, rows 5 and 6,
text named); mixed plain-number panels next to currency panels with the footrow text in a currency column
(on main the currency column is not numeric, so this one differs from main). The two disjoint-panel pins
remain.

**Gate comparison, main vs branch, 105 frozen pages:** the same 9 additions as round 2 (lift p481; upright
woodford p786/p800/p802, ljungvist p7, gomez-cram p10, theodoridis p371/p545/p1203; all rendered and checked in
round 2, all real faults), **0 predicate removals and 0 lost TNC rows**, both asserted mechanically in the
measurement script (`removals` empty; per page, the set of TNC row ids on main is a subset of the branch's).
116 TNC rows added over those pages in total (most on the reference-list and index pages).

**Mutations** (external copy of src + tests + pyproject; canary asserting `socr.__file__` inside the copy;
uncapped anchor count == 1 asserted; file restored after). A no-mutation baseline run is part of the script
(92 passed): without it I misread an always-failing pin as "killed by every mutant" (the mixed pin first fed
`$`-prefixed source words, which do not pair; the source prints bare numbers, only the markdown carries the sign).

| mutant | killed by |
|---|---|
| M0 baseline | none (92 passed) |
| M1 no canonical in `_cell_kind` | currency, star, `_cell_kind`, currency-panels, mixed pin |
| M2 canonical only | `test_footnote_markers_and_stars_on_numbers` |
| M3 panel columns over all rows | both disjoint-panel pins, mixed pin |
| M4 no minimum filled rows | single-filled-cell pin |
| M5 no shared-support clause | both disjoint-panel pins, mixed pin |
| **M8 drop the main-rule half of the union** | **`test_notes_sharing_a_column_set_are_still_caught` (Astra's pin)** |
| M9 drop the panel-rule half | both disjoint-panel pins, mixed pin |
| M6 whole file = main | 8: classifier, star, `_cell_kind`, both panel pins, empty-column, single-cell, mixed |
| M7 `>=` for `>` | `test_numeric_column_needs_more_than_half_the_data_rows` |

Full suite (default OLLAMA_HOST, nohup, one complete run on the final tree): 6066 passed, 2 skipped, 4 xfailed. An earlier run in this round had 1 failure: the mixed pin as first written (see above), fixed before this run.
