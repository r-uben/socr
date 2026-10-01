# GH-917: `text_in_numeric_column` in the native-first ship gate (2026-10-01)

Branch `fix/917-text-in-numeric-column` from origin/main c6a9e95 (includes #926 and the #929
cleanup; ancestry verified with `git merge-base --is-ancestor c6a9e95 origin/main`). The rotated
quarantine (#918) is NOT lifted here.

## Why

The quarantine-lift audit (issue #917, last comment) ran all 449 rotated pages with the quarantine
bypassed: 13 SHIP, all `2017__fama__ap.pdf`; a Fable vision audit found every number correct and
8 of 13 structurally wrong (caption, footnote, equation fragments or "(Continued)" emitted as grid
rows inside numeric columns, plus extra empty columns). Data: `~/.local/state/socr-housekeeping/lift/ship/`.

## What changed

`src/socr/tables/ship_gate.py`: one new DEFER-only predicate, `text_in_numeric_column`, recorded
like the others (it joins `native_ship_gate`'s fault tuple, so the plan reason is
`ship_gate:text_in_numeric_column` and the existing `native_ship_gate_deferred` event carries it;
no new emit site, no new audit kind). Output-side, per markdown block, everything derived from the
block's own rows:

- **Cell kinds** (`_cell_kind`): `number` (a value with its dressing: sign glyph with or without a
  space, parentheses/brackets, `*`, dagger, `%`, thousands commas, at most `_MARKER_MAX_LETTERS` = 1
  trailing letter, so `0.23*`, `0.23a`, `(0.12)`, `- 0.28`), `text` (carries a letter), `other` (no
  letter, no value: `-`, an em dash, a range fragment `1927-`), `empty`.
- **Placeholders**: a text the SAME column holds in two or more rows (`_PLACEHOLDER_MIN_ROWS`) is that
  column's own placeholder (`n.a.` where the table prints it). No vocabulary: prose does not repeat
  verbatim down one column.
- **Data rows**: paired uniquely to a source row, at least `_MIN_CORE_LANES` number cells, and values
  (numbers, dashes, placeholders) in more than half of the columns the paired rows establish as numeric.
  Fewer than two data rows: abstain.
- **Numeric columns**: more than half of the data rows hold a number there.
- **Exempt**: every row above the first data row (the header band, whatever it holds); a row before the
  last data row whose first non-empty cell is left of the first numeric column (a panel label occupying
  the label columns); number-with-marker cells, dashes, placeholders, standard-error parentheses.
- **Fault**: any other row below the first data row with a text cell in a numeric column.

No caption word list, no vocabulary. Named constants only: `_MARKER_MAX_LETTERS`,
`_PLACEHOLDER_MIN_ROWS`; "most" is the strict majority (`2 * count > total`).

Design iterations kept honest (first run, before eyeballing the new fires): the first version counted
text cells against numbers for the whole row and derived placeholders from data rows. Eyeballing the
upright new fires showed two false fires (eskildsen p15, rows of `n.a.` and an accidental data row from
a panel label with a year range, kim_muhn p52) and a reference-list page; the coverage rule and the
column-repeat placeholder rule came from that. The final version dropped the `N > text` row rule (the
coverage rule subsumes it, and no test could pin it).

## Measurement (counts, basenames and pages only; `scripts/measure_tnc.py`, `socr.__file__` asserted inside the worktree)

### The 13 lift pages (all `2017__fama__ap.pdf`), stopped vs Fable verdicts

| verdict | pages | stopped by this predicate |
|---|---|---|
| wrong | 8 | **4** (p435, p570, p592, p780) |
| cosmetic | 4 | 0 (p368, p782, p784, p792 still ship) |
| correct | 1 | 0 (p562 ships) |

**Target not met: 4 of the 8 wrong pages still ship** (p481, p553, p589, p591). All four have their
caption, equation fragments or note paragraph ABOVE the first data row (between the title and the
column headings). The exemption for the header band is in the contract, and no output-side feature
separates a column heading from caption or equation text there: both put one cell per column in the
numeric columns; word, cell and lane counts, one-word-per-lane (the H-V3 test), operators and hyphenation
were all tried on these pages and none separate them (equation row `+ T | = + bY(t) | + | + T)` on p591
and p589 is lane-aligned like a header; the legitimate multi-word header `Y(t) = D(t)/P(t - 1)` is not).
p553 was stopped by the first version only through an accidental data row (`tau, t | 1 | 1 | 1 | 1`,
the equation's subscripts, which pair with the source); the coverage rule removed it. I kept the
correct rule and did not tune for that page.

### 35 rotated q917 pages (fable.jsonl: 14 wrong, 20 cosmetic, 1 correct)

| | pages | stopped (any predicate) | stopped by this predicate | stopped ONLY by it (new vs main) |
|---|---|---|---|---|
| wrong | 14 | 14 | 0 additional | 0 (the 14/14 is #926's) |
| cosmetic | 20 | 12 | 4 | 4 (ids 13, 19, 25, 27 = Fama p435, p570, p592, p780) |
| correct | 1 | 0 | 0 | 0 |

The four new rotated fires are the same Fama pages as lift ids 001, 005, 008, 009; the audit read
the q917 versions as cosmetic and the lift re-run as wrong (footnote or "(Continued)" rows inside
numeric columns). Per-predicate on the 35: `text_in_numeric_column` 19 pages (15 co-fire with
`label_row_missing`, `sign_detached`, `header_band_missing`).

### 92 upright census pages (census.jsonl deduped by (doc, page))

| predicate | pages |
|---|---|
| data_row_missing | 6 |
| label_row_missing | 20 |
| header_band_missing | 17 |
| foreign_direction | 2 |
| **text_in_numeric_column** | **27** |
| any | 47 (was 40 on main: +7) |

New fires vs main (`text_in_numeric_column` is the ONLY predicate firing): **7 pages**, renders and
markdown in `~/.local/state/socr-housekeeping/gh917/tnc_newfires/` (`up_*.png`, `up_*.md`):

| page | reading |
|---|---|
| 2020__bybee_kelly_manela p36 | real: table Note paragraph absorbed as rows in numeric columns |
| 2023__hansen_kazinnik p28 | real: Note paragraph absorbed as rows |
| 2024__defiore_maurin_mijakovic_sandri p26 | real: grid shattered (values shifted into the header row, significance tails as rows) |
| 2025__gomez-cram_jensen_kung p8 | real: Notes paragraph emitted as rows |
| 2026__meyer_wesseler p86 | real: Notes paragraph emitted as rows |
| 2026__theodoridis__machine_learning p548 | real: bibliography page emitted as a table |
| 2025__costello_levy_nikolaev p55 | false-ish: a mid-table panel row `Panel B. Data for continued (sample firms' prior to 2007)` spanning into numeric columns of the second block; costs one model read |

6 real, 1 false-ish. My reading from the markdown for all seven and from the render for defiore p26
(I did not view each render). A page with the same content in v1 of the rule that fired falsely and was
fixed (eskildsen p15, kim_muhn p52, theodoridis p371/p545/p1203) does not fire in the final measurement.

### `empty_extra_column`: NOT implemented (ambiguous)

Prototype (`scripts/measure_emptycol.py`): a column empty in EVERY data row. It would stop three of the
four still-shipping wrong lift pages (p481, p589, p591), but it is ambiguous, so it is skipped:

- Fires on 14 of 20 rotated cosmetic pages and on 30 of 92 upright pages. Fable calls the rotated
  cases cosmetic (numbers intact, consistent extra column), so it would defer pages for a cosmetic
  reason on a corpus where the cost of a false DEFER is a model read per page.
- Why a column is empty in every data row: a header word, a split fragment (`)` of `t(a0 tau )`), or a
  spanning heading centred over a gap gives the rowizer a cluster in a column of its own. The grid
  carries no x-extent per column, so "this column has no source lane" cannot be tested on the output;
  the lane set (`_table_geometry`) is numeric-only, so a centred spanning header (a legitimate
  source column) has no numeric lane either and cannot be told from a stray fragment.
- 6 of its upright-only fires were whole-page-as-table output (herskovic p29, levy p105,
  liu_cao_flake p49/p50/p62, harren_kilic_zhang p66), which is a real defect but not a defect it
  characterises.

## Tests (`tests/test_gh917_text_in_numeric_column.py`, 22 tests, hermetic: synthetic word tuples, no provider)

Difference pins (`_plan` with the gate on, the gate patched off to prove the exact-pass it overrides
exists, and the clean grid): footnote row in numeric columns; the same text as one label cell ships;
`(Continued)` in the last column; fault detail names row, column and text; sub-header between data
rows; signed numbers `- 0.253` establish the column. Must-not-fire controls: header band above the first
data row; panel label in the label column between data rows (the same cells after the last data row
fire); panel label spanning numeric columns; marker and star numbers (`0.23*`, `**`, dagger, `0.23a`,
`(0.23)`, `[0.23]`, `12.5%`; `0.23abc` fires); dash placeholders (four glyphs); range fragments;
standard-error parentheses; a column's own repeated `n.a.` (a single occurrence or one-per-column
fires: documented limit); a different text is not covered by a placeholder; placeholders after numbers
keep a row a data row; a panel label with a year range is not a data row. Derivation: more-than-half
(3 of 6 columns does not, 4 does); fewer than two data rows abstains; a row is data only if the source
pairs it; a one-number row is not data; two label cells and two values is a data row.

## Mutation (copy of src + tests + pyproject in a temp dir, `socr.__file__` canary inside the copy, uncapped `anchor.count == 1`; `scripts/mutate_tnc.py`)

17 of 17 killed against `tests/test_gh917_text_in_numeric_column.py`, canary passed each time. The first
run had two survivors (sign glyphs: the test only signed some rows; "row with as much text as numbers is
data": later removed with the rule it guarded). Both fixed or removed before the final run.

| mutant | failing tests |
|---|---|
| predicate call removed from `native_ship_gate` | 13 |
| header band not exempt (`range(0, ...)`) | 2 |
| label-column exemption removed | 3 |
| label exemption also after the last data row | 2 |
| label exemption in any column | 1 |
| marker letters unbounded | 1 |
| marker letters not allowed | 1 |
| sign glyphs not part of a number | 1 |
| dash cell is text | 2 |
| placeholders never repeat | 2 |
| one occurrence is a placeholder | 13 |
| majority is half (`>=`) | 1 |
| one data row is enough | 2 |
| pairing not required | 1 |
| one-number row is a candidate | 1 |
| coverage not required | 1 |
| dashes and placeholders do not cover | 1 |

## Results

Focused: this file 22 passed. Full suite (default OLLAMA_HOST, nohup, polled, one complete run on the
final tree): **6055 passed, 2 skipped, 4 xfailed, 0 failed** (2091 s; the earlier gh916 + gh917 files
are inside it and pass unchanged: the new predicate does not fire on their fixtures).
`uvx ruff@0.16.0 format --check .` clean.

## Follow-up / decision for the orchestrator

- **The lift criterion is not met by this predicate alone**: 4 of 8 wrong pages still ship (header-zone
  caption/equation text). Options: (A) add `empty_extra_column` (stops 3 more of the 4 wrong; ambiguous;
  defers ~14/20 rotated cosmetic pages); (B) a header-vs-caption discriminator that needs a source-side
  signal I could not find on these pages; (C) keep the quarantine for rotated pages whose first data row
  sits below non-heading text (not measurable on the output grid).
- The quarantine (#918) stays. A re-audit of the 13 is still the lift protocol.
