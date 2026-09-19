# GH-700 — two-column pages withhold left-column prose behind the right column

## What changed

`socr/tables/row_corroboration.py`:

- Added `_median_adjacent_word_gap` and `_split_band_by_column_gap`, and a new
  `_prose_bands_with_columns(words)` that takes `cluster_band_words`'s own
  y-clustered bands and splits any band at a horizontal gap wider than
  `ALIGNED_RUN_GAP_MAX_WORD_SPACES` (reused from `socr.core.born_digital`,
  GH-592) times the page's own median adjacent-word gap.
- `partition_prose_bands` now builds its bands via `_prose_bands_with_columns`
  instead of calling `cluster_band_words` directly. `cluster_band_words` and
  `baseline_bands` themselves are UNCHANGED.

`tests/tables/test_row_corroboration.py`: three new tests —
`test_single_column_page_bands_are_byte_identical_to_cluster_band_words`
(written first, per the ticket), `test_two_column_prose_and_table_row_split_into_separate_bands`
(the ticket's own repro), `test_paragraph_between_two_tables_not_swallowed`
(the rejected-fix trap, re-derived as a construction-level regression guard).

`tests/test_gh649_scanned_prose_recovery.py`: `TestTwoColumnPagesFailSafe`'s
docstring updated from "documented limitation" to "resolved"; added
`test_left_column_prose_now_ships`, an end-to-end pin on the same fixture the
class already used to document the loss.

Four new tests total, not three: the reviewer's own `--collect-only` node-id
diff between `f8949f1` and this tree caught that I undercounted
`test_left_column_prose_now_ships` in an earlier status report. The four
node ids present only in this tree:

- `tests/tables/test_row_corroboration.py::test_two_column_prose_and_table_row_split_into_separate_bands`
- `tests/tables/test_row_corroboration.py::test_single_column_page_bands_are_byte_identical_to_cluster_band_words`
- `tests/tables/test_row_corroboration.py::test_paragraph_between_two_tables_not_swallowed`
- `tests/test_gh649_scanned_prose_recovery.py::TestTwoColumnPagesFailSafe::test_left_column_prose_now_ships`

## Why the split is scoped to `partition_prose_bands`, not `cluster_band_words`

The ticket's open question: does the column split belong in the shared
`cluster_band_words` (so `baseline_bands` / `corroborate_rows` get it too), or
only on the `partition_prose_bands` path.

Measured before choosing, per the reviewer's constraint. I temporarily patched
`cluster_band_words` itself with the identical column-gap split and ran it
against:

1. The existing `tests/tables/test_row_corroboration.py` suite (15 tests,
   synthetic fixtures spacing "columns" 60pt apart with no smaller word-gap
   sample in the same population) — all 15 still passed. Inconclusive: those
   fixtures never populate a *smaller* gap alongside the wide one, so the
   median-gap yardstick just measures the wide gap itself and the threshold
   never trips.
2. A realistic econ-table row built by hand: a multi-word label with tight
   inline spacing (3pt gaps, e.g. "Total loans and leases") followed by a wide
   gutter (90pt pitch) to its first numeric column, corroborated against
   `corroborate_rows`. Unpatched: `bound=1, total=1, clears=True`. Patched
   (shared clusterer): `bound=0, total=1, clears=False` — the row's numeric
   tokens are no longer a contiguous run on one native band once the gutter
   before the first value column exceeds the same threshold that would catch
   a genuine page-column gutter.

There is no way to tell "this page has two article columns" from "this table
has a wide inter-column gutter" by geometry alone. `baseline_bands`'s row
matching needs a candidate row's full ordered token run intact on ONE band —
a false split there silently drops a corroborating row's `bound` count, which
is exactly the kind of defect this module exists to prevent. `cluster_band_words`
and `baseline_bands` are left untouched; the split lives only in
`_prose_bands_with_columns`, used solely by `partition_prose_bands`. The cost
of a false split on THAT path is cheap (an extra withheld-run marker around a
zero-digit label fragment, never a lost row or a misbound value), so scoping
the split there is the more conservative choice under "no way to tell them
apart from geometry alone."

## Single-column non-regression

`test_single_column_page_bands_are_byte_identical_to_cluster_band_words`
asserts `partition_prose_bands`'s band construction (words per band, band
count) is IDENTICAL to `cluster_band_words`'s own bands on a page whose
inline word spacing never crosses the derived threshold — written before the
column-split code, per the ticket.

## Two-table trap re-checked

`partition_prose_bands`'s own docstring records the REJECTED fix: withholding
every zero-token band inside a withheld band's y-span, which swallowed a
paragraph printed between two tables. GH-700's split never inspects y-span —
it only ever splits a band at a horizontal gap WITHIN that one band's own
line — so it cannot reintroduce that failure by construction. Added
`test_paragraph_between_two_tables_not_swallowed` as a concrete regression
pin anyway: two single-column numeric table blocks with a two-line paragraph
between them; the paragraph ships, both tables' rows are withheld.

## Consequence disclosed, not fixed here

On the ticket's own two-column fixture, `partition_prose_bands` now emits an
INTERLEAVED band sequence (left-line1, right-line1, left-line2, right-line2,
…) rather than true whole-page column-major reading order (all of the left
column, then all of the right column). `manifest.native_prose_floor_text`
joins consecutive PROSE bands into one paragraph and stamps one marker per
contiguous WITHHELD run — with every line alternating prose/withheld, the
two-column fixture now emits one marker per withheld row instead of one
marker for the whole table. No content is lost (this is what
`test_left_column_prose_now_ships` and `test_no_printed_value_reaches_the_page`
both check), but the shipped markdown is more marker-verbose on a two-column
page than a true column-aware reflow would produce. Whole-page column-major
reordering is out of this ticket's scope (band CONSTRUCTION for the
withholding decision, not page reading-order reconstruction) and is not
attempted here.

## Verification

- `~/venvs/socr/bin/pytest tests/tables/test_row_corroboration.py
  tests/test_gh649_scanned_prose_recovery.py
  tests/test_gh652_witness_extent_abstention.py -q -o
  pythonpath=/tmp/wt-700/src` — 83 passed.
- Wider sweep of every test file importing `row_corroboration` /
  `cluster_band_words` / `baseline_bands` / `partition_prose_bands` (15
  files) — 275 passed.
- Full suite: `~/venvs/socr/bin/pytest tests -q -o
  pythonpath=/tmp/wt-700/src` — 5707 passed, 4 xfailed, 0 failed.
- Baseline reconciled by NAME, not by count alone, after the reviewer's own
  `pytest tests -q --collect-only` diff between `f8949f1` and this tree
  (5707 vs. 5711 collected; my own earlier full-run totals, 5703 passed / 4
  xfailed on `f8949f1` vs. 5707 passed / 4 xfailed here, agree with that —
  5703 + 4 = 5707 collected on base, 5707 + 4 = 5711 here). Delta: **+4**,
  matching the four new tests exactly (listed above); zero renamed or
  re-expanded parametrised cases, zero tests only in base.
- Mutation check: copied `src` + `tests` to `/tmp/mut700` (outside the repo),
  asserted the anchor
  (`"    for band in _prose_bands_with_columns(words):\n"`) occurs exactly
  once in `row_corroboration.py` before editing, reverted it to
  `cluster_band_words(words)`, and confirmed via a canary
  (`os.path.realpath(socr.__file__)` under the mutant path) that the mutant
  source was actually loaded. The guard FAILED as required:
  `test_two_column_prose_and_table_row_split_into_separate_bands` and
  `TestTwoColumnPagesFailSafe::test_left_column_prose_now_ships` both failed
  against the reverted source. Mutant directory deleted after the check.
- `uvx ruff@0.16.0 format --check .` clean (one file needed reformatting —
  `tests/tables/test_row_corroboration.py`'s new `line_words` signature —
  reformatted and re-checked clean).

## Rejected

- Splitting the shared `cluster_band_words` (see measurement above — breaks
  `corroborate_rows` on a realistic table row).
- Withholding every zero-token band inside a withheld band's y-span (already
  rejected upstream of this ticket; not attempted).
