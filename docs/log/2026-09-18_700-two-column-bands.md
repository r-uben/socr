# GH-700 — two-column pages withhold left-column prose behind the right column

**STATUS (2026-09-18, mid-revision): NOT DONE.** Everything below "What
changed" through "Rejected" describes the FIRST committed version
(`8b3311c`), which review found defective (see the two sections at the
bottom of this file, added during revision, for what superseded it). The
working tree currently contains a SECOND, also-incomplete attempt reusing
`_detect_column_gutter` (GH-152) plus a `LABEL_COLUMN_WIDTH_SHARE` gate that
is measured-not-shipped (see below). Holding for direction; nothing past
this point is final.

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

---

## Revision round 1 (2026-09-18): reviewer-reproduced dilution defect

Reviewer independently reproduced a real defect in the version above:
`_median_adjacent_word_gap` pools EVERY y-clustered band's internal word-gaps
into one page-wide median, so a dense table anywhere on the page can dilute
that median and raise the split threshold past a genuine two-column gutter
elsewhere on the page. Reproduced on a 20-row x 6-column dense table plus a
separate two-column region sharing one baseline, table column pitch swept:

```
pitch=30.0  median_word_gap=30.0  min_gutter_needed(2x)=60.0   true_page_gutter_width=180.0  -> gutter=310.0  (SPLIT FIRES)
pitch=40.0  median_word_gap=40.0  min_gutter_needed(2x)=80.0   true_page_gutter_width=130.0  -> gutter=335.0  (SPLIT FIRES)
pitch=45.0  median_word_gap=45.0  min_gutter_needed(2x)=90.0   true_page_gutter_width=105.0  -> gutter=347.5 (SPLIT FIRES)
pitch=50.0  median_word_gap=50.0  min_gutter_needed(2x)=100.0  true_page_gutter_width=80.0   -> gutter=None  (NO SPLIT — reviewer's exact repro, unfixed)
```

Direction: reuse `_detect_column_gutter` (`socr/tables/reconstruct.py:1959`,
GH-152) instead of a private median rule — it finds the ONE x-interval no
word's bbox crosses ANYWHERE on the page, a whole-page structural check
rather than a per-band statistical one, so a distant table's own density
cannot dilute it the same way. Applied: `_prose_bands_with_columns` now
calls `_detect_column_gutter(words)` once over the whole page and splits
each y-band only where it has words on both sides of that one x.

This closes the pitch-30/40/45 cases above, but NOT pitch-50: sizing
`_detect_column_gutter`'s own `min_gutter_pt` threshold is `_median_word_gap`
(reconstruct.py:1930), which pools per-`(block_no, line_no)` gaps across the
WHOLE page the same way the rejected v1 rule did — the dilution is inherited
from GH-152's own shared code, not eliminated. Both of `_detect_column_gutter`'s
existing production callers (`reconstruct.py:204`, `:2265`, inside
`find_tables`) already invoke it on `page.get_text("words")` unscoped, so
this is latent, shipped GH-152 behaviour, not something GH-700 introduces.
Filed separately as **#828** (references #824 and #700) rather than fixed
here: fixing it touches shared code with two callers this ticket does not
own, and per the ruling below (asymmetry of the two failure directions), a
missed split (fail-closed, no new loss) is the acceptable residual, not
something to spend more scope closing.

## Revision round 2 (2026-09-18): reviewer-reproduced label/value contamination — the headline defect

Reviewer reproduced a SEVERER defect in the `_detect_column_gutter` reuse:
on an ORDINARY single-column table (one row label, N numeric columns), the
label->value gutter is ITSELF a clean, page-wide, never-crossed x-interval —
geometrically indistinguishable from a genuine two-column page from
x-projection alone. `_detect_column_gutter` correctly finds it and the reuse
splits every row in half, tagging the label half `is_prose=True`. Reproduced
on a 4-row table ("Total loans and leases", 90pt label->value gutter, values
20pt apart), numeric-column count swept:

```
ncols=1  pre=4  post=8  FALSE SPLIT
ncols=2  pre=4  post=4  ok (accidental: 2 qualifying gutters -> _detect_column_gutter itself abstains)
ncols=3  pre=4  post=8  FALSE SPLIT
ncols=5  pre=4  post=8  FALSE SPLIT
```

Why this is worse than the pitch-50 miss: `manifest.native_prose_floor_text`
(manifest.py:2566) joins consecutive prose-tagged bands into ONE paragraph.
A false-split label band is tagged prose (it carries no digit) and is
concatenated straight into the surrounding body text, indistinguishable from
a real sentence — table content misrepresented as prose on a citation
corpus. A missed split (band stays whole, withheld under the marker) is
GH-700's pre-fix, disclosed behaviour: visible, fail-closed, no new loss.
The two error directions are not symmetric, and every remaining ambiguity in
this ticket resolves toward NOT splitting.

`_has_row_labels` (reconstruct.py:2002, GH-152's own guard against
mis-splitting a wide single table into two independent tables) does not
discriminate the two cases either — measured, not assumed: on the false-split
table's post-split halves, `_has_row_labels(left)=False` and
`_has_row_labels(right)=False`; on the genuine two-column prose case, ALSO
`False`/`False` on both sides. It answers "does this side have its own row
label column", which neither case's halves do (the label side has no
numeric lane to be positioned against; the value side has no non-numeric
tokens at all), so it returns the same verdict for the case that must split
and the case that must not.

## `LABEL_COLUMN_WIDTH_SHARE` gate — measured, NOT shipped

Reviewer's proposed second gate: `LABEL_COLUMN_WIDTH_SHARE = 0.65`
(`socr/core/born_digital.py:493`, GH-592), applied the same way
`born_digital._try_aligned_run` already does at born_digital.py:1009 (median
line width left of the gutter vs. median line width right of it), but to the
opposite decision — refuse the split (rather than accept a label/value
pairing) when the left side is the narrow one. Implemented on a throwaway
basis to produce these numbers; **not committed, not shipping** per the
reviewer's explicit ruling that this is a finding to record, not a fix to
adopt, until it is known whether ncols=1 is the *only* residual (four
hand-built fixtures are not a population; that determination needs its own
measurement this ticket has not done).

Measured, left/right median line width per fixture (median across y-bands
with content on both sides of the gutter):

```
ncols=1              gutter=102.0  left=47.00  right=12.00   ratio=3.917  -> ALLOW split  (WRONG direction)
ncols=2              gutter=None   (no split candidate — _detect_column_gutter itself abstains)
ncols=3              gutter=102.0  left=47.00  right=76.00   ratio=0.618  -> REFUSE split (correct)
ncols=5              gutter=102.0  left=47.00  right=140.00  ratio=0.336  -> REFUSE split (correct)
target 2-col prose   gutter=239.0  left=68.00  right=24.00   ratio=2.833  -> ALLOW split  (correct)
```

**Derivation-transfer failure, stated explicitly:** `LABEL_COLUMN_WIDTH_SHARE`
was measured on a population where the label is narrow BY CONSTRUCTION — an
abbreviation ("Mr."/"Ms.") beside a full name, Fed fixture ratio 0.51 (see
the constant's own derivation block). GH-700's row labels are the opposite
shape: a full descriptive phrase ("Total loans and leases") beside a single
short number. With 2+ numeric columns the value side's span (first value's
x0 to last value's x1) grows past the label's width and the gate correctly
refuses the split; with exactly one numeric column the value side is just
that one number's own character width, narrower than the label, and the
gate answers backwards. The constant transfers for the multi-column-value
population it happens to also fit, not by design for this domain.

## Digit-carrying band fraction — measured, numbers only, not a predicate

Reviewer's falsification probe, on the same five fixtures: for the whole
page, what share of y-bands (pre-split, `cluster_band_words`'s own bands)
carry at least one printed digit; and for the region strictly left of the
detected gutter, how many consecutive bands are entirely digit-free.

```
ncols=1              page_bands=4  bands_with_digit=4/4  left_digit_free_bands=4/4  longest_consecutive_run=4
ncols=2              page_bands=4  bands_with_digit=4/4  gutter=None -> left-region N/A
ncols=3              page_bands=4  bands_with_digit=4/4  left_digit_free_bands=4/4  longest_consecutive_run=4
ncols=5              page_bands=4  bands_with_digit=4/4  left_digit_free_bands=4/4  longest_consecutive_run=4
target 2-col prose   page_bands=1  bands_with_digit=1/1  left_digit_free_bands=1/1  longest_consecutive_run=1
```

Falsified, as the reviewer asked to check rather than confirm: EVERY
fixture's left-of-gutter region is 100% digit-free, table label column and
genuine two-column prose alike — the table's own label column ("Total loans
and leases" etc., correctly free of digits) and the prose column's left
side look identical on this measure. Digit-carrying fraction does not
discriminate the two cases on these fixtures; whatever the eventual signal
is, it is not this one, at least not on this population.

## `detected_table_count` / `detected_table_bboxes` at `partition_prose_bands` — traced, not inferred

Reviewer asked whether a page reaching `partition_prose_bands` via
`native_prose_floor_text` could ever carry upstream table-geometry evidence
(`p.detected_table_count` / `p.detected_table_bboxes`) that this fix could
consult instead of guessing from word geometry alone. Traced the single
write site rather than trusting the docstring's one worked example
(Fed 1989-11-14 p3): `ps.detected_table_count` / `ps.detected_table_bboxes`
are assigned in exactly one place in the codebase,
`DocumentState.apply_born_digital` (`socr/core/state.py:625`,
lines 702-704), and that assignment sits inside `if pa.is_born_digital:`
(state.py:649). `native_prose_floor_text`'s only caller runs inside a branch
gated `not p.is_born_digital` (manifest.py:3099-3101). So for every page
that reaches this code path, the copy never ran, and both attributes sit at
their dataclass defaults (`0` / `[]`, state.py:190-191) — always, by
construction, not merely in the one documented example. There is no
upstream table-bbox evidence available to this fix at this call site.

## Crossing-line measurement — measured by reviewer, reproduced independently

Reviewer's hypothesis: any full-width line on a page (caption, footnote,
running header) that spans across the label->value x makes
`_detect_column_gutter` correctly abstain (fail closed), because that
line's own words occupy the x-range a false gutter would need to be clean.
Fixture: label "Total loans and leases" as four words, 20pt pitch (15pt
wide, 5pt gaps), ending x=125; value columns starting x=199 at 17pt pitch;
two-column prose page of six y-bands, left column x=50..245, right column
values x=320..402, one band sharing a baseline with a right-side value; all
words as PyMuPDF 8-tuples. Reviewer's numbers, reproduced independently
below (own run, same fixture spec):

```
table ncols=1/3/5 +caption above (wide)   gutter=None  pre=5  post_bands=5  prose=0
table ncols=1     +caption below (wide)   gutter=None  pre=5  post_bands=5  prose=0
two-col prose, 6 bands (no header)        gutter=282.5 pre=6  post_bands=7  prose=6
two-col prose +full-width header          gutter=None  pre=7  post_bands=7  prose=6
bare table (no caption), ncols=1/2/3/5    gutter=162.0 pre=4  post_bands=8  prose=4  (FALSE SPLIT, all four)
```

**Reviewer's own numbers reproduced exactly**, once two fixture-construction
bugs on my own side were fixed: (1) my first attempt built the caption/header
too narrow (extent only ~x=10 to ~123-151), so it never actually crossed the
gutter region it was meant to block — widened it to explicitly span the full
relevant x-range (`end_x=245` for the table case, `end_x=402` for the
two-column case); (2) a digit embedded in the header text ("14", "1989",
"3") made the merged header band itself non-prose (withheld), undercounting
`prose` by one — removed digits from the header fixture. After both fixes,
every number above matches the reviewer's independently.

**Bare-table ncols discrepancy, reconciled:** an earlier round of my own
measurement (reported before this one) had shown `ncols=2` abstaining
(`gutter=None`) rather than false-splitting, with `LABEL_COLUMN_WIDTH_SHARE`
appearing to "save" ncols=3/5. Reproducing the reviewer's EXACT pitch (label
ending x=125, values starting x=199 at 17pt pitch, vs. my earlier looser,
inconsistent value spacing) reproduces the reviewer's numbers exactly: FALSE
SPLIT at every ncols in {1,2,3,5}, no accidental abstention anywhere. The
earlier divergence was my own fixture's looser/inconsistent value-column
pitch, not a genuine disagreement about `_detect_column_gutter`'s or the
width-share gate's behaviour.

**Two conclusions (reviewer's), confirmed by reproduction:**

1. The crossing-line hypothesis holds for the three cases it is meant to
   cover — a full-width caption above or below an otherwise-bare table (at
   ncols 1, 3, and 5) makes `_detect_column_gutter` correctly abstain, so
   the label/value false-split defect (Revision round 2, above) does not
   fire on any page that also carries a full-width line crossing that x.
2. Its stated killer also fires, and is benign rather than fatal: a
   full-width header on a GENUINE two-column prose page also collapses that
   page's real gutter (`gutter=None`), so the fix is lost there too — but
   only as a missed split (the shared band stays whole, withheld under the
   marker), not as a false split. Per the asymmetry ruling above (Revision
   round 2), a missed split reproduces GH-700's pre-fix, disclosed,
   fail-closed behaviour — acceptable, not corruption.

**Real-fixture data point — one more instance of the abstention mechanism,
does not touch the corrupting direction:** measured `_detect_column_gutter`
against `tests/fixtures/table_repair/ce_like_p4.pdf` (a
synthetic-but-realistic single page combining two distinct tables of
different column counts/positions plus a four-line prose paragraph at the
bottom, 25 native y-bands via `cluster_band_words`). Result: `gutter=None`
already, with no crossing-line fixture needed — page heterogeneity alone
defeats "one x-interval no word anywhere on the page crosses." This shows
the detector abstaining (the fix not firing) on one more shape; it says
nothing about the corrupting direction, which is already settled by
`table_ladder/binding_shift_doc.pdf` below (a real PDF, demonstrated
corruption). **Frequency claim: unmeasured, no corpus access on this
machine.** One synthetic fixture is not evidence that real corruption is
rare — there is no corpus here to count against, and this data point must
not be read as softening the ruling below.

## Ruling (2026-09-19): the split approach does not ship — reverted to HEAD

Team-lead verified independently, not relayed, against a real PDF already in
this repo (`tests/fixtures/table_ladder/binding_shift_doc.pdf`), running the
working tree's (uncommitted, v2) `_detect_column_gutter`-based split with the
source canary asserted:

```
page 1: 12 words  gutter=219.52  _is_label_column_gutter=False  raw_bands=5 -> post_bands=9
        PROSE: ['Clean','table','(control','page)']
        PROSE: ['Label']  ['Value']  ['Revenue']  ['Costs']  ['Margin']
page 2: 19 words  gutter=293.00  _is_label_column_gutter=False  raw_bands=6 -> post_bands=11
        PROSE: ['Label','OLS']   PROSE: ['IV']
```

v2 ships a real table's row labels as body prose on a fixture already in the
repo, not a synthetic artefact; on page 2 the chosen gutter lands INSIDE the
value block, between "OLS" and "IV", shipping half a header row as prose.
This is the exact severity class the ticket was raised over — content
misrepresented as prose, not merely withheld.

**Ruling: the split approach does not ship**, for two reasons together, not
one:

1. It corrupts, demonstrated on real repo data above (v2 only — see below).
2. It does not fire on the realistic shape of the bug it was reported
   against. The "Crossing-line measurement" section above (case 4) and the
   second-model consult's own finding (labelled B1) reach this
   independently, by different mechanisms: mine is that a realistic
   right-column's own inter-value gaps register as extra candidate
   gutters (fragmentation, `_detect_column_gutter` returns `None` because
   MORE than one qualifying gutter exists); the consult's is that a
   labelled right-hand table has its own label->value gap, giving two
   uncrossed gutters on the page. Either way, `_detect_column_gutter`
   abstains on a realistic two-column page and the original GH-700 loss
   stands unfixed.

A fix that corrupts the safe case and abstains on the target case is not a
partial fix. `src/socr/tables/row_corroboration.py` reverted to `HEAD`
(`git checkout -- src/socr/tables/row_corroboration.py`), which restores it
to `8b3311c` — the FIRST committed version (`_median_adjacent_word_gap` +
`_split_band_by_column_gap`, described at the top of this file), not
pre-ticket baseline. Re-measured `8b3311c`'s own behaviour against the same
real fixture, post-revert, to confirm it does NOT reproduce this specific
corruption:

```
page 1: words=12 raw_bands=5 post_bands=5 (no split fired)
   PROSE ['Clean', 'table', '(control', 'page)']
   PROSE ['Label', 'Value']
   withheld ['Revenue', '120'] / ['Costs', '45'] / ['Margin', '75']
page 2: words=19 raw_bands=6 post_bands=6 (no split fired)
   withheld ['Shift', 'table', '(GH-273', 'shape)']
   PROSE ['Label', 'OLS', 'IV']
   withheld ['RowA', '11', '21'] / ['RowB', ...] / ['RowC', ...] / ['RowD', ...]
```

`8b3311c` does not split at all on this fixture (`post_bands == raw_bands`
on both pages) — it ships the header row whole as PROSE (a different,
pre-existing residual: a two-column-shaped header/label row is not
recognised as table content and ships unsplit, which is the same class of
gap GH-700 was opened to close, not new corruption). It does not exhibit
the mid-row false split (`gutter` landing between "OLS" and "IV") that v2
did. The disclosed pitch-50 dilution residual (Revision round 1, above)
remains v1's own known limitation. Width-share gate numbers are NOT
reshipped; they remain in this log only, per the ruling.

## `_has_row_labels` — three measurements, unreconciled

Do not resolve by picking one story; recording all three as measured,
against different fixtures, explicitly unreconciled:

1. **Mine** (four hand-built ncols=1/2/3/5 label/value fixtures and the
   target two-column prose fixture, "Revision round 2" above):
   `_has_row_labels(left)=False`, `_has_row_labels(right)=False` on EVERY
   fixture, both the false-split table cases and the genuine two-column
   case — no discrimination observed.
2. **Second-model consult** (its Case B, a right-hand column that is itself
   a labelled table, including the GH-649 fixture per team-lead's relay):
   `_has_row_labels(right)=True`. Team-lead has not independently verified
   this claim and does not assert it here.
3. **Team-lead**, on the real PDF above (`binding_shift_doc.pdf`):
   page 1 (`['Label','Value']` vs. `['Revenue','120']` etc.) →
   `left=False, right=False`; page 2 (`['Label','OLS']` vs. `['IV']` etc.,
   the GH-273-shaped shift table) → `left=True, right=False`.

These three do not agree even on sign for a labelled-right-column case
(mine: False; team-lead's real PDF page 2: True for LEFT, not right, in a
different split geometry than the consult's Case B). Left open, not
resolved — whatever picks this back up must re-derive it against a shared
fixture rather than trust any one of these three numbers.

## Proposal filed, not implemented

Filed **#829** as a PROPOSAL, not a bug, per the ruling that
`_detect_column_gutter`'s abstention on a realistic two-column page is the
detector correctly declining a job it was never built for, not a fourth
defect to patch. Two separable strands, neither implemented, neither
measured as sufficient:

- **Strand A** (second-model consult's): select WHICH of several candidate
  gutters is the page gutter by asking whether the region right of it
  carries its own row labels (`_has_row_labels`), gating the split on the
  same predicate. Geometry-only. Unmeasured hazard, stated by the consult:
  a table whose first value column is textual in most rows (e.g. 8 of 10
  `n.a.`) flips the predicate and false-splits; nobody has measured how
  often that shape occurs in the corpus.
- **Strand B** (team-lead's, from reconciling the two
  `detected_table_count`/`detected_table_bboxes` traces — mine, that the
  copy onto `PageState` in `state.py:702-704` is gated
  `if pa.is_born_digital:`; the consult's, that `_detect_table_regions`
  itself runs on EVERY page and is stamped onto the `PageAssessment`
  unconditionally at `born_digital.py:2881-2883`): carry
  `detected_table_count`/`detected_table_bboxes` onto `PageState`
  unconditionally rather than only for born-digital pages, then withhold
  inside the detected bbox and ship outside it — no gutter, no split, no
  new predicate, reuses a detection the codebase already performs and
  already trusts for born-digital pages. Explicitly UNVERIFIED: whether
  `_detect_table_regions` is any good on a scanned page, which is the
  population `partition_prose_bands` actually serves via
  `native_prose_floor_text`'s `not p.is_born_digital` gate — it may be
  near-useless without native text, in which case the unconditional copy
  ships a field that is reliably empty and buys nothing. Also noted:
  moving the copy out of the `is_born_digital` branch touches a widely-read
  state field and needs its own before/after difference test, not a pinned
  tuple.

Neither strand is to be implemented before its stated open question is
measured.

**Corpus-frequency question — not answered.** Searched this machine for a
real Fed/ECB minutes PDF or an accessible copy of the corpus this ticket's
citation-fidelity concern is about, to ground-truth "realistic pitch" and to
attempt the frequency count directly. Found none reachable from this
worktree or session: the four PDFs under `tests/fixtures/` in this repo are
all synthetic fixtures built for other tickets (table_repair, table_ladder,
replay_binding), and no real Fed/ECB source corpus is present on this
machine outside of unrelated research-library PDFs (papers about the Fed,
not primary-source minutes/SEP documents) and other agents' unrelated
`/tmp/astra735*-corpus` directories (not touched, not part of this ticket).
This question is left open, pending either access to the real corpus or
folding-in of the second-model consult's own measurement.
