# GH-700: column-aware native baseline banding

## What changed

`row_corroboration.cluster_band_words` clustered native words into baseline
bands by y-centre across the **full page width**. On a two-column page a
left-column prose line and a right-column table row on the same y therefore
landed in one band; the band's numeric tokens (from the table side) made the
whole band table-shaped, so the fail-closed prose/table partition
(`partition_prose_bands`, and everything downstream of it in
`manifest._recover_scanned_prose_when_no_table_geometry`) withheld the left
column's prose along with the right column's digits. Safe direction (no
printed value leaked), but content loss on the left column — exactly what
#700 tracks.

Fix: `cluster_band_words` now detects column geometry first, reusing
`reconstruct._detect_column_gutter` (GH-152) unchanged — same
`ALIGNED_RUN_GAP_MAX_WORD_SPACES` yardstick, same "any spanning word rules
the gutter out" fail-closed behaviour, no new threshold. When a single
gutter is found, each side's words are clustered independently (factored
into `_cluster_band_words_single_column`, the original full-width logic) and
returned left-column-bands-first-top-to-bottom then
right-column-bands-first-top-to-bottom — the same left-band-then-right-band
convention `reconstruct.py`'s own two-table split already returns (see that
function's docstring). No gutter, or any exception while detecting one,
falls through unchanged to the original single full-width clustering.

Because every caller of `cluster_band_words` (`baseline_bands`,
`partition_prose_bands`, `native_prose_floor_text`) goes through this one
function, the fix applies uniformly without touching any of those three.

## Round 2: review found a regression, opt-in scoping fixes it

Reproduced before touching anything (per instructions): `corroborate_rows`
reaches `cluster_band_words` through `baseline_bands` with REGION-SCOPED
table words (`baseline_bands(words_in_region(words, region))` —
`corroborate_rows`'s own docstring says so at the top). A single wide table
with two column GROUPS separated by a gap wider than
`ALIGNED_RUN_GAP_MAX_WORD_SPACES * median_word_gap` — a `Mean | SD` block
beside a `Q1 | Q3` block, an ordinary shape — has exactly the x-gap
`_detect_column_gutter` looks for. Splitting it in two does not recover two
independent structures; it cuts every row's own cells in half at the same y,
so `match_rows_monotonic` can never find a candidate row's full numeric run
in either half. Measured on a synthetic 5-row x 5-column table
(`xs = [40, 100, 160, 470, 530]`, correct byte-for-byte markdown
transcription):

| | `bound` | `total` | `native_numeric_rows` | `unbound_rows` |
|---|---|---|---|---|
| pre-#700 / desired | 5 | 5 | 5 | `((),)` |
| #700 round 1 (regression, reproduced) | 0 | 5 | 10 | `((0, 1, 2, 3, 4),)` |
| #700 round 2 (this fix) | 5 | 5 | 5 | `((),)` |

**Decision: scope the split to the one call site that actually has a page,
not a tuned threshold.** Added `column_aware: bool = False` to
`cluster_band_words`; `baseline_bands` (and therefore every region-scoped
table caller: `corroborate_rows`, plus the two other `baseline_bands`
call sites in `row_corroboration.py:597` and `manifest.py:2247`, both also
region-scoped) keeps the default and is completely unaffected by #700 —
verified by inspection of every `baseline_bands`/`cluster_band_words` call
site in `src/`. `partition_prose_bands` (the #649/#652 prose-floor lane,
always page-scoped words) and `manifest.native_region_text` (its sole
caller passes `prose_words`, itself page-scoped, and the two functions must
never disagree about where a line begins per `native_region_text`'s own
docstring) both now pass `column_aware=True` explicitly.

**Why not the "two independent columns" discriminator instead** (the other
candidate offered): `_has_row_labels` (GH-152's own two-table-vs-one-wide-
table discriminator) requires a numeric anchor lane on BOTH sides to define
"positionally left of the numeric lane" — it returns `False` outright when a
side has zero numeric words. #700's own target fixture is exactly that: the
left column is pure prose, zero numeric tokens. Gating on `_has_row_labels`
on both sides would silently un-fix #700 while fixing the regression, i.e.
trade one loss for the other rather than closing both. A "do bands on
either side share y" check was also considered and rejected: it does not
discriminate here either — a genuine two-column PAGE can coincidentally
row-align (short lines, similar font) the same way a table's rows always
do by construction; the opt-in placement is the one distinction grounded in
what's actually known at each call site (does this caller hold a whole page
or a table's own region), not in gap geometry, so it needs no threshold at
all.

`partition_prose_bands`'s docstring updated (was: "Bands run top of page to
bottom" unconditionally, which becomes false once a gutter fires) to say so
explicitly and point at `cluster_band_words`'s `column_aware` docstring for
the detector.

## Files

- `src/socr/tables/row_corroboration.py` — `cluster_band_words` gains
  `column_aware: bool = False`; new `_cluster_band_words_single_column`
  holds the original full-width logic; `partition_prose_bands` opts in
  (`column_aware=True`) and its docstring is corrected; `baseline_bands`
  is untouched (stays default `False`).
- `src/socr/core/manifest.py` — `native_region_text` opts in
  (`column_aware=True`), matching `partition_prose_bands`.
- `tests/test_gh649_scanned_prose_recovery.py` — `TestTwoColumnPagesFailSafe`:
  updated docstring to the new expectation, added
  `test_the_left_columns_prose_ships_instead_of_hiding_behind_the_marker`
  (pins the fix), renamed the fabrication test for clarity. Both pre-existing
  no-leak assertions (`test_no_printed_value_reaches_the_page`, the renamed
  fabrication test) are untouched in substance.
- `tests/tables/test_row_corroboration.py` — new
  `TestSingleWideTableSurvivesColumnAwareBanding` (round 2's regression pin).

## Verification

- `tests/test_gh649_scanned_prose_recovery.py`: 52 passed.
- `tests/tables/test_row_corroboration.py`, `tests/test_gh152_column_aware_rowize.py`,
  `tests/test_gh152_reconstruct_band_clip.py`, `tests/test_gh652_prose_witness_trust.py`,
  `tests/test_gh652_review_renderer_literal_escapes.py`,
  `tests/test_gh652_witness_extent_abstention.py`: 273 passed.
- Full suite (`PYTHONPATH=/tmp/wt-700/src ~/venvs/socr/bin/pytest -q`):
  `5740 passed, 4 xfailed` in 278.62s.
- Format gate: `cd /tmp/wt-700 && uvx ruff@0.16.0 format --check .` →
  `761 files already formatted`.

## Mutation proof

**Round 1** (page-wide split itself): copied `src`+`tests` to
`/tmp/mutant-700` (outside the repo), reverted only the `cluster_band_words`
change there. Confirmed the mutant copy's own source was exercised
(`sys.path` showed `/private/tmp/mutant-700/src` ahead of the venv's
editable-install entry during collection — `/tmp` is a symlink to
`/private/tmp` on this machine, which tripped an early canary using a bare
`/tmp/...` prefix; fixed by checking both prefixes). Collected 52 tests in
both the mutant and the real worktree (no silent skip). Result:
`test_the_left_columns_prose_ships_instead_of_hiding_behind_the_marker`
failed (left prose absent, bare marker only) while the two no-leak/
no-fabrication tests in the same class still passed.

**Round 2** (opt-in scoping): rather than a second full external mutant
copy, flipped `cluster_band_words`'s default to `column_aware: bool = True`
in place (the exact shape of "the scoping guard is absent"), ran only
`tests/tables/test_row_corroboration.py -k TestSingleWideTableSurvivesColumnAwareBanding`,
confirmed it failed with `native_numeric_rows=10, bound=0` (byte-identical to
the round-1 regression numbers above), then restored the file from a
pre-edit backup and re-ran the file to confirm the restore was exact
(273-test file returns to green). This is a same-process default-flip
mutation, not an external-tree copy, because the fault this round is a
default-value regression in the same function already covered by round 1's
external-tree proof; the guard under test — whether `baseline_bands`'s
callers see the split — is entirely local to this one function, so no
import-resolution risk (the trap round 1's canary caught) applies here.

## Deviations / follow-ups

- Round 1: none. `_detect_column_gutter` was reused exactly as instructed;
  no new threshold was introduced.
- Round 2: `cluster_band_words` gained a boolean parameter rather than
  staying a zero-argument primitive; documented as opt-in scoping, not a
  second heuristic, in its own docstring.
- Reading order for a two-column page ships left-column-then-right-column
  (not y-interleaved). This matches the convention `reconstruct.py`'s own
  two-table split already uses and documents as a known, separately-scoped
  remainder (true left-to-right interleaving needs a change to that file's
  own sort). Left as-is here for the same reason: out of #700's scope.
