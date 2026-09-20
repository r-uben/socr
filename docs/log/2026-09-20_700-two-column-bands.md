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

## Files

- `src/socr/tables/row_corroboration.py` — `cluster_band_words` column split;
  new `_cluster_band_words_single_column` holds the original logic.
- `tests/test_gh649_scanned_prose_recovery.py` — `TestTwoColumnPagesFailSafe`:
  updated docstring to the new expectation, added
  `test_the_left_columns_prose_ships_instead_of_hiding_behind_the_marker`
  (pins the fix), renamed the fabrication test for clarity. Both pre-existing
  no-leak assertions (`test_no_printed_value_reaches_the_page`, the renamed
  fabrication test) are untouched in substance.

## Verification

- `tests/test_gh649_scanned_prose_recovery.py`: 52 passed.
- `tests/test_gh152_column_aware_rowize.py`, `tests/test_gh152_reconstruct_band_clip.py`,
  `tests/tables/test_row_corroboration.py`, `tests/test_gh652_prose_witness_trust.py`,
  `tests/test_gh652_review_renderer_literal_escapes.py`,
  `tests/test_gh652_witness_extent_abstention.py`: 220 passed.
- Full suite (`PYTHONPATH=/tmp/wt-700/src ~/venvs/socr/bin/pytest -q`):
  `5739 passed, 4 xfailed` in 297.99s.
- Format gate: `cd /tmp/wt-700 && uvx ruff@0.16.0 format --check .` →
  `760 files already formatted`.

## Mutation proof

Copied both `src` and `tests` to `/tmp/mutant-700` (outside the repo),
reverted only the `cluster_band_words` change there (removed the
`_detect_column_gutter` import and the column split, restoring the original
full-width-only clustering; kept the updated tests). Confirmed the mutant
copy's own source was exercised (`sys.path` shows
`/private/tmp/mutant-700/src` ahead of the venv's editable-install entry
during collection — `/tmp` is a symlink to `/private/tmp` on this machine,
which tripped an early canary using a bare `/tmp/...` prefix; fixed by
checking both prefixes). Collected 52 tests in both the mutant and the real
worktree (no silent skip).

Result on the mutant: `test_the_left_columns_prose_ships_instead_of_hiding_behind_the_marker`
fails (the left prose is absent, replaced by the bare marker) while the two
no-leak/no-fabrication tests in the same class still pass — the new test is
the one carrying the fix, and the existing safety guarantees are independent
of it.

## Deviations / follow-ups

- None. `_detect_column_gutter` was reused exactly as instructed; no new
  threshold was introduced.
- Reading order for a two-column page ships left-column-then-right-column
  (not y-interleaved). This matches the convention `reconstruct.py`'s own
  two-table split already uses and documents as a known, separately-scoped
  remainder (true left-to-right interleaving needs a change to that file's
  own sort). Left as-is here for the same reason: out of #700's scope.
