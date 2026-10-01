# GH-902: rotated tables rowized 180 degrees flipped (wrong rotation sign)

## Cause

`upright_rotation_for(page)` is the correction that makes text upright. The word-geometry
code applied it with the wrong sign: `rowize_from_word_list` rotated words by `-rotation`
(and output rects by `+rotation`), and `native_first.upright_words_for_page` did the same
for the witness words. For 90 and 270 pages the grid came out reversed in rows and
columns. For 180 the sign is irrelevant (-180 == +180), and 0 is a no-op.

## Empirical sign (synthetic pymupdf pages, text drawn with rotate=0/90/180/270)

| drawn at | upright_rotation_for | `_rotate_word_bbox` +rotation | -rotation |
| --- | --- | --- | --- |
| 0 | 0 | upright | upright (no-op) |
| 90 | 90 | upright reading order | fully reversed |
| 180 | 180 | upright | upright |
| 270 | 270 | upright reading order | fully reversed |

So +rotation is correct for both 90 and 270. Output rects go back with -rotation.

Real page, Fama p368 (index 367, rot=90): after the fix the shipped grid has 28 rows,
labels in the leading column, and the last row is the printed last row (label first,
last cell 7.62). Before: labels last, rows reversed (issue comment).

## Changes

- `src/socr/tables/reconstruct.py`: words rotated by `+rotation`, rects back by `-rotation`.
- `src/socr/tables/native_first.py`: `upright_words_for_page` rotates by `+rotation`.
- Other `upright_rotation_for` users rotate raster pixmaps via `Matrix.prerotate` and
  were not touched.

## Fixtures were encoding the bug

Two existing fixtures laid the 90/270 grids out 180 degrees off the real geometry, which
the wrong sign undid, so their tests passed against the bug:

- `tests/test_reconstruct.py::_create_synthetic_table_page`: the 90 and 270 layouts were
  swapped. The matrix test also now asserts the page measures as the rotation it names.
- `tests/test_rotated_native_table_first.py::_rotated_dense_forecast_pdf` replaced by
  `_forecast_pdf(path, rotation)` which maps an upright layout onto the page with
  `_place` (0, 90, 270). Existing tests keep the 90 case.

## New tests (`TestRotationSign`, 90 and 270)

- rowized markdown equals the upright twin's, labels in reading order; the old sign gives
  a different grid (difference pin).
- `attempt_rotated_native_table` markdown equals the upright twin's, plan is SHIP, labels
  in reading order.
- witness words (`upright_words_for_page`) sort into the upright twin's reading order.
- region rect encloses the table words in page coordinates.

## Finding: the verifier cannot see a flipped grid

The requested guard "`plan_native_table` on a 180-flipped grid against corrected words
must not be SHIP" is not achievable. Measured on the 90 twin with correct witness words:
reversed rows, reversed cells, and both together all return SHIP / exact_pass. The verifier
pairs rows by number multiset, so it is orientation-blind. The sign is therefore observable
only in geometry, which is why the witness is pinned directly. Whether the verifier should
learn row order / label binding is a separate question (not done here).

## Mutation checks (copy of src, tests, pyproject in /tmp; canary on realpath of `socr.__file__`; anchor count == 1)

| mutation | result |
| --- | --- |
| revert sign at `rowize_from_word_list` (words) | 7 failed (4 new, matrix test, 2 more); canary passed |
| revert sign at rect-back only | 2 failed (`test_region_rect_encloses...` 90, 270) |
| revert sign at `upright_words_for_page` only | 2 failed (witness-order test 90, 270) |

## Real-corpus re-measure (every rotated page in ~/papers/pdf, CPU, `retry.py`)

| | before (retry.jsonl) | after (retry_after.jsonl) |
| --- | --- | --- |
| pages with a rotation | 449 | 449 |
| SHIP | 29 | 35 |
| refuse | 112 | 115 |
| defer | 59 | 50 |
| no attempt | 185 | 185 |
| errors | 64 | 64 |
| SHIP, labels first | 3 | 24 |
| SHIP, labels last | 24 | 7 |

The 7 remaining labels-last SHIP pages (Fama pp46-51, Martens p16) were inspected by row
order: they are lists/tables whose text column is genuinely last (rank, counts, year,
reference) or have a header row first; reading order is correct. The heuristic is a
false positive on them. SHIP-to-refuse 7, defer-to-ship 7, refuse-to-ship 10 reflect the
grid now being upright so the verifier sees the true row structure.

Files: outputs in `~/.local/state/socr-housekeeping/gh902/retry_after.{jsonl,log}`.
