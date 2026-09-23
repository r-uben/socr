# GH-887 — re-attach a minus sign the column boundary split off (2026-09-23)

Branch `fix/887-reattach-detached-sign`.

## Found by

The #846 corpus measurement (Fable, 2026-09-23). #846's own shape — a margin page number or
footnote marker absorbed into a data cell — was not observed (0 of 62,321 cells). This sibling
was: some PDFs set a negative number's minus sign as its own one-glyph word, and text-strategy
`find_tables` put a column boundary between the sign and the digits. The sign shipped at the end
of the left neighbour's cell and the value shipped unsigned. The numeric-multiset guards cannot
see it, because both pieces stay on the page.

Reproduced independently on `2017__fama__ap.pdf` p551 with `origin/main`'s
`reconstruct_table_regions`: 67 cell pairs ending/starting that way. Confirmed on the rendered
page by eye.

## Evidence for the rule

On the affected page, all 76 sign-then-digit word pairs on one line abut **exactly** (gap 0.00pt,
also 0.00 in units of the digit word's glyph advance). Two separate numbers in adjacent columns
were never flush (minimum 0.24 glyph advances). So the repair keys on geometric contact — the
sign's right edge meets or overlaps the digits' left edge, same text line — and needs no tolerance
constant. A placeholder dash in a cell of its own sits a column gap away and is left alone.

## What

`reconstruct._reattach_detached_signs(grid, table, words)`, called on the raw `table.extract()`
grid before `_clean_grid` and before the destroyed-token check. Only a trailing sign glyph (U+2212,
en dash, ASCII hyphen) of one cell and a digit-leading next cell, only with contact evidence from
the native words inside each cell's rectangle, never when a cell rectangle is missing.

## Collateral, measured

Every table page in the library (2,614 pages, 1,865 regions), `origin/main` vs this branch:

- pages whose shipped tables changed: **5**, all the affected Fama pages (p488, p537, p542, p551, p700);
- region count changed: 0; cell-separator count changed: 0;
- on those pages, row and cell counts identical; every changed cell is a sign move — 112 single
  moves plus 3 cells in a chain (receiving a sign from the left and passing their own to the right).
- None of the 5 tables is rejected by the destroyed-token check after the repair.

Fable's count also listed one cell in a second document (Forney 1988 p14); this change does not
alter that page's output. Not investigated further — one cell, and the contact criterion did not
fire there.

## Tests

`tests/test_gh887_reattach_detached_signs.py`, 8 tests on the helper with stand-in cell rectangles
and native words — the corpus is copyrighted and the split only arises on particular real
layouts. Mutation seen to fail (out-of-repo copy, import canary): replacing the contact evidence
with `True` fails the placeholder-dash and different-line tests.
