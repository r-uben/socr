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
