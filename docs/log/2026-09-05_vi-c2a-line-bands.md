# VI-C2a — row-band / column-edge / ordinal-origin helpers

Base: `origin/main@cc1c632` (rebased off `a601728`). Branch `feat/vi-C2a-line-bands`. Design:
`docs/plans/verifier-independence/logs/2026-09-05_C1-design.md` §(a) (rev 4).

## What changed

`src/socr/tables/locate.py`:

- `RowBand` (frozen dataclass: `y0`, `y1`, `source` ∈ `{"rule", "line"}`).
- `row_bands_from_rules(rules, region)` — pairs consecutive horizontal rules inside
  `region` into bands.
- `row_bands_from_lines(page, region)` — groups PDF text lines inside `region` into bands.
  Two lines join only when their boxes overlap in y by more than the overlap any two
  adjacent printed rows share. Adjacent rows are consecutive unique-baseline groups at
  the region's line pitch (the modal unique-baseline gap). Fewer than three lines
  cannot establish a pitch → `[]` (abstain).
- `row_bands(page, region)` — the C1 §(f) decision-1 dispatcher (rules-else-lines): trusts
  `row_bands_from_rules`'s output only when it corresponds 1:1 with `row_bands_from_lines`'s
  (same count, each rule band contains its line-band counterpart); otherwise returns the
  line bands. No text layer ⇒ `[]` (abstain input, never a guess).
- `ordinal_origin(page, region)` — the y of the **second** horizontal-rule group in
  `region`. Two consecutive rules are one drawn border iff (a) no text-line baseline in
  the region lies between them **and** (b) their gap is smaller than the smallest
  text-line height in the region (a gap no printed line could fit in). No text lines
  → `None` (cannot certify an origin). `None` also when fewer than two rules or two
  groups exist.
- `label_column_edge(page, region)` — `R = min x0` over every non-leftmost text line of
  any printed row, shrunk to a whitespace edge (iterate while some line straddles `R`).
  `None` on a one-column region, when no such `R > region.x0` exists, or when `R`
  collapses onto the leftmost text (a wrapped label's own `x0` is not a column edge).
- `band_index_for(bands, y_mid)` — index of the unique band containing `y_mid`; `None`
  if none or more than one contains the point (abstain).
- `_horizontal_rules` keeps its 3-tuple contract; the thickness 4-tuple plumbing is gone
  (`agentic.py`, `header_cut.py`, `test_header_cut.py` untouched).

`src/socr/tables/binding.py`: `BindingResult` gains two read-only fields,
`native_rows: list[_NativeRow]` and `row_binding: dict[int, int]` — `bind()`'s own
`_native_rows` output and `_bind_rows` mapping, set once right after they are computed, on
every return path. No new computation, no geometry, nothing else in `binding.py` changed.

## Why two rules are one drawn border

C1 §(a) reads "rules closer than a rule thickness are one drawn border." Stroke width
fails on doc01 (0.50 pt stroke vs a 2.50 pt doubled-rule gap). Half the smallest body
font is a corpus-fitted constant (splits a 4 pt pair on 6 pt type; merges distinct 8 pt
rules on 24 pt type). A ratio jump on the sorted gaps needs a distribution: with exactly
two gaps it always declares a class break, so a plain 3-rule 15/45 table merges top+mid
and returns the bottomrule as origin.

The criterion that does not need a distribution: two consecutive rules are one drawn
border iff (a) no text-line baseline in the region lies between them **and** (b) their
gap is smaller than the smallest text-line height in the region — a gap no printed line
could fit in. Otherwise they are distinct. If neither condition can be evaluated (no
text lines) `ordinal_origin` returns `None` (abstain), never a guess. Doubled booktabs
pairs on the frozen corpus have no baseline between them and a 2.4–2.5 pt gap below
every line height on the page; every other consecutive pair has a baseline between them
and/or a gap larger than a line. C1's measured origins are unchanged.

Row bands use the same page-derived discipline: two lines share a band only when their
boxes overlap in y by more than the overlap adjacent printed rows share, adjacent being
consecutive unique-baseline groups at the modal unique-baseline gap. A uniform 9.5 pt
pitch on 10 pt type is six printed rows, six bands — not three.

## Verification

- `tests/test_locate_line_bands.py`: ruled / booktabs / no-rules / one-column fixtures,
  `band_index_for` (including overlapping bands → `None`), wrapped-label `R` is `None`,
  6 pt doubled pair → header-rule origin, 24 pt distinct rules with a baseline between
  them → second-rule origin, 3-rule 15/45 → midrule, doubled 2.5+20 → second group,
  2.5/3.0 only → `None`, 6 rows at 9.5 pt on 10 pt type → 6 bands, subscript under a
  label → same band as the label. Frozen-corpus check included. Origins:

  | table | origin (computed) | origin (C1 §(a)) |
  |---|---|---|
  | doc01 p2 | 116.30 | 116.3 |
  | doc02 p3 | 123.90 | 123.9 |
  | doc02 p4 | 123.90 | 123.9 |
  | doc03 p1 | 241.50 | 241.5 |
  | doc04 p3 | None (scanned, no vector rules) | None |
  | doc05 p1 | 121.0 (120.97) | 121.0 |
  | doc07 p1 | 121.0 (120.97) | 121.0 |

  `row_bands` returned a non-empty band list for all 7. The overlap-vs-pitch rule left
  doc04 p3 at 12 bands and doc05 p1 at 17 — unchanged; those two counts are now pinned
  in the corpus test.

- A1 harness byte-identity: ran `python -m socr.benchmark.replay_binding
  ~/Data/socr/ladder-run2-2026-09-04` from a throwaway detached worktree pinned at
  `origin/main@cc1c632` and from this worktree: **diff is empty**.
- Full suite: `4148 passed, 4 xfailed` (`~/venvs/socr/bin/pytest tests/ -q`,
  `PYTHONPATH=<worktree>/src`).
- `uvx ruff@0.16.0 format --check .`: clean (576 files).

## Deviations from the ticket text (superseded by the design doc per dispatch instructions)

- Ticket text names `row_bands(page, region)` / `label_column_edge(bands, region)` /
  `ordinal_origin(page, region)`; the design's "New helpers for C2a" block (§(a)) is more
  specific and was implemented verbatim (`row_bands_from_rules`, `row_bands_from_lines`,
  `row_bands` dispatcher, `ordinal_origin`, `label_column_edge`, `band_index_for`) — all
  present, ticket's `row_bands`/`label_column_edge` names covered.
- `BindingResult.native_rows` element type is `_NativeRow` (module-private dataclass) —
  exposed as-is, per the ticket's "no new computation" constraint; consumers read its
  public-shaped fields (`row_path`, `label_bbox`, `lane_bboxes`, `multiset`, `y`, …).
