# VI-C2a — row-band / column-edge / ordinal-origin helpers

Base: `main@a601728`. Branch `feat/vi-C2a-line-bands`. Design:
`docs/plans/verifier-independence/logs/2026-09-05_C1-design.md` §(a) (rev 4).

## What changed

`src/socr/tables/locate.py`:

- `RowBand` (frozen dataclass: `y0`, `y1`, `source` ∈ `{"rule", "line"}`).
- `row_bands_from_rules(rules, region)` — pairs consecutive horizontal rules inside
  `region` into bands.
- `row_bands_from_lines(page, region)` — groups PDF text lines inside `region` into bands;
  a new line joins the current band when its baseline is within the smaller of (the
  band's first/anchor font size, its own) of that *anchor* baseline, not of the previous
  line (C1 §(a)'s row-band separator, non-transitive).
- `row_bands(page, region)` — the C1 §(f) decision-1 dispatcher (rules-else-lines): trusts
  `row_bands_from_rules`'s output only when it corresponds 1:1 with `row_bands_from_lines`'s
  (same count, each rule band contains its line-band counterpart); otherwise returns the
  line bands. No text layer ⇒ `[]` (abstain input, never a guess).
- `ordinal_origin(page, region)` — the y of the **second** horizontal-rule group in
  `region`. Two rules are one drawn border (e.g. booktabs' doubled `\toprule`) when their
  gap sits on the small side of a **natural break in the region's own consecutive
  inter-rule gaps**: sort the gaps, collapse float-twin copies of one magnitude, and
  split at the largest ratio jump that actually separates two gap classes. A single gap
  class (no doubled rules) has no break and nothing merges. No font-size input, no
  literal point threshold. `None` when fewer than two rules, or fewer than two groups,
  exist.
- `label_column_edge(page, region)` — `R = min x0` over every non-leftmost text line of
  any printed row, shrunk to a whitespace edge (iterate while some line straddles `R`).
  `None` on a one-column region, when no such `R > region.x0` exists, or when `R`
  collapses onto the leftmost text (a wrapped label's own `x0` is not a column edge).
- `band_index_for(bands, y_mid)` — index of the band containing `y_mid`, else `None`.
- `_horizontal_rules` is now a thin wrapper over the new `_horizontal_rules_with_thickness`
  (keeps every existing caller's 3-tuple contract byte-identical; `agentic.py`,
  `header_cut.py`, `test_header_cut.py` untouched).

`src/socr/tables/binding.py`: `BindingResult` gains two read-only fields,
`native_rows: list[_NativeRow]` and `row_binding: dict[int, int]` — `bind()`'s own
`_native_rows` output and `_bind_rows` mapping, set once right after they are computed, on
every return path. No new computation, no geometry, nothing else in `binding.py` changed.

## Why the merge distance is a natural break in the region's own rule gaps

C1 §(a) reads "rules closer than a rule thickness are one drawn border." Tried literally
(merge iff `gap < max(own stroke widths)`) first — it does **not** reproduce the design's
own measured origin on doc01 (0.50 pt stroke width vs a 2.50 pt gap between the doubled
top rules: `2.50 < 0.50` is false, so the pair never merges, and the "second group" becomes
the doubled rule's own second line at y=84.05, not the cmidrule at y=116.30 the design
reports). A second reading — half the smallest body font size in the region — reproduces
the corpus but is a corpus-fitted constant: on a 6 pt table with a 4 pt doubled-rule gap
it splits the pair (origin lands on the second hairline); on a 24 pt table with two
distinct rules 8 pt apart it merges them (origin disappears). The merge distance is
therefore taken from the region's **own consecutive inter-rule gaps** only: sort them,
collapse float-twin copies of one magnitude (a doubled pair measured at both top and
bottom), and split at the largest ratio jump that actually separates two gap classes. A
small-end jump is a class break only when it dominates later jumps (`r0 > max(later)²`);
when a later jump is larger (a huge body span, doc02) the search recurses on the small
side. A single gap class (doc05/doc07 — no doubled rules; or two distinct rules with
nothing else) has no break and nothing merges. No font-size input, no literal. This
reproduces C1's measured origins and both constructed counterexamples.

## Verification

- `tests/test_locate_line_bands.py`: original 4 synthetic-fixture tests (ruled / booktabs /
  no-rules / one-column) plus `band_index_for`, the frozen-corpus check, and four
  reviewer-gap tests (6 pt doubled-rule pair → origin is the header rule; 24 pt two
  distinct rules 8 pt apart → origin is the second rule; 6 rows at a 9.5 pt step on
  10 pt type → 6 bands under the anchor rule; wrapped-label row → `label_column_edge`
  is `None`). Locally, with `~/Data/socr/ladder-run2-2026-09-04` present: **10 passed**,
  including the corpus check, which reproduces C1's exact measured origins:

  | table | origin (computed) | origin (C1 §(a)) |
  |---|---|---|
  | doc01 p2 | 116.30 | 116.3 |
  | doc02 p3 | 123.90 | 123.9 |
  | doc02 p4 | 123.90 | 123.9 |
  | doc03 p1 | 241.50 | 241.5 |
  | doc04 p3 | None (scanned, no vector rules) | None |
  | doc05 p1 | 121.0 (120.97) | 121.0 |
  | doc07 p1 | 121.0 (120.97) | 121.0 |

  `row_bands` returned a non-empty band list for all 7 (`band_count > 0` asserted; no
  reference band-count table exists in C1 §(d) to pin exactly, so this is a sanity bound,
  not an exact-count regression guard). The anchor clustering rule left doc04 p3 at 12
  bands and doc05 p1 at 17 — unchanged from pairwise-adjacent clustering.

- A1 harness byte-identity: ran `python -m socr.benchmark.replay_binding
  ~/Data/socr/ladder-run2-2026-09-04` from a throwaway detached worktree pinned at
  `main@a601728` (this branch's own parent — the team-lead-suggested
  `.worktrees/socr-vi-B2` pin, `4c1b284`/PR #594, predates **A2** (#602), so diffing
  against it conflates A2's already-merged, expected effect with C2a's; used the correct
  base instead) and from this worktree: **diff is empty**.
- Full suite: `4143 passed, 4 xfailed` (`~/venvs/socr/bin/pytest tests/ -q`,
  `PYTHONPATH=<worktree>/src`).
- `uvx ruff@0.16.0 format --check .`: clean (575 files).

## Deviations from the ticket text (superseded by the design doc per dispatch instructions)

- Ticket text names `row_bands(page, region)` / `label_column_edge(bands, region)` /
  `ordinal_origin(page, region)`; the design's "New helpers for C2a" block (§(a)) is more
  specific and was implemented verbatim (`row_bands_from_rules`, `row_bands_from_lines`,
  `row_bands` dispatcher, `ordinal_origin`, `label_column_edge`, `band_index_for`) — all
  present, ticket's `row_bands`/`label_column_edge` names covered.
- `BindingResult.native_rows` element type is `_NativeRow` (module-private dataclass) —
  exposed as-is, per the ticket's "no new computation" constraint; consumers read its
  public-shaped fields (`row_path`, `label_bbox`, `lane_bboxes`, `multiset`, `y`, …).
