# VI-C2a — row-band / column-edge / ordinal-origin helpers

Base: `main@a601728`. Branch `feat/vi-C2a-line-bands`. Design:
`docs/plans/verifier-independence/logs/2026-09-05_C1-design.md` §(a) (rev 4).

## What changed

`src/socr/tables/locate.py`:

- `RowBand` (frozen dataclass: `y0`, `y1`, `source` ∈ `{"rule", "line"}`).
- `row_bands_from_rules(rules, region)` — pairs consecutive horizontal rules inside
  `region` into bands.
- `row_bands_from_lines(page, region)` — groups PDF text lines inside `region` into bands;
  two lines merge into one band when their baselines differ by less than the smaller of
  their two font sizes (C1 §(a)'s row-band separator).
- `row_bands(page, region)` — the C1 §(f) decision-1 dispatcher (rules-else-lines): trusts
  `row_bands_from_rules`'s output only when it corresponds 1:1 with `row_bands_from_lines`'s
  (same count, each rule band contains its line-band counterpart); otherwise returns the
  line bands. No text layer ⇒ `[]` (abstain input, never a guess).
- `ordinal_origin(page, region)` — the y of the **second** horizontal-rule group in
  `region`. Two rules are one drawn border (e.g. booktabs' doubled `\toprule`) when their
  gap is smaller than **half the smallest font size of any text line in the region** — a
  page-derived quantity, not a literal. That threshold was picked because the actual
  doubled-rule gap on the frozen corpus (2.39–2.50 pt across doc01/doc02/doc03, three
  different table styles) sits far below any genuinely distinct rule gap (≥16.87 pt) and
  far below half of any body font size measured on those pages (~4 pt at an 8 pt body
  font) — the individual rule's own stroke width (0.25–0.94 pt) does **not** span that
  gap and was tried first; it does not reproduce C1's measured origins (see below). A
  region with rules but no text falls back to the widest rule's own thickness.
  `None` when fewer than two rules, or fewer than two groups, exist.
- `label_column_edge(page, region)` — `R = min x0` over every non-leftmost text line of
  any printed row, shrunk to a whitespace edge (iterate while some line straddles `R`).
  `None` on a one-column region or when no such `R > region.x0` exists.
- `band_index_for(bands, y_mid)` — index of the band containing `y_mid`, else `None`.
- `_horizontal_rules` is now a thin wrapper over the new `_horizontal_rules_with_thickness`
  (keeps every existing caller's 3-tuple contract byte-identical; `agentic.py`,
  `header_cut.py`, `test_header_cut.py` untouched).

`src/socr/tables/binding.py`: `BindingResult` gains two read-only fields,
`native_rows: list[_NativeRow]` and `row_binding: dict[int, int]` — `bind()`'s own
`_native_rows` output and `_bind_rows` mapping, set once right after they are computed, on
every return path. No new computation, no geometry, nothing else in `binding.py` changed.

## Why the "rule thickness" reading in the design text was revised

C1 §(a) reads "rules closer than a rule thickness are one drawn border." Tried literally
(merge iff `gap < max(own stroke widths)`) first — it does **not** reproduce the design's
own measured origin on doc01 (0.50 pt stroke width vs a 2.50 pt gap between the doubled
top rules: `2.50 < 0.50` is false, so the pair never merges, and the "second group" becomes
the doubled rule's own second line at y=84.05, not the cmidrule at y=116.30 the design
reports). Reverse-engineered against all three doubled-rule tables in the corpus
(doc01, doc02 p3/p4, doc03): the actual merge gap is consistently 2.39–2.50 pt regardless
of the individual rule's stroke width (0.25–0.94 pt across those tables), while the next
distinct rule gap is always ≥16.87 pt. Half of the smallest body font size in the region
(~4 pt at 8 pt type) sits cleanly inside that gap on every one of the three tables, and
also on doc05/doc07 (no doubled rule — smallest real gap there is 24.09 pt, well above any
font-derived threshold, so nothing wrongly merges). Documented in the docstring; flagging
this as a design-text correction the reviewer should confirm.

## Verification

- `tests/test_locate_line_bands.py` (new): 4 synthetic-fixture tests (ruled fixture, one
  band per row = `row_bands_from_rules`'s bands; booktabs fixture, one band per row from
  lines with `ordinal_origin` = the midrule; no-rules fixture, `ordinal_origin` is `None`;
  one-column fixture, `label_column_edge` is `None`) plus `band_index_for` and a
  frozen-corpus check (`@pytest.mark.skipif` when the corpus dir is absent — CI has no
  corpus). Locally, with `~/Data/socr/ladder-run2-2026-09-04` present: **6 passed**,
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
  not an exact-count regression guard).

- A1 harness byte-identity: ran `python -m socr.benchmark.replay_binding
  ~/Data/socr/ladder-run2-2026-09-04` from a throwaway detached worktree pinned at
  `main@a601728` (this branch's own parent — the team-lead-suggested
  `.worktrees/socr-vi-B2` pin, `4c1b284`/PR #594, predates **A2** (#602), so diffing
  against it conflates A2's already-merged, expected effect with C2a's; used the correct
  base instead) and from this worktree: **diff is empty**.
- Full suite: `4139 passed, 4 xfailed` (`~/venvs/socr/bin/pytest tests/ -q`,
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
