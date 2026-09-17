# GH-808 — extend the #746/#807 guard to `'qu'` (quad) items

## What changed

`Mark.curve: bool = False` is renamed `Mark.asserts_segment: bool = True` — a
property of the drawing operator, not an operator name, per the review note:
"the operator does not assert a straight segment" covers `'c'` (curve) and
`'qu'` (quad) alike, and a future multi-point operator joins the same flag
rather than needing a third one. `page_marks` sets
`asserts_segment=item[0] not in ("c", "qu")` at the one `Mark(...)`
construction site (keyword argument, so the rename shifts nothing for any
other field). `Mark.horizontal`/`.vertical` check `not self.asserts_segment`
in place of the old `self.curve` check — same short-circuit, same effect on
`'c'` items, now also applied to `'qu'`.

## Why this is the same hole, geometrically

`_item_bbox` bounds a `'qu'` item by its `.rect` — the bounding rect of all
four corners, exactly analogous to how it bounds `'c'` by all four control
points. Four corners that are themselves collinear and level yield a
zero-height bbox indistinguishable, by coordinates alone, from an ordinary
horizontal run — the #746 case, one operator over.

## Measured, not assumed: why extending is safe

1. **The census the ticket asked for.** Across both corpora (`in`: 23 pages,
   `in-minutes`: 175 pages) `'qu'` items are 100% inert: 195 real quads
   found, all in `in-minutes`, 0 filled, 0 dashed, 0 with a near-degenerate
   bbox. No corpus page currently reaches the new refusal path.
2. **Every consumer of `.horizontal`/`.vertical` was traced, not guessed at
   from the curve case:**
   - `_stroked_horizontals` (axis candidates, `chart_reader.py:315`) already
     excludes every filled mark (`not m.filled`) before consulting
     `.horizontal` — irrelevant to a filled quad bar.
   - `read_legend`'s swatch admission test is `m.filled or m.horizontal` — an
     `or`; a filled quad swatch is admitted regardless of what
     `.horizontal` says.
   - `_resting_bars`/`read_solid_series` — the actual bar reader — select a
     filled mark by bbox resting on the axis and `.filled` alone. Neither
     consults `.horizontal`/`.vertical` at all. Pinned directly:
     `test_a_filled_non_segment_mark_still_rests_as_a_bar`.
   So a filled quad drawn as a bar was never at risk from this change, by
   construction of the consumers — not by the absence of a corpus example.
3. **PyMuPDF itself never emits a filled `'qu'` item.** Measured directly
   (`shape.draw_quad(...)`, `shape.finish(fill=..., width=0)`): a filled
   quad is always decomposed into its four `'l'` edges at draw time — the
   `'qu'` operator is only kept for an UNFILLED (stroke-only) quad. This is
   consistent with, and upstream of, the corpus census: a filled quad
   reaching this reader as `'qu'` is not merely rare, it does not occur for
   any generator PyMuPDF can parse. `test_a_degenerate_quad_does_not_assert_a_segment`
   constructs the degenerate case the only way it can occur — unfilled,
   dashed, four distinct collinear-level corners (duplicate corners
   simplify to `'l'` items before reaching this reader, measured, and would
   test PyMuPDF's own upstream simplification rather than this guard).

## Files

- `src/socr/figures/chart_reader.py` — `Mark`, `page_marks`.
- `tests/test_gh635_chart_reader.py` — two new tests:
  - `test_a_degenerate_quad_does_not_assert_a_segment` — a stroked, dashed
    quad with four distinct, collinear, level corners; asserts the resulting
    `Mark` has `asserts_segment=False` and both `.horizontal`/`.vertical`
    False, mirroring the #746 curve tests.
  - `test_a_filled_non_segment_mark_still_rests_as_a_bar` — constructs a
    `Mark(filled=True, asserts_segment=False)` directly (no PDF fixture
    exists that reaches this reader that way, per the finding above) and
    asserts `_resting_bars` still selects it. This is the "would extending
    the guard cost a legitimately-read filled bar" question, answered
    mechanically rather than left to a case the corpus (and PyMuPDF) cannot
    produce.

## Test results

`test_gh635_chart_reader.py`: 52 passed before this change's two new tests
(the #807 count), 54 passed after. Full suite (measured):

    5625 passed, 4 xfailed

(5623 after #807 + 2 new tests here.)

`uvx ruff@0.16.0 format --check .`: clean, 733 files.

## Mutation round

Copied `src/`, `tests/`, `pyproject.toml` to `/tmp/mutant-808`, outside the
repo. Added the same realpath canary as the #807 round; confirmed it passed
against this mutant tree's own source.

Anchor: `asserts_segment=item[0] not in ("c", "qu"),` in `page_marks` —
confirmed `grep -c` == 1 before mutating. Reverted to
`asserts_segment=item[0] != "c",` (the pre-#808 behaviour: only `'c'`
refuses, `'qu'` still asserts a segment).

    1 failed, 54 passed
    FAILED test_a_degenerate_quad_does_not_assert_a_segment
      AssertionError: [Mark(filled=False, dashed=True, width=2.0,
      x0=10.0, y0=50.0, x1=40.0, y1=50.0, asserts_segment=True)]
      assert 0 == 1

Exactly the intended test reddened — the degenerate quad's `Mark` reverts to
`asserts_segment=True`, which is precisely the pre-#808 hole (it would read
as an ordinary horizontal run). Every other test, including #746's and
#807's curve tests, stayed green. Deleted `/tmp/mutant-808` after.

## Scope note

This does not touch `_item_bbox`'s quad handling (`item[1].rect`, already
correct — analogous to the untouched four-control-point curve bound) and
does not add a new refusal *message*: a degenerate quad now refuses through
the same `unread`/`shape_unreadable` paths #746 and #807 already wired up,
with the same named reasons ("neither a horizontal run nor a vertical
riser" / "a shape this reader cannot decompose"), since those paths key off
`.horizontal`/`.vertical`/`.filled`, not off which operator produced the
`Mark`.
