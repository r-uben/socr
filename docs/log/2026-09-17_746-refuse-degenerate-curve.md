# GH-746 — a degenerate curve must refuse, not read as a straight run

## What changed

`_item_bbox` correctly bounds a `'c'` item by all four control points, so an
ordinary curve (endpoints level, control points off the level) already yields
a non-degenerate bbox and refuses. But a curve whose four control points are
themselves collinear AND level yields a zero-height bbox indistinguishable
from an ordinary horizontal run by coordinates alone — it was silently read
as data.

Per the ticket's constraint, `Mark(...)` construction (`chart_reader.py:254`)
already discards the item kind, so the refusal cannot live downstream in the
consumer that decomposes marks into runs/risers — by the time a consumer sees
a `Mark`, curve-ness is gone. Fixed it at the `Mark` boundary instead:

- `Mark` gains a `curve: bool = False` field (`chart_reader.py:129`).
- `Mark.horizontal` and `Mark.vertical` both short-circuit to `False` when
  `curve` is set, regardless of what the bbox measures.
- `page_marks` sets `curve=item[0] == "c"` at construction (the one call site
  that builds a `Mark`).

Every consumer of `.horizontal`/`.vertical` (axis-finding, run/riser
collection, the `unread` refusal branch at `chart_reader.py:1373`) inherits
the fix for free — a curve can no longer be mistaken for an axis line, a run,
or a riser, degenerate or not. `_item_bbox`'s four-control-point bound is
untouched, as required (mutation-demonstrated load-bearing in #739).

## Files

- `src/socr/figures/chart_reader.py`
- `tests/test_gh635_chart_reader.py` — two new tests:
  - `test_a_collinear_level_curve_still_refuses` — the #746 fixture: a
    dashed staircase whose one run is replaced by a bezier item whose four
    control points are collinear and level (`build_chart(...,
    dashed_curve_at=1, dashed_curve_level=True)`). Must read `UNRESOLVED`
    with the same "cannot decompose" reason every undecomposable piece
    surfaces.
  - `test_a_plain_run_still_reads` — the other direction: an ordinary dashed
    staircase with no curve item at all still reads its counts.
  - `build_chart` gained a `dashed_curve_level` parameter (default `False`,
    preserves the existing `test_a_curve_inside_a_compound_path_still_refuses`
    ordinary-curve case unchanged).

## Synthetic fixture, necessarily

No corpus page reaches this path (SEP census across 23 pages is `{'l':
1359}`, zero curves) and the repo is public with a copyrighted corpus, so the
fixture is constructed: `page.new_shape().draw_bezier(p1, mid, mid, p2)`
with `mid` left ON the segment's own level (no vertical offset), producing a
`'c'` item whose four points are literally collinear and level — matching the
ticket's measured example (`bbox=(10.0, 50.0, 40.0, 50.0) height=0.0`).

## Test results

Baseline (main@ab18464, measured before any edit, via `git stash` /
`git stash pop` in the same worktree):

    5620 passed, 4 xfailed in 256.05s

After the fix:

    5622 passed, 4 xfailed in 247.16s

Delta is exactly the two new tests; nothing else moved.

## Mutation round

Copied `src/`, `tests/`, and `pyproject.toml` (its `pythonpath = ["src"]`
shadows an external `PYTHONPATH`) to `/tmp/mutant-746`, outside the repo.
Added `tests/test_zz746_mutant_canary.py` asserting `os.path.realpath(socr.
__file__)` starts under the mutant tree's own `os.path.realpath(...)` — a
prefix check that would false-fail on `/tmp` vs `/private/tmp` without
`realpath` on both sides.

Anchor for the mutation: `curve=item[0] == "c",` in `page_marks`. Confirmed
`grep -c` == 1 before mutating. Reverted it to `curve=False,` (the pre-fix
behaviour: kind never reaches `Mark`).

Ran `PYTHONPATH=/tmp/mutant-746/src pytest tests/test_zz746_mutant_canary.py
tests/test_gh635_chart_reader.py -k "canary or curve or plain_run"`:

    1 failed, 3 passed
    FAILED test_a_collinear_level_curve_still_refuses
      AssertionError: ['4', '0', '4', '0', '0'] != {'UNRESOLVED'}

The canary passed (mutant source loaded), the ordinary-curve refusal test
still passed, the plain-run test still passed, and exactly the new guard
reddened — reproducing the ticket's own described failure mode (the
degenerate curve read as a straight run, silently). Deleted `/tmp/mutant-746`
after.

## Confirmation

The refusal reaches `SeriesReading.detail` with the same named reason every
other undecomposable piece uses ("the outline is drawn as a path this reader
cannot decompose ... is neither a horizontal run nor a vertical riser") — not
a silent drop. `presence=PRESENT`, `status=UNRESOLVED` per bin, matching the
existing ordinary-curve test's contract exactly.

## Where I found the orchestrator's framing right, unchanged

Everything in the ticket held: kind is discarded at `Mark` construction, the
fix belongs at or before that boundary, `_item_bbox`'s four-point bound is
correctly out of scope, and the corpus genuinely cannot reach this path
(hence the synthetic-only fixture).

## GH-807 — the #746 guard applied to a legend swatch made the series vanish

Found by review ("Astra"), not by a test: the #746 guard is applied to a
mark's OWN geometry, and a legend swatch is a mark. A dashed series named by
a swatch drawn as a collinear, level curve now failed `read_legend`'s
`m.filled or m.horizontal` test — the same way a data mark does — so
`read_legend` dropped it before it was ever named. No `LegendEntry` means
`read_chart_page` never calls `read_dashed_series` for that name, so no
`SeriesReading` at all was produced: not `UNRESOLVED`, just absent. That is
worse in kind than the bug #746 fixed — a wrong count is at least a count;
this was nothing, in either the published markdown or the JSON metadata.

### Fix

Not exempting legend swatches from the curve guard (that would reinstate the
original #746 defect one scope up: a curved swatch would again pass through
as an ordinary horizontal run). Instead, `read_legend`'s per-mark loop now
distinguishes "this mark is a riser, exclude it" from "this mark is
undecomposable, but it may still be the swatch a row names":

- `shape_unreadable = not m.filled and not m.horizontal and not m.vertical` —
  true only for a mark that is neither filled, nor a run, nor (now
  correctly, per #746) a curve masquerading as a riser. Vertical risers keep
  their existing exclusion unchanged (`m.vertical` stays `True` for them,
  so `shape_unreadable` stays `False` and the early `continue` still fires —
  the #735 riser-vs-tick-label fix is untouched).
- A `shape_unreadable` mark is carried through naming (same row-adjacency
  logic as any swatch) instead of being dropped immediately. If it ends up
  named, it is appended to a new `unresolved: list[tuple[str, str]]` output
  — `(name, detail)` — instead of becoming a `LegendEntry`.
- `read_legend` now returns `(entries, swatches, unresolved)`; its one caller
  (`read_chart_page`) threads `unresolved` through a per-region
  `unresolved_swatches` dict, and — in the panel-building loop, before the
  "nothing was read" refusal check — appends
  `SeriesReading(name=name, style=DASHED_STROKE,
  presence=PRESENCE_UNRESOLVED, detail=detail)` for every unresolved name
  the group's legend-donor region produced that a confident entry didn't
  already cover.

This reuses two surfaces that already handle non-`PRESENT` series generically
and needed no new rendering code: `panel_block()` already prints
`Series "name": presence — detail.` for every series whose presence isn't
`PRESENT`, and `PanelReading.to_dict()["series"]` always serializes every
`SeriesReading` regardless of presence. The bug was that neither surface
ever received the series to render — not that either surface mishandled it.

### Files

- `src/socr/figures/chart_reader.py` — `read_legend`, `read_chart_page`.
- `tests/test_gh635_chart_reader.py` —
  `test_an_unresolvable_legend_swatch_surfaces_instead_of_vanishing`, plus a
  `dashed_swatch_as_curve` parameter on `build_chart` that draws the
  legend's own dashed swatch as a degenerate bezier in place of a straight
  `draw_line`.

### Test evidence

The new test asserts at both boundaries required: (1) `read_chart_page`'s
output — `panel.series` contains a `SeriesReading` for the dashed name with
`presence == PRESENCE_UNRESOLVED` and a non-empty `detail`, alongside the
solid series still reading `PRESENT` (the panel is not refused outright); and
(2) the published-artifact boundary — the name and `PRESENCE_UNRESOLVED`
both appear in `panel_block(panel)`'s markdown, and `panel.to_dict()`'s
`series` list carries the same name with `presence == "unresolved"`.

Full `test_gh635_chart_reader.py`: 51 passed before this fix's test was
added, 52 passed after (the new test, nothing else moved).

### Mutation round

Copied `src/`, `tests/`, `pyproject.toml` to `/tmp/mutant-807`, outside the
repo. Added a realpath-based canary (`test_zz_mutant_canary.py`) confirming
`socr.__file__` resolves under the mutant tree, not the real editable
install; it passed.

Anchor: `shape_unreadable = not m.filled and not m.horizontal and not
m.vertical` in `read_legend` — confirmed `grep -c` == 1 before mutating.
Reverted to the pre-fix behaviour (`shape_unreadable = False`; the early
`continue` restored to its original unconditional `if not (m.filled or
m.horizontal): continue`, dropping a curve-shaped dashed swatch the same way
it did before #807's fix).

    1 failed, 52 passed
    FAILED test_an_unresolvable_legend_swatch_surfaces_instead_of_vanishing
      AssertionError: the dashed series vanished instead of being surfaced
      assert None is not None

Exactly the reported failure mode reproduced, exactly the one test reddened
(the #746 curve/plain-run tests and everything else stayed green). Deleted
`/tmp/mutant-807` after.

### Two corrections to record, per review

1. **"Degenerate" undersells the guard's actual reach.** `Mark.horizontal`
   and `.vertical` were never gated on exactly zero height/width — they were
   always gated on `<= self.tolerance`, where `tolerance` is derived from the
   stroke's own width (half the stroke width; see `Mark.tolerance`). The
   #746 guard therefore refuses any curve whose bbox falls within that
   stroke-width-derived band, not literally only a bbox that measures
   `0.0` — the fixture happens to draw an exact `0.0` because that is the
   simplest way to construct the case, but the guard's blast radius is
   wider than "exactly degenerate." Worth stating precisely rather than
   letting "degenerate" imply an exact-zero special case.
2. **The guard is best-effort, not a complete classification of every
   originally-curved PDF operator.** `item[0] == "c"` at the point
   `_item_bbox`/`page_marks` inspect it is what MuPDF's `get_drawings()`
   hands back — and MuPDF is free to have already simplified some
   genuinely-degenerate Bezier curves into `'l'` items upstream, before
   they are ever seen as `'c'`. Where that happens, this guard cannot see
   the curve at all: it classifies what MuPDF chose to report, not what
   the original PDF content stream drew. This does not weaken the fix (the
   corpus census confirms zero curves survive to this reader today, so no
   known case is affected), but it means the guard should not be read as
   "socr refuses every degenerate curve a PDF could contain" — only every
   one MuPDF still reports as `'c'` by the time it reaches this module.
