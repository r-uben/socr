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
