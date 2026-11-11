"""#1282 review fixes (carbon review 3, verified in LAND-1282/CARBON-VERIFY.md).

The repair writes a number into a cell, so every case below asks one thing: is a cell whose
true value is not the written one ever rewritten?  Each wrong-repair shape has a control that
differs in one thing and shows the real repair still happens (a refusal is the safe side, but a
guard that refuses everything would pass the wrong-repair tests too).

- B (finding 1): a block that has to scan a region shared with another block gets no repair.
- Z (finding 2b): a point drop is the same digit string apart from the point.
- N (findings 2a, 3): another page line with the row's exact label that is close to the row
  makes the hit ambiguous.

Calls go through ``propose_page_point_repairs`` with the regions the pipeline passes
(``[b.bbox for b in locate_tables(page)] or None``), on synthetic fitz pages at rotation
0/90/270 (``test_gh1267_row_binding_check.build_page``).  No model, no provider.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

fitz = pytest.importorskip("fitz")

from native_table_fixtures import place  # noqa: E402
from test_gh1267_row_binding_check import (  # noqa: E402
    HEIGHT,
    LINE_PITCH,
    ROTATIONS,
    V0,
    WIDTH,
    build_page,
)

from socr.tables import row_binding_check as rbc  # noqa: E402
from socr.tables.locate import locate_tables  # noqa: E402
from socr.tables.reconcile import find_table_blocks  # noqa: E402
from socr.tables.row_binding_check import (  # noqa: E402
    REFUSE_SHARED_REGION,
    point_dropped,
    propose_page_point_repairs,
)


def block(rows):
    """A markdown table with an empty heading row (nothing to judge) and *rows*."""
    width = 1 + max(len(vals) for _, vals in rows)
    head = "|" + "  |" * width + "\n|" + " --- |" * width
    return head + "\n" + "\n".join("| " + " | ".join([lab, *vals]) + " |" for lab, vals in rows)


def live(page, markdown, regions="locate"):
    """What ``UnifiedPipeline._repair_dropped_points`` calls."""
    if regions == "locate":
        regions = [tuple(b.bbox) for b in locate_tables(page)] or None
    return propose_page_point_repairs(page, markdown, regions)


def cell(markdown, block_index, label, col, occurrence=1):
    """Value cell *col* (0-based among value cells) of the *occurrence*-th row labelled *label*
    in table block *block_index*."""
    lines = markdown.splitlines()
    b = find_table_blocks(markdown)[block_index]
    seen = 0
    for i in range(b.start, b.end + 1):
        cs = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        if cs and cs[0] == label:
            seen += 1
            if seen == occurrence:
                return cs[1 + col]
    raise AssertionError((block_index, label, occurrence))


def raw_box(rotation, v0, v1, u0=60.0, u1=520.0):
    """The raw-page-frame box of the upright band v0..v1 (what ``locate_tables`` returns)."""
    pts = [place(u, v, rotation, WIDTH, HEIGHT) for u in (u0, u1) for v in (v0, v1)]
    return (
        min(p[0] for p in pts),
        min(p[1] for p in pts),
        max(p[0] for p in pts),
        max(p[1] for p in pts),
    )


def one_box_per_table(rotation, bands):
    """Patch for a ``process()`` / offline-tool test on a ruleless synthetic page: the repair
    receives one raw-frame box per upright band in *bands* where ``locate_tables`` found none,
    i.e. the regions the pipeline passes when every table has its own detected box."""
    boxes = [raw_box(rotation, v0, v1) for v0, v1 in bands]
    real = rbc.propose_page_point_repairs

    def _boxed(page, text, regions):
        assert regions is None, regions  # what locate_tables gives the ruleless page
        return real(page, text, boxes)

    return patch.object(rbc, "propose_page_point_repairs", _boxed)


def rules(page, rotation, vs):
    """Booktabs-style full-width horizontal rules at upright heights *vs*."""
    for v in vs:
        a = place(60.0, v, rotation, WIDTH, HEIGHT)
        b = place(520.0, v, rotation, WIDTH, HEIGHT)
        page.draw_line(fitz.Point(*a), fitz.Point(*b), width=0.5)


def why(refusals):
    return {w for _, refs in refusals for f in refs for w in f.why}


# ---------------------------------------------------------------------------
# B (finding 1): two table blocks, one region
# ---------------------------------------------------------------------------

PANEL_A = ("Real GDP", ["-.7", "2.5", "3.0"])
# panel B really prints -7 with the minus a separate word (`-` `7`): no text-layer word is `-7`
PANEL_B_PAGE = ("Real GDP", ["- 7", "2.5", "3.0"])
PANEL_B_SHIPPED = ("Real GDP", ["-7", "2.5", "3.0"])  # the model read it faithfully
F1_NATIVE = [("Percent change", []), PANEL_A, ("Billions", []), PANEL_B_PAGE]
F1_MD = block([PANEL_A]) + "\n\nBillions\n\n" + block([PANEL_B_SHIPPED]) + "\n"

# The same shape with three detached minus signs: panel B's own line now differs from the row in
# three values (signs), so no OTHER guard sees it as the row's own line; only B stands between
# panel A's values and panel B's cells.
PANEL_A3 = ("Real GDP", ["-.7", "-.2", "-.3", "2.5", "3.0"])
PANEL_B3_PAGE = ("Real GDP", ["- 7", "- 2", "- 3", "2.5", "3.0"])
PANEL_B3_SHIPPED = ("Real GDP", ["-7", "-2", "-3", "2.5", "3.0"])
F1_3_NATIVE = [("Percent change", []), PANEL_A3, ("Billions", []), PANEL_B3_PAGE]
F1_3_MD = block([PANEL_A3]) + "\n\nBillions\n\n" + block([PANEL_B3_SHIPPED]) + "\n"


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_b_ruleless_page_panel_b_is_not_rewritten_from_panel_a(rotation):
    page = build_page(F1_NATIVE, rotation)
    assert locate_tables(page) == []  # regions=None: both blocks scan the whole page
    new, reps, refs = live(page, F1_MD)
    assert cell(new, 1, "Real GDP", 0) == "-7", (reps, refs)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_b_one_ruled_box_over_both_panels_panel_b_is_not_rewritten(rotation):
    page = build_page(F1_NATIVE, rotation)
    rules(page, rotation, [V0 - 12, V0 + 3.5 * LINE_PITCH])
    new, reps, refs = live(page, F1_MD)
    assert cell(new, 1, "Real GDP", 0) == "-7", (len(locate_tables(page)), reps, refs)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_b_three_detached_minus_signs_are_not_rewritten_from_panel_a(rotation):
    page = build_page(F1_3_NATIVE, rotation)
    new, reps, refs = live(page, F1_3_MD)
    assert [cell(new, 1, "Real GDP", k) for k in range(3)] == ["-7", "-2", "-3"], (reps, refs)
    assert REFUSE_SHARED_REGION in why(refs)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_b_is_a_named_refusal_of_exactly_the_shared_region(rotation):
    """A correct repair (block 0 prints -.7, the model shipped -7) is refused when two blocks
    share the page, and made when the same block has a box of its own.  Difference only."""
    page = build_page(F1_NATIVE, rotation)
    shipped = (
        block([("Real GDP", ["-7", "2.5", "3.0"])])
        + "\n\nBillions\n\n"
        + block([("Exports", ["1.0", "2.0", "3.0"])])
        + "\n"
    )
    new, reps, refs = live(page, shipped, regions=None)
    assert reps == [] and new == shipped, reps
    assert why(refs) == {REFUSE_SHARED_REGION} and {i for i, _ in refs} == {0, 1}
    own_boxes = [
        raw_box(rotation, V0 - 10, V0 + 1.5 * LINE_PITCH),
        raw_box(rotation, V0 + 2.5 * LINE_PITCH, V0 + 4 * LINE_PITCH),
    ]
    new2, reps2, refs2 = live(page, shipped, regions=own_boxes)
    assert [(r.old, r.new) for _, rs in reps2 for r in rs] == [("-7", "-.7")], refs2
    assert cell(new2, 0, "Real GDP", 0) == "-.7"
    assert REFUSE_SHARED_REGION not in why(refs2)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_b_control_attached_minus_condition_1_holds(rotation):
    """Control: `-7` printed as one word; condition 1 (an exact line on the page) refuses."""
    page = build_page(
        [("Percent change", []), PANEL_A, ("Billions", []), PANEL_B_SHIPPED], rotation
    )
    new, reps, _ = live(page, F1_MD)
    assert reps == [] and new == F1_MD


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_b_control_one_box_per_block_scans_its_own_box(rotation):
    """Control: one box per panel.  Block 1 scans only its own box; panel A is out of reach."""
    page = build_page(F1_3_NATIVE, rotation)
    regions = [
        raw_box(rotation, V0 - 10, V0 + 1.5 * LINE_PITCH),
        raw_box(rotation, V0 + 2.5 * LINE_PITCH, V0 + 4 * LINE_PITCH),
    ]
    new, reps, refs = live(page, F1_3_MD, regions=regions)
    assert [cell(new, 1, "Real GDP", k) for k in range(3)] == ["-7", "-2", "-3"], (reps, refs)
    assert REFUSE_SHARED_REGION not in why(refs)
