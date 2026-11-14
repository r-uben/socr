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
    page = build_page(
        [("Percent change", []), PANEL_A, ("Billions", []), ("Exports", ["1.0", "2.0", "3.0"])],
        rotation,
    )
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


# ---------------------------------------------------------------------------
# Z (finding 2b): a point drop keeps every digit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shipped,native",
    [("70", ".070"), ("70", "0.070"), ("7", ".07"), ("7", ".007"), ("-7", "-.07")],
)
def test_z_a_zero_lost_after_the_point_is_not_a_point_drop(shipped, native):
    """`.070` read as `70` loses the zero after the point as well as the point."""
    assert not point_dropped(rbc._canon_number(shipped), rbc._canon_number(native))


@pytest.mark.parametrize(
    "shipped,native",
    [
        ("-7", "-.7"),
        ("-7", "-0.7"),
        ("4", ".4"),
        ("12", "1.2"),
        ("70", "7.0"),
        ("125", "1.25"),
        ("125", "12.5"),
        ("-2.", "-.2"),
    ],
)
def test_z_control_a_point_drop_by_design_is_still_one(shipped, native):
    assert point_dropped(rbc._canon_number(shipped), rbc._canon_number(native))


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_z_a_row_whose_line_has_a_zero_after_the_point_is_not_repaired(rotation):
    """Through the entry: the page prints `.07`, the model shipped `7`.  Writing `.07` would
    rest on a pair that is not a point drop; the row is refused.  Control: `.7` is repaired."""

    def case(printed):
        page = build_page(
            [("Real GDP", ["1.1", "2.2", "3.3"]), ("Rate", [printed, "1.5", "2.5"])], rotation
        )
        shipped = block([("Real GDP", ["1.1", "2.2", "3.3"]), ("Rate", ["7", "1.5", "2.5"])]) + "\n"
        return live(page, shipped)

    new, reps, refs = case(".07")
    assert reps == [] and cell(new, 0, "Rate", 0) == "7", (reps, refs)
    new2, reps2, _ = case(".7")
    assert cell(new2, 0, "Rate", 0) == ".7", reps2


# ---------------------------------------------------------------------------
# N (findings 2a, 3): a repeated row label, the row's own line not a hit
# ---------------------------------------------------------------------------
# Fed tables run `Real GDP / Previous / Final sales / Previous` with no valueless line between, and
# on a page where the model drops points the rows in between are not BOUND either, so no reading-
# order anchor separates the two `Previous` lines.


def noheads(own, own_shipped, sib, sib_shipped):
    native = [
        ("Real GDP", ["-.4", "1.1", "1.2"]),
        ("Previous", own),
        ("Final sales", ["-.5", "5.1", "5.2"]),
        ("Previous", sib),
    ]
    shipped = (
        block(
            [
                ("Real GDP", ["-4", "1.1", "1.2"]),  # the model dropped this point too
                ("Previous", own_shipped),
                ("Final sales", ["-5", "5.1", "5.2"]),  # and this one
                ("Previous", sib_shipped),
            ]
        )
        + "\n"
    )
    return native, shipped


OWN_LINES = pytest.mark.parametrize(
    "own",
    [["- 7", "2.5", "3.0"], ["-7.1", "2.5", "3.0"]],
    ids=["detached_minus", "model_misread_-7.1"],
)


@pytest.mark.parametrize("rotation", ROTATIONS)
@OWN_LINES
def test_n_section_1_previous_is_not_overwritten_from_section_2(rotation, own):
    """Section 1 `Previous` really prints `- 7` (or `-7.1`); section 2 `Previous` prints `-.7`
    with the same other values.  `-.7` is section 2's number."""
    native, shipped = noheads(
        own, ["-7", "2.5", "3.0"], ["-.7", "2.5", "3.0"], ["-.7", "2.5", "3.1"]
    )
    new, reps, refs = live(build_page(native, rotation), shipped)
    assert cell(new, 0, "Previous", 0) == "-7", (reps, refs)
    assert rbc.REFUSE_RIVAL_LINE in why(refs)


@pytest.mark.parametrize("rotation", ROTATIONS)
@OWN_LINES
def test_n_the_correct_repairs_around_it_survive(rotation, own):
    """Once `Previous` is not bound to section 2's line, `Real GDP` and `Final sales` are
    repaired from their own lines (before the fix the wrong binding refused `Final sales`)."""
    native, shipped = noheads(
        own, ["-7", "2.5", "3.0"], ["-.7", "2.5", "3.0"], ["-.7", "2.5", "3.1"]
    )
    new, reps, refs = live(build_page(native, rotation), shipped)
    assert cell(new, 0, "Real GDP", 0) == "-.4", (reps, refs)
    assert cell(new, 0, "Final sales", 0) == "-.5", (reps, refs)


@pytest.mark.parametrize("rotation", ROTATIONS)
@pytest.mark.parametrize("true,sib", [("70", "7.0"), ("125", "1.25"), ("125", "12.5")])
def test_n_a_true_integer_is_not_rewritten_from_a_sibling_with_a_point(rotation, true, sib):
    """Section 1 `Previous` prints *true* and the model read it right but misread another cell,
    so its own line is no hit; section 2 prints *sib*, the same digits with a point."""
    native, shipped = noheads(
        [true, "2.5", "3.1"], [true, "2.5", "3.0"], [sib, "2.5", "3.0"], [sib, "2.5", "3.2"]
    )
    new, reps, refs = live(build_page(native, rotation), shipped)
    assert cell(new, 0, "Previous", 0) == true, (reps, refs)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_n_a_clipped_box_does_not_write_the_siblings_12_5_for_1_25(rotation):
    """Finding 3: section 1 prints 1.25 (the model read 125); section 2 prints 12.5.  The box
    starts below section 1's line, so section 2's line is the only hit inside it; section 1's
    own line is still on the page, a point-only hit, so the row is refused."""
    native, shipped = noheads(
        ["1.25", "2.5", "3.0"],
        ["125", "2.5", "3.0"],
        ["12.5", "2.5", "3.0"],
        ["12.5", "2.5", "3.1"],
    )
    page = build_page(native, rotation)
    clipped = [raw_box(rotation, V0 + 1.5 * LINE_PITCH, V0 + 4 * LINE_PITCH)]
    new, reps, refs = live(page, shipped, regions=clipped)
    assert cell(new, 0, "Previous", 0) == "125", (reps, refs)
    assert rbc.REFUSE_RIVAL_LINE in why(refs)
    # control: unclipped, both lines are hits and the row is refused as ambiguous
    new2, reps2, refs2 = live(page, shipped)
    assert cell(new2, 0, "Previous", 0) == "125", (reps2, refs2)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_n_control_a_section_heading_row_between_still_holds(rotation):
    """The review's own 2a shape (a valueless heading row between the sections) was already
    refused by condition 6; it stays refused."""
    native = [
        ("GDP", []),
        ("Previous", ["- 7", "2.5", "3.0"]),
        ("Unemployment", []),
        ("Previous", ["-.7", "2.5", "3.0"]),
    ]
    shipped = (
        block(
            [
                ("GDP", []),
                ("Previous", ["-7", "2.5", "3.0"]),
                ("Unemployment", []),
                ("Previous", ["-.7", "2.5", "3.1"]),
            ]
        )
        + "\n"
    )
    new, reps, refs = live(build_page(native, rotation), shipped)
    assert cell(new, 0, "Previous", 0) == "-7", (reps, refs)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_n_control_the_minus_point_two_repair_still_happens(rotation):
    """The defect the repair exists for (2014-01 Tealbook p92: Japan `-.2` read as `-2`)."""
    native = [("Euro area", ["1.0", "-.9", "1.4", "1.5"]), ("Japan", ["1.1", "-.2", "-.3", "1.2"])]
    shipped = (
        block([("Euro area", ["1.0", "-.9", "1.4", "1.5"]), ("Japan", ["1.1", "-2", "-3", "1.2"])])
        + "\n"
    )
    new, reps, refs = live(build_page(native, rotation), shipped)
    assert [cell(new, 0, "Japan", k) for k in (1, 2)] == ["-.2", "-.3"], (reps, refs)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_n_control_a_repeated_label_far_from_the_row_does_not_refuse(rotation):
    """A repeated label is not refused by itself.  Section 2's `Previous` differs from section
    1's row in more than MAX_RIVAL_MISMATCHES values and is no point-only hit: section 1 is
    repaired.  The difference: with section 2 one value closer, the same row is refused."""
    far = rbc.MAX_RIVAL_MISMATCHES + 1
    row = ["-.7", "2.5", "3.0", "4.0", "5.0"]
    shipped_row = ["-7", *row[1:]]

    def case(n_differ):
        sib = row[:1] + [f"{9 + k}.9" if k < n_differ else v for k, v in enumerate(row[1:])]
        native = [("Real GDP", ["1.1", "1.2", "1.3", "1.4", "1.5"]), ("Previous", row)]
        native += [("Final sales", ["2.1", "2.2", "2.3", "2.4", "2.5"]), ("Previous", sib)]
        shipped = block([native[0], ("Previous", shipped_row), native[2], ("Previous", sib)])
        return live(build_page(native, rotation), shipped + "\n")

    # sib: -.7 then `far` values changed; vs the shipped row it differs in the point and `far`
    new, reps, refs = case(far - 1)
    assert cell(new, 0, "Previous", 0) == "-.7", (reps, refs)
    new2, reps2, refs2 = case(far - 2)
    assert cell(new2, 0, "Previous", 0) == "-7", (reps2, refs2)
    assert rbc.REFUSE_RIVAL_LINE in why(refs2)


# ---------------------------------------------------------------------------
# N at process(): the arms differ exactly by the correct repairs
# ---------------------------------------------------------------------------

# Six lines: the harness sends four-line pages to the native lane (no model table to repair).
F2A_NATIVE = [
    ("Real GDP", ["-.4", "1.1", "1.2"]),
    ("Previous", ["- 7", "2.5", "3.0"]),
    ("Final sales", ["-.5", "5.1", "5.2"]),
    ("Previous", ["-.7", "2.5", "3.0"]),
    ("Exports", ["2.1", "2.2", "2.3"]),
    ("Imports", ["3.1", "3.2", "3.3"]),
]
F2A_SHIPPED = [
    ("Real GDP", ["-4", "1.1", "1.2"]),
    ("Previous", ["-7", "2.5", "3.0"]),
    ("Final sales", ["-5", "5.1", "5.2"]),
    ("Previous", ["-.7", "2.5", "3.1"]),
    *F2A_NATIVE[4:],
]


@pytest.mark.parametrize("providers", [True, False], ids=["ladder", "no_ladder"])
def test_n_at_process_the_step_repairs_the_own_line_rows_and_leaves_previous(tmp_path, providers):
    """Arms differ only by the repair step (the control arm replaces it with a no-op).  With the
    ladder, they differ in exactly the `Real GDP` and `Final sales` cells; section 1's
    `Previous` ships the model's `-7` in both.  With an empty ladder no model reads the page and
    the arms are identical.  Rotation 0 only: the harness sends this page to the native lane at
    270 (no model table)."""
    from test_gh1267_row_binding_pipeline import _events, _fragment, _run
    from test_gh1282_point_repair_pipeline import KIND, _noop

    from socr.pipeline.orchestrator import UnifiedPipeline

    markdown = block(F2A_SHIPPED) + "\n"
    kw = dict(check_runs=True, native=F2A_NATIVE, markdown=markdown, providers=providers)
    _, _, _, on, pdf = _run(tmp_path, None, 0, out_name="on", **kw)
    with patch.object(UnifiedPipeline, "_repair_dropped_points", _noop):
        _, _, _, off, _ = _run(tmp_path, None, 0, out_name="off", pdf=pdf, **kw)
    if not providers:
        assert _fragment(on) == _fragment(off) and _events(on, KIND) == []
        return
    f_on, f_off = _fragment(on), _fragment(off)
    assert cell(f_on, 0, "Previous", 0) == cell(f_off, 0, "Previous", 0) == "-7"
    assert (cell(f_on, 0, "Real GDP", 0), cell(f_off, 0, "Real GDP", 0)) == ("-.4", "-4")
    assert (cell(f_on, 0, "Final sales", 0), cell(f_off, 0, "Final sales", 0)) == ("-.5", "-5")
    rows = [r for e in _events(on, KIND) for r in e["data"]["rows"]]
    assert sorted((r["label"], r["old"], r["new"]) for r in rows) == [
        ("Final sales", "-5", "-.5"),
        ("Real GDP", "-4", "-.4"),
    ]


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_n_a_clipped_box_with_three_dropped_points_is_refused_by_the_point_only_rule(rotation):
    """Section 1's own line differs from the row in three values (all dropped points), more
    than MAX_RIVAL_MISMATCHES, so only the point-only clause names it as a rival.  The box hides
    it; section 2's `12.5` would be written for the true `1.25`."""
    own = ["1.25", "-.2", "-.3", "2.5", "3.0"]
    native = [
        ("Real GDP", ["1.1", "1.2", "1.3", "1.4", "1.5"]),
        ("Previous", own),
        ("Final sales", ["2.1", "2.2", "2.3", "2.4", "2.5"]),
        ("Previous", ["12.5", "-.2", "-.3", "2.5", "3.0"]),
    ]
    shipped = block(
        [
            native[0],
            ("Previous", ["125", "-2", "-3", "2.5", "3.0"]),
            native[2],
            ("Previous", ["12.5", "-.2", "-.3", "2.5", "3.1"]),
        ]
    )
    assert len(own) - 2 > rbc.MAX_RIVAL_MISMATCHES  # three differences, two agreeing
    page = build_page(native, rotation)
    clipped = [raw_box(rotation, V0 + 1.5 * LINE_PITCH, V0 + 4 * LINE_PITCH)]
    new, reps, refs = live(page, shipped + "\n", regions=clipped)
    assert cell(new, 0, "Previous", 0) == "125", (reps, refs)
    assert rbc.REFUSE_RIVAL_LINE in why(refs)
