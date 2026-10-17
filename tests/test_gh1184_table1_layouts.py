"""#1184: the projection table read as a witness, across the layouts it is printed in.

Hermetic by construction: every table here is DRAWN with PyMuPDF on a grid, at
coordinates this file chooses, so the geometry the parser walks is known rather
than sampled. No provider, no network, no Ollama, no model call -- the parser
is deterministic and reads a text layer.

Why a grid and not lines of text. #1155's parser counted a row's values and
compared the total with the header's column total. #1184 replaces that with
per-column binding (orchestrator ruling), because a count cannot tell a cell
the table never printed from one the parser lost, so it rejected both: on
2010-04-28 the core PCE rows, whose longer-run cells the Fed does not collect
(that release's own footnote 3), and on 2008-10-29 every row, because its
groups are stacked and a row legitimately carries half the columns.

The real releases these fixtures imitate, and the measurements behind the
constants, are in ``scratch/routing-baseline/leads/1184/`` --
``design.md`` section 3, and ``probes/feasibility2.txt`` for the geometry.
"""

from __future__ import annotations

import fitz
import pytest

from socr.figures.chart_witness import (
    REFUSED_COLUMNS_UNRESOLVED,
    REFUSED_GROUPS_UNRESOLVED,
    REFUSED_NO_RANGE_GROUP,
    REJECTED_MEDIAN_OUTSIDE_CENTRAL,
    REJECTED_MEDIAN_OUTSIDE_RANGE,
    REJECTED_MISSING_GROUP,
    REJECTED_ORPHAN_PRIOR_ROW,
    REFUSED_UNSUPPORTED_ROTATION,
    REJECTED_CENTRAL_OUTSIDE_RANGE,
    REJECTED_TWO_CELLS_IN_A_COLUMN,
    REJECTED_UNBOUND_CELL,
    LAYOUT_SIDE_BY_SIDE_GROUPS,
    LAYOUT_STACKED_BLOCKS,
    parse_table1_witness,
)

_FONT = "helv"
_SIZE = 9.0
#: Column pitch and the gap between groups, in points. Chosen to be comfortably
#: wider than a cell -- the real releases print 48.8pt (2010-04-28) and 26.4 to
#: 43.0pt (2015-09-17) -- so the fixtures are not quietly tighter than the
#: documents they stand for.
_PITCH = 60.0
_GROUP_GAP = 40.0
#: The table sits well right of the page edge, so a fixture can draw a prose
#: column beside it the way 2008-10-29 p12 does, instead of on top of its labels.
_PROSE_U = 30.0
_LABEL_U = 250.0
_FIRST_COLUMN_U = 360.0


def _width(text: str) -> float:
    return fitz.get_text_length(text, fontname=_FONT, fontsize=_SIZE)


def _column_u(group_index: int, year_index: int, years: int, pitches=None) -> float:
    """Where one column's centre lands.

    ``pitches`` gives each group its OWN column pitch. The releases do not print
    the groups at one width -- on 2015-09-17 the median group holds a single
    figure per column and sits 26.4pt apart while the range group holds a pair
    and sits 41.4pt apart -- and a fixture with one pitch everywhere cannot
    exercise the code that derives the trailing bound from the LAST group.
    """
    origin = _FIRST_COLUMN_U
    for earlier in range(group_index):
        origin += years * (pitches[earlier] if pitches else _PITCH) + _GROUP_GAP
    return origin + year_index * (pitches[group_index] if pitches else _PITCH)


def _draw(path, placements, *, rotate=0):
    """Write a one-page PDF placing each ``(line, u_centre, text)``.

    ``line`` numbers rows down the page and ``u_centre`` is where the token's
    centre must land along a row, in the table's own reading frame. For the
    rotated page the same (line, u) pair is mapped onto the rotated axes, so a
    fixture can be drawn upright and rotated from ONE description and the two
    compared.
    """
    doc = fitz.open()
    page = doc.new_page(width=1000, height=792)
    for line, centre, text in placements:
        if not text:
            continue
        width = _width(text)
        if rotate == 0:
            page.insert_text((centre - width / 2, 60 + line * 18), text, fontsize=_SIZE)
        elif rotate == 90:
            # Text runs upward from its insertion point, so a token placed at y
            # occupies [y - width, y]; its centre in the (u = -y) frame is
            # width/2 - y. Solving for the wanted centre puts the table in the
            # negative-u half of the page, exactly as 2015-09-17 p13 lands.
            page.insert_text(
                (60 + line * 18, width / 2 - (centre - 960)), text, fontsize=_SIZE, rotate=90
            )
        else:
            raise AssertionError(f"fixture does not draw rotation {rotate}")
    doc.save(path)
    doc.close()
    return fitz.open(path)


def _side_by_side(
    path, *, years, groups, rows, rotate=0, extra=(), pitches=None, trailing_label=None
):
    """The 2010-04-28 / 2015-09-17 shape: every group printed beside the others.

    ``rows`` is ``(label, {(group_index, year_index): cell_text})``; a pair left
    out is a column the table does not print there. ``trailing_label`` adds one
    further column to EVERY group, named by something other than a year, the
    way the releases print "Longer run"; its cells use year index
    ``len(years)``.
    """
    span_count = len(years) + (1 if trailing_label else 0)
    placements = [(0, _FIRST_COLUMN_U, "Table 1. Economic projections")]
    for g, group in enumerate(groups):
        edges = [_column_u(g, y, span_count, pitches) for y in (0, span_count - 1)]
        placements.append((1, sum(edges) / 2, group))
    for g in range(len(groups)):
        for y, year in enumerate(years):
            placements.append((2, _column_u(g, y, span_count, pitches), year))
        if trailing_label:
            placements.append((2.6, _column_u(g, len(years), span_count, pitches), trailing_label))
    placements.append((2, _LABEL_U, "Variable"))
    for index, (label, cells) in enumerate(rows):
        line = 3 + index
        placements.append((line, _LABEL_U, label))
        for (g, y), text in cells.items():
            placements.append((line, _column_u(g, y, span_count, pitches), text))
    placements.extend(extra)
    return _draw(path, placements, rotate=rotate)


def _stacked(path, *, years, blocks):
    """The 2008-10-29 shape: one year header, then one block per group."""
    placements = [(0, _FIRST_COLUMN_U, "Table 1. Economic projections"), (1, _LABEL_U, "Variable")]
    for y, year in enumerate(years):
        placements.append((1, _column_u(0, y, len(years)), year))
    line = 2
    for group, rows in blocks:
        placements.append((line, _FIRST_COLUMN_U, group))
        line += 1
        for label, cells in rows:
            placements.append((line, _LABEL_U, label))
            for y, text in cells.items():
                placements.append((line, _column_u(0, y, len(years)), text))
            line += 1
    return _draw(path, placements)


#: One logical table, in the fixtures' own terms: two years, three groups, two
#: rows. Every test below starts from this and changes exactly one thing.
_YEARS = ("2010", "2011")
_GROUPS = ("Median", "Central tendency", "Range")
_ROWS = (
    (
        "Change in real GDP",
        {
            (0, 0): "3.5",
            (0, 1): "3.9",
            (1, 0): "3.2 to 3.7",
            (1, 1): "3.4 to 4.5",
            (2, 0): "2.7 to 4.0",
            (2, 1): "3.0 to 4.6",
        },
    ),
    (
        "January projection",
        {
            (0, 0): "3.1",
            (0, 1): "3.9",
            (1, 0): "2.8 to 3.5",
            (1, 1): "3.4 to 4.5",
            (2, 0): "2.3 to 4.0",
            (2, 1): "2.7 to 4.7",
        },
    ),
)
#: What the fixture above must yield. The RANGE group is the witness, never the
#: central tendency and never the median.
_EXPECTED = {
    ("change in real gdp", "current", "2010"): (2.7, 4.0),
    ("change in real gdp", "current", "2011"): (3.0, 4.6),
    ("change in real gdp", "prior", "2010"): (2.3, 4.0),
    ("change in real gdp", "prior", "2011"): (2.7, 4.7),
}


def _parse(doc):
    witness = parse_table1_witness(doc[0])
    doc.close()
    return witness


# --------------------------------------------------------------------------------------
# The three layouts the issue names. Each is the SAME table; only its printing moves.
# --------------------------------------------------------------------------------------


def test_the_side_by_side_layout_yields_the_range_of_every_column(tmp_path):
    witness = _parse(_side_by_side(tmp_path / "side.pdf", years=_YEARS, groups=_GROUPS, rows=_ROWS))
    assert witness.refusal == ""
    assert witness.layout == LAYOUT_SIDE_BY_SIDE_GROUPS
    assert witness.rows == _EXPECTED
    assert witness.prior_months == frozenset({"January"})


def test_the_stacked_block_layout_yields_THE_SAME_witness(tmp_path):
    """2008-10-29's shape. The printing changes; the witness must not.

    The parser on main refused every row of this layout ("4 cells, header
    declares 8"), which is why the release yielded no witness at all.
    """
    blocks = [
        (group, [(label, {y: cells[(g, y)] for y in range(len(_YEARS))}) for label, cells in _ROWS])
        for g, group in enumerate(_GROUPS)
    ]
    stacked = _parse(_stacked(tmp_path / "stacked.pdf", years=_YEARS, blocks=blocks))
    side = _parse(_side_by_side(tmp_path / "side.pdf", years=_YEARS, groups=_GROUPS, rows=_ROWS))
    assert stacked.layout == LAYOUT_STACKED_BLOCKS
    assert side.layout == LAYOUT_SIDE_BY_SIDE_GROUPS
    assert stacked.rows == side.rows == _EXPECTED, "the layout moved, the witness did not"


def test_the_rotated_layout_yields_THE_SAME_witness(tmp_path):
    """2015-09-17 draws Table 1 at 90 degrees inside an unrotated page.

    ``page.rotation`` is 0 there, so the rotation has to come from the text
    direction the page records. The parser on main refused this layout outright.
    """
    rotated = _parse(
        _side_by_side(tmp_path / "rot.pdf", years=_YEARS, groups=_GROUPS, rows=_ROWS, rotate=90)
    )
    upright = _parse(_side_by_side(tmp_path / "up.pdf", years=_YEARS, groups=_GROUPS, rows=_ROWS))
    assert rotated.refusal == "", rotated.refusal
    assert rotated.rows == upright.rows == _EXPECTED, "the page turned, the witness did not"


# --------------------------------------------------------------------------------------
# A cell the table never printed, versus a cell this parser lost
# --------------------------------------------------------------------------------------


def test_a_column_the_table_never_prints_costs_only_its_own_key(tmp_path):
    """2010-04-28's core PCE rows: longer-run core PCE is not collected.

    The count rule rejected the whole row for this, losing the years the table
    DOES print. A blank column is a hole: no witness for that one key, no
    rejection of the row.
    """
    rows = tuple((label, {k: v for k, v in cells.items() if k != (2, 1)}) for label, cells in _ROWS)
    witness = _parse(_side_by_side(tmp_path / "hole.pdf", years=_YEARS, groups=_GROUPS, rows=rows))
    gone = {k: v for k, v in _EXPECTED.items() if k[2] != "2011"}
    assert witness.rows == gone, "only the blank column's keys are absent"
    assert witness.rejected == (), "and no row was rejected for it"


def test_a_cell_that_lost_one_of_its_two_figures_is_REFUSED_by_name(tmp_path):
    """The orchestrator's condition, half one: a MISSING figure must be named.

    This is the case a hole must not be confused with. Dropping one figure of
    "2.7 to 4.0" leaves "4.0" printed where the pair's centre was, which is half
    a cell away from the column -- measured on the releases, a dropped figure
    moves a cell 10.10pt while a real cell sits 1.10pt from its column, against
    a 5.05-5.55pt tolerance. So it binds to nothing and the row is refused.
    """
    rows = ((_ROWS[0][0], {k: v for k, v in _ROWS[0][1].items() if k != (2, 0)}),)
    # Where the "4.0" of "2.7 to 4.0" is actually printed: the end of the cell,
    # not its middle. Re-centring it would be a fixture that tests nothing.
    survivor = _column_u(2, 0, len(_YEARS)) + _width("2.7 to 4.0") / 2 - _width("4.0") / 2
    witness = _parse(
        _side_by_side(
            tmp_path / "lost.pdf",
            years=_YEARS,
            groups=_GROUPS,
            rows=rows,
            extra=((3, survivor, "4.0"),),
        )
    )
    assert witness.rows == {}, "a row with a lost figure yields no witness at all"
    assert [reason for _, _, reason in witness.rejected] == [REJECTED_UNBOUND_CELL]


def test_an_EXTRA_value_printed_off_the_grid_is_REFUSED_by_name(tmp_path):
    """The orchestrator's condition, half two: an extra value must be named.

    Printed in the gap between two groups, so no column is occupied twice --
    that way only the binding rule can catch it, and a mutant that drops the
    binding rule is not rescued by the duplicate rule (Argon, design review
    P1-5).
    """
    stray = _column_u(0, len(_YEARS) - 1, len(_YEARS)) + _GROUP_GAP / 2
    clean = _parse(_side_by_side(tmp_path / "clean.pdf", years=_YEARS, groups=_GROUPS, rows=_ROWS))
    witness = _parse(
        _side_by_side(
            tmp_path / "stray.pdf",
            years=_YEARS,
            groups=_GROUPS,
            rows=_ROWS,
            extra=((3, stray, "9.9"),),
        )
    )
    assert clean.rows == _EXPECTED
    assert ("change in real gdp", "current", "2010") not in witness.rows, (
        "the row carrying the stray value yields nothing"
    )
    assert REJECTED_UNBOUND_CELL in [reason for _, _, reason in witness.rejected]


def test_a_whole_cell_printed_off_its_column_is_REFUSED_by_name(tmp_path):
    """The cell is complete and wide; only its POSITION is wrong.

    This is the case the distance tolerance alone catches. A whole range cell
    is about 37pt wide here, so displacing it by 12pt still leaves it covering
    its column -- the mutant run recorded in ``probes/mutants.txt`` showed that
    without this test the tolerance can be widened fourfold and nothing fails,
    because the other half of the binding rule (the cell must cover the
    column's centre) was quietly doing all the work.

    A displaced cell is a wrong number, not a missing one: it publishes one
    column's figures under another column's year.
    """
    displaced = _column_u(2, 0, len(_YEARS)) + 12.0
    rows = ((_ROWS[0][0], {k: v for k, v in _ROWS[0][1].items() if k != (2, 0)}),)
    witness = _parse(
        _side_by_side(
            tmp_path / "displaced.pdf",
            years=_YEARS,
            groups=_GROUPS,
            rows=rows,
            extra=((3, displaced, "2.7 to 4.0"),),
        )
    )
    assert ("change in real gdp", "current", "2010") not in witness.rows
    assert [reason for _, _, reason in witness.rejected] == [REJECTED_UNBOUND_CELL]


def test_a_SECOND_value_in_one_column_is_REFUSED_by_name(tmp_path):
    """Two cells landing in one column is its own named refusal, not a silent pick."""
    beside = _column_u(2, 0, len(_YEARS)) + _width("2.7 to 4.0") / 2
    witness = _parse(
        _side_by_side(
            tmp_path / "double.pdf",
            years=_YEARS,
            groups=_GROUPS,
            rows=_ROWS,
            extra=((3, beside, "9.9"),),
        )
    )
    reasons = [reason for _, _, reason in witness.rejected]
    assert REJECTED_TWO_CELLS_IN_A_COLUMN in reasons or REJECTED_UNBOUND_CELL in reasons
    assert ("change in real gdp", "current", "2010") not in witness.rows


# --------------------------------------------------------------------------------------
# Printing traps the releases actually contain
# --------------------------------------------------------------------------------------


def test_a_collapsed_range_in_the_range_group_reads_as_a_degenerate_interval(tmp_path):
    """A release prints a collapsed range as ONE figure ("2.0", not "2.0 to 2.0").

    The degenerate cell is put in the RANGE group deliberately: in the median or
    the central tendency a dropped cell is indistinguishable from a hole, so the
    test would pass against a parser that skipped it (Argon, design review P1-5).
    """
    collapsed = ((_ROWS[0][0], {**_ROWS[0][1], (0, 1): "4.6", (1, 1): "4.6", (2, 1): "4.6"}),)
    spelled = (
        (
            _ROWS[0][0],
            {**_ROWS[0][1], (0, 1): "4.6", (1, 1): "4.6 to 4.6", (2, 1): "4.6 to 4.6"},
        ),
    )
    one = _parse(
        _side_by_side(tmp_path / "collapsed.pdf", years=_YEARS, groups=_GROUPS, rows=collapsed)
    )
    two = _parse(
        _side_by_side(tmp_path / "spelled.pdf", years=_YEARS, groups=_GROUPS, rows=spelled)
    )
    assert one.rows[("change in real gdp", "current", "2011")] == (4.6, 4.6)
    assert one.rows == two.rows, "one figure or two, the published range is the same"


def test_prose_sharing_a_band_with_a_row_does_not_change_that_rows_witness(tmp_path):
    """2008-10-29 p12 draws Table 1 beside a prose column.

    Its band for the June sub-row BEGINS "unemployment, and inflation in 2008,
    2009, 2010, and" -- so a row's label is not the start of its band, and
    taking the leading tokens binds the previous meeting's figures under a label
    read out of a sentence (Argon, design review P0-3.1).
    """
    prose = (
        (3, _PROSE_U + _width("unemployment, and") / 2, "unemployment, and"),
        (4, _PROSE_U + _width("inflation in 2008, 2009, and") / 2, "inflation in 2008, 2009, and"),
    )
    clean = _parse(_side_by_side(tmp_path / "clean.pdf", years=_YEARS, groups=_GROUPS, rows=_ROWS))
    beside = _parse(
        _side_by_side(
            tmp_path / "beside.pdf", years=_YEARS, groups=_GROUPS, rows=_ROWS, extra=prose
        )
    )
    assert clean.rows == _EXPECTED
    assert beside.rows == clean.rows, "prose arrived in the band; the witness did not move"


def test_a_range_is_not_paired_ACROSS_an_empty_column(tmp_path):
    """With the middle cell blank, the figures either side must not become a range.

    A band-wide regex pairs the last figure of one column with the first of the
    column beyond the gap and publishes that as a range (Argon, design review
    P0-2.3). Cells are built from three CONSECUTIVE tokens instead.
    """
    years = ("2010", "2011", "2012")
    rows = (
        (
            "Change in real GDP",
            {
                (0, 0): "3.5",
                (0, 1): "3.9",
                (0, 2): "4.1",
                (1, 0): "3.2 to 3.7",
                (1, 2): "3.9 to 4.3",
                (2, 0): "2.7 to 4.0",
                (2, 2): "3.5 to 4.6",
            },
        ),
    )
    witness = _parse(_side_by_side(tmp_path / "gap.pdf", years=years, groups=_GROUPS, rows=rows))
    assert witness.rows == {
        ("change in real gdp", "current", "2010"): (2.7, 4.0),
        ("change in real gdp", "current", "2012"): (3.5, 4.6),
    }
    assert ("change in real gdp", "current", "2011") not in witness.rows


# --------------------------------------------------------------------------------------
# The row must still prove itself, and a caller must still find it
# --------------------------------------------------------------------------------------


def test_a_row_whose_nesting_breaks_is_not_a_witness(tmp_path):
    """Moved from the #1155 file: central tendency outside its own range."""
    broken = ((_ROWS[0][0], {**_ROWS[0][1], (2, 0): "1.0 to 1.5"}),)
    witness = _parse(
        _side_by_side(tmp_path / "broken.pdf", years=_YEARS, groups=_GROUPS, rows=broken)
    )
    assert witness.rows == {}
    assert [reason for _, _, reason in witness.rejected] == [REJECTED_CENTRAL_OUTSIDE_RANGE]


def test_a_series_is_the_prior_meeting_only_when_its_own_month_matches(tmp_path):
    """Moved from the #1155 file: every series is month-prefixed, so "has a month" is not a test."""
    witness = _parse(
        _side_by_side(tmp_path / "months.pdf", years=_YEARS, groups=_GROUPS, rows=_ROWS)
    )
    assert witness.series_row_for("January projections") == "prior"
    assert witness.series_row_for("April projections") == "current"


def test_a_caller_whose_variable_name_is_longer_than_the_tables_key_still_finds_it(tmp_path):
    """2015-09-17 wraps "Core PCE inflation4", so its data band reads only "Core PCE".

    The caller holds the full name, so the caller's name is tested for STARTING
    WITH the key, not the reverse.
    """
    rows = (("Core PCE", _ROWS[0][1]),)
    witness = _parse(
        _side_by_side(tmp_path / "wrapped.pdf", years=_YEARS, groups=_GROUPS, rows=rows)
    )
    assert witness.variable_for("Core PCE inflation") == "core pce"
    assert witness.range_for("Core PCE inflation", "current", "2010") == (2.7, 4.0)
    assert witness.variable_for("Unemployment rate") is None


# --------------------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------------------


def test_a_table_with_no_range_group_witnesses_nothing(tmp_path):
    """The witness IS the published range. Without that group there is nothing to witness with."""
    rows = tuple((label, {k: v for k, v in cells.items() if k[0] != 2}) for label, cells in _ROWS)
    witness = _parse(
        _side_by_side(tmp_path / "norange.pdf", years=_YEARS, groups=_GROUPS[:2], rows=rows)
    )
    assert witness.rows == {}
    assert witness.refusal == REFUSED_NO_RANGE_GROUP


def test_an_angle_this_parser_does_not_read_refuses_by_name(monkeypatch):
    """The corpus is 1,021 documents over more decades than these three.

    An unrecognised layout must refuse with a name, never improvise: a wrong
    number is worse than a missing one.
    """
    import socr.core.born_digital as born_digital

    monkeypatch.setattr(born_digital, "upright_rotation_for", lambda page: 37)
    doc = fitz.open()
    doc.new_page(width=612, height=792)
    witness = parse_table1_witness(doc[0])
    doc.close()
    assert witness.rows == {}
    assert witness.refusal == REFUSED_UNSUPPORTED_ROTATION


@pytest.mark.parametrize("rotation", [0, 90])
def test_the_fixture_really_lands_where_the_test_says_it_does(tmp_path, rotation):
    """A guard on the fixtures themselves, not on the parser.

    These tests mean nothing if the helper does not put a token's centre where
    it was asked to. An earlier #1155 test passed against an ``if True else``
    expression in the test itself; a fixture that silently mislays its tokens is
    the same failure wearing geometry.
    """
    from socr.figures.chart_witness import _bands

    doc = _side_by_side(
        tmp_path / f"probe{rotation}.pdf",
        years=_YEARS,
        groups=_GROUPS,
        rows=_ROWS,
        rotate=rotation,
    )
    _, bands = _bands(doc[0])
    doc.close()
    wanted = _column_u(0, 0, len(_YEARS)) - (960 if rotation == 90 else 0)
    centres = [
        (u0 + u1) / 2 for tokens in bands.values() for u0, u1, text in tokens if text == "2010"
    ]
    assert centres, "the header's own year token must be on the page"
    assert min(abs(c - wanted) for c in centres) < 1.0, (
        f"the fixture placed 2010 at {centres}, not at {wanted}"
    )


# --------------------------------------------------------------------------------------
# PR #1195 review, blocker 1: a lost cell is not a blank column
# --------------------------------------------------------------------------------------


def test_a_range_contradicting_its_own_median_does_not_ship(tmp_path):
    """The reviewer's case: with the central cell lost, nothing checked the range.

    The self-check compared neighbouring groups only -- range against central,
    central against median -- so a missing central cell left the range with
    nothing above it. Measured on head c9e38d8: GDP 2010 shipped (4.1, 4.6) on
    a row whose own median is 3.5, with ``rejected`` empty.
    """
    cells = {k: v for k, v in _ROWS[0][1].items() if k != (1, 0)}
    rows = ((_ROWS[0][0], {**cells, (2, 0): "4.1 to 4.6"}),)
    witness = _parse(
        _side_by_side(tmp_path / "contradicts.pdf", years=_YEARS, groups=_GROUPS, rows=rows)
    )
    assert ("change in real gdp", "current", "2010") not in witness.rows
    assert witness.rows == {}, "and the row ships nothing at all, not just that year"
    assert [reason for _, _, reason in witness.rejected] == [REJECTED_MISSING_GROUP]


def test_a_year_missing_a_declared_group_is_a_lost_cell_not_a_hole(tmp_path):
    """A column the release never prints is blank in EVERY group.

    Longer-run core PCE is uncollected, so that column prints no median, no
    central tendency and no range. A gap in one group only cannot be that, so
    it is a cell this parser lost and the row is refused by name -- while the
    blank-column case keeps working (see the hole test above).
    """
    rows = ((_ROWS[0][0], {k: v for k, v in _ROWS[0][1].items() if k != (0, 1)}),)
    witness = _parse(
        _side_by_side(tmp_path / "lostgroup.pdf", years=_YEARS, groups=_GROUPS, rows=rows)
    )
    assert witness.rows == {}
    assert [reason for _, _, reason in witness.rejected] == [REJECTED_MISSING_GROUP]


def test_a_row_carrying_only_the_range_group_ships_nothing(tmp_path):
    """Both checking groups lost at once must not read as a clean row."""
    rows = ((_ROWS[0][0], {k: v for k, v in _ROWS[0][1].items() if k[0] == 2}),)
    witness = _parse(
        _side_by_side(tmp_path / "rangeonly.pdf", years=_YEARS, groups=_GROUPS, rows=rows)
    )
    assert witness.rows == {}
    assert [reason for _, _, reason in witness.rejected] == [REJECTED_MISSING_GROUP]


def test_a_table_with_no_central_group_still_checks_median_against_range(tmp_path):
    """Where no central tendency is declared, the median is the only check left.

    The missing-group rule cannot help here -- nothing is missing, the table
    simply prints two groups -- so this is what the DIRECT median-in-range test
    is for.
    """
    groups = (_GROUPS[0], _GROUPS[2])
    rows = (
        (
            "Change in real GDP",
            {(0, 0): "3.5", (0, 1): "3.9", (1, 0): "1.0 to 1.5", (1, 1): "3.0 to 4.6"},
        ),
    )
    witness = _parse(
        _side_by_side(tmp_path / "medrange.pdf", years=_YEARS, groups=groups, rows=rows)
    )
    assert witness.rows == {}
    assert [reason for _, _, reason in witness.rejected] == [REJECTED_MEDIAN_OUTSIDE_RANGE]


def test_a_median_outside_its_central_tendency_is_not_a_witness(tmp_path):
    """The third nesting rule, which no test reached before (PR #1195 review)."""
    rows = ((_ROWS[0][0], {**_ROWS[0][1], (0, 0): "2.9"}),)
    witness = _parse(_side_by_side(tmp_path / "medct.pdf", years=_YEARS, groups=_GROUPS, rows=rows))
    assert witness.rows == {}
    assert [reason for _, _, reason in witness.rejected] == [REJECTED_MEDIAN_OUTSIDE_CENTRAL]


# --------------------------------------------------------------------------------------
# PR #1195 review, blocker 2: a rejected row lends its name to nothing
# --------------------------------------------------------------------------------------


def test_a_rejected_row_does_not_hand_its_sub_row_to_the_variable_above(tmp_path):
    """The reviewer's four-row fixture, and the wrong number it exposed.

    A sub-row names a MEETING, so it only means anything under the variable row
    above it. With that row rejected, the reviewer measured
    ``range_for("Unemployment rate", "prior", "2010")`` returning GDP's
    (2.3, 4.0): one variable's figures under another variable's key, which is a
    wrong number, not a missing one. The sub-row must not fall back on its own
    printed label either -- that publishes a variable called "january
    projection".
    """
    unemployment = {
        (0, 0): "9.3",
        (0, 1): "8.3",
        (1, 0): "9.1 to 9.5",
        (1, 1): "8.1 to 8.5",
        (2, 0): "8.6 to 10.0",
        (2, 1): "7.2 to 8.8",
    }
    unemployment_prior = {
        (0, 0): "9.6",
        (0, 1): "8.4",
        (1, 0): "9.5 to 9.7",
        (1, 1): "8.2 to 8.5",
        (2, 0): "8.6 to 10.0",
        (2, 1): "7.2 to 8.8",
    }
    rows = (
        ("Unemployment rate", unemployment),
        ("January projection", unemployment_prior),
        ("Change in real GDP", {k: v for k, v in _ROWS[0][1].items() if k != (2, 0)}),
        ("January projection", _ROWS[1][1]),
    )
    stray = _column_u(2, 0, len(_YEARS)) + _GROUP_GAP / 2
    witness = _parse(
        _side_by_side(
            tmp_path / "fourrow.pdf",
            years=_YEARS,
            groups=_GROUPS,
            rows=rows,
            extra=((5, stray, "2.7 to 4.0"),),
        )
    )
    assert witness.range_for("Unemployment rate", "prior", "2010") == (8.6, 10.0), (
        "the unemployment sub-row keeps its OWN figures"
    )
    assert witness.range_for("Change in real GDP", "current", "2010") is None
    assert ("january projection", "current", "2010") not in witness.rows, (
        "and no variable is invented out of a sub-row's printed label"
    )
    reasons = [reason for _, _, reason in witness.rejected]
    assert REJECTED_ORPHAN_PRIOR_ROW in reasons
    assert REJECTED_UNBOUND_CELL in reasons


# --------------------------------------------------------------------------------------
# PR #1195 review: gates that were in the code but in no test
# --------------------------------------------------------------------------------------


def test_header_labels_that_do_not_repeat_evenly_refuse(tmp_path):
    """Each group must print the same columns, or the table is not understood."""
    gap = (_column_u(0, 1, len(_YEARS)) + _column_u(1, 0, len(_YEARS))) / 2
    witness = _parse(
        _side_by_side(
            tmp_path / "uneven.pdf",
            years=_YEARS,
            groups=_GROUPS,
            rows=_ROWS,
            extra=((2, gap, "Longer run"),),
        )
    )
    assert witness.rows == {}
    assert witness.refusal == REFUSED_COLUMNS_UNRESOLVED


def test_two_group_names_over_one_block_refuse(tmp_path):
    """Which group governs those columns is not established, so nothing ships."""
    witness = _parse(
        _side_by_side(
            tmp_path / "twonames.pdf",
            years=_YEARS,
            groups=_GROUPS,
            rows=_ROWS,
            extra=((1, _column_u(0, 1, len(_YEARS)), "Range"),),
        )
    )
    assert witness.rows == {}
    assert witness.refusal == REFUSED_GROUPS_UNRESOLVED


def test_text_printed_past_the_last_column_does_not_become_a_column(tmp_path):
    """2015-09-17 p13 runs "Federal Open Market Committee" down the page edge.

    In the table's own rotated frame that text sits past the last column, where
    the trailing "Longer run" column would be looked for.
    """
    beyond = _column_u(len(_GROUPS) - 1, len(_YEARS) - 1, len(_YEARS)) + 2 * _PITCH
    witness = _parse(
        _side_by_side(
            tmp_path / "edge.pdf",
            years=_YEARS,
            groups=_GROUPS,
            rows=_ROWS,
            extra=((2, beyond, "Committee"),),
        )
    )
    assert witness.rows == _EXPECTED, "the edge text changed nothing"


def test_groups_printed_at_different_widths_still_resolve(tmp_path):
    """2015-09-17's median columns sit 26.4pt apart and its range columns 41.4pt.

    The trailing column's bound has to come from the LAST group's pitch. Taken
    from the first, the bound falls short of the last group's "Longer run" and
    the table refuses.
    """
    pitches = [30.0, 60.0, 60.0]
    rows = (
        (
            "Change in real GDP",
            {
                (0, 0): "3.5",
                (0, 1): "3.9",
                (0, 2): "3.6",
                (1, 0): "3.2 to 3.7",
                (1, 1): "3.4 to 4.5",
                (1, 2): "3.3 to 3.9",
                (2, 0): "2.7 to 4.0",
                (2, 1): "3.0 to 4.6",
                (2, 2): "2.5 to 4.1",
            },
        ),
    )
    witness = _parse(
        _side_by_side(
            tmp_path / "pitches.pdf",
            years=_YEARS,
            groups=_GROUPS,
            rows=rows,
            pitches=pitches,
            trailing_label="Longer run",
        )
    )
    assert witness.refusal == ""
    assert witness.rows == {
        ("change in real gdp", "current", "2010"): (2.7, 4.0),
        ("change in real gdp", "current", "2011"): (3.0, 4.6),
        ("change in real gdp", "current", "Longer run"): (2.5, 4.1),
    }


def test_a_stray_pair_above_the_table_is_not_taken_for_its_first_row(tmp_path):
    """The first data band must be ON the grid, or the header is read short.

    The column names printed below the year header -- "Longer run" on its own
    band, as 2015-09-17 prints it -- are gathered between the header and the
    first data row. Treat a stray line as that first row and they fall outside
    the window and are lost.
    """
    noise = _column_u(0, 0, len(_YEARS) + 1) + _PITCH / 2
    rows = (
        (
            "Change in real GDP",
            {
                (0, 0): "3.5",
                (0, 1): "3.9",
                (0, 2): "3.6",
                (1, 0): "3.2 to 3.7",
                (1, 1): "3.4 to 4.5",
                (1, 2): "3.3 to 3.9",
                (2, 0): "2.7 to 4.0",
                (2, 1): "3.0 to 4.6",
                (2, 2): "2.5 to 4.1",
            },
        ),
    )
    witness = _parse(
        _side_by_side(
            tmp_path / "noise.pdf",
            years=_YEARS,
            groups=_GROUPS,
            rows=rows,
            trailing_label="Longer run",
            extra=((2.3, noise, "7.7"), (2.3, noise + _PITCH, "8.8")),
        )
    )
    assert witness.refusal == ""
    assert witness.rows[("change in real gdp", "current", "Longer run")] == (2.5, 4.1)


def test_a_declared_hole_printed_off_the_grid_is_REFUSED(tmp_path):
    """An "n.a." is bound like a figure, so a misplaced one is caught.

    Both callers used to filter holes out before binding, which made the branch
    that reads them dead code: an "n.a." could sit anywhere, or on top of a
    figure's column, and nothing noticed (PR #1195 review).
    """
    stray = _column_u(2, 0, len(_YEARS)) + _GROUP_GAP / 2
    witness = _parse(
        _side_by_side(
            tmp_path / "hole.pdf",
            years=_YEARS,
            groups=_GROUPS,
            rows=_ROWS,
            extra=((3, stray, "n.a."),),
        )
    )
    assert ("change in real gdp", "current", "2010") not in witness.rows
    assert REJECTED_UNBOUND_CELL in [reason for _, _, reason in witness.rejected]


def test_a_row_whose_first_column_prints_a_hole_keeps_its_own_label(tmp_path):
    """The label ends at the row's first CELL, holes included (PR #1195 review).

    Anchoring on the first FIGURE instead leaves any "n.a." printed before it
    inside the label run, so the row is filed under "change in real gdp n.a."
    and a caller asking for "Change in real GDP" gets nothing. The reviewer
    found the anchor could be switched back to figures with all tests still
    green; this is the row that notices.
    """
    rows = ((_ROWS[0][0], {**_ROWS[0][1], (0, 0): "n.a."}),)
    witness = _parse(
        _side_by_side(tmp_path / "firsthole.pdf", years=_YEARS, groups=_GROUPS, rows=rows)
    )
    assert witness.rejected == (), "a declared hole is not a lost cell"
    assert witness.range_for("Change in real GDP", "current", "2010") == (2.7, 4.0)
    assert {key[0] for key in witness.rows} == {"change in real gdp"}
