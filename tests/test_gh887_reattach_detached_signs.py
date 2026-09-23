"""GH-887: a minus sign set as its own glyph must not be left behind in the
neighbouring cell when ``find_tables`` puts a column boundary between it and its
digits.

Measured on the papers library (2026-09-23): 5 pages of one document shipped
tables where ``-0.25`` came out as ``0.25`` with a stray sign in the cell to its
left -- 115 cells. The numeric-multiset guards cannot see it: both pieces are
still on the page. With this fix, across every one of the library's 2,614 table
pages, exactly those 5 pages change, no region/row/cell count changes, and every
changed cell is a sign move.

The corpus is copyrighted, and ``find_tables``' text strategy only produces the
split on particular real layouts, so these tests drive
``_reattach_detached_signs`` directly with a stand-in table (cell rectangles) and
native words -- the exact inputs production hands it.
"""

from __future__ import annotations

from dataclasses import dataclass

from socr.tables.reconstruct import _reattach_detached_signs

MINUS = "−"
EN_DASH = "–"


@dataclass
class _Row:
    cells: list


@dataclass
class _Table:
    rows: list


def _word(x0, x1, text, *, y0=100.0, y1=110.0, block=0, line=0, n=0):
    return (x0, y0, x1, y1, text, block, line, n)


# Two cells side by side on one row: left spans x 50-150, right spans x 150-250.
_LEFT = (50.0, 98.0, 150.0, 112.0)
_RIGHT = (150.0, 98.0, 250.0, 112.0)
_TABLE = _Table(rows=[_Row(cells=[_LEFT, _RIGHT])])


def test_a_sign_touching_its_digits_is_moved_back():
    words = [
        _word(60, 90, "0.00029"),
        _word(140, 146, MINUS),  # sign sits in the LEFT cell ...
        _word(146, 170, "0.25"),  # ... flush against the digits in the right cell
    ]
    grid = [["0.00029 " + MINUS, "0.25"]]
    assert _reattach_detached_signs(grid, _TABLE, words) == [["0.00029", MINUS + "0.25"]]


def test_an_en_dash_used_as_minus_is_moved_too():
    words = [_word(60, 90, "1.5"), _word(140, 146, EN_DASH), _word(146, 170, "3.45")]
    grid = [["1.5 " + EN_DASH, "3.45"]]
    assert _reattach_detached_signs(grid, _TABLE, words) == [["1.5", EN_DASH + "3.45"]]


def test_a_placeholder_dash_separated_by_a_gap_is_left_alone():
    """A dash that is its own cell value (``-`` meaning 'not available') sits a
    column gap away from the next number. Moving it would invent a negative."""
    words = [_word(120, 126, EN_DASH), _word(160, 180, "0.25")]
    grid = [[EN_DASH, "0.25"]]
    assert _reattach_detached_signs(grid, _TABLE, words) == grid


def test_a_sign_on_a_different_text_line_is_left_alone():
    words = [_word(140, 146, MINUS, line=0), _word(146, 170, "0.25", line=1)]
    grid = [[MINUS, "0.25"]]
    assert _reattach_detached_signs(grid, _TABLE, words) == grid


def test_nothing_changes_when_no_cell_ends_in_a_sign():
    words = [_word(60, 90, "0.5"), _word(160, 180, "0.25")]
    grid = [["0.5", "0.25"]]
    assert _reattach_detached_signs(grid, _TABLE, words) == grid


def test_a_chain_moves_each_sign_one_cell_right():
    """Measured on the real pages: a cell can receive a sign from its left and
    hand its own trailing sign to its right, in the same row."""
    c0, c1, c2 = (
        (50.0, 98.0, 150.0, 112.0),
        (150.0, 98.0, 250.0, 112.0),
        (250.0, 98.0, 350.0, 112.0),
    )
    table = _Table(rows=[_Row(cells=[c0, c1, c2])])
    words = [
        _word(60, 90, "1.1"),
        _word(140, 146, EN_DASH),
        _word(146, 170, "2.2"),
        _word(240, 246, EN_DASH),
        _word(246, 270, "3.3"),
    ]
    grid = [["1.1 " + EN_DASH, "2.2 " + EN_DASH, "3.3"]]
    assert _reattach_detached_signs(grid, table, words) == [
        ["1.1", EN_DASH + "2.2", EN_DASH + "3.3"]
    ]


def test_the_input_grid_is_not_mutated():
    words = [_word(140, 146, MINUS), _word(146, 170, "0.25")]
    grid = [[MINUS, "0.25"]]
    snapshot = [row[:] for row in grid]
    _reattach_detached_signs(grid, _TABLE, words)
    assert grid == snapshot


def test_missing_cell_geometry_is_a_no_op():
    """``find_tables`` reports ``None`` for spanned cells; never guess without a box."""
    table = _Table(rows=[_Row(cells=[None, _RIGHT])])
    words = [_word(140, 146, MINUS), _word(146, 170, "0.25")]
    grid = [[MINUS, "0.25"]]
    assert _reattach_detached_signs(grid, table, words) == grid


def test_a_hyphen_flush_on_both_sides_is_a_range_not_a_minus():
    """PR #888 review: "1990-2000" set as three flush words. The hyphen touches
    the digits after it, but it touches the number BEFORE it too -- the shape of a
    range. Moving it would invent a negative year."""
    words = [_word(110, 140, "1990"), _word(140, 146, "-"), _word(146, 170, "2000")]
    grid = [["1990-", "2000"]]
    assert _reattach_detached_signs(grid, _TABLE, words) == grid


def test_a_sign_with_space_before_it_and_flush_after_it_is_a_minus():
    """The complementary case, and the typographic rule itself: space on the left,
    contact on the right is how a minus is set -- this is what the real pages show."""
    words = [_word(60, 90, "1990"), _word(140, 146, "-"), _word(146, 170, "2000")]
    grid = [["1990 -", "2000"]]
    assert _reattach_detached_signs(grid, _TABLE, words) == [["1990", "-2000"]]
