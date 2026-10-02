"""GH-949: a grid with no table geometry that carries fewer rows than its source.

A grid holding only a table's shaded highlight row has one unique pair, so
``_table_geometry`` is None and every geometric predicate stays silent while the rest
of the table ships as loose lines. Each test pins a DIFFERENCE between two runs of the
same gate that change only the thing under test. The gate is pure (no provider, no
ollama), so nothing here depends on the CI environment.
"""

from __future__ import annotations

import pytest
from native_table_fixtures import HEADER, ROWS, UNCHECKED
from test_gh916_native_ship_gate import _gate, _md, _words

from socr.tables import ship_gate

GEOMETRYLESS = ship_gate.GEOMETRYLESS_BLOCK
ROW_WORDS = _words([HEADER] + ROWS)
#: A grid of the header and the first data row only: one unique pair, no geometry.
LONE_ROW_MD = _md(HEADER, ROWS[:1])


def _geos(words, md):
    blocks = ship_gate._output_blocks(md)
    src = ship_gate._source_rows(words)
    return ship_gate._block_geometries(ship_gate._unique_pairs(blocks, src), src)


def test_lone_row_grid_has_no_geometry_and_fires() -> None:
    assert _geos(ROW_WORDS, LONE_ROW_MD) == [None]
    assert GEOMETRYLESS in _gate(ROW_WORDS, LONE_ROW_MD)


def test_same_grid_over_a_source_of_just_that_row_is_quiet() -> None:
    """Difference: the grid is identical, only the source's extra rows are gone."""
    lone_words = _words([HEADER] + ROWS[:1])
    assert _geos(lone_words, LONE_ROW_MD) == [None]
    assert GEOMETRYLESS not in _gate(lone_words, LONE_ROW_MD)
    assert GEOMETRYLESS in _gate(ROW_WORDS, LONE_ROW_MD)


def test_label_only_rows_do_not_stand_in_for_numeric_rows() -> None:
    """Difference: the same grid padded with rows of words and no numbers still fires."""
    labels = [[r[0], "", "", "", ""] for r in ROWS[1:]]
    padded = _md(HEADER, ROWS[:1] + labels)
    assert len(ship_gate._output_blocks(padded)[0]) >= len(ROWS) + 1
    assert _geos(ROW_WORDS, padded) == [None]
    assert GEOMETRYLESS in _gate(ROW_WORDS, padded)


def test_complete_grid_is_quiet() -> None:
    md = _md(HEADER, ROWS)
    assert GEOMETRYLESS not in _gate(ROW_WORDS, md)


def test_block_with_geometry_is_untouched() -> None:
    """A grid with geometry that drops rows is the older predicates' business, not this one."""
    md = _md(HEADER, ROWS[:2])
    assert _geos(ROW_WORDS, md)[0] is not None
    gate = _gate(ROW_WORDS, md)
    assert GEOMETRYLESS not in gate
    assert ship_gate.DATA_ROW_MISSING in gate


def test_only_adds_faults(monkeypatch) -> None:
    """The new predicate's faults are the only difference from the gate without it."""
    full = ship_gate.native_ship_gate(ROW_WORDS, LONE_ROW_MD, line_dirs=UNCHECKED)
    monkeypatch.setattr(ship_gate, "geometryless_block_faults", lambda *a, **k: [])
    without = ship_gate.native_ship_gate(ROW_WORDS, LONE_ROW_MD, line_dirs=UNCHECKED)
    assert all(f in full for f in without)
    assert {f["predicate"] for f in full if f not in without} == {GEOMETRYLESS}


def test_distant_rows_outside_the_reach_do_not_count() -> None:
    """Difference: the same extra rows, moved beyond the outward reach, stop firing."""
    near = _words([HEADER] + ROWS[:1]) + _words(ROWS[1:], start_line=2, y_start=100.0 + 14.0 * 3)
    far_y = 100.0 + 14.0 * (3 + ship_gate._PANEL_GAP_ROWS * 4)
    far = _words([HEADER] + ROWS[:1]) + _words(ROWS[1:], start_line=2, y_start=far_y)
    assert GEOMETRYLESS in _gate(near, LONE_ROW_MD)
    assert GEOMETRYLESS not in _gate(far, LONE_ROW_MD)


@pytest.mark.parametrize("width", [5, 8])
def test_short_word_tuples_do_not_raise(width) -> None:
    """Words cut to ``(x0, y0, x1, y1, text)`` neither raise nor lose the fault."""
    words = [tuple(w[:width]) for w in ROW_WORDS]
    faults = ship_gate.native_ship_gate(words, LONE_ROW_MD, line_dirs=UNCHECKED)
    assert ship_gate.GATE_ERROR not in {f["predicate"] for f in faults}
    assert GEOMETRYLESS in {f["predicate"] for f in faults}
