"""#1374: a page whose band order cannot be confirmed ships the layer's line-order body.

Owner ruling on #1374: when the floor's reading order cannot be confirmed, the page may ship the
words in the order the PDF text layer stores them, with every detected table box still withheld
and the page still marked. Scope is the ``order_gate`` ONLY.

Why it needs a second body: the #1261 in-order fallback withholds every line whose x-range
overlaps a box (the span rule), so on a page whose table box spans the page width it ships
nothing and abstains. Measured on the 114 order-gate pages of the 156-document Fed run: no line
ships on any of them. The line-order body drops the span rule, replaces it with a continuation
walk and an edge-overlap leak guard, and is tried only after the span body and the bands body
have both failed.

Every claim is a DIFFERENCE between two runs in one process (#253/#257), never a value measured
on one machine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_gh1266_floor_geometry import MARKER, PNG, _build, _whole_page  # noqa: E402

from socr.core import manifest as M  # noqa: E402
from socr.core.manifest import (  # noqa: E402
    FLOOR_LINE_ORDER_NOTE,
    has_line_order_notice,
    structure_class_floor_text,
    table_floor_text_for_source,
    withheld_table_floor_text,
)

LEFT = [
    "Left column opens the discussion of output.",
    "It continues with a second sentence here.",
    "A third sentence closes the upper left text.",
]
RIGHT = [
    "Right column starts with inflation news.",
    "The next right sentence covers wages today.",
    "A third right sentence follows after that.",
]
GRID = [
    ("Measure", "2003", "2004", "2005", "2006"),
    ("Real GDP", "2.6", "4.1", "3.3", "2.9"),
    ("Final sales", "2.4", "4.0", "3.1", "2.7"),
]
GRID_TOKENS = ("Measure", "Real", "GDP", "Final", "sales", "2.6", "4.1", "2.4", "4.0", "3.3")
#: Full page width: every line of the page lies in its horizontal span.
GRID_BOX = (60.0, 118.0, 560.0, 170.0)
NOTE_UNDER_GRID = "1. Source note under the grid, 2003 figures."
TAIL_LEFT = "Tail of the left column closes the page."
TAIL_RIGHT = "Tail of the right column closes it too."
EDGE_LINE = "Clipped 7.77 edge words here"


def _page(page, *, edge=False) -> None:
    """Column-major: the whole left column, then the whole right one, in stream order."""
    page.insert_text((72, 60), "\n".join(LEFT), fontsize=9)
    for r, row in enumerate(GRID):
        for c, cell in enumerate(row):
            page.insert_text((76 + c * 100, 130 + r * 16), cell, fontsize=9)
    page.insert_text((72, 185), NOTE_UNDER_GRID, fontsize=9)
    page.insert_text((72, 260), TAIL_LEFT, fontsize=9)
    for i, text in enumerate(RIGHT):
        page.insert_text((330, 60 + i * 16), text, fontsize=9)
    page.insert_text((330, 260), TAIL_RIGHT, fontsize=9)
    if edge:
        # Its centre is below the box (y > 172) but its glyph box crosses the box edge.
        page.insert_text((72, 178), EDGE_LINE, fontsize=9)


def _own_lines(body: str) -> list[str]:
    return [
        ln
        for ln in body.splitlines()
        if ln.strip() and not ln.lstrip().startswith((">", "[page", "![", "["))
    ]


def _without_line_order(monkeypatch) -> None:
    """The ONLY thing toggled: the line-order body (the table_span=False call) is unavailable."""
    real = M._native_order_prose_body

    def _only_span(p, page_num, *, marker_line, png_ref, table_span=True):
        if not table_span:
            return None
        return real(p, page_num, marker_line=marker_line, png_ref=png_ref, table_span=table_span)

    monkeypatch.setattr(M, "_native_order_prose_body", _only_span)


def _floor(ps) -> str:
    return withheld_table_floor_text(ps, 1)


# ---------------------------------------------------------------------------
# what comes back
# ---------------------------------------------------------------------------


def test_line_order_body_ships_prose_in_layer_order(tmp_path: Path, monkeypatch) -> None:
    """DIFFERENCE: same page, only the line-order body toggled. Off: the whole-page marker (the
    span body abstains on a full-width box and the bands body welds the columns). On: the
    prose in the order the layer stores it, with the table and its note withheld."""
    _state, ps = _build(tmp_path, _page, [GRID_BOX])
    # the premise: the span body has nothing to ship, and the bands body fails the order gate
    assert M._native_order_prose_body(ps, 1, marker_line=MARKER, png_ref=PNG) is None, (
        "the span body shipped: this page no longer reaches the line-order body"
    )
    with monkeypatch.context() as mp:
        _without_line_order(mp)
        off = _floor(ps)
    on = _floor(ps)
    assert _whole_page(off), off
    assert not _whole_page(on)
    assert has_line_order_notice(on, 1)
    assert not has_line_order_notice(off, 1)
    assert M._is_prose_recovery_body(on, 1)
    assert _own_lines(on) == [*LEFT, TAIL_LEFT, *RIGHT, TAIL_RIGHT], _own_lines(on)
    for token in GRID_TOKENS:
        assert token not in on, f"table content {token!r} shipped"
    assert NOTE_UNDER_GRID not in on, "the numbered note under the grid shipped"
    assert MARKER in on


def test_a_body_the_order_gate_cannot_confirm_keeps_the_marker(tmp_path: Path) -> None:
    """DIFFERENCE: the same page with its ``native_text`` in another order than the layer's
    words keeps the whole-page marker; the line-order body is held to the same order gate."""
    _s, ps = _build(tmp_path, _page, [GRID_BOX], name="ok.pdf")
    _s2, bad = _build(tmp_path, _page, [GRID_BOX], name="bad.pdf")
    bad.native_text = "\n".join(reversed(bad.native_text.splitlines()))
    ok_body, bad_body = _floor(ps), _floor(bad)
    assert not _whole_page(ok_body)
    assert _whole_page(bad_body), bad_body


# ---------------------------------------------------------------------------
# the leak guard
# ---------------------------------------------------------------------------


def test_no_word_overlapping_a_box_ships(tmp_path: Path) -> None:
    """A line whose glyphs cross a box edge but whose centre is outside is not 'inside' for the
    band rule; the line-order body must still withhold it. DIFFERENCE between a page with the
    clipped line and the same page without it: the clipped line never ships, and the rest does."""
    clipped_box = (60.0, 118.0, 560.0, 172.0)
    _s, with_edge = _build(
        tmp_path, lambda pg: _page(pg, edge=True), [clipped_box], name="edge.pdf"
    )
    _s2, without = _build(tmp_path, _page, [clipped_box], name="plain.pdf")
    # the premise: the band rule alone does not see the clipped line
    words = [w for w in with_edge.native_words if w[4] == "7.77"]
    assert words, "fixture lost the clipped word"
    w = words[0]
    cy = (w[1] + w[3]) / 2.0
    assert clipped_box[3] < cy and w[1] < clipped_box[3], (w, clipped_box)
    body = _floor(with_edge)
    assert has_line_order_notice(body, 1)
    assert "7.77" not in body, "a word overlapping a withheld box shipped"
    assert _own_lines(body) == _own_lines(_floor(without))


def test_a_clipped_word_beside_the_box_does_not_ship(tmp_path: Path) -> None:
    """The continuation walk only reaches bands above and below a box. A line on a baseline
    BETWEEN the table's rows whose first word crosses the box's right edge (centre outside) is
    in no inside band, so only the overlap guard withholds it. DIFFERENCE: the same page without
    that line ships the same prose."""
    narrow = (60.0, 118.0, 400.0, 170.0)

    def grid_page(page, *, clip=False) -> None:
        page.insert_text((72, 60), "\n".join(LEFT), fontsize=9)
        for r, row in enumerate(GRID):
            for c, cell in enumerate(row[:3]):
                page.insert_text((76 + c * 100, 130 + r * 16), cell, fontsize=9)
        page.insert_text((72, 260), TAIL_LEFT, fontsize=9)
        for i, text in enumerate(RIGHT):
            page.insert_text((330, 60 + i * 16), text, fontsize=9)
        page.insert_text((330, 260), TAIL_RIGHT, fontsize=9)
        if clip:
            page.insert_text((394, 138), "end 7.77 and more words here", fontsize=9)

    _s, ps = _build(tmp_path, lambda pg: grid_page(pg, clip=True), [narrow], name="clip.pdf")
    _s2, plain = _build(tmp_path, grid_page, [narrow], name="noclip.pdf")
    end = [w for w in ps.native_words if w[4] == "end"][0]
    assert end[0] < narrow[2] < (end[0] + end[2]) / 2.0, (end, narrow)
    assert M._native_order_prose_body(ps, 1, marker_line=MARKER, png_ref=PNG) is None, (
        "the span body shipped: this page no longer reaches the line-order body"
    )
    body = _floor(ps)
    assert has_line_order_notice(body, 1)
    assert "7.77" not in body and not any("end 7.77" in ln for ln in _own_lines(body))
    assert _own_lines(body) == _own_lines(_floor(plain)) != []


def test_line_order_body_leaks_no_numeric_token_of_any_box(tmp_path: Path) -> None:
    """Every word overlapping a detected box is absent from the body, for every box."""
    second = (60.0, 230.0, 560.0, 250.0)

    def two_tables(page) -> None:
        _page(page, edge=True)
        page.insert_text((76, 244), "Second", fontsize=9)
        page.insert_text((200, 244), "9.91", fontsize=9)
        page.insert_text((300, 244), "8.82", fontsize=9)

    _s, ps = _build(tmp_path, two_tables, [(60.0, 118.0, 560.0, 172.0), second])
    body = _floor(ps)
    assert has_line_order_notice(body, 1)
    boxes = ps.detected_table_bboxes
    shipped = set(body.replace("\\", "").split())
    for w in ps.native_words:
        overlapping = any(
            min(w[2], b[2]) > max(w[0], b[0]) and min(w[3], b[3]) > max(w[1], b[1]) for b in boxes
        )
        if overlapping and any(ch.isdigit() for ch in w[4]):
            assert w[4] not in shipped, f"numeric token {w[4]!r} from inside a box shipped"


# ---------------------------------------------------------------------------
# the other gates still refuse
# ---------------------------------------------------------------------------


def _arm_pair(ps, monkeypatch) -> tuple[str, str]:
    with monkeypatch.context() as mp:
        _without_line_order(mp)
        off = _floor(ps)
    return off, _floor(ps)


def _reasons(ps) -> list[str]:
    reasons: list[str] = []
    table_floor_text_for_source(
        ps,
        1,
        ps.native_text,
        prose_recovery=True,
        geometry_floor=True,
        chart_floor_offered=False,
        refusals=reasons,
    )
    return reasons


def test_control_arm_the_line_order_body_changes_this_page(tmp_path: Path, monkeypatch) -> None:
    _s, ps = _build(tmp_path, _page, [GRID_BOX])
    off, on = _arm_pair(ps, monkeypatch)
    assert off != on


def _numeric_rows_below(page) -> None:
    _page(page)
    for i in range(6):
        y = 320 + i * 14
        page.insert_text((72, y), f"Series{i}", fontsize=9)
        for c, v in enumerate(("1.1", "2.2", "3.3")):
            page.insert_text((200 + c * 60, y), f"{v}{i}", fontsize=9)


@pytest.mark.parametrize(
    "gate",
    ["rotated_lane", "rotated_page", "no_boxes", "box_unsane", "structure_outside"],
)
def test_the_other_gates_still_refuse(tmp_path: Path, monkeypatch, gate: str) -> None:
    """DIFFERENCE: for each gate other than the order gate, toggling the line-order body changes
    nothing (identical bytes, the whole-page marker, and the refusal names that gate)."""
    draw = _numeric_rows_below if gate == "structure_outside" else _page
    _s, ps = _build(tmp_path, draw, [GRID_BOX], name=f"{gate}.pdf")
    expected = {
        "rotated_lane": "rotated_lane",
        "rotated_page": "rotated_page",
        "no_boxes": "no_boxes_no_chart_floor",
        "box_unsane": "box_unsane",
        "structure_outside": "structure_outside",
    }[gate]
    if gate == "rotated_lane":
        ps.native_table_lane_refused_rotated = True
    elif gate == "rotated_page":
        ps.page_rotation = 90
    elif gate == "no_boxes":
        ps.detected_table_count = 0
        ps.detected_table_bboxes = []
    elif gate == "box_unsane":
        ps.detected_table_bboxes = [(500.0, 700.0, 520.0, 710.0)]
        ps.detected_table_count = 1
    off, on = _arm_pair(ps, monkeypatch)
    assert off == on
    assert _whole_page(on), on
    assert _reasons(ps)[0] == expected, _reasons(ps)


# ---------------------------------------------------------------------------
# marking
# ---------------------------------------------------------------------------


def test_the_page_is_marked_as_loudly_as_any_recovered_floor_page(tmp_path: Path) -> None:
    """The finalized page of a line-order body: ERROR, not audit-passed, the typed recovery flag
    set, and the named note on the output (page, sidecar and manifest read these)."""
    state, ps = _build(tmp_path, _page, [GRID_BOX])
    ps.table_ladder_disposition = M.FailureMode.TABLE_WITHHELD
    out = M._winning_page_output(state, 1)
    assert has_line_order_notice(out.text, 1), out.text
    assert out.status == M.PageStatus.ERROR
    assert out.audit_passed is False
    assert out.scanned_prose_recovered is True
    assert FLOOR_LINE_ORDER_NOTE in out.audit_notes
    assert MARKER in out.text


def test_structure_class_floor_ships_the_same_body(tmp_path: Path) -> None:
    """The shared floor: the S1 ending's helper returns the same line-order body."""
    _s, ps = _build(tmp_path, _page, [GRID_BOX])
    assert has_line_order_notice(structure_class_floor_text(ps, 1, chart_floor_offered=False), 1)
