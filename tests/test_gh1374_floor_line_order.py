"""#1374: a page whose band order cannot be confirmed ships the layer's line-order body.

Owner ruling on #1374: when the floor's reading order cannot be confirmed, the page may ship the
words in the order the PDF text layer stores them, with every detected table box still withheld
and the page still marked. Scope is the ``order_gate`` ONLY.

Why it needs a second body: the #1261 in-order fallback withholds every line whose x-range
overlaps a box (the span rule), so on a page whose table box spans the page width it ships
nothing and abstains. Measured on the 114 order-gate pages of the 156-document Fed run: no line
ships on any of them. The line-order body drops the span rule and decides what continues a table
by ROW PITCH (never by what a row says), plus an edge-overlap guard.

Review round (PR #1376): the first revision decided continuation by digits (the #1261 round 2
design). Constructed pages showed it shipping label-only rows, ``n.a.`` rows, one-line memo rows,
half of a ruled cell, and a clipped row released by a sentence in the other column. Each such
input is a test below; every claim is a DIFFERENCE between two runs in one process (#253/#257).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_gh1266_floor_geometry import (  # noqa: E402
    MARKER,
    PNG,
    _build,
    _restore_and_assemble,
    _run,
    _whole_page,
)

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
#: Full page width: every line of the page lies in its horizontal span. The grid rows sit at
#: baselines 130, 146, 162: a row pitch of 16 pt.
GRID_BOX = (60.0, 118.0, 560.0, 170.0)
NOTE_UNDER_GRID = "1. Source note under the grid, 2003 figures."
TAIL_LEFT = "Tail of the left column closes the page."
TAIL_RIGHT = "Tail of the right column closes it too."
EDGE_LINE = "Clipped 7.77 edge words here"


def _base(page, *, note=True) -> None:
    """Column-major: the whole left column, then the whole right one, in stream order."""
    page.insert_text((72, 60), "\n".join(LEFT), fontsize=9)
    for r, row in enumerate(GRID):
        for c, cell in enumerate(row):
            page.insert_text((76 + c * 100, 130 + r * 16), cell, fontsize=9)
    if note:
        # 14 pt under the last grid row: inside the table's own row pitch (16).
        page.insert_text((72, 176), NOTE_UNDER_GRID, fontsize=9)
    page.insert_text((72, 260), TAIL_LEFT, fontsize=9)
    for i, text in enumerate(RIGHT):
        page.insert_text((330, 60 + i * 16), text, fontsize=9)
    page.insert_text((330, 260), TAIL_RIGHT, fontsize=9)


def _with(extra, *, note=True):
    def draw(page) -> None:
        _base(page, note=note)
        extra(page)

    return draw


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


def _reaches_line_order(ps) -> None:
    assert M._native_order_prose_body(ps, 1, marker_line=MARKER, png_ref=PNG) is None, (
        "the span body shipped: this page no longer reaches the line-order body"
    )


# ---------------------------------------------------------------------------
# what comes back
# ---------------------------------------------------------------------------


def test_line_order_body_ships_prose_in_layer_order(tmp_path: Path, monkeypatch) -> None:
    """DIFFERENCE: same page, only the line-order body toggled. Off: the whole-page marker (the
    span body abstains on a full-width box and the bands body welds the columns). On: the
    prose in the order the layer stores it, with the table and its note withheld."""
    _state, ps = _build(tmp_path, _base, [GRID_BOX])
    _reaches_line_order(ps)
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
    assert NOTE_UNDER_GRID not in on, "the numbered note 14 pt under the grid shipped"
    assert MARKER in on


def test_a_body_the_order_gate_cannot_confirm_keeps_the_marker(tmp_path: Path) -> None:
    """DIFFERENCE: the same page with its ``native_text`` in another order than the layer's
    words keeps the whole-page marker; the line-order body is held to the same order gate."""
    _s, ps = _build(tmp_path, _base, [GRID_BOX], name="ok.pdf")
    _s2, bad = _build(tmp_path, _base, [GRID_BOX], name="bad.pdf")
    bad.native_text = "\n".join(reversed(bad.native_text.splitlines()))
    ok_body, bad_body = _floor(ps), _floor(bad)
    assert not _whole_page(ok_body)
    assert _whole_page(bad_body), bad_body


def test_a_box_with_one_band_has_no_pitch_and_the_body_abstains(tmp_path: Path) -> None:
    """No pitch is invented: a box with a single band centred inside it keeps the marker.
    DIFFERENCE against the same page whose box holds the whole grid."""
    one_row = (60.0, 124.0, 560.0, 136.0)
    _s, single = _build(tmp_path, _base, [one_row], name="one.pdf")
    _s2, full = _build(tmp_path, _base, [GRID_BOX], name="full.pdf")
    assert has_line_order_notice(_floor(full), 1)
    assert _whole_page(_floor(single)), _floor(single)


# ---------------------------------------------------------------------------
# the table continues past its box: every reviewer input ships nothing of the table
# ---------------------------------------------------------------------------

#: (id, drawn rows, tokens that must not ship). Rows are drawn below the grid box at a pitch
#: no wider than the grid's own (16 pt): 14 pt, so by geometry they are the table's own rows.
CONTINUATIONS = [
    (
        "label_then_total_row",  # Astra round 3: a label-only row, then a total row
        [
            ((76, 176), "Pension liabilities"),
            ((76, 190), "Total pension obligation"),
            ((300, 190), "12 million dollars"),
        ],
        ("Pension liabilities", "Total pension obligation", "12 million dollars"),
    ),
    (
        "long_label_then_total_row",  # round 4: a longer label reopens a threshold
        [
            ((76, 176), "Pension liabilities for retired employees"),
            ((76, 190), "Total pension obligation"),
            ((300, 190), "12 million dollars"),
        ],
        ("Pension liabilities for retired", "Total pension obligation", "12 million dollars"),
    ),
    (
        "categorical_row",  # a categorical cell
        [((76, 176), "Coverage included"), ((76, 190), "Yes")],
        ("Coverage included", "Yes"),
    ),
    (
        "ruled_two_line_cell",  # round 5: half of a multi-line ruled cell
        [
            ((76, 176), "Liabilities for retired employees,"),
            ((76, 190), "for retired employees."),
            ((76, 204), "net of plan assets."),
        ],
        ("for retired employees.", "net of plan assets."),
    ),
    (
        "memo_rows",  # three one-line rows under a memo heading
        [
            ((76, 176), "Memo items"),
            ((76, 190), "Core PCE inflation 1.9 percent"),
            ((76, 204), "Unemployment rate 4.5 percent"),
            ((76, 218), "Federal funds rate 5.25 percent"),
        ],
        ("Memo items", "Core PCE inflation", "Unemployment rate", "Federal funds rate"),
    ),
    (
        "word_cells",  # n.a. / up / down / unchanged cells with their labels
        [
            ((76, 176), "Retail sales"),
            ((300, 176), "up"),
            ((76, 190), "Housing starts"),
            ((300, 190), "down"),
            ((76, 204), "Exports"),
            ((300, 204), "n.a."),
            ((76, 218), "Prices"),
            ((300, 218), "unchanged"),
        ],
        ("Retail sales", "Housing starts", "Exports", "Prices", "unchanged", "n.a."),
    ),
    (
        "unboxed_word_table",  # an unboxed word-cell table under the boxed one
        [
            ((76, 176), "District"),
            ((180, 176), "Activity"),
            ((300, 176), "Prices"),
            ((76, 190), "Boston"),
            ((180, 190), "Modest"),
            ((300, 190), "Stable"),
            ((76, 204), "Chicago"),
            ((180, 204), "Slight"),
            ((300, 204), "Firm"),
        ],
        ("District", "Boston", "Modest", "Chicago", "Slight", "Firm", "Stable"),
    ),
]


@pytest.mark.parametrize("name,rows,tokens", CONTINUATIONS, ids=[c[0] for c in CONTINUATIONS])
def test_a_table_continuing_at_row_pitch_ships_none_of_its_rows(
    tmp_path: Path, name: str, rows, tokens
) -> None:
    """DIFFERENCE: the same page without the rows ships the same prose; with them, none of the
    rows' words ship. The rows lie within the grid's own row pitch of the box, so they are
    table by geometry, whatever they say (digits, words, ``n.a.``, a sentence ending)."""

    def draw(page) -> None:
        for (x, y), text in rows:
            page.insert_text((x, y), text, fontsize=9)

    _s, with_rows = _build(tmp_path, _with(draw, note=False), [GRID_BOX], name=f"{name}.pdf")
    _s2, without = _build(tmp_path, _with(lambda pg: None, note=False), [GRID_BOX], name="b.pdf")
    _reaches_line_order(with_rows)
    body = _floor(with_rows)
    assert has_line_order_notice(body, 1)
    for token in tokens:
        assert token not in body, f"{name}: table row text {token!r} shipped"
    assert _own_lines(body) == _own_lines(_floor(without)) != []


def test_prose_beyond_the_row_pitch_still_ships(tmp_path: Path) -> None:
    """DIFFERENCE: the pitch rule is not a blanket. A paragraph set further below the grid than
    its own row spacing ships; the same paragraph set at row pitch does not."""
    para = "A paragraph set well below the table keeps its words."

    def far(page) -> None:
        page.insert_text((72, 220), para, fontsize=9)

    def near(page) -> None:
        page.insert_text((72, 176), para, fontsize=9)

    _s, far_ps = _build(tmp_path, _with(far, note=False), [GRID_BOX], name="far.pdf")
    _s2, near_ps = _build(tmp_path, _with(near, note=False), [GRID_BOX], name="near.pdf")
    assert para in _own_lines(_floor(far_ps))
    assert para not in _own_lines(_floor(near_ps))


def test_a_sentence_in_the_other_column_does_not_release_a_clipped_row(tmp_path: Path) -> None:
    """Reviewer finding 2. Two boxes, one per column; a row clipped out of the left box sits
    inside its row pitch, and a sentence of the right column sits between them in y. The walk
    skips a band with no line in the box's x-span instead of stopping at it. DIFFERENCE: the
    same page without the right-column sentence withholds the row too."""
    left_box = (70.0, 118.0, 300.0, 172.0)
    right_box = (328.0, 218.0, 560.0, 270.0)

    def draw(page, *, sentence: bool) -> None:
        page.insert_text((72, 60), "\n".join(LEFT), fontsize=9)
        for r, y in enumerate((126, 147, 168)):
            page.insert_text((76, y), ("Measure", "Real GDP", "Final sales")[r], fontsize=9)
            page.insert_text((200, y), ("2003", "2.6", "2.4")[r], fontsize=9)
        page.insert_text((76, 188), "Total pension obligation", fontsize=9)
        page.insert_text((200, 188), "12 million dollars", fontsize=9)
        for r, y in enumerate((226, 247, 268)):
            page.insert_text((334, y), ("Measure", "Real GDP", "Final sales")[r], fontsize=9)
            page.insert_text((470, y), ("2003", "2.6", "2.4")[r], fontsize=9)
        for i, text in enumerate(RIGHT):
            page.insert_text((330, 60 + i * 16), text, fontsize=9)
        if sentence:
            page.insert_text(
                (330, 179), "A right column sentence sits level with the row.", fontsize=9
            )
        page.insert_text((72, 320), TAIL_LEFT, fontsize=9)

    _s, with_s = _build(
        tmp_path, lambda pg: draw(pg, sentence=True), [left_box, right_box], name="s.pdf"
    )
    _s2, without = _build(
        tmp_path, lambda pg: draw(pg, sentence=False), [left_box, right_box], name="n.pdf"
    )
    for ps in (with_s, without):
        _reaches_line_order(ps)
    a, b = _floor(with_s), _floor(without)
    for body in (a, b):
        assert has_line_order_notice(body, 1)
        assert "Total pension obligation" not in body and "12 million dollars" not in body
    assert "A right column sentence" in a


def test_a_label_between_two_boxes_at_row_pitch_is_table(tmp_path: Path) -> None:
    """The ``Liabilities`` leak of MPR 2024-07 p53: a label printed in the gap between two
    boxes. DIFFERENCE against the same page without it: only that line moves."""
    upper = (60.0, 118.0, 560.0, 168.0)
    lower = (60.0, 181.0, 560.0, 228.0)

    def draw(page, *, label: bool) -> None:
        _base(page, note=False)
        for r, y in enumerate((192, 208, 224)):
            page.insert_text((76, y), ("Measure", "Real GDP", "Final sales")[r], fontsize=9)
            page.insert_text((300, y), ("2003", "2.6", "2.4")[r], fontsize=9)
        if label:
            page.insert_text((76, 178), "Liabilities", fontsize=9)

    _s, with_l = _build(tmp_path, lambda pg: draw(pg, label=True), [upper, lower], name="l.pdf")
    _s2, without = _build(tmp_path, lambda pg: draw(pg, label=False), [upper, lower], name="n.pdf")
    body = _floor(with_l)
    assert has_line_order_notice(body, 1)
    assert "Liabilities" not in body
    assert _own_lines(body) == _own_lines(_floor(without))


def test_a_numerals_withheld_line_withholds_the_lines_in_its_band(tmp_path: Path) -> None:
    """``band_span`` on the line-order path: a row printed as a label line plus a values line
    in ONE band, outside every box, is withheld whole. DIFFERENCE against the same row without
    the values line (the label alone is a sentence-like line and ships)."""

    def draw(page, *, values: bool) -> None:
        _base(page, note=False)
        page.insert_text(
            (72, 300), "Real GDP growth is projected to ease over the coming year", fontsize=9
        )
        if values:
            for c, v in enumerate(("2.6", "4.1", "3.3", "2.9")):
                page.insert_text((330 + c * 40, 300), v, fontsize=9)

    _s, with_v = _build(tmp_path, lambda pg: draw(pg, values=True), [GRID_BOX], name="v.pdf")
    _s2, without = _build(tmp_path, lambda pg: draw(pg, values=False), [GRID_BOX], name="nv.pdf")
    a, b = _floor(with_v), _floor(without)
    assert has_line_order_notice(a, 1) and has_line_order_notice(b, 1)
    assert "Real GDP growth is projected to ease over the coming year" in b
    assert "Real GDP growth is projected to ease over the coming year" not in a


# ---------------------------------------------------------------------------
# the leak guard
# ---------------------------------------------------------------------------


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
    _reaches_line_order(ps)
    body = _floor(ps)
    assert has_line_order_notice(body, 1)
    assert "7.77" not in body and not any("end 7.77" in ln for ln in _own_lines(body))
    assert _own_lines(body) == _own_lines(_floor(plain)) != []


def test_a_word_clipping_a_box_edge_does_not_ship(tmp_path: Path) -> None:
    """A line whose glyphs cross the box's top edge but whose centre is outside is not 'inside'
    for the band rule. It never ships. DIFFERENCE against the same page without it."""
    box = (60.0, 118.0, 560.0, 170.0)

    def edge(page) -> None:
        page.insert_text((72, 119), EDGE_LINE, fontsize=9)

    _s, with_edge = _build(tmp_path, _with(edge), [box], name="edge.pdf")
    _s2, without = _build(tmp_path, _base, [box], name="plain.pdf")
    w = [x for x in with_edge.native_words if x[4] == "7.77"][0]
    assert (w[1] + w[3]) / 2.0 < box[1] < w[3], (w, box)
    body = _floor(with_edge)
    assert has_line_order_notice(body, 1)
    assert "7.77" not in body, "a word overlapping a withheld box shipped"
    assert _own_lines(body) == _own_lines(_floor(without))


def test_line_order_body_leaks_no_numeric_token_of_any_box(tmp_path: Path) -> None:
    """Every word overlapping a detected box is absent from the body, for every box."""
    second = (60.0, 222.0, 560.0, 252.0)

    def two_tables(page) -> None:
        _base(page)
        page.insert_text((72, 119), EDGE_LINE, fontsize=9)
        for y, (a, b, c) in ((232, ("Second", "9.91", "8.82")), (246, ("Third", "7.13", "6.64"))):
            page.insert_text((76, y), a, fontsize=9)
            page.insert_text((200, y), b, fontsize=9)
            page.insert_text((300, y), c, fontsize=9)

    _s, ps = _build(tmp_path, two_tables, [GRID_BOX, second])
    body = _floor(ps)
    assert has_line_order_notice(body, 1)
    shipped = set(body.replace("\\", "").split())
    for w in ps.native_words:
        overlapping = any(
            min(w[2], b[2]) > max(w[0], b[0]) and min(w[3], b[3]) > max(w[1], b[1])
            for b in ps.detected_table_bboxes
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
    _s, ps = _build(tmp_path, _base, [GRID_BOX])
    off, on = _arm_pair(ps, monkeypatch)
    assert off != on


def _numeric_rows_below(page) -> None:
    _base(page)
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
    draw = _numeric_rows_below if gate == "structure_outside" else _base
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
    state, ps = _build(tmp_path, _base, [GRID_BOX])
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
    _s, ps = _build(tmp_path, _base, [GRID_BOX])
    assert has_line_order_notice(structure_class_floor_text(ps, 1, chart_floor_offered=False), 1)


def test_the_floor_line_order_pages_reader_names_only_line_order_pages() -> None:
    """The reader behind the audit event, the document note and the CLI line: DIFFERENCE between a
    recovered page with the notice and one without. (A mutant that returns nothing kills it.)"""
    from types import SimpleNamespace

    from socr.pipeline.orchestrator import UnifiedPipeline

    def rec(page: int, text: str, recovered: bool = True):
        return SimpleNamespace(
            output=SimpleNamespace(page_num=page, text=text, scanned_prose_recovered=recovered)
        )

    banner = M.prose_recovered_flag(1, born_digital=True, keep_sentences=True)
    notice = M.LINE_ORDER_NOTICE.format(page_num=1)
    with_notice = rec(1, f"{banner}\n\n{notice}\n\nbody")
    plain = rec(2, M.prose_recovered_flag(2, born_digital=True, keep_sentences=True) + "\n\nbody")
    assert UnifiedPipeline._floor_line_order_pages([with_notice, plain]) == [1]
    note = UnifiedPipeline._scanned_prose_recovered_note([with_notice, plain])
    assert note and "floor_line_order_body" in note and "page(s) 1:" in note.split(";")[-1]
    assert UnifiedPipeline._floor_line_order_pages([plain]) == []
    assert "floor_line_order_body" not in (
        UnifiedPipeline._scanned_prose_recovered_note([plain]) or ""
    )


# ---------------------------------------------------------------------------
# process(), both provider states, and a real restore
# ---------------------------------------------------------------------------


def _wide_table_pdf(path: Path) -> Path:
    """Two welded columns, a full-width ruled grid the real detector finds, row-pitch rows below."""
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 60), "\n".join(LEFT))
    for r, row in enumerate(GRID):
        for c, cell in enumerate(row):
            page.insert_text((80 + c * 90, 134 + r * 22), cell, fontsize=9)
    for r in range(4):
        page.draw_line(fitz.Point(76, 116 + r * 22), fitz.Point(526, 116 + r * 22))
    for c in range(6):
        page.draw_line(fitz.Point(76 + c * 90, 116), fitz.Point(76 + c * 90, 116 + 3 * 22))
    page.insert_text((72, 200), "Pension liabilities", fontsize=9)
    page.insert_text((72, 222), "Total pension obligation", fontsize=9)
    page.insert_text((300, 222), "12 million dollars", fontsize=9)
    page.insert_text((72, 300), TAIL_LEFT)
    for i, text in enumerate(RIGHT):
        page.insert_text((330, 60 + i * 16), text)
    page.insert_text((330, 300), TAIL_RIGHT)
    doc.save(str(path))
    doc.close()
    return path


def test_process_in_both_provider_states_marks_every_surface_and_restores(
    tmp_path: Path, monkeypatch
) -> None:
    """DIFFERENCE through process(): line-order body on versus off, a populated ladder and an
    empty one (the CI state: no provider). The shipped lines are the same in both provider states;
    the toggle alone moves them. The notice, the page note, the audit event and the document note
    are all present; then a FRESH pipeline restores the page (no words) and re-assembles:
    fragment and final .md are byte-identical to the live run's."""
    import json

    pdf = _wide_table_pdf(tmp_path / "doc.pdf")
    by_state = {}
    for ladder in (True, False):
        tag = "ladder" if ladder else "empty"
        on, on_meta, out = _run(tmp_path, pdf, monkeypatch, ladder=ladder, tag=f"on-{tag}")
        with monkeypatch.context() as mp:
            _without_line_order(mp)
            off, off_meta, _ = _run(tmp_path, pdf, monkeypatch, ladder=ladder, tag=f"off-{tag}")
        assert on_meta["failure_mode"] == off_meta["failure_mode"], "route differs between arms"
        assert has_line_order_notice(on, 1), on
        assert not has_line_order_notice(off, 1)
        assert on != off, f"{tag}: the toggle changed nothing"
        for token in (
            *GRID_TOKENS,
            "Pension liabilities",
            "Total pension obligation",
            "12 million",
        ):
            assert token not in on, f"{tag}: table content {token!r} shipped"
        assert FLOOR_LINE_ORDER_NOTE in (on_meta["winning_output"].get("audit_notes") or [])
        assert on_meta["winning_output"].get("scanned_prose_recovered") is True
        audit = json.loads(next(iter(out.rglob("audit_log.json"))).read_text())
        assert any(e["kind"] == "floor_line_order_body" for e in audit["events"]), tag
        meta = json.loads(next(iter(out.rglob("metadata.json"))).read_text())
        assert "floor_line_order_body" in json.dumps(meta), tag
        manifest = json.loads(next(iter(out.rglob("manifest.json"))).read_text())
        assert [e.get("floor_line_order_body") for e in manifest["entries"].values()] == [True], tag
        by_state[ladder] = (on, on_meta, out)
    assert _own_lines(by_state[True][0]) == _own_lines(by_state[False][0])

    frag, meta, out = by_state[True]
    frag_path = next(iter(out.rglob("pages/00001.md")))
    finals = [p for p in out.rglob("*.md") if p.parent.name != "pages"]
    assert finals, "the live run wrote no final .md"
    final_text = finals[0].read_text()
    frag_path.unlink()
    finals[0].unlink()
    _restore_and_assemble(pdf, meta, out, monkeypatch, expected_body=frag)
    assert frag_path.read_text() == frag
    assert [p for p in out.rglob("*.md") if p.parent.name != "pages"][0].read_text() == final_text


def test_the_manifest_entry_records_the_line_order_body() -> None:
    """``floor_line_order_body`` rides the manifest entry only when true (byte-compatible when
    false) and survives a round trip."""
    from socr.core.manifest import ManifestEntry, PageFingerprint

    fp = PageFingerprint(pdf_file_hash="h", page_num=1, render_dpi=0, engine="native")
    plain = ManifestEntry(page_num=1, blob_ref="x", fingerprint=fp)
    marked = ManifestEntry(page_num=1, blob_ref="x", fingerprint=fp, floor_line_order_body=True)
    assert "floor_line_order_body" not in plain.to_dict()
    assert marked.to_dict()["floor_line_order_body"] is True
    assert ManifestEntry.from_dict(marked.to_dict()).floor_line_order_body is True
    assert ManifestEntry.from_dict(plain.to_dict()).floor_line_order_body is False
