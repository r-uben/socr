"""#1261: a two-column page whose table is withheld keeps its prose, in the layer's own order.

The born-digital geometry floor ships the words outside every detected box. Its bands are
clustered on y only, so a page drawn column-major welds its columns and the reading-order gate
sent the whole page to the marker (136 pages, 20,318 text-layer words on the 51270eb5 Fed run).
The fallback re-emits the SAME shippable words in the layer's own line order, after the box
sanity and the structure-outside gates, and under the same order gate.

What stays refused is asserted next to what comes back, as DIFFERENCES between two runs in one
process, never as a tuple measured on one machine (#253/#257).
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
    _whole_page,
)

from socr.core import manifest as M  # noqa: E402
from socr.core.manifest import (  # noqa: E402
    BORN_DIGITAL_KEEP_SENTENCES_FLAG,
    _winning_page_output,
    structure_class_floor_text,
    table_floor_credential_holds,
    table_floor_text_for_source,
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
LEFT_TAIL = "Tail of the left column after the table."
RIGHT_TAIL = "Tail of the right column."
SHARES_BASELINE = "Right text on a grid baseline."
GRID = [("Measure", "2003", "2004"), ("Real GDP", "2.6", "4.1"), ("Final sales", "2.4", "4.0")]
GRID_BOX = (70.0, 118.0, 230.0, 160.0)
GRID_TOKENS = ("Measure", "Real", "GDP", "Final", "sales", "2.6", "4.1", "2.4", "4.0")


def _columns(page, *, extra=None) -> None:
    """Column-major: the whole left column, then the whole right one, in stream order."""
    for i, text in enumerate(LEFT):
        page.insert_text((72, 60 + i * 16), text, fontsize=9)
    for r, row in enumerate(GRID):
        for c, cell in enumerate(row):
            page.insert_text((76 + c * 50, 130 + r * 16), cell, fontsize=9)
    page.insert_text((72, 260), LEFT_TAIL, fontsize=9)
    if extra:
        extra(page)
    for i, text in enumerate(RIGHT):
        page.insert_text((330, 60 + i * 16), text, fontsize=9)
    page.insert_text((330, 132), SHARES_BASELINE, fontsize=9)
    page.insert_text((330, 260), RIGHT_TAIL, fontsize=9)


def _numeric_rows_below(page) -> None:
    for i in range(6):
        y = 300 + i * 14
        page.insert_text((72, y), f"Series{i}", fontsize=9)
        for c, v in enumerate(("1.1", "2.2", "3.3")):
            page.insert_text((200 + c * 60, y), f"{v}{i}", fontsize=9)


def _own_lines(body: str) -> list[str]:
    return [
        ln
        for ln in body.splitlines()
        if ln.strip() and not ln.lstrip().startswith((">", "[page", "![", "["))
    ]


def _without_fallback(monkeypatch) -> None:
    monkeypatch.setattr(M, "_native_order_prose_body", lambda *a, **k: None)


# ---------------------------------------------------------------------------
# what comes back
# ---------------------------------------------------------------------------


def test_the_fallback_returns_the_prose_in_the_layers_order(tmp_path: Path, monkeypatch) -> None:
    """DIFFERENCE: same page, only the fallback toggled. Off: the whole-page marker. On:
    exactly the shippable prose, column by column, with the table and the baseline-sharing
    sentence withheld behind a marker."""
    _state, ps = _build(tmp_path, _columns, [GRID_BOX])
    with monkeypatch.context() as mp:
        _without_fallback(mp)
        off = structure_class_floor_text(ps, 1)
    on = structure_class_floor_text(ps, 1)
    assert _whole_page(off), off
    assert not _whole_page(on)
    assert BORN_DIGITAL_KEEP_SENTENCES_FLAG.format(page_num=1) in on
    assert _own_lines(on) == [*LEFT, LEFT_TAIL, *RIGHT, RIGHT_TAIL], _own_lines(on)
    for token in GRID_TOKENS:
        assert token not in on, f"table content {token!r} shipped"
    assert SHARES_BASELINE not in on, "a sentence on a table row's baseline shipped"
    assert MARKER in on


def test_every_word_the_fallback_ships_is_a_word_the_bands_body_would_ship(
    tmp_path: Path,
) -> None:
    """The fallback changes the ORDER of the surviving words and admits none: on a page whose
    bands body passes the order gate (one column), the two bodies carry the same lines."""
    from socr.core.manifest import native_prose_floor_text

    def one_column(page) -> None:
        for i, text in enumerate(LEFT):
            page.insert_text((72, 60 + i * 16), text, fontsize=9)
        for r, row in enumerate(GRID):
            for c, cell in enumerate(row):
                page.insert_text((76 + c * 50, 130 + r * 16), cell, fontsize=9)
        page.insert_text((72, 260), LEFT_TAIL, fontsize=9)

    _state, ps = _build(tmp_path, one_column, [GRID_BOX])
    bands = native_prose_floor_text(ps, 1, marker_line=MARKER, png_ref=PNG, keep_sentences=True)
    ordered = M._native_order_prose_body(ps, 1, marker_line=MARKER, png_ref=PNG)
    assert bands is not None and ordered is not None
    assert _own_lines(ordered) == _own_lines(bands)


# ---------------------------------------------------------------------------
# what stays refused
# ---------------------------------------------------------------------------


def test_the_fallback_is_held_to_the_order_gate(tmp_path: Path) -> None:
    """The fallback ships words in the layer's own order, and the layer's own order is the
    page's ``native_text``. DIFFERENCE: the same page with its ``native_text`` in another
    order (a layer whose block order and word order disagree) keeps the whole-page marker."""
    _s, ps = _build(tmp_path, _columns, [GRID_BOX], name="o.pdf")
    reasons: list[str] = []
    ok = table_floor_text_for_source(
        ps,
        1,
        ps.native_text,
        prose_recovery=True,
        geometry_floor=True,
        chart_floor_offered=False,
        refusals=reasons,
    )
    assert not _whole_page(ok) and reasons == []
    ps.native_text = "\n".join(reversed(ps.native_text.splitlines()))
    body = table_floor_text_for_source(
        ps,
        1,
        ps.native_text,
        prose_recovery=True,
        geometry_floor=True,
        chart_floor_offered=False,
        refusals=reasons,
    )
    assert _whole_page(body), body
    assert reasons == ["order_gate"], reasons


def test_structure_printed_outside_every_box_still_gates_the_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    """The structure-outside gate runs BEFORE the fallback: a column-major page with unboxed
    numeric rows is the whole-page marker with the fallback on, and the recorded reason is the
    gate, not the order check. DIFFERENCE against the same page without the rows."""
    _gated, gated_ps = _build(
        tmp_path, lambda pg: _columns(pg, extra=_numeric_rows_below), [GRID_BOX], name="g.pdf"
    )
    _clean, clean_ps = _build(tmp_path, _columns, [GRID_BOX], name="c.pdf")
    reasons: list[str] = []
    body = table_floor_text_for_source(
        gated_ps,
        1,
        gated_ps.native_text,
        prose_recovery=True,
        geometry_floor=True,
        chart_floor_offered=False,
        refusals=reasons,
    )
    assert _whole_page(body), body
    assert reasons == ["structure_outside"], reasons
    assert "Series" not in body
    assert not _whole_page(structure_class_floor_text(clean_ps, 1))


def test_an_unsane_box_still_gates_the_fallback(tmp_path: Path) -> None:
    """A box that captures no table band is not the table it claims to bound: whole page."""
    far_box = (70.0, 400.0, 230.0, 440.0)  # empty region: no band inside it
    _s, ps = _build(tmp_path, _columns, [far_box])
    reasons: list[str] = []
    body = table_floor_text_for_source(
        ps,
        1,
        ps.native_text,
        prose_recovery=True,
        geometry_floor=True,
        chart_floor_offered=False,
        refusals=reasons,
    )
    assert _whole_page(body)
    assert reasons == ["box_unsane"], reasons


def test_rotation_prior_ocr_and_scans_stay_refused(tmp_path: Path) -> None:
    ok_state, ok_ps = _build(tmp_path, _columns, [GRID_BOX], name="ok.pdf")
    assert not _whole_page(structure_class_floor_text(ok_ps, 1))
    _r, rotated = _build(tmp_path, _columns, [GRID_BOX], name="r.pdf")
    rotated.page_rotation = 90
    assert _whole_page(structure_class_floor_text(rotated, 1))
    _p, prior = _build(tmp_path, _columns, [GRID_BOX], name="p.pdf")
    prior.prior_ocr_layer = True
    assert _whole_page(structure_class_floor_text(prior, 1))
    _s, scan = _build(tmp_path, _columns, [GRID_BOX], born_digital=False, name="s.pdf")
    scan_body = structure_class_floor_text(scan, 1)
    assert BORN_DIGITAL_KEEP_SENTENCES_FLAG.format(page_num=1) not in scan_body


def test_a_layer_line_holding_shipped_and_withheld_words_is_withheld_whole(
    tmp_path: Path,
) -> None:
    """A baseline shared by prose and a table: one native line whose first word sits in a
    prose band and whose second sits inside the box. The line is withheld, not shipped."""
    _s, ps = _build(tmp_path, _columns, [GRID_BOX])
    # (x0, y0, x1, y1, text, block, line, word): one layer line, two bands.
    ps.native_words = [
        (72.0, 60.0, 100.0, 70.0, "Intro", 0, 0, 0),
        (104.0, 60.0, 140.0, 70.0, "sentence", 0, 0, 1),
        (72.0, 62.0, 100.0, 72.0, "Sales", 1, 0, 0),
        (80.0, 130.0, 110.0, 140.0, "12.5", 1, 0, 1),
        (80.0, 150.0, 110.0, 160.0, "13.5", 2, 0, 0),
    ]
    body = M._native_order_prose_body(ps, 1, marker_line=MARKER, png_ref=PNG)
    assert body is not None
    assert "12.5" not in body and "Sales" not in body
    assert "Intro sentence" in body


# ---------------------------------------------------------------------------
# credential, resume
# ---------------------------------------------------------------------------


def test_the_floor_credential_holds_and_a_second_selection_is_byte_identical(
    tmp_path: Path,
) -> None:
    _s, ps = _build(tmp_path, _columns, [GRID_BOX])
    first = _winning_page_output(_s, 1)
    second = _winning_page_output(_s, 1)
    assert first.text == second.text
    assert _own_lines(first.text) == [*LEFT, LEFT_TAIL, *RIGHT, RIGHT_TAIL]
    assert first.table_floor_body is True and table_floor_credential_holds(first)


def _column_pdf(path: Path) -> Path:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    for i, text in enumerate(LEFT):
        page.insert_text((72, 60 + i * 16), text)
    rows = [["Real GDP", "2.6", "4.1"], ["Previous", "2.9", "3.9"], ["Final sales", "2.4", "4.0"]]
    for r, cells in enumerate(rows):
        for c, cell in enumerate(cells):
            page.insert_text((80 + c * 50, 130 + r * 22), cell, fontsize=9)
    for r in range(4):
        page.draw_line(fitz.Point(76, 116 + r * 22), fitz.Point(226, 116 + r * 22))
    for c in range(4):
        page.draw_line(fitz.Point(76 + c * 50, 116), fitz.Point(76 + c * 50, 116 + 3 * 22))
    page.insert_text((72, 260), LEFT_TAIL)
    for i, text in enumerate(RIGHT):
        page.insert_text((330, 60 + i * 16), text)
    page.insert_text((330, 260), RIGHT_TAIL)
    doc.save(str(path))
    doc.close()
    return path


def test_process_in_both_provider_states_and_a_real_restore(tmp_path: Path, monkeypatch) -> None:
    """DIFFERENCE through process(): fallback on versus off, a populated ladder and an empty
    one. The page's own lines are the same in both provider states; the toggle alone moves them.
    Then a FRESH pipeline restores the page (no words) and re-assembles: fragment and final
    .md are byte-identical to the live run's."""
    from test_gh1266_floor_geometry import _run

    pdf = _column_pdf(tmp_path / "doc.pdf")
    on_by_state = {}
    for ladder in (True, False):
        tag = "ladder" if ladder else "empty"
        on, on_meta, out = _run(tmp_path, pdf, monkeypatch, ladder=ladder, tag=f"on-{tag}")
        with monkeypatch.context() as mp:
            _without_fallback(mp)
            off, off_meta, _ = _run(tmp_path, pdf, monkeypatch, ladder=ladder, tag=f"off-{tag}")
        assert on_meta["failure_mode"] == off_meta["failure_mode"], "route differs between arms"
        assert on != off, f"{tag}: the toggle changed nothing"
        # The written fragment joins a paragraph's lines: compare the word sequence.
        want = " ".join([*LEFT, LEFT_TAIL, *RIGHT, RIGHT_TAIL]).split()
        assert " ".join(_own_lines(on)).split() == want, _own_lines(on)
        assert not _own_lines(off), "the off arm shipped prose"
        for token in ("Real", "GDP", "2.6", "4.1", "Final"):
            assert token not in on, f"{tag}: table content shipped"
        on_by_state[ladder] = (on, on_meta, out)
    assert _own_lines(on_by_state[True][0]) == _own_lines(on_by_state[False][0])

    frag, meta, out = on_by_state[True]
    frag_path = next(iter(out.rglob("pages/00001.md")))
    finals = [p for p in out.rglob("*.md") if p.parent.name != "pages"]
    assert finals, "the live run wrote no final .md"
    final_text = finals[0].read_text()
    frag_path.unlink()
    finals[0].unlink()
    _restore_and_assemble(pdf, meta, out, monkeypatch, expected_body=frag)
    assert frag_path.read_text() == frag
    assert [p for p in out.rglob("*.md") if p.parent.name != "pages"][0].read_text() == final_text


def test_a_loose_value_line_in_a_prose_band_is_withheld(tmp_path: Path) -> None:
    """A layer line is narrower than its band: the band "Rate label  +-0.3  +-0.7" has words
    outnumbering numerals and ships under the bands rule, but its values are separate layer
    lines. Each line is held to the keep-sentences rule on its own words, so a lone value does
    not ship as a line (MPR 2008-07 p45 on the Fed run). DIFFERENCE: the same lone value inside
    a sentence ships."""
    _s, ps = _build(tmp_path, _columns, [GRID_BOX])
    ps.native_words = [
        (72.0, 60.0, 130.0, 70.0, "Unemployment", 0, 0, 0),
        (132.0, 60.0, 150.0, 70.0, "rate", 0, 0, 1),
        (152.0, 60.0, 190.0, 70.0, "overall", 0, 0, 2),
        (200.0, 60.0, 220.0, 70.0, "\u00b10.3", 1, 0, 0),
        (240.0, 60.0, 260.0, 70.0, "\u00b10.7", 2, 0, 0),
        (72.0, 80.0, 100.0, 90.0, "Growth", 3, 0, 0),
        (102.0, 80.0, 112.0, 90.0, "in", 3, 0, 1),
        (114.0, 80.0, 140.0, 90.0, "2003", 3, 0, 2),
        (142.0, 80.0, 190.0, 90.0, "stayed", 3, 0, 3),
        (192.0, 80.0, 230.0, 90.0, "firm", 3, 0, 4),
    ]
    body = M._native_order_prose_body(ps, 1, marker_line=MARKER, png_ref=PNG)
    assert body is not None
    assert "\u00b10.3" not in body and "\u00b10.7" not in body
    assert "Unemployment rate overall" in body
    assert "Growth in 2003 stayed firm" in body


def test_a_layer_line_that_parses_as_table_syntax_is_withheld(tmp_path: Path) -> None:
    """A header, separator and delimited row printed as pipes are a table in the layer's own
    bytes; they would assemble into a grid over content nothing verified. DIFFERENCE: the
    sentence beside them ships."""
    _s, ps = _build(tmp_path, _columns, [GRID_BOX])
    ps.native_words = [
        (72.0, 60.0, 100.0, 70.0, "Intro", 0, 0, 0),
        (104.0, 60.0, 150.0, 70.0, "sentence", 0, 0, 1),
        (72.0, 80.0, 150.0, 90.0, "| Name | Remark |", 1, 0, 0),
        (72.0, 100.0, 150.0, 110.0, "|---|---|", 2, 0, 0),
        (72.0, 120.0, 150.0, 130.0, "| Alpha | beta |", 3, 0, 0),
    ]
    body = M._native_order_prose_body(ps, 1, marker_line=MARKER, png_ref=PNG)
    assert body is not None
    assert "|" not in body
    assert "Intro sentence" in body
