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
#: Prose inside a table's horizontal span ships only as a closed paragraph (two or more layer
#: lines of one block, the last ending in a full stop), so the tail is one.
LEFT_TAIL_LINES = (LEFT_TAIL, "Its second line closes the page.")
RIGHT_TAIL = "Tail of the right column."
SHARES_BASELINE = "Right text on a grid baseline."
GRID = [("Measure", "2003", "2004"), ("Real GDP", "2.6", "4.1"), ("Final sales", "2.4", "4.0")]
GRID_BOX = (70.0, 118.0, 230.0, 160.0)
GRID_TOKENS = ("Measure", "Real", "GDP", "Final", "sales", "2.6", "4.1", "2.4", "4.0")


def _columns(page, *, extra=None) -> None:
    """Column-major: the whole left column, then the whole right one, in stream order."""
    page.insert_text((72, 60), "\n".join(LEFT), fontsize=9)
    for r, row in enumerate(GRID):
        for c, cell in enumerate(row):
            page.insert_text((76 + c * 50, 130 + r * 16), cell, fontsize=9)
    page.insert_text((72, 260), "\n".join(LEFT_TAIL_LINES), fontsize=9)
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
    assert _own_lines(on) == [*LEFT, *LEFT_TAIL_LINES, *RIGHT, RIGHT_TAIL], _own_lines(on)
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
        page.insert_text((72, 60), "\n".join(LEFT), fontsize=9)
        for r, row in enumerate(GRID):
            for c, cell in enumerate(row):
                page.insert_text((76 + c * 50, 130 + r * 16), cell, fontsize=9)
        page.insert_text((72, 260), "\n".join(LEFT_TAIL_LINES), fontsize=9)

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
        (142.0, 60.0, 160.0, 70.0, "of", 0, 0, 2),
        (162.0, 60.0, 190.0, 70.0, "the", 0, 0, 3),
        (192.0, 60.0, 230.0, 70.0, "page", 0, 0, 4),
        (72.0, 40.0, 100.0, 50.0, "Second", 0, 1, 0),
        (102.0, 40.0, 140.0, 50.0, "line", 0, 1, 1),
        (142.0, 40.0, 190.0, 50.0, "ends.", 0, 1, 2),
        (72.0, 62.0, 100.0, 72.0, "Sales", 1, 0, 0),
        (80.0, 130.0, 110.0, 140.0, "12.5", 1, 0, 1),
        (80.0, 150.0, 110.0, 160.0, "13.5", 2, 0, 0),
    ]
    body = M._native_order_prose_body(ps, 1, marker_line=MARKER, png_ref=PNG)
    assert body is not None
    assert "12.5" not in body and "Sales" not in body
    assert "Intro sentence of the page" in body


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
    assert _own_lines(first.text) == [*LEFT, *LEFT_TAIL_LINES, *RIGHT, RIGHT_TAIL]
    assert first.table_floor_body is True and table_floor_credential_holds(first)


def _column_pdf(path: Path) -> Path:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 60), "\n".join(LEFT))
    rows = [["Real GDP", "2.6", "4.1"], ["Previous", "2.9", "3.9"], ["Final sales", "2.4", "4.0"]]
    for r, cells in enumerate(rows):
        for c, cell in enumerate(cells):
            page.insert_text((80 + c * 50, 130 + r * 22), cell, fontsize=9)
    for r in range(4):
        page.draw_line(fitz.Point(76, 116 + r * 22), fitz.Point(226, 116 + r * 22))
    for c in range(4):
        page.draw_line(fitz.Point(76 + c * 50, 116), fitz.Point(76 + c * 50, 116 + 3 * 22))
    page.insert_text((72, 260), "\n".join(LEFT_TAIL_LINES))
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
        want = " ".join([*LEFT, *LEFT_TAIL_LINES, *RIGHT, RIGHT_TAIL]).split()
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


def test_a_loose_value_line_and_its_column_are_withheld(tmp_path: Path) -> None:
    """A layer line is narrower than its band: the band "Rate label  +-0.3  +-0.7" has words
    outnumbering numerals and ships under the bands rule, but its values are separate layer
    lines. Each line is held to the keep-sentences rule on its own words, so a lone value does
    not ship as a line (MPR 2008-07 p45 on the Fed run), and the label on the value's side of the
    band goes with it (the stub of a table whose values were withheld). DIFFERENCE: the sentence
    carrying a number, on its own band, ships."""
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
    ps.detected_table_bboxes = [(500.0, 500.0, 520.0, 520.0)]
    body = M._native_order_prose_body(ps, 1, marker_line=MARKER, png_ref=PNG)
    assert body is not None
    assert "\u00b10.3" not in body and "\u00b10.7" not in body
    assert "Unemployment" not in body
    assert "Growth in 2003 stayed firm" in body


def test_a_layer_line_that_parses_as_table_syntax_is_withheld(tmp_path: Path) -> None:
    """A header, separator and delimited row printed as pipes are a table in the layer's own
    bytes; they would assemble into a grid over content nothing verified. DIFFERENCE: the
    sentence beside them ships."""
    _s, ps = _build(tmp_path, _columns, [GRID_BOX])
    ps.native_words = [
        (72.0, 60.0, 100.0, 70.0, "Intro", 0, 0, 0),
        (104.0, 60.0, 150.0, 70.0, "sentence", 0, 0, 1),
        (152.0, 60.0, 170.0, 70.0, "of", 0, 0, 2),
        (172.0, 60.0, 200.0, 70.0, "the", 0, 0, 3),
        (202.0, 60.0, 240.0, 70.0, "page", 0, 0, 4),
        (72.0, 40.0, 100.0, 50.0, "Second", 0, 1, 0),
        (102.0, 40.0, 140.0, 50.0, "line", 0, 1, 1),
        (142.0, 40.0, 190.0, 50.0, "ends.", 0, 1, 2),
        (72.0, 80.0, 150.0, 90.0, "| Name | Remark |", 1, 0, 0),
        (72.0, 100.0, 150.0, 110.0, "|---|---|", 2, 0, 0),
        (72.0, 120.0, 150.0, 130.0, "| Alpha | beta |", 3, 0, 0),
    ]
    body = M._native_order_prose_body(ps, 1, marker_line=MARKER, png_ref=PNG)
    assert body is not None
    assert "|" not in body
    assert "Intro sentence of the page" in body


# ---------------------------------------------------------------------------
# review round on PR #1324
# ---------------------------------------------------------------------------

#: A box taller than the grid, and the two ruled rows the detector left OUTSIDE it.
CLIPPED_BOX = (70.0, 118.0, 320.0, 170.0)
CLIPPED_LABELS = ("Total pension obligation", "Funded ratio")
CLIPPED_VALUES = ("12 million dollars", "80 percent")


def _clipped_rows(page) -> None:
    for i, (label, value) in enumerate(zip(CLIPPED_LABELS, CLIPPED_VALUES)):
        y = 190 + i * 16
        page.insert_text((76, y), label, fontsize=9)
        page.insert_text((230, y), value, fontsize=9)


def test_table_rows_clipped_out_of_the_box_do_not_ship_as_prose(
    tmp_path: Path, monkeypatch
) -> None:
    """Astra's blocker: ruled final rows with loose numeric cells and units (one numeral, two
    alphabetic tokens) sit just below a clipped box, in a column-major page. The structure gate
    does not see them (no recurring numeric lanes) and the bands body was refused by the order
    gate, so nothing used to withhold them. DIFFERENCE: the same page without the rows; and the
    prose after the rows still ships."""
    _clip, clipped = _build(
        tmp_path, lambda pg: _columns(pg, extra=_clipped_rows), [CLIPPED_BOX], name="k.pdf"
    )
    _base, base = _build(tmp_path, _columns, [CLIPPED_BOX], name="b.pdf")
    reasons: list[str] = []
    body = table_floor_text_for_source(
        clipped,
        1,
        clipped.native_text,
        prose_recovery=True,
        geometry_floor=True,
        chart_floor_offered=False,
        refusals=reasons,
    )
    assert reasons == [], reasons
    assert not _whole_page(body)
    for cell in (*CLIPPED_LABELS, *CLIPPED_VALUES, "12", "80"):
        assert cell not in body, f"clipped table row shipped: {cell!r}"
    ref = structure_class_floor_text(base, 1)
    assert _own_lines(body) == _own_lines(ref) == [*LEFT, *LEFT_TAIL_LINES, *RIGHT, RIGHT_TAIL]


def test_a_line_that_continues_a_box_is_withheld_but_the_prose_after_it_ships(
    tmp_path: Path,
) -> None:
    """The walk away from a box stops at the first band whose lines in the box's span carry no
    numeral: a numeral-free sentence under the clipped rows is prose."""
    _s, ps = _build(tmp_path, lambda pg: _columns(pg, extra=_clipped_rows), [CLIPPED_BOX])
    body = M._native_order_prose_body(ps, 1, marker_line=MARKER, png_ref=PNG)
    assert body is not None
    assert LEFT_TAIL in body and "pension" not in body


def test_a_box_whose_geometry_cannot_be_read_makes_the_body_abstain(tmp_path: Path) -> None:
    """Exclusion cannot be established without the box: no prose, not unexcluded prose."""
    _s, ps = _build(tmp_path, _columns, [GRID_BOX])
    assert M._native_order_prose_body(ps, 1, marker_line=MARKER, png_ref=PNG) is not None
    ps.detected_table_bboxes = [(70.0, 118.0, 230.0)]
    assert M._native_order_prose_body(ps, 1, marker_line=MARKER, png_ref=PNG) is None


def _words_line(texts, y, x0, block, line=0):
    out, x = [], x0
    for k, t in enumerate(texts):
        out.append((x, y, x + 7.0 * len(t), y + 10.0, t, block, line, k))
        x += 7.0 * len(t) + 3.0
    return out


def test_a_table_row_beside_prose_is_not_diluted_by_the_prose(tmp_path: Path) -> None:
    """Fable 1 and 2: ``Real GDP 2.3`` and ``2.5`` on one baseline beside a left-column sentence.
    The band reads as prose (words outnumber numerals), the line ``Real GDP 2.3`` alone does too,
    but ``2.5`` is a loose value, so the row's column is withheld together: nothing ships
    partly. DIFFERENCE: the left-column sentence on the same baseline ships."""
    _s, ps = _build(tmp_path, _columns, [GRID_BOX])
    ps.native_words = [
        *_words_line(["Left", "column", "sentence", "is", "here"], 300.0, 72.0, 0),
        *_words_line(["Real", "GDP", "2.3"], 300.0, 330.0, 1),
        *_words_line(["2.5"], 300.0, 450.0, 2),
        *_words_line(["Other", "words", "stay", "put", "here."], 320.0, 72.0, 0, line=1),
    ]
    ps.detected_table_bboxes = [(500.0, 500.0, 520.0, 520.0)]
    body = M._native_order_prose_body(ps, 1, marker_line=MARKER, png_ref=PNG)
    assert body is not None
    assert "Real" not in body and "2.3" not in body and "2.5" not in body
    assert "Left column sentence is here" in body and "Other words stay put here." in body


def test_a_line_with_no_word_of_two_letters_is_withheld(tmp_path: Path) -> None:
    """Fable 3: shredded inline math (``t``, ``y``, ``+``) is symbols out of position, not a
    sentence. DIFFERENCE: the sentence around them ships."""
    _s, ps = _build(tmp_path, _columns, [GRID_BOX])
    ps.native_words = [
        *_words_line(["The", "rate", "follows", "the", "path"], 60.0, 72.0, 0),
        *_words_line(["t"], 80.0, 72.0, 1),
        *_words_line(["y", "+"], 100.0, 72.0, 2),
        *_words_line(["Policy", "stays", "steady", "today"], 120.0 + 200.0, 72.0, 3),
    ]
    ps.detected_table_bboxes = [(500.0, 500.0, 520.0, 520.0)]
    body = M._native_order_prose_body(ps, 1, marker_line=MARKER, png_ref=PNG)
    assert body is not None
    assert "The rate follows the path" in body and "Policy stays steady today" in body
    assert "\nt\n" not in body and "y +" not in body


def test_one_table_marker_per_box_run_and_a_neutral_mark_elsewhere(tmp_path: Path) -> None:
    """Fable 4: the table marker stands for a run that holds a box line. A page number or a
    baseline-sharing line is not an unverifiable table and gets a neutral mark. DIFFERENCE: the
    same run of withheld lines with and without a box in it."""
    _s, ps = _build(tmp_path, _columns, [GRID_BOX])
    body = M._native_order_prose_body(ps, 1, marker_line=MARKER, png_ref=PNG)
    assert body is not None
    assert body.count(MARKER) == 1
    assert M._LINE_WITHHELD_MARK in body
    # A page number is a numeral-only line, in no box: a neutral mark, and the page still
    # carries the table notice once (the box run).
    ps.native_words = [
        *_words_line(["Opening", "sentence", "of", "the", "page"], 60.0, 72.0, 0),
        *_words_line(["49"], 80.0, 72.0, 1),
        *_words_line(["Closing", "sentence", "of", "the", "page"], 100.0, 72.0, 2),
        *_words_line(["1.5"], 120.0, 72.0, 3),
    ]
    ps.detected_table_bboxes = [(500.0, 500.0, 520.0, 520.0)]
    boxless_run = M._native_order_prose_body(ps, 1, marker_line=MARKER, png_ref=PNG)
    assert boxless_run is not None
    assert boxless_run.count(MARKER) == 1, "the page must carry the table notice at least once"
    assert M._LINE_WITHHELD_MARK in boxless_run


def test_a_paragraph_breaks_where_the_layer_block_changes(tmp_path: Path) -> None:
    """Fable 5: consecutive shipped lines of different layer blocks are not one paragraph (a
    heading or the next column's first line welded onto a sentence). DIFFERENCE: two lines of
    one block stay together."""
    _s, ps = _build(tmp_path, _columns, [GRID_BOX])
    ps.native_words = [
        *_words_line(["First", "line", "of", "block", "zero"], 60.0, 72.0, 0, line=0),
        *_words_line(["Second", "line", "of", "block", "zero"], 76.0, 72.0, 0, line=1),
        *_words_line(["Heading", "of", "block", "one"], 92.0, 72.0, 1, line=0),
    ]
    ps.detected_table_bboxes = [(500.0, 500.0, 520.0, 520.0)]
    body = M._native_order_prose_body(ps, 1, marker_line=MARKER, png_ref=PNG)
    assert body is not None
    assert "First line of block zero\nSecond line of block zero\n\nHeading of block one" in body


# ---------------------------------------------------------------------------
# review round 3 on PR #1324: the walk stops only at a sentence; no row fragment ships
# ---------------------------------------------------------------------------


def _subheading_then_row(page) -> None:
    """Astra 1: a numeral-free subheading, then a numeric row INSIDE the box's span."""
    page.insert_text((76, 182), "Pension liabilities", fontsize=9)
    page.insert_text((76, 206), "Total pension obligation", fontsize=9)
    page.insert_text((230, 206), "12 million dollars", fontsize=9)


def _categorical_row(page) -> None:
    """Astra 2: a categorical row under the clipped box."""
    page.insert_text((76, 190), "Coverage included", fontsize=9)
    page.insert_text((230, 190), "Yes", fontsize=9)


@pytest.mark.parametrize(
    ("extra", "cells"),
    [
        (_subheading_then_row, ("Pension", "Total", "12", "million", "dollars")),
        (_categorical_row, ("Coverage", "included", "Yes")),
    ],
    ids=["numeral-free-subheading", "categorical-row"],
)
def test_the_walk_goes_on_past_text_that_is_not_a_sentence(tmp_path: Path, extra, cells) -> None:
    """Astra's two reproductions: a subheading or a categorical row is not a sentence, so the
    walk from the box does not stop there and the row after it is withheld. DIFFERENCE: the same
    page without the rows ships the same prose, and the real sentence below them still ships."""
    _s, withrows = _build(
        tmp_path, lambda pg: _columns(pg, extra=extra), [CLIPPED_BOX], name="r.pdf"
    )
    _b, base = _build(tmp_path, _columns, [CLIPPED_BOX], name="n.pdf")
    body = M._native_order_prose_body(withrows, 1, marker_line=MARKER, png_ref=PNG)
    ref = M._native_order_prose_body(base, 1, marker_line=MARKER, png_ref=PNG)
    assert body is not None and ref is not None
    for cell in cells:
        assert cell not in body, f"table text shipped: {cell!r}"
    assert _own_lines(body) == _own_lines(ref)
    assert LEFT_TAIL in body


def test_a_split_table_row_beside_prose_ships_no_fragment(tmp_path: Path) -> None:
    """Fable 1: cell gaps (60 pt) wider than the gutter (20 pt), so no single cut isolates the
    row. ``2.5`` is a loose value and its neighbours up to the prose go with it. DIFFERENCE: the
    same row without the ``2.5`` cell ships ``Real GDP 2.3``, so the cell is the cause."""
    _s, ps = _build(tmp_path, _columns, [GRID_BOX])

    def body_for(cells: list[tuple[list[str], float]]) -> str:
        ps.native_words = [
            *_words_line(["Policy", "path", "stays", "steady", "here"], 300.0, 72.0, 0),
            *[
                w
                for k, (texts, x0) in enumerate(cells)
                for w in _words_line(texts, 300.0, x0, 1 + k)
            ],
            *_words_line(["Other", "words", "stay", "put", "here."], 340.0, 72.0, 0, line=1),
        ]
        ps.detected_table_bboxes = [(500.0, 500.0, 520.0, 520.0)]
        out = M._native_order_prose_body(ps, 1, marker_line=MARKER, png_ref=PNG)
        assert out is not None
        return out

    row = (["Real", "GDP", "2.3"], 280.0)
    gap = 60.0
    x_end = row[1] + 7.0 * 10 + 3.0 * 2
    with_cell = body_for([row, (["2.5"], x_end + gap)])
    without = body_for([row])
    assert "Real GDP 2.3" in without
    for frag in ("Real", "GDP", "2.3", "2.5"):
        assert frag not in with_cell, f"row fragment shipped: {frag!r}"
    assert "Policy path stays steady here" in with_cell
    assert "Other words stay put here." in with_cell


# ---------------------------------------------------------------------------
# review round 4 on PR #1324: a word count is not a table boundary; only a closed paragraph
# ships inside a span
# ---------------------------------------------------------------------------

LONG_LABEL = "Pension liabilities for retired employees"
LONG_CATEGORICAL = "Coverage for all employees included"


def _long_label_rows(page) -> None:
    """Astra round 4, 1: a five-word-plus stub label above numeric rows, in the box's span."""
    page.insert_text((76, 182), LONG_LABEL, fontsize=9)
    page.insert_text((76, 206), "Total pension obligation", fontsize=9)
    page.insert_text((230, 206), "12 million dollars", fontsize=9)


def _long_label_one_block(page) -> None:
    page.insert_text((76, 182), f"{LONG_LABEL}\nTotal pension obligation", fontsize=9)
    page.insert_text((230, 206), "12 million dollars", fontsize=9)


def _long_categorical(page) -> None:
    """Astra round 4, 2."""
    page.insert_text((76, 190), LONG_CATEGORICAL, fontsize=9)
    page.insert_text((230, 190), "Yes", fontsize=9)


@pytest.mark.parametrize(
    ("extra", "cells"),
    [
        (_long_label_rows, ("Pension", "Total", "12", "million", "dollars")),
        (_long_label_one_block, ("Pension", "Total", "12", "million", "dollars")),
        (_long_categorical, ("Coverage", "employees", "Yes")),
    ],
    ids=["long-label-rows", "long-label-one-block", "long-categorical"],
)
def test_a_long_label_is_not_a_sentence_and_ships_nothing(tmp_path: Path, extra, cells) -> None:
    """DIFFERENCE: the same page without the rows ships the same lines. Through the public
    entry point as well as the body."""
    _s, withrows = _build(
        tmp_path, lambda pg: _columns(pg, extra=extra), [CLIPPED_BOX], name="l.pdf"
    )
    _b, base = _build(tmp_path, _columns, [CLIPPED_BOX], name="m.pdf")
    body = table_floor_text_for_source(
        withrows,
        1,
        withrows.native_text,
        prose_recovery=True,
        geometry_floor=True,
        chart_floor_offered=False,
        refusals=[],
    )
    ref = M._native_order_prose_body(base, 1, marker_line=MARKER, png_ref=PNG)
    assert body is not None and ref is not None
    for cell in cells:
        assert cell not in body, f"table text shipped: {cell!r}"
    assert _own_lines(body) == _own_lines(ref)


def _span_words(rows: list[tuple[list[str], float, int, int]]) -> list[tuple]:
    """(texts, y, block, line) rows at x=72, inside GRID_BOX's horizontal span."""
    filler = _words_line(["Other", "column", "keeps", "the", "page", "trusted"], 400.0, 330.0, 9)
    return [
        *[w for texts, y, block, line in rows for w in _words_line(texts, y, 72.0, block, line)],
        *filler,
    ]


def _in_span_body(ps, rows) -> str:
    ps.native_words = _span_words(rows)
    ps.detected_table_bboxes = [GRID_BOX]
    body = M._native_order_prose_body(ps, 1, marker_line=MARKER, png_ref=PNG)
    assert body is not None
    return body


def _off_span_body(ps, rows) -> str:
    """The same words, in a box far from them: the span rule does not apply."""
    ps.native_words = _span_words(rows)
    ps.detected_table_bboxes = [(500.0, 500.0, 520.0, 520.0)]
    body = M._native_order_prose_body(ps, 1, marker_line=MARKER, png_ref=PNG)
    assert body is not None
    return body


CLOSED = [
    (["First", "line", "of", "a", "paragraph"], 200.0, 0, 0),
    (["Last", "line", "ends", "here."], 216.0, 0, 1),
]


def test_a_closed_two_line_paragraph_after_a_table_still_ships(tmp_path: Path) -> None:
    """(a)+(b)+(c) all hold: it ships in the span."""
    _s, ps = _build(tmp_path, _columns, [GRID_BOX])
    body = _in_span_body(ps, CLOSED)
    assert "First line of a paragraph\nLast line ends here." in body


def test_a_one_line_block_in_a_span_is_withheld_whatever_it_ends_with(tmp_path: Path) -> None:
    """(b). DIFFERENCE: the same line outside every span ships."""
    _s, ps = _build(tmp_path, _columns, [GRID_BOX])
    rows = [(["One", "closed", "sentence", "ends", "here."], 200.0, 0, 0)]
    assert "One closed sentence ends here." not in _in_span_body(ps, rows)
    assert "One closed sentence ends here." in _off_span_body(ps, rows)


def test_a_block_without_a_final_stop_in_a_span_is_withheld(tmp_path: Path) -> None:
    """(c): two layer lines of one block, the last a stub label with no final punctuation.
    DIFFERENCE: the same words outside every span ship."""
    _s, ps = _build(tmp_path, _columns, [GRID_BOX])
    rows = [
        (["Pension", "liabilities", "for", "retired", "employees"], 200.0, 0, 0),
        (["Total", "pension", "obligation"], 216.0, 0, 1),
    ]
    assert "Pension" not in _in_span_body(ps, rows)
    assert "Pension liabilities for retired employees" in _off_span_body(ps, rows)


def test_a_block_holding_a_withheld_line_ships_none_of_its_lines_in_a_span(
    tmp_path: Path,
) -> None:
    """(a), and the cubic P1 shape: a shredded-math line (no word of two letters) cannot release
    the lines after it. DIFFERENCE: outside every span the good lines of the block ship."""
    _s, ps = _build(tmp_path, _columns, [GRID_BOX])
    rows = [
        (["t"], 200.0, 0, 0),
        (["Table", "labels", "follow", "the", "math"], 216.0, 0, 1),
        (["Last", "line", "ends", "here."], 232.0, 0, 2),
    ]
    inside = _in_span_body(ps, rows)
    assert "Table labels" not in inside and "Last line" not in inside
    outside = _off_span_body(ps, rows)
    assert "Table labels follow the math" in outside and "Last line ends here." in outside


def test_a_closing_quote_after_the_full_stop_counts_as_final(tmp_path: Path) -> None:
    _s, ps = _build(tmp_path, _columns, [GRID_BOX])
    rows = [
        (["She", "said", "the", "rate", "would", "hold"], 200.0, 0, 0),
        (["and", "then", "she", "left", 'the "room."'], 216.0, 0, 1),
    ]
    assert "She said the rate would hold" in _in_span_body(ps, rows)
