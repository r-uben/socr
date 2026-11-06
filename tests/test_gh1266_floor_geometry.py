"""#1266: the born-digital structure-class floor ships no table line as loose text.

``splice_all_table_regions`` replaced only the parsed GFM blocks, so every un-parsed table
line shipped verbatim (2002-12-10 greenbook-part1 p10: header words and 16 rows inside the
box; 2002-09-24 greenbook-part2 p94: 9 data rows below a header-only box). The floor now
takes the geometry lane: a shared gate (rotation, count, per-box sanity, table structure
outside the boxes) sends the page to the whole-page marker, otherwise a band outside every
box ships unless it is mostly numerals (owner ruling 2026-11-05).

Pages here are built with PyMuPDF so the words carry real geometry. Every behavioural test
pins a DIFFERENCE: the same page run twice in one process, changing only the thing under test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_gh317_structure_class_floor import _state  # noqa: E402

from socr.core.manifest import (  # noqa: E402
    BORN_DIGITAL_KEEP_SENTENCES_FLAG,
    BORN_DIGITAL_PROSE_RECOVERED_FLAG,
    SCANNED_PROSE_RECOVERED_FLAG,
    _escaped_native_line,
    _select_and_finalize_page,
    _winning_page_output,
    structure_class_floor_text,
)
from socr.core.result import PageOutput, PageStatus  # noqa: E402
from socr.tables import row_corroboration as RC  # noqa: E402

MARKER = "[page 1 failed: unverifiable table — see image]"
PNG = "![Failed table page 1](figures/failed_table_p1.png)"

SENTENCE_WITH_YEAR = "Growth in 2003 stayed firm across the major regions."
NUMBERS_ONLY = "2003 2004 2005"
TAIL_SENTENCE = "The paragraph after the table has no figures at all."
GRID_ROWS = [
    ("Measure", "2003", "2004"),
    ("Real GDP", "2.6", "4.1"),
    ("Previous", "2.9", "3.9"),
    ("Final sales", "2.4", "4.0"),
    ("Consumption", "3.1", "3.8"),
]
GRID_BOX = (70.0, 128.0, 330.0, 128.0 + 18.0 * len(GRID_ROWS) + 6.0)
GRID_TOKENS = ("Real GDP", "Final sales", "Consumption", "2.6", "4.1", "3.9", "Measure")


def _grid(page, y0: float = 140.0) -> None:
    for r, row in enumerate(GRID_ROWS):
        for c, cell in enumerate(row):
            page.insert_text((80 + c * 100, y0 + r * 18), cell, fontsize=9)


def _build(tmp_path: Path, draw, boxes, *, born_digital: bool = True, name: str = "p.pdf"):
    """One page state with REAL word geometry and the given detected boxes."""
    fitz = pytest.importorskip("fitz")
    path = tmp_path / name
    doc = fitz.open()
    page = doc.new_page()
    draw(page)
    words = page.get_text("words")
    native_text = page.get_text()
    doc.save(str(path))
    doc.close()
    state = _state(path, native_text=native_text)
    ps = state.pages[1]
    ps.is_born_digital = born_digital
    ps.native_words = words
    ps.d3_floor_png_ref = PNG
    ps.detected_table_count = len(boxes)
    ps.detected_table_bboxes = list(boxes)
    ps.table_bbox_sane = True  # the union verdict; the new gate does not read it
    return state, ps


def _draw_p10(page) -> None:
    page.insert_text((72, 60), SENTENCE_WITH_YEAR, fontsize=9)
    page.insert_text((72, 100), NUMBERS_ONLY, fontsize=9)
    _grid(page)
    page.insert_text((72, 260), TAIL_SENTENCE, fontsize=9)


def _draw_p94(page) -> None:
    """A header-only box with the data rows BELOW it, outside any box."""
    page.insert_text((72, 40), SENTENCE_WITH_YEAR, fontsize=9)
    page.insert_text((80, 80), "2000", fontsize=9)
    page.insert_text((180, 80), "2001", fontsize=9)
    for i in range(9):
        y = 130 + i * 18
        page.insert_text((72, y), f"Indicator{i}", fontsize=9)
        for c, v in enumerate(("1.1", "2.2", "3.3")):
            page.insert_text((200 + c * 60, y), f"{v}{i}", fontsize=9)


P94_BOX = (70.0, 66.0, 300.0, 88.0)


def _whole_page(body: str) -> bool:
    return body.strip() == f"{MARKER}\n\n{PNG}"


# ---------------------------------------------------------------------------
# the cut (predicate + constants)
# ---------------------------------------------------------------------------


def _band(*tokens):
    return [(0.0, 0.0, 1.0, 1.0, t, 0, 0, i) for i, t in enumerate(tokens)]


def test_the_cut_is_a_named_one_to_one_numeral_to_word_rule() -> None:
    assert RC.MOSTLY_NUMERALS_MIN_NUMERALS == 1
    assert RC.MOSTLY_NUMERALS_MIN_NUMERALS_PER_WORD == 1


@pytest.mark.parametrize(
    ("tokens", "withheld"),
    [
        (("2003", "2004", "2005"), True),  # all numerals
        (("Growth", "in", "2003", "stayed", "firm"), False),  # a sentence with a year
        (("rose", "2.5%"), True),  # tie: numerals == words -> withheld (>=, not >)
        (("rose", "to", "2.5%"), False),
        (("-", "|"), False),  # no numeral: this rule withholds nothing
        ((), False),
    ],
)
def test_band_is_mostly_numerals(tokens, withheld: bool) -> None:
    assert RC.band_is_mostly_numerals(_band(*tokens)) is withheld


def test_keep_sentences_changes_only_the_outside_box_numeral_rule() -> None:
    """DIFFERENCE: the same words partitioned with and without ``keep_sentences``."""
    words = []
    for i, line in enumerate((SENTENCE_WITH_YEAR, NUMBERS_ONLY, TAIL_SENTENCE)):
        for j, tok in enumerate(line.split()):
            words.append((72.0 + 40 * j, 20.0 * i, 100.0 + 40 * j, 20.0 * i + 10, tok, 0, i, j))
    default = [(p, " ".join(w[4] for w in b)) for p, b in RC.partition_prose_bands(words)]
    kept = [
        (p, " ".join(w[4] for w in b))
        for p, b in RC.partition_prose_bands(words, keep_sentences=True)
    ]
    assert default == [(False, SENTENCE_WITH_YEAR), (False, NUMBERS_ONLY), (True, TAIL_SENTENCE)]
    assert kept == [(True, SENTENCE_WITH_YEAR), (False, NUMBERS_ONLY), (True, TAIL_SENTENCE)]


def test_keep_sentences_still_withholds_inside_a_box_and_categorical_rows() -> None:
    """The new rule never loosens the box rule or the categorical rule."""
    words = []
    for i, line in enumerate(("Total share of firms", "Fixed effects Yes Yes")):
        for j, tok in enumerate(line.split()):
            words.append((72.0 + 40 * j, 20.0 * i, 100.0 + 40 * j, 20.0 * i + 10, tok, 0, i, j))
    box = (60.0, -5.0, 600.0, 15.0)  # covers only the first line
    parts = RC.partition_prose_bands(words, keep_sentences=True, table_boxes=[box])
    assert [p for p, _b in parts] == [False, False]  # box rule, categorical rule
    free = RC.partition_prose_bands(words, keep_sentences=True)
    assert [p for p, _b in free] == [True, False]  # categorical alone still withholds


def test_the_per_box_census_counts_one_answer_token_as_a_table_band() -> None:
    assert RC.bears_categorical_answer(_band("generation", "of", "heat", "Yes")) is True
    assert RC.bears_categorical_row(_band("generation", "of", "heat", "Yes")) is False
    assert RC.bears_categorical_answer(_band("plain", "words")) is False


# ---------------------------------------------------------------------------
# p10 shape: whole table in a box, a year-bearing sentence outside it
# ---------------------------------------------------------------------------


def test_p10_shaped_page_ships_no_table_line_and_ships_the_prose(tmp_path: Path) -> None:
    state, _ps = _build(tmp_path, _draw_p10, [GRID_BOX])
    body = structure_class_floor_text(state.pages[1], 1)
    assert BORN_DIGITAL_KEEP_SENTENCES_FLAG.format(page_num=1) in body
    assert SENTENCE_WITH_YEAR in body, "a sentence carrying a year was withheld"
    assert TAIL_SENTENCE in body
    assert NUMBERS_ONLY not in body, "a mostly-numerals band outside the box shipped"
    for tok in GRID_TOKENS:
        assert tok not in body, f"table content {tok!r} shipped as loose text"
    assert MARKER in body


def test_scope_switch_a_scan_keeps_the_any_digit_rule(tmp_path: Path) -> None:
    """DIFFERENCE: the same page, only ``is_born_digital`` changes."""
    born, _ = _build(tmp_path, _draw_p10, [GRID_BOX], name="b.pdf")
    scan, scan_ps = _build(tmp_path, _draw_p10, [GRID_BOX], born_digital=False, name="s.pdf")
    born_body = structure_class_floor_text(born.pages[1], 1)
    scan_body = structure_class_floor_text(scan.pages[1], 1)
    assert SENTENCE_WITH_YEAR in born_body
    assert SENTENCE_WITH_YEAR not in scan_body, "a scan used the born-digital rule"
    assert SCANNED_PROSE_RECOVERED_FLAG.format(page_num=1) in scan_body
    for tok in GRID_TOKENS:
        assert tok not in born_body and tok not in scan_body


# ---------------------------------------------------------------------------
# the shared gate: one test per clause, each against its own control
# ---------------------------------------------------------------------------


def test_p94_shaped_page_is_whole_page_by_the_outside_lanes(tmp_path: Path) -> None:
    state, _ps = _build(tmp_path, _draw_p94, [P94_BOX])
    body = structure_class_floor_text(state.pages[1], 1)
    assert _whole_page(body), body
    assert "Indicator" not in body
    # control: the same words with no data rows below ship the sentence.
    ctl, _ = _build(
        tmp_path,
        lambda pg: _draw_p94_header_only(pg),
        [P94_BOX],
        name="c.pdf",
    )
    assert SENTENCE_WITH_YEAR in structure_class_floor_text(ctl.pages[1], 1)


def _draw_p94_header_only(page) -> None:
    page.insert_text((72, 40), SENTENCE_WITH_YEAR, fontsize=9)
    page.insert_text((80, 80), "2000", fontsize=9)
    page.insert_text((180, 80), "2001", fontsize=9)


def _draw_categorical_outside(page) -> None:
    _draw_p10(page)
    page.insert_text((72, 300), "Industry fixed effects Yes Yes Yes", fontsize=9)


def test_a_categorical_row_outside_every_box_gates_the_page(tmp_path: Path) -> None:
    gated, _ = _build(tmp_path, _draw_categorical_outside, [GRID_BOX], name="g.pdf")
    ok, _ = _build(tmp_path, _draw_p10, [GRID_BOX], name="o.pdf")
    assert _whole_page(structure_class_floor_text(gated.pages[1], 1))
    assert SENTENCE_WITH_YEAR in structure_class_floor_text(ok.pages[1], 1)


def test_rotation_gates_the_page(tmp_path: Path) -> None:
    rotated, ps = _build(tmp_path, _draw_p10, [GRID_BOX], name="r.pdf")
    ok, _ = _build(tmp_path, _draw_p10, [GRID_BOX], name="k.pdf")
    ps.page_rotation = 90
    assert _whole_page(structure_class_floor_text(rotated.pages[1], 1))
    assert SENTENCE_WITH_YEAR in structure_class_floor_text(ok.pages[1], 1)


def test_count_and_box_disagreement_gates_the_page(tmp_path: Path) -> None:
    state, ps = _build(tmp_path, _draw_p10, [GRID_BOX], name="n.pdf")
    ok, _ = _build(tmp_path, _draw_p10, [GRID_BOX], name="m.pdf")
    ps.detected_table_count = 2  # one box for two detected tables
    assert _whole_page(structure_class_floor_text(state.pages[1], 1))
    assert SENTENCE_WITH_YEAR in structure_class_floor_text(ok.pages[1], 1)


def _draw_prose_majority_box(page) -> None:
    page.insert_text((72, 40), SENTENCE_WITH_YEAR, fontsize=9)
    for i, line in enumerate(
        (
            "This note explains the exercise in words",
            "and continues with more words here",
            "and finishes the explanation of method",
        )
    ):
        page.insert_text((80, 100 + i * 18), line, fontsize=9)
    page.insert_text((80, 160), "Row 1.5 2.5", fontsize=9)


def _draw_sentences_only(page) -> None:
    page.insert_text((72, 40), SENTENCE_WITH_YEAR, fontsize=9)
    page.insert_text((72, 60), TAIL_SENTENCE, fontsize=9)


PROSE_MAJORITY_BOX = (70.0, 88.0, 330.0, 168.0)


def _draw_pure_prose_box(page) -> None:
    page.insert_text((72, 40), SENTENCE_WITH_YEAR, fontsize=9)
    for i, line in enumerate(("Words only inside this box", "and more words inside it")):
        page.insert_text((80, 100 + i * 18), line, fontsize=9)


PURE_PROSE_BOX = (70.0, 88.0, 330.0, 140.0)


def _draw_text_answer_box(page) -> None:
    """p77 shape: a Yes/No answer grid, ONE answer per row, no numeral anywhere inside."""
    page.insert_text((72, 40), TAIL_SENTENCE, fontsize=9)
    for i, (q, a) in enumerate(
        (("generation of electricity", "Yes"), ("generation of heat", "No"), ("cooling", "No"))
    ):
        page.insert_text((80, 100 + i * 18), q, fontsize=9)
        page.insert_text((300, 100 + i * 18), a, fontsize=9)


TEXT_ANSWER_BOX = (70.0, 88.0, 400.0, 150.0)


@pytest.mark.parametrize(
    ("draw", "box", "ships"),
    [
        (_draw_prose_majority_box, PROSE_MAJORITY_BOX, False),  # too large per box
        (_draw_pure_prose_box, PURE_PROSE_BOX, False),  # no numeric, no categorical token
        (_draw_text_answer_box, TEXT_ANSWER_BOX, True),  # one Yes/No per row is a table band
        (_draw_sentences_only, (400.0, 500.0, 520.0, 560.0), False),  # an empty box
        (_draw_p10, GRID_BOX, True),
    ],
)
def test_per_box_sanity(tmp_path: Path, draw, box, ships: bool) -> None:
    state, _ps = _build(tmp_path, draw, [box])
    body = structure_class_floor_text(state.pages[1], 1)
    assert (not _whole_page(body)) is ships, body
    if ships:
        assert BORN_DIGITAL_KEEP_SENTENCES_FLAG.format(page_num=1) in body


def test_the_union_artefact_is_what_the_per_box_clause_removes(tmp_path: Path) -> None:
    """Two sane boxes with a sentence between them: the union reads 'too large'."""

    def draw(page) -> None:
        _grid(page, y0=100.0)
        for i in range(12):  # more prose bands between the boxes than numeric bands in both
            page.insert_text(
                (72, 210 + i * 14), f"Words printed between the boxes, line {i}x.", fontsize=9
            )
        _grid(page, y0=420.0)

    box1 = (70.0, 88.0, 330.0, 88.0 + 18.0 * len(GRID_ROWS) + 6.0)
    box2 = (70.0, 408.0, 330.0, 408.0 + 18.0 * len(GRID_ROWS) + 6.0)
    state, ps = _build(tmp_path, draw, [box1, box2])
    from socr.core.manifest import _each_table_box_sane, _table_bbox_sane

    ps.table_bbox_sane = None
    assert _table_bbox_sane(ps) is False  # the union verdict (unchanged function)
    assert _each_table_box_sane(ps) is True
    assert not _whole_page(structure_class_floor_text(ps, 1))


def test_a_page_without_words_fails_closed(tmp_path: Path) -> None:
    state, ps = _build(tmp_path, _draw_p10, [GRID_BOX])
    live = structure_class_floor_text(ps, 1)
    ps.native_words = []
    assert SENTENCE_WITH_YEAR in live
    assert _whole_page(structure_class_floor_text(ps, 1))
    from socr.core.manifest import _structure_outside_boxes

    assert _structure_outside_boxes(ps) is True  # no words is no evidence, on its own


# ---------------------------------------------------------------------------
# ordered-list escape: digit-bearing prose now ships
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "shipped"),
    [
        ("1. Introduction to the method", "1\\. Introduction to the method"),
        ("2005) The year in review", "2005\\) The year in review"),
        ("0.5 percentage point higher", "0.5 percentage point higher"),
        ("In 2005. Then more", "In 2005. Then more"),
    ],
)
def test_an_ordered_list_opener_is_escaped_and_nothing_else_is(line: str, shipped: str) -> None:
    assert _escaped_native_line(line) == shipped


# ---------------------------------------------------------------------------
# resume / re-finalize
# ---------------------------------------------------------------------------


def test_finalize_twice_and_with_the_saved_body_is_byte_identical(tmp_path: Path) -> None:
    state, _ps = _build(tmp_path, _draw_p10, [GRID_BOX])
    first = _winning_page_output(state, 1)
    second = _winning_page_output(state, 1)
    assert first.text == second.text
    assert SENTENCE_WITH_YEAR in first.text
    again = _select_and_finalize_page(state, 1, saved_text=first.text)
    assert again.output.text == first.text
    assert again.output.scanned_prose_recovered is True


def _restored(tmp_path: Path, body: str, *, typed: bool, name: str):
    """A page rebuilt from its sidecar: boxes and count, but no ``native_words``."""
    state, ps = _build(tmp_path, _draw_p10, [GRID_BOX], name=name)
    ps.native_words = []
    # round 4: ``typed`` = what a floor writer sets -- the recovery flag AND the floor
    # credential with a digest of these bytes; the words-less replay needs both.
    from socr.core.manifest import table_floor_digest_of

    out = PageOutput(
        page_num=1,
        text=body,
        status=PageStatus.ERROR,
        engine="native",
        audit_passed=False,
        scanned_prose_recovered=typed,
        table_floor_body=typed,
        table_floor_digest=table_floor_digest_of(body) if typed else "",
    )
    ps.attempts = [out]
    ps.best_output = out
    return ps


def test_a_restored_born_digital_recovery_replays_and_anything_else_fails_closed(
    tmp_path: Path,
) -> None:
    live_state, live_ps = _build(tmp_path, _draw_p10, [GRID_BOX], name="live.pdf")
    body = structure_class_floor_text(live_ps, 1)
    assert SENTENCE_WITH_YEAR in body
    # replay: typed credential + born-digital banner -> the frozen bytes stand
    assert (
        structure_class_floor_text(_restored(tmp_path, body, typed=True, name="a.pdf"), 1) == body
    )
    # DIFFERENCE 1: no typed credential -> the same bytes are NOT adopted
    assert _whole_page(
        structure_class_floor_text(_restored(tmp_path, body, typed=False, name="b.pdf"), 1)
    )
    # DIFFERENCE 2: a scan-banner recovery is not replayed by the born-digital lane
    scan_body = body.replace(
        BORN_DIGITAL_KEEP_SENTENCES_FLAG.format(page_num=1),
        SCANNED_PROSE_RECOVERED_FLAG.format(page_num=1),
    )
    assert _whole_page(
        structure_class_floor_text(_restored(tmp_path, scan_body, typed=True, name="c.pdf"), 1)
    )


# ---------------------------------------------------------------------------
# process(): hermetic, through the real pipeline
# ---------------------------------------------------------------------------


def _process_pdf(path: Path) -> Path:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 60), SENTENCE_WITH_YEAR)
    rows = [["Real GDP", "2.6", "4.1"], ["Previous", "2.9", "3.9"], ["Final sales", "2.4", "4.0"]]
    for r, cells in enumerate(rows):
        for c, cell in enumerate(cells):
            page.insert_text((80 + c * 110, 130 + r * 22), cell, fontsize=9)
    for r in range(4):
        page.draw_line(fitz.Point(76, 116 + r * 22), fitz.Point(416, 116 + r * 22))
    for c in range(4):
        page.draw_line(fitz.Point(76 + c * 110, 116), fitz.Point(76 + c * 110, 116 + 3 * 22))
    page.insert_text((72, 260), TAIL_SENTENCE)
    doc.save(str(path))
    doc.close()
    return path


def test_process_ships_the_sentences_and_no_table_line_and_replays(
    tmp_path: Path, monkeypatch
) -> None:
    """Run 2 goes through the PAGE-level path, not the document-level skip.

    Measured before this fix: a second ``process()`` over the untouched output directory was
    ``DocumentStatus.SKIPPED`` with zero ``_load_terminal_page`` calls, so the replay claim
    could not fail (#1267 merge gate, Fable P1). The root index is now deleted between the
    runs, which reopens the document; the premise is asserted. The five-page, both-ending
    version is ``tests/test_gh1266_page_level_resume.py``.
    """
    from ocr_output_contract import RootIndex
    from test_gh1144_prose_survives_the_table_floor import _GridEngine, _pipeline

    from socr.core.result import DocumentStatus
    from socr.pipeline import orchestrator as orch
    from socr.pipeline.orchestrator import UnifiedPipeline

    pdf = _process_pdf(tmp_path / "doc.pdf")
    monkeypatch.setattr(orch, "get_engine", lambda engine_type: _GridEngine())
    out = tmp_path / "out"
    bodies = []
    loads: list[int] = []
    real_load = UnifiedPipeline._load_terminal_page

    def _load(self, state, page_num, output_dir):
        loads.append(page_num)
        return real_load(self, state, page_num, output_dir)

    for run in range(2):
        if run == 1:
            RootIndex(out).path.unlink()
            monkeypatch.setattr(UnifiedPipeline, "_load_terminal_page", _load)
        result = _pipeline().process(pdf, output_dir=out)
        bodies.append(next(iter(out.rglob("pages/00001.md"))).read_text())
    assert result.status is not DocumentStatus.SKIPPED, "run 2 was the document-level skip"
    assert 1 in loads, "run 2 never consulted the page ledger"
    body = bodies[0]
    assert SENTENCE_WITH_YEAR in body and TAIL_SENTENCE in body
    for tok in ("Real GDP", "Final sales", "2.6", "4.1", "3.9"):
        assert tok not in body
    assert "failed: unverifiable table" in body
    assert bodies[0] == bodies[1], "a second run over the same output changed the shipped page"


def test_the_gh520_splice_is_retired_on_born_digital_pages(tmp_path: Path) -> None:
    """DIFFERENCE: the same page through the lane and through the old splice.

    The GH-520 guard PASSES here (one detected table, one parsed block, one region), so the
    old code spliced ``native_text`` and shipped its un-parsed line ``Loose ...`` verbatim.
    The lane reads the words instead and withholds it (mostly numerals).
    """
    from test_gh520_regional_floor_splice import _floored_page

    from socr.core.manifest import table_floor_text_for_source
    from socr.tables.reconcile import find_table_blocks

    text = (
        f"{TAIL_SENTENCE}\n\n| a | b |\n|---|---|\n| 1.5 | 2.5 |\n| 3.5 | 4.5 |\n\nLoose 7.7 8.8\n"
    )
    assert len(find_table_blocks(text)) == 1
    state, ps = _floored_page(tmp_path, text, detected=1, geometry=True)
    lane = structure_class_floor_text(ps, 1)
    old = table_floor_text_for_source(ps, 1, ps.native_text, prose_recovery=True)
    assert "Loose" in old, "premise: the old splice ships the un-parsed line"
    assert "Loose" not in lane, "the retired splice's leak shipped"
    assert TAIL_SENTENCE in lane


# ---------------------------------------------------------------------------
# review round 1
# ---------------------------------------------------------------------------


def test_a_suspect_layer_takes_the_same_lane_as_an_admissible_one(tmp_path: Path) -> None:
    """#1156's recorded exception overrules Argon #4: "unknown is not a finding".

    A born-digital page whose prior-OCR scan FAILED (``prior_ocr_layer_suspect``, no positive
    finding) keeps its layer's prose under the mark. The same page, only the suspect flag
    differs, ships the SAME keep-sentences body; a POSITIVE prior-OCR page is unchanged
    (whole-page marker, returned before this lane).
    """
    from socr.core.state import layer_is_admissible_evidence

    ok, ok_ps = _build(tmp_path, _draw_p10, [GRID_BOX], name="adm.pdf")
    sus, sus_ps = _build(tmp_path, _draw_p10, [GRID_BOX], name="sus.pdf")
    sus_ps.prior_ocr_layer_suspect = True
    assert layer_is_admissible_evidence(ok_ps) and not layer_is_admissible_evidence(sus_ps), (
        "premise: the suspect flag is really set and really inadmissible"
    )
    ok_body = structure_class_floor_text(ok_ps, 1)
    sus_body = structure_class_floor_text(sus_ps, 1)
    assert SENTENCE_WITH_YEAR in ok_body
    assert sus_body == ok_body, "a suspect layer took a different lane"
    for tok in GRID_TOKENS:
        assert tok not in sus_body

    pos, pos_ps = _build(tmp_path, _draw_p10, [GRID_BOX], name="pos.pdf")
    pos_ps.prior_ocr_layer = True
    assert _whole_page(structure_class_floor_text(pos_ps, 1))


def test_the_splice_is_retired_on_a_suspect_born_digital_page_too(tmp_path: Path) -> None:
    """The coordinator's correction: a suspect page must not fall back to the splice."""
    from test_gh520_regional_floor_splice import _floored_page

    from socr.core.manifest import table_floor_text_for_source

    text = (
        f"{TAIL_SENTENCE}\n\n| a | b |\n|---|---|\n| 1.5 | 2.5 |\n| 3.5 | 4.5 |\n\nLoose 7.7 8.8\n"
    )
    state, ps = _floored_page(tmp_path, text, detected=1, geometry=True)
    ps.prior_ocr_layer_suspect = True
    old = table_floor_text_for_source(ps, 1, ps.native_text, prose_recovery=True)
    assert "Loose" in old, "premise: the old splice ships the un-parsed line"
    body = structure_class_floor_text(ps, 1)
    assert "Loose" not in body, "a suspect page fell back to the splice remainder"
    assert TAIL_SENTENCE in body


def _each_sane(boxes, tmp_path: Path, name: str = "bx.pdf") -> bool:
    from socr.core.manifest import _each_table_box_sane

    _state_, ps = _build(tmp_path, _draw_p10, [GRID_BOX], name=name)
    ps.detected_table_bboxes = list(boxes)
    ps.detected_table_count = len(boxes)
    return _each_table_box_sane(ps)


_NAN = float("nan")
_INF = float("inf")


@pytest.mark.parametrize(
    ("boxes", "sane"),
    [
        ([GRID_BOX], True),  # control
        ([(70.0, _NAN, 330.0, 230.0)], False),  # non-finite
        ([(70.0, 128.0, _INF, 230.0)], False),  # non-finite
        ([(330.0, 128.0, 70.0, 230.0)], False),  # inverted x
        ([(70.0, 230.0, 330.0, 128.0)], False),  # inverted y
        ([(70.0, 128.0, 70.0, 230.0)], False),  # zero width
        ([(70.0, 128.0, 330.0, 128.0)], False),  # zero height
        ([GRID_BOX, GRID_BOX], False),  # duplicate
        ([GRID_BOX, (200.0, 150.0, 500.0, 300.0)], False),  # positive-area overlap
        ([(70.0, 128.0, 230.0, 230.0), (230.0, 128.0, 330.0, 230.0)], True),  # touching only
    ],
)
def test_box_validation_fails_closed(tmp_path: Path, boxes, sane: bool) -> None:
    assert _each_sane(boxes, tmp_path) is sane


def test_recovered_pages_are_named_by_their_own_kind() -> None:
    """Argon #5. A born-digital recovery is not a 'scanned page' on the CLI or in metadata."""
    from test_gh697_prose_recovery_surfacing import _recovered_output, _record

    from socr.pipeline.orchestrator import UnifiedPipeline

    scan = _recovered_output(page_num=1)
    born = _recovered_output(page_num=2)
    born.text = f"{BORN_DIGITAL_PROSE_RECOVERED_FLAG.format(page_num=2)}\n\nSome prose."
    note_scan = UnifiedPipeline._scanned_prose_recovered_note([_record(scan)])
    note_born = UnifiedPipeline._scanned_prose_recovered_note([_record(born)])
    assert note_scan is not None and note_born is not None
    assert "page(s) 2" in note_born and "withheld table" in note_born
    assert "born-digital" in note_born and "scan" not in note_born.lower().replace(
        "scanned_prose", ""
    )
    assert "born-digital" not in note_scan
    mixed = UnifiedPipeline._scanned_prose_recovered_note([_record(scan), _record(born)])
    assert "page(s) 1" in mixed and "page(s) 2" in mixed


def test_the_cli_line_names_a_born_digital_recovery_as_born_digital(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch
) -> None:
    from test_gh697_prose_recovery_surfacing import _assemble, _recovered_output, _record

    from socr.pipeline.orchestrator import UnifiedPipeline

    # hermetic: assemble fingerprints the run, which would probe the judge model
    monkeypatch.setattr(UnifiedPipeline, "_resolve_judge_model", lambda *a, **k: "")

    born = _recovered_output(page_num=1)
    born.text = f"{BORN_DIGITAL_PROSE_RECOVERED_FLAG.format(page_num=1)}\n\nSome prose."
    _assemble(tmp_path, [_record(born)], page_count=1)
    out = capsys.readouterr().out
    assert "unverified recovered prose" in out and "[1]" in out
    assert "born-digital page(s)" in out
    assert "scanned page(s)" not in out


# ---------------------------------------------------------------------------
# process()-level differential and real restore
# ---------------------------------------------------------------------------


def _gated_process_pdf(path: Path) -> Path:
    """A real lined grid (detected as a box) with unboxed numeric rows below it: the gate's
    outside-lanes clause sends this page to the whole-page marker."""
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 60), SENTENCE_WITH_YEAR)
    rows = [["Real GDP", "2.6", "4.1"], ["Previous", "2.9", "3.9"], ["Final sales", "2.4", "4.0"]]
    for r, cells in enumerate(rows):
        for c, cell in enumerate(cells):
            page.insert_text((80 + c * 110, 130 + r * 22), cell, fontsize=9)
    for r in range(4):
        page.draw_line(fitz.Point(76, 116 + r * 22), fitz.Point(416, 116 + r * 22))
    for c in range(4):
        page.draw_line(fitz.Point(76 + c * 110, 116), fitz.Point(76 + c * 110, 116 + 3 * 22))
    for i in range(8):
        y = 260 + i * 16
        page.insert_text((72, y), f"Series{i}", fontsize=9)
        for c, v in enumerate(("1.1", "2.2", "3.3")):
            page.insert_text((200 + c * 70, y), f"{v}{i}", fontsize=9)
    doc.save(str(path))
    doc.close()
    return path


def _run(tmp_path: Path, pdf: Path, monkeypatch, *, ladder: bool, tag: str):
    import json

    from test_gh1144_prose_survives_the_table_floor import _GridEngine, _pipeline

    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch

    monkeypatch.setattr(orch, "get_engine", lambda engine_type: _GridEngine())
    pipe = _pipeline()
    pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL] if ladder else []
    out = tmp_path / tag
    pipe.process(pdf, output_dir=out)
    frag = next(iter(out.rglob("pages/00001.md"))).read_text()
    meta = json.loads(next(iter(out.rglob("pages/00001.json"))).read_text())
    return frag, meta, out


def test_process_differential_the_1266_path_on_and_off(tmp_path: Path, monkeypatch) -> None:
    """DIFFERENCE: the same PDF through process(), only the #1266 geometry path toggled,
    under BOTH provider states (a populated ladder and an empty one).

    Route premise, asserted from the sidecar in every arm: the page ends at the
    structure-class floor (the empty ladder reaches it too: no reading to accept). The toggle
    changes the shipped bytes (old splice versus lane) and the provider state changes
    nothing: the page's own content lines are the same across the two states.
    """
    from socr.core import manifest as M

    pdf = _process_pdf(tmp_path / "doc.pdf")
    banner = BORN_DIGITAL_KEEP_SENTENCES_FLAG.format(page_num=1)
    on_by_state = {}
    for ladder in (True, False):
        tag = "ladder" if ladder else "empty"
        on_frag, on_meta, _ = _run(tmp_path, pdf, monkeypatch, ladder=ladder, tag=f"on-{tag}")
        with monkeypatch.context() as mp:
            real = M.table_floor_text_for_source
            mp.setattr(
                M,
                "table_floor_text_for_source",
                lambda *a, **k: real(*a, **{**k, "geometry_floor": False}),
            )
            off_frag, off_meta, _ = _run(
                tmp_path, pdf, monkeypatch, ladder=ladder, tag=f"off-{tag}"
            )
        assert on_meta["failure_mode"] == off_meta["failure_mode"], "route differs between arms"
        assert "structure_class" in on_meta["failure_mode"] or "judge" in on_meta["failure_mode"]
        assert banner in on_frag
        assert on_frag != off_frag, f"{tag}: the toggle changed nothing"
        for tok in ("Real GDP", "Final sales", "2.6", "4.1"):
            assert tok not in on_frag, f"{tag}: table content shipped"
        on_by_state[ladder] = on_frag

    # The provider state adds only socr's own furniture (the page judge's "unchecked" mark,
    # and where the page image is embedded versus linked); the page's own lines are the same.
    def _own_lines(body: str) -> list[str]:
        return [
            ln
            for ln in body.splitlines()
            if ln.strip() and not ln.lstrip().startswith((">", "![", "[page"))
        ]

    assert _own_lines(on_by_state[True]) == _own_lines(on_by_state[False]), (
        "the provider state changed the lane's page"
    )
    assert "page_judge=UNKNOWN" in on_by_state[True] and "page_judge" not in on_by_state[False]


def _restore_and_assemble(
    pdf: Path, meta: dict, out: Path, monkeypatch, *, expected_body, neuter=False
):
    """A FRESH pipeline restores the page (no ``native_words``), finalizes it, assembles.

    ``neuter`` replaces both fragment writers with no-ops for this phase only (the live run
    already happened), which is how the guard's bite is shown without mutating the source.
    """
    from test_gh1144_prose_survives_the_table_floor import _pipeline

    from socr.core.document import DocumentHandle
    from socr.core.manifest import finalized_page_records
    from socr.core.state import DocumentState
    from socr.pipeline.orchestrator import UnifiedPipeline

    fresh = _pipeline()
    fresh._scan_root = pdf.parent
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    fresh._restore_terminal_page_state(state, 1, PageOutput.from_dict(meta["winning_output"]), out)
    assert not (state.pages[1].native_words or []), "premise: a restored page has no words"
    # No file is read here at all: the restored page's finalized body, with no saved body and
    # no fragment on disk, is the live page's bytes (the replay arm, or a fail-closed marker).
    assert finalized_page_records(state)[0].output.text == expected_body, (
        "a words-less restored page did not finalize to the live bytes"
    )
    with monkeypatch.context() as mp:
        if neuter:
            mp.setattr(UnifiedPipeline, "_rewrite_all_fragments", lambda *a, **k: None)
            mp.setattr(UnifiedPipeline, "_flush_page_fragment", lambda *a, **k: None)
        fresh._phase_assemble(state, out)
    return state


@pytest.mark.parametrize("shape", ["recovered", "whole-page"])
def test_a_real_restore_reproduces_the_live_run_byte_for_byte(
    tmp_path: Path, monkeypatch, shape: str
) -> None:
    """Live run -> sidecar -> a FRESH pipeline restores the page with no ``native_words`` ->
    finalize -> assemble. The page fragment and the final .md are byte-identical to the live
    run's, and the surfaces (banner, ``scanned_prose_recovered``, status) agree.

    The compared artifacts are DELETED before the restore's assemble, so equality cannot come
    from a stale file: the fragment must be freshly written (and exist), the final .md must
    exist in the recovered arm and be ABSENT in the whole-page arm (a one-page document whose
    only page is the marker writes none).
    """
    import json

    make = _process_pdf if shape == "recovered" else _gated_process_pdf
    pdf = make(tmp_path / "doc.pdf")
    frag, meta, out = _run(tmp_path, pdf, monkeypatch, ladder=True, tag="live")

    def _final_path(root: Path):
        found = [p for p in root.rglob("*.md") if p.parent.name != "pages"]
        return found[0] if found else None

    live_final = _final_path(out)
    final_md = live_final.read_text() if live_final else None
    assert (final_md is not None) is (shape == "recovered"), "live final .md presence"
    banner = BORN_DIGITAL_KEEP_SENTENCES_FLAG.format(page_num=1)
    assert (banner in frag) is (shape == "recovered"), frag
    if shape == "whole-page":
        assert "Series" not in frag and "failed: unverifiable table" in frag

    # delete what we compare against; the sidecar stays (the restore reads it)
    frag_path = next(iter(out.rglob("pages/00001.md")))
    frag_path.unlink()
    if live_final is not None:
        live_final.unlink()
    assert not frag_path.exists() and _final_path(out) is None

    _restore_and_assemble(pdf, meta, out, monkeypatch, expected_body=frag)

    if shape == "recovered":
        # A document with content: assemble rewrites every fragment. (A one-page document whose
        # only page is the marker has no final text, and by design assemble then writes no
        # fragment; its in-loop flush wrote the one deleted above, so that arm is judged by
        # the words-less finalize assertion in the helper and by the absent final .md below.)
        assert frag_path.exists(), "the restore's assemble did not rewrite the page fragment"
        frag2 = frag_path.read_text()
    else:
        frag2 = frag
    meta2 = json.loads(next(iter(out.rglob("pages/00001.json"))).read_text())
    final2_path = _final_path(out)
    assert frag2 == frag, "the page fragment changed across a real restore"
    if shape == "recovered":
        assert final2_path is not None, "the final .md was not re-assembled"
        assert final2_path.read_text() == final_md, "the final .md changed across a real restore"
    else:
        assert final2_path is None, "a whole-page document gained a final .md"
    for key in ("status", "failure_mode"):
        assert meta2[key] == meta[key], key
    assert meta2["winning_output"].get("scanned_prose_recovered") == meta["winning_output"].get(
        "scanned_prose_recovered"
    )
    assert (banner in frag2) is (shape == "recovered")


def test_the_restore_guard_bites_when_assemble_writes_no_fragment(
    tmp_path: Path, monkeypatch
) -> None:
    """DIFFERENCE: the restore flow with the fragment writers neutered during the restore's
    assemble. The fragment deleted beforehand must then be missing, which is exactly what the
    real restore test's "assemble rewrote the fragment" assertion would catch."""
    pdf = _process_pdf(tmp_path / "doc.pdf")
    frag, meta, out = _run(tmp_path, pdf, monkeypatch, ladder=True, tag="live")
    frag_path = next(iter(out.rglob("pages/00001.md")))
    results = {}
    for neuter in (False, True):
        frag_path.unlink()
        _restore_and_assemble(pdf, meta, out, monkeypatch, expected_body=frag, neuter=neuter)
        results[neuter] = frag_path.exists()
    assert results == {False: True, True: False}, results


# ---------------------------------------------------------------------------
# merge gate (Fable P2): the keep-sentences banner says what its body does
# ---------------------------------------------------------------------------

#: The pre-merge-gate banner bytes, written out literally so a change to them is visible.
OLD_TAIL = " — the paragraphs below are this page's own text layer; every numeric row is withheld]"


def test_the_scan_and_any_digit_banners_keep_their_bytes_and_keep_sentences_gets_its_own() -> None:
    from socr.core.manifest import SCANNED_PROSE_RECOVERED_FLAG, prose_recovered_flag

    assert SCANNED_PROSE_RECOVERED_FLAG == "[page {page_num}: unverified scan" + OLD_TAIL
    assert BORN_DIGITAL_PROSE_RECOVERED_FLAG == "[page {page_num}: unverified extraction" + OLD_TAIL
    assert prose_recovered_flag(1, born_digital=False, keep_sentences=True) == (
        SCANNED_PROSE_RECOVERED_FLAG.format(page_num=1)
    ), "a scan never gets the keep-sentences wording"
    new = prose_recovered_flag(1, born_digital=True, keep_sentences=True)
    assert new == BORN_DIGITAL_KEEP_SENTENCES_FLAG.format(page_num=1)
    assert "every numeric row is withheld" not in new
    assert "sentences carrying figures ship unchecked" in new


def test_the_floor_body_carries_the_keep_sentences_banner(tmp_path: Path) -> None:
    """DIFFERENCE: the same page as a born-digital floor (keep-sentences body, new banner,
    and a sentence carrying a year ships) and as a scan (old banner, the year sentence
    withheld)."""
    born, _ = _build(tmp_path, _draw_p10, [GRID_BOX], name="kb.pdf")
    scan, _ = _build(tmp_path, _draw_p10, [GRID_BOX], born_digital=False, name="ks.pdf")
    born_body = structure_class_floor_text(born.pages[1], 1)
    scan_body = structure_class_floor_text(scan.pages[1], 1)
    assert born_body.startswith(BORN_DIGITAL_KEEP_SENTENCES_FLAG.format(page_num=1))
    assert SENTENCE_WITH_YEAR in born_body
    assert scan_body.startswith(SCANNED_PROSE_RECOVERED_FLAG.format(page_num=1))
    assert SENTENCE_WITH_YEAR not in scan_body


def test_an_old_form_born_digital_body_still_replays_and_is_still_recognised(
    tmp_path: Path,
) -> None:
    """Frozen bodies written before the merge gate carry the old born-digital banner; the
    words-less replay, the recovery recogniser, the page-kind split and the meter all still
    accept it."""
    from socr.benchmark import acceptance
    from socr.core.manifest import _is_prose_recovery_body, prose_recovery_is_born_digital

    _live, live_ps = _build(tmp_path, _draw_p10, [GRID_BOX], name="ol.pdf")
    body = structure_class_floor_text(live_ps, 1)
    old_body = body.replace(
        BORN_DIGITAL_KEEP_SENTENCES_FLAG.format(page_num=1),
        BORN_DIGITAL_PROSE_RECOVERED_FLAG.format(page_num=1),
    )
    assert old_body != body, "premise: the live body carries the new banner"
    assert structure_class_floor_text(
        _restored(tmp_path, old_body, typed=True, name="o.pdf"), 1
    ) == (old_body)
    assert _is_prose_recovery_body(old_body, 1) and prose_recovery_is_born_digital(old_body, 1)
    old_line = BORN_DIGITAL_PROSE_RECOVERED_FLAG.format(page_num=1)
    new_line = BORN_DIGITAL_KEEP_SENTENCES_FLAG.format(page_num=1)
    for line in (old_line, new_line):
        assert acceptance.line_is_mark(line, 1), line
        assert "scanned_flag" in acceptance.families_of(line, 1), line
