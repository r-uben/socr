"""#1402 (rule B, proposed): a scanned page's earlier OCR layer is confirmed by a fresh reading
when the two agree apart from Markdown marks and line-break hyphens, and the scan's ink outside
the layer could not be a character.

Every case is a DIFFERENCE: from the #1381 exact comparison (what rule B newly accepts), from
the strict ink fact (what rule B newly excuses), or from a control built the same way (what rule
B still refuses). The ``process()`` cases run the same document twice and change only whether
the new code path executes.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import fitz
import pytest

import socr
from socr.core import text_coverage as tc
from socr.core import verdict as verdict_mod
from socr.core.text_coverage import analyze_page
from socr.core.verdict import (
    PRIOR_OCR_AGREEMENT_KIND,
    _reading_matches_layer_exactly,
    _reading_matches_layer_normalised,
)

from test_gh1381_prior_ocr_agreement import (
    LAYER,
    SCAN_LINES,
    _agrees,
    _layer_of,
    _page,
    _run,
    _scan,
)
from test_gh1383_scan_ink_check import ROWS, _scan_doc

FONT_PT = 10
#: A row of the fixture page far below the text rows and far from every number.
EMPTY_ROW_Y = 600
PAGE_NUMBER_Y = 740


def test_loaded_source_is_this_checkout() -> None:
    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


# --- the text comparison: what rule B newly accepts ------------------------------------


@pytest.mark.parametrize(
    ("layer", "reading"),
    [
        ("Summary\nRates rose to 3.5 per cent.", "# Summary\nRates rose to 3.5 per cent."),
        ("Summary\nRates rose to 3.5 per cent.", "### Summary\nRates rose to 3.5 per cent."),
        ("Rates rose to 3.5 per cent.", "Rates rose to **3.5** per cent."),
        ("Rates rose to 3.5 per cent.", "Rates *rose* to 3.5 per cent."),
        (
            "Rate 3 3.5\nGrowth 1.1 -.7",
            "| Rate | 3 | 3.5 |\n|---|---|---|\n| Growth | 1.1 | -.7 |",
        ),
        ("Rates rose.\nGrowth slowed.", "Rates rose.\n\n---\n\nGrowth slowed."),
        ("reviewed condi-\ntions in the economy", "reviewed conditions in the economy"),
        ("reviewed conditions in the economy", "reviewed condi-\ntions in the economy"),
        ("the long-\nterm outlook", "the long-term outlook"),
        ("the long-\nterm outlook", "the long-\nterm outlook"),
    ],
)
def test_markup_and_line_break_hyphens_no_longer_decide(layer: str, reading: str) -> None:
    assert not _reading_matches_layer_exactly(reading, layer), "the exact rule already took it"
    assert _reading_matches_layer_normalised(reading, layer)
    assert _agrees(reading, _page(layer))


# --- the text comparison: what rule B still refuses ------------------------------------


@pytest.mark.parametrize(
    ("layer", "reading"),
    [
        # a Markdown-looking character of the LAYER is a character of the page
        ("# 7 items were listed", "7 items were listed"),
        ("Rates | 3.5", "Rates 3.5"),
        ("Rates **3.5", "Rates 3.5"),
        # markup does not hide a changed character
        ("Rates rose to 3.5 per cent.", "Rates rose to **3.6** per cent."),
        ("Rates rose to 3.5 per cent.", "# Rates rose to 35 per cent."),
        ("Rate 3 3.5", "| Rate | 3 | 35 |"),
        ("Rate 3 -3.5", "| Rate | 3 | 3.5 |"),
        # a lone footnote asterisk has no partner and is kept
        ("Preliminary.", "*Preliminary."),
        # digits are never joined across a line break
        ("from 1990-\n91 onward", "from 199091 onward"),
        ("from 1990-\n91 onward", "from 1990-91 onward"),
        ("rose 3-\nfold", "rose 3fold"),
        # a break that continues in upper case is not a word break
        ("the Bush-\nClinton years", "the BushClinton years"),
        # a different word across the break
        ("reviewed condi-\ntions in", "reviewed conditons in"),
        ("reviewed condi-\ntions in", "reviewed condi tions in"),
        # the comparison's own marker cannot be smuggled in
        ("reviewed conditions in", "reviewed condi\x01tions in"),
    ],
)
def test_everything_else_still_keeps_the_page_warned(layer: str, reading: str) -> None:
    assert _agrees(layer, _page(layer)), "the control must agree, or the case below is vacuous"
    assert not _reading_matches_layer_normalised(reading, layer)
    assert not _agrees(reading, _page(layer))


def test_added_or_dropped_content_inside_markup_is_still_compared() -> None:
    assert _agrees("# " + LAYER)
    assert not _agrees("# Minutes\n" + LAYER)
    assert not _agrees(LAYER + "\n\n| 9.9 |")


# --- the ink condition at the clause ----------------------------------------------------


def test_either_ink_fact_satisfies_completeness_and_neither_does_not() -> None:
    def page(strict: bool, below_letter: bool):
        return _page(
            text_hidden_layer_ink_accounted=strict,
            text_hidden_layer_ink_below_letter=below_letter,
        )

    assert _agrees(LAYER, page(True, False))
    assert _agrees(LAYER, page(False, True))
    assert not _agrees(LAYER, page(False, False))


# --- the ink check itself: one change to the same scan at a time -----------------------


def _facts(doc: fitz.Document) -> tuple[bool, bool]:
    """``(the strict #1381 fact, rule B's fact)`` for the fixture page."""
    cov = analyze_page(doc[0])
    return cov.hidden_layer_ink_accounted, cov.hidden_layer_ink_below_letter


def _letter_height_px(doc: fitz.Document) -> float:
    """The page's letter height as the check derives it, for sizing a fixture mark."""
    page = doc[0]
    gray, mat = tc._render_gray(page)
    dark, _light = tc._polarity_masks(gray)
    vis, inv = tc.visible_char_boxes(page)
    _l, stats, _r, covered, _g = tc._residual_components(
        dark, tc._paint(dark.shape, [*vis, *inv], page, mat)
    )
    assert len(covered) >= tc.H_MED_MIN_COVERED, "fixture too small for a page median"
    return float(__import__("numpy").median(stats[covered, 3])) * (
        tc.LETTER_H_PX / tc.GLYPH_H_P50_PX
    )


def _speck(x: float, y: float, doc_for_scale: fitz.Document | None = None):
    """Draws a square mark of glyph width and under letter height at (x, y), in the raster."""
    side_px = tc.GLYPH_W_P02_PX + 1
    side_pt = side_px / tc.SCALE
    if doc_for_scale is not None:
        assert side_px < _letter_height_px(doc_for_scale), "fixture speck is letter sized"
    return lambda s: s.draw_rect(fitz.Rect(x, y, x + side_pt, y + side_pt), color=None, fill=0)


def test_control_scan_passes_both_ink_checks() -> None:
    assert _facts(_scan_doc()) == (True, True)


def test_a_speck_away_from_every_number_is_excused() -> None:
    doc = _scan_doc(extra_ink=_speck(400, EMPTY_ROW_Y, _scan_doc()))
    assert _facts(doc) == (False, True)


def test_the_same_speck_beside_a_number_is_not_excused() -> None:
    """An unread decimal point or sign: the speck of the test above, moved to just left of a
    number that stands in the layer."""
    number_x = 300

    def layer_number(p):
        p.insert_text((number_x, EMPTY_ROW_Y), "75", fontsize=FONT_PT, render_mode=3)

    def ink(s):
        s.insert_text((number_x, EMPTY_ROW_Y), "75", fontsize=FONT_PT)
        _speck(number_x - 4, EMPTY_ROW_Y - 4)(s)

    def ink_far(s):
        s.insert_text((number_x, EMPTY_ROW_Y), "75", fontsize=FONT_PT)
        _speck(number_x + 150, EMPTY_ROW_Y - 4)(s)

    assert _facts(_scan_doc(extra_ink=ink_far, extra_page=layer_number))[1] is True
    assert _facts(_scan_doc(extra_ink=ink, extra_page=layer_number))[1] is False


def test_an_unread_minus_sign_is_not_excused() -> None:
    """Raster ``-7 percent`` whose layer is ``7 percent``: both readings can drop the sign."""
    x = 300
    sign_w = fitz.get_text_length("-", fontsize=FONT_PT)

    def layer_text(p):
        p.insert_text((x + sign_w, EMPTY_ROW_Y), "7 percent", fontsize=FONT_PT, render_mode=3)

    signed = _scan_doc(
        extra_ink=lambda s: s.insert_text((x, EMPTY_ROW_Y), "-7 percent", fontsize=FONT_PT),
        extra_page=layer_text,
    )
    unsigned = _scan_doc(
        extra_ink=lambda s: s.insert_text((x + sign_w, EMPTY_ROW_Y), "7 percent", fontsize=FONT_PT),
        extra_page=layer_text,
    )
    assert _facts(unsigned)[1] is True
    assert _facts(signed)[1] is False


def _dash(x: float, y: float):
    """Draws a dash in the raster: as low as the lowest glyph, wider than a letter is tall and
    narrower than a rule is long, whatever the page's letter height turns out to be."""
    letter_h = _letter_height_px(_scan_doc())
    width_pt = (letter_h + tc.GLYPH_H_P98_PX) / 2 / tc.SCALE
    height_pt = tc.GLYPH_H_P02_PX / tc.SCALE
    assert letter_h < tc.GLYPH_H_P98_PX, "fixture cannot place a dash between letter and rule"
    return lambda s: s.draw_rect(fitz.Rect(x, y, x + width_pt, y + height_pt), color=None, fill=0)


def test_a_dash_away_from_a_page_number_is_not_excused() -> None:
    """Lower than a letter, as wide as one: a dash, wherever it is not a page number's."""
    assert _facts(_scan_doc(extra_ink=_dash(400, EMPTY_ROW_Y)))[1] is False


def test_a_letter_outside_the_layer_is_not_excused() -> None:
    doc = _scan_doc(extra_ink=lambda s: s.insert_text((400, EMPTY_ROW_Y), "x", fontsize=FONT_PT))
    assert _facts(doc)[1] is False


def test_a_dropped_page_label_is_not_excused() -> None:
    """#1403: a page label in the raster that neither the layer nor the reading has."""
    doc = _scan_doc(
        extra_ink=lambda s: s.insert_text((300, PAGE_NUMBER_Y), "V-3", fontsize=FONT_PT)
    )
    assert _facts(doc)[1] is False
    doc = _scan_doc(
        extra_ink=lambda s: s.insert_text((300, PAGE_NUMBER_Y), "-2-", fontsize=FONT_PT)
    )
    assert _facts(doc)[1] is False


PAGE_NUMBER_X = 280
ROW_WORD_X = 200
#: A layer set smaller than the print: its boxes stop short of the row's last character, the
#: way an old OCR layer's boxes miss a page number's closing dash.
SHRUNK_LAYER_PT = 8


def _page_number_scan(
    layer_number: str,
    ink_number: str,
    *,
    layer_pt: float = FONT_PT,
    layer_dx: float = 0.0,
    row_word: str = "",
) -> fitz.Document:
    """A scan whose bottom row prints *ink_number*; the layer reads *layer_number* there, in
    *layer_pt* type, *layer_dx* points to the right. *row_word* stands on the same row, in both."""

    def ink(s):
        s.insert_text((PAGE_NUMBER_X, PAGE_NUMBER_Y), ink_number, fontsize=FONT_PT)
        if row_word:
            s.insert_text((ROW_WORD_X, PAGE_NUMBER_Y), row_word, fontsize=FONT_PT)

    def layer(p):
        p.insert_text(
            (PAGE_NUMBER_X + layer_dx, PAGE_NUMBER_Y),
            layer_number,
            fontsize=layer_pt,
            render_mode=3,
        )
        if row_word:
            p.insert_text((ROW_WORD_X, PAGE_NUMBER_Y), row_word, fontsize=FONT_PT, render_mode=3)

    return _scan_doc(extra_ink=ink, extra_page=layer)


def test_a_closing_dash_the_page_numbers_text_has_is_excused() -> None:
    """Raster ``-35-``, layer text ``-35-`` whose boxes stop before the closing dash: the strict
    fact fails on the dash, rule B's does not."""
    assert _facts(_page_number_scan("-35-", "-35-")) == (True, True)
    assert _facts(_page_number_scan("-35-", "-35-", layer_pt=SHRUNK_LAYER_PT)) == (False, True)


def test_a_dash_the_page_numbers_text_does_not_have_is_not_excused() -> None:
    """The layer reads ``-35`` and the raster prints ``-35-``: the closing dash was never read."""
    assert _facts(_page_number_scan("-35", "-35"))[1] is True
    assert _facts(_page_number_scan("-35", "-35-"))[1] is False


def test_a_sign_before_a_lone_number_is_not_a_page_number_dash() -> None:
    """Raster ``-35`` alone on its row over a layer ``35``: a negative number lost its sign."""
    sign_w = fitz.get_text_length("-", fontsize=FONT_PT)
    assert _facts(_page_number_scan("35", "35"))[1] is True
    assert _facts(_page_number_scan("35", "-35", layer_dx=sign_w))[1] is False


def test_the_same_dash_beside_a_number_that_shares_its_row_is_not_excused() -> None:
    """``Total -35-`` is a table row, not a page number: its trailing dash may be content."""
    lone = _page_number_scan("-35-", "-35-", layer_pt=SHRUNK_LAYER_PT)
    shared = _page_number_scan("-35-", "-35-", layer_pt=SHRUNK_LAYER_PT, row_word="Total")
    assert _facts(_page_number_scan("-35-", "-35-", row_word="Total"))[1] is True
    assert (_facts(lone)[1], _facts(shared)[1]) == (True, False)


def test_a_drawing_that_is_not_a_rule_is_not_excused() -> None:
    def box(p):
        p.draw_rect(fitz.Rect(300, 300, 360, 340), color=(1, 1, 1), fill=None)

    assert _facts(_scan_doc(extra_page=box))[1] is False


def test_a_page_without_an_invisible_layer_has_no_fact() -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for row, line in enumerate(ROWS):
        page.insert_text((72, 100 + row * 14), line, fontsize=FONT_PT)
    assert analyze_page(page).hidden_layer_ink_below_letter is False


def test_an_error_in_the_check_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a, **_k):
        raise RuntimeError("words unavailable")

    monkeypatch.setattr(tc, "_layer_words_px", boom)
    assert _facts(_scan_doc())[1] is False


# --- process(): only whether the new code path executes differs ------------------------


def _confirmed(arm: dict) -> bool:
    return arm["sidecar"].get(PRIOR_OCR_AGREEMENT_KIND) is True


def test_process_a_markdown_heading_no_longer_blocks_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = _scan(tmp_path / "scan.pdf", SCAN_LINES, SCAN_LINES)
    reading = "# " + _layer_of(pdf)
    with monkeypatch.context() as m:
        m.setattr(verdict_mod, "_reading_matches_layer_normalised", lambda *_a: False)
        before = _run(tmp_path / "before", pdf, reading, monkeypatch)
    after = _run(tmp_path / "after", pdf, reading, monkeypatch)
    wrong = _run(tmp_path / "wrong", pdf, reading.replace("3.5", "3.6"), monkeypatch)
    assert before["sidecar"].get("prior_ocr_layer_suspect") is True, "not a prior-OCR scan"
    assert (_confirmed(before), _confirmed(after), _confirmed(wrong)) == (False, True, False)
    assert (before["sidecar"]["status"], after["sidecar"]["status"]) == ("warning", "success")


def test_process_a_speck_no_longer_blocks_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = _scan_doc(extra_ink=_speck(400, EMPTY_ROW_Y), layer=SCAN_LINES, ink=SCAN_LINES)
    pdf = tmp_path / "scan.pdf"
    doc.save(pdf)
    reading = _layer_of(pdf)
    with monkeypatch.context() as m:
        m.setattr(tc, "scan_residual_ink_holds_no_character", lambda *_a: False)
        before = _run(tmp_path / "before", pdf, reading, monkeypatch)
    after = _run(tmp_path / "after", pdf, reading, monkeypatch)
    assert after["sidecar"].get("text_hidden_layer_ink_accounted") is not True, (
        "the strict fact took the page: the speck is not in the fixture"
    )
    assert after["sidecar"].get("text_hidden_layer_ink_below_letter") is True
    assert (_confirmed(before), _confirmed(after)) == (False, True)


# --- the report on a finished run: read-only, the run's own predicate ------------------


def _tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def _doc_dir(out_dir: Path) -> Path:
    return next(iter(out_dir.rglob("pages"))).parent


def test_agreement_report_finds_what_a_run_of_this_code_confirms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run made WITHOUT rule B ships the page marked; the report, with rule B, names it. A
    page whose reading is wrong is not named. Nothing in the run's output changes."""
    from socr.pipeline.refinalize_prior_ocr import agreement_report_document

    pdf = _scan(tmp_path / "scan.pdf", SCAN_LINES, SCAN_LINES)
    reading = "# " + _layer_of(pdf)
    with monkeypatch.context() as m:
        m.setattr(verdict_mod, "_reading_matches_layer_normalised", lambda *_a: False)
        old = _run(tmp_path / "old", pdf, reading, monkeypatch)
        wrong = _run(tmp_path / "wrong", pdf, reading.replace("3.5", "3.6"), monkeypatch)
    assert not _confirmed(old) and not _confirmed(wrong)
    assert old["sidecar"]["failure_mode"] == "prior_ocr_layer_unverified"

    digest = _tree_digest(tmp_path / "old")
    counts: Counter = Counter()
    assert agreement_report_document(_doc_dir(tmp_path / "old"), tmp_path, counts) == [1]
    assert agreement_report_document(_doc_dir(tmp_path / "wrong"), tmp_path, counts) == []
    assert counts == {"would be confirmed by agreement": 1, "stays marked": 1}
    assert _tree_digest(tmp_path / "old") == digest

    # the report is the predicate, not a second rule: without rule B it names nothing
    with monkeypatch.context() as m:
        m.setattr(verdict_mod, "_reading_matches_layer_normalised", lambda *_a: False)
        assert agreement_report_document(_doc_dir(tmp_path / "old"), tmp_path, Counter()) == []


def test_agreement_report_refuses_a_pdf_that_is_not_the_runs_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from socr.pipeline.refinalize_prior_ocr import agreement_report_document

    pdf = _scan(tmp_path / "scan.pdf", SCAN_LINES, SCAN_LINES)
    with monkeypatch.context() as m:
        m.setattr(verdict_mod, "_reading_matches_layer_normalised", lambda *_a: False)
        _run(tmp_path / "old", pdf, "# " + _layer_of(pdf), monkeypatch)
    other = tmp_path / "other"
    other.mkdir()
    _scan(other / "scan.pdf", SCAN_LINES[:2], SCAN_LINES[:2])
    counts: Counter = Counter()
    assert agreement_report_document(_doc_dir(tmp_path / "old"), other, counts) == []
    assert list(counts) == ["not checked: PDF under --pdf-dir is not the run's input (checksum)"]


def test_agreement_report_counts_an_unfinished_page_and_does_not_judge_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import shutil

    from socr.pipeline.refinalize_prior_ocr import agreement_report_document

    pdf = _scan(tmp_path / "scan.pdf", SCAN_LINES, SCAN_LINES)
    with monkeypatch.context() as m:
        m.setattr(verdict_mod, "_reading_matches_layer_normalised", lambda *_a: False)
        _run(tmp_path / "old", pdf, "# " + _layer_of(pdf), monkeypatch)
    shutil.copytree(tmp_path / "old", tmp_path / "unfinished")
    sidecar_path = next(iter((tmp_path / "unfinished").rglob("pages/00001.json")))
    sidecar_path.write_text(json.dumps({**json.loads(sidecar_path.read_text()), "terminal": False}))
    counts: Counter = Counter()
    assert agreement_report_document(_doc_dir(tmp_path / "old"), tmp_path, Counter()) == [1]
    assert agreement_report_document(_doc_dir(tmp_path / "unfinished"), tmp_path, counts) == []
    assert counts == {"not checked: page record is not final": 1}


def test_only_the_pages_own_prior_ocr_mark_is_set_aside() -> None:
    from socr.core.marks import page_unchecked_clause_mark
    from socr.pipeline.refinalize_prior_ocr import _without_own_prior_ocr_mark

    own = page_unchecked_clause_mark(3, "prior_ocr_layer", "UNKNOWN", "page", "read once")
    other = page_unchecked_clause_mark(3, "text_layer_numbers", "UNKNOWN", "page", "no layer")
    assert _without_own_prior_ocr_mark(f"{own}\n\nbody 3.5", 3) == "body 3.5"
    kept = _without_own_prior_ocr_mark(f"{other}\n\n{own}\n\nbody 3.5", 3)
    assert other in kept and own not in kept
    # another page's line, or a line inside the body, is not this page's mark
    assert _without_own_prior_ocr_mark(f"{own}\n\nbody", 4) == f"{own}\n\nbody"
    assert _without_own_prior_ocr_mark(f"body\n\n{own}", 3) == f"body\n\n{own}"
