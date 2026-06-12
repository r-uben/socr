"""Math recovery: region grouping, LaTeX cleaning, and prose-preserving splice.

The OCR call is injected (no model needed); only geometry/text logic is tested.
"""

from __future__ import annotations

import fitz

from socr.math.recover import (
    clean_latex,
    corrupt_math_line_rects,
    recover_math_regions,
    splice_math,
)

# '=' -> '¼', '(' -> 'ð', ')' -> 'Þ', '+' -> 'þ'
_EQ = "PðA or BÞ ¼ PðAÞ þ PðBÞ"


def _page(lines: list[tuple[float, str]]):
    """Build a page; each (y, text) is one line at x=72."""
    doc = fitz.open()
    page = doc.new_page()
    for y, text in lines:
        page.insert_text((72, y), text, fontsize=10)
    return doc, page


def test_clean_latex_strips_fences_and_dollars():
    assert clean_latex("```latex\nP(A) = 1\n```") == "P(A) = 1"
    assert clean_latex("$$P(A) = 1$$") == "P(A) = 1"
    assert clean_latex("$x$\n$y$") == "x\ny"
    assert clean_latex("  P(A)=1  ") == "P(A)=1"


def test_corrupt_math_lines_detected_and_prose_skipped():
    _doc, page = _page(
        [
            (100, "This is clean prose with no math at all."),
            (140, _EQ),
            (180, "More clean prose here."),
        ]
    )
    rects = corrupt_math_line_rects(page)
    assert len(rects) == 1  # only the corrupt line


def test_adjacent_corrupt_lines_merge_into_one_region():
    _doc, page = _page([(100, _EQ), (112, _EQ), (124, _EQ)])  # tight spacing -> merge
    rects = corrupt_math_line_rects(page)
    assert len(rects) == 1


def test_distant_corrupt_lines_stay_separate():
    _doc, page = _page([(100, _EQ), (400, _EQ)])  # far apart -> two regions
    rects = corrupt_math_line_rects(page)
    assert len(rects) == 2


def test_no_corrupt_math_returns_empty():
    _doc, page = _page([(100, "Ordinary prose."), (140, "f(x) = a + b clean.")])
    assert corrupt_math_line_rects(page) == []


def test_recover_uses_injected_ocr():
    _doc, page = _page([(100, "prose"), (140, _EQ)])
    regions = recover_math_regions(page, ocr=lambda png, **kw: "P(A) = 1")
    assert len(regions) == 1
    assert regions[0][1] == "P(A) = 1"


def test_splice_replaces_math_keeps_prose_verbatim():
    _doc, page = _page([(100, "Clean prose line one."), (140, _EQ), (180, "Clean prose line two.")])
    regions = recover_math_regions(page, ocr=lambda png, **kw: "P(A \\text{ or } B) = P(A) + P(B)")
    native = "Clean prose line one.\n" + _EQ + "\nClean prose line two."
    out = splice_math(page, native, regions)
    assert "Clean prose line one." in out
    assert "Clean prose line two." in out
    assert "P(A \\text{ or } B) = P(A) + P(B)" in out
    assert "$$" in out
    assert "¼" not in out and "ð" not in out  # no mojibake survives


def test_splice_empty_latex_drops_corruption():
    _doc, page = _page([(100, "Prose."), (140, _EQ)])
    regions = recover_math_regions(page, ocr=lambda png, **kw: "")  # model failed
    out = splice_math(page, "Prose.\n" + _EQ, regions)
    assert "¼" not in out and "ð" not in out  # corrupted glyphs never reach corpus
    assert "Prose." in out
