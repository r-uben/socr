"""#1270 (B4, class 1): a list bullet is not unrecovered mathematics.

Before: ``has_unmapped_math_glyphs = pua_count > 0`` counted every private-use codepoint,
font-blind, so SymbolMT / Wingdings list bullets (U+F0B7, U+F0A7, U+F074, U+F06E) marked
292 of 298 ``native_glyphs=FAIL`` pages in run 51270eb5 as "unrecovered mathematics".
Now the flag follows the span-level maths count (#960's ``_math_private_use_count``).

Scope: only the PUA clause (``_native_glyph_clause``). ``NATIVE_GARBLED_MATH`` and
``NATIVE_MINUS_AS_DIGIT`` also map to ``native_glyphs`` and are untouched here.

Hermetic: a fitz PDF whose text and span layers are faked through the extractor seam (a base
font cannot encode a private-use codepoint). The pipeline arm patches
``_available_engines_for_agentic`` in BOTH provider states, pins the engines and the judge.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest
from test_gh165_unresolved_math_outcome import _prose_pdf, _run
from verdict_marks_helper import assert_differs_only_by_marks

from socr.core.born_digital import BornDigitalDetector, count_pua_chars
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import PageOutput, PageStatus
from socr.core.state import PageState
from socr.core.verdict import _native_glyph_clause

BULLET = chr(0xF0B7)
WINGDINGS_BULLET = chr(0xF0A7)
MATH_MINUS = chr(0xF02D)

_PROSE = (
    "Optimists set a lift-off date and the heterogeneous agents consume more, with enough "
    "trailing words to stay above the floor for a born digital clean text layer classification."
)


def _assess(tmp_path: Path, monkeypatch, *, glyph: str, font: str, scan_fails: bool = False):
    """Run the REAL ``_assess_page`` on a page whose text layer carries ``glyph`` in a span of
    ``font``. Only the glyph's font differs between arms."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = _prose_pdf(tmp_path / "doc.pdf")
    doc = fitz.open(str(pdf))
    page = doc[0]
    original = page.get_text
    crafted = f"{_PROSE}\n{glyph} first list item\n"

    def fake(*args, **kwargs):
        mode = args[0] if args else kwargs.get("option", "text")
        if mode == "text":
            return crafted
        if mode == "dict":
            spans = [{"font": font, "text": f"{glyph} first list item", "bbox": (0, 0, 1, 1)}]
            return {"blocks": [{"type": 0, "lines": [{"spans": spans}]}]}
        return original(*args, **kwargs)

    monkeypatch.setattr(page, "get_text", fake)
    if scan_fails:
        monkeypatch.setattr(
            "socr.core.born_digital.detect_garbled_math",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("scan failed")),
        )
    pa = BornDigitalDetector()._assess_page(page, 1)
    doc.close()
    return pa


@pytest.mark.parametrize(
    ("glyph", "font"),
    [(BULLET, "SymbolMT"), (WINGDINGS_BULLET, "Wingdings-Regular"), (chr(0xF074), "Wingdings")],
)
def test_a_bullet_is_not_a_math_glyph_but_is_still_reported(tmp_path, monkeypatch, glyph, font):
    pa = _assess(tmp_path, monkeypatch, glyph=glyph, font=font)
    assert pa.is_born_digital
    assert count_pua_chars(pa.native_text) == 1, "premise: the codepoint is in the shipped text"
    assert pa.has_unmapped_math_glyphs is False
    assert not any("unmapped math glyphs" in n for n in pa.notes)
    assert any("list bullets or dingbats" in n for n in pa.notes), "never silent"
    assert pa.needs_ocr_enhancement is False


def test_only_the_font_differs_between_a_bullet_and_maths(tmp_path, monkeypatch):
    """Same codepoint-count, same text; a font that is not a Symbol-encoded font owns its own
    glyph, which may be maths (the #960 rule)."""
    bullet = _assess(tmp_path / "a", monkeypatch, glyph=BULLET, font="SymbolMT")
    maths = _assess(tmp_path / "b", monkeypatch, glyph=BULLET, font="CIDFont+F6")
    real_minus = _assess(tmp_path / "c", monkeypatch, glyph=MATH_MINUS, font="SymbolMT")
    assert bullet.native_text == maths.native_text
    assert bullet.has_unmapped_math_glyphs is False
    assert maths.has_unmapped_math_glyphs is True
    assert real_minus.has_unmapped_math_glyphs is True
    assert any("unmapped math glyphs" in n for n in real_minus.notes)


def test_a_failed_scan_keeps_the_font_blind_answer(tmp_path, monkeypatch):
    """Fail closed: an unreadable page is unknown, not clean."""
    with_pua = _assess(tmp_path / "a", monkeypatch, glyph=BULLET, font="SymbolMT", scan_fails=True)
    assert with_pua.garbled_math_scan_failed is True
    assert with_pua.has_unmapped_math_glyphs is True
    no_pua = _assess(tmp_path / "b", monkeypatch, glyph="-", font="SymbolMT", scan_fails=True)
    assert no_pua.has_unmapped_math_glyphs is False


def test_the_clause_follows_the_flag_and_only_the_flag() -> None:
    out = PageOutput(
        page_num=1, text="x", status=PageStatus.WARNING, engine="native", audit_passed=True
    )
    off, on = PageState(page_num=1), PageState(page_num=1)
    on.has_unmapped_math_glyphs = True
    assert _native_glyph_clause(off, out) is None
    assert _native_glyph_clause(on, out) is not None


@pytest.mark.parametrize("providers", [[PROFILE_QWEN_LOCAL], []], ids=["provider", "no-provider"])
def test_the_shipped_page_differs_only_by_the_mark(tmp_path, monkeypatch, providers) -> None:
    """Shipping boundary, both provider states: the bullet page and the maths page carry the
    SAME bytes; the bullet page loses the ``native_glyphs`` mark and the maths page keeps it."""
    from socr.core.manifest import finalized_page_record

    bullet_pa = _assess(tmp_path / "pa", monkeypatch, glyph=BULLET, font="SymbolMT")
    # same codepoint, same text; only the font of the span carrying it differs
    maths_pa = _assess(tmp_path / "pb", monkeypatch, glyph=BULLET, font="CIDFont+F6")
    results = {}
    for name, pa in (("bullet", bullet_pa), ("maths", maths_pa)):
        _pipe, state, _res, _out = _run(
            tmp_path / name,
            native_text=pa.native_text,
            damaged=pa.has_unmapped_math_glyphs,
            providers=providers,
            tag=name,
        )
        rec = finalized_page_record(state, 1)
        results[name] = (
            rec.output.status,
            [c.check.value for c in rec.verdict.clauses if c.result.value == "FAIL"],
            rec.output.text,
        )
    assert "native_glyphs" in results["maths"][1]
    assert "native_glyphs" not in results["bullet"][1]
    assert results["bullet"][0] is PageStatus.SUCCESS
    assert results["maths"][0] is PageStatus.WARNING
    # the page differs only by the mark: same native bytes, private-use codepoint included
    assert_differs_only_by_marks(results["maths"][2], results["bullet"][2], 1)
