"""Text-layer encoding-corruption detection (broken font/ToUnicode maps).

Catches the failure mode the garbage-char check misses: valid characters in wrong
positions ("Journal"->"Joumal", "1997"->"(/997)"). Two tiers: flag mild (e.g. a
broken header font) for visibility; escalate pervasive corruption to OCR.
"""

from __future__ import annotations

import fitz

from socr.core.born_digital import BornDigitalDetector


def _page(text_lines: list[str]):
    doc = fitz.open()
    page = doc.new_page()
    y = 80
    for ln in text_lines:
        page.insert_text((60, y), ln, fontsize=9)
        y += 16
    return doc, page


_CLEAN = "The estimated coefficient on schooling is 0.082 and significant at the 1 percent level"
# Broken-encoding signatures: dropped spaces (mid-word caps), '/' for '1', run-ons.
_CORRUPT = (
    "The coefficienton schoolingis 0./82 significantAt the / percent across /204 firmsInThe set"
)


def test_score_zero_on_clean_text():
    det = BornDigitalDetector()
    text = " ".join([_CLEAN] * 6)
    assert det._encoding_corruption_ratio(text) == 0.0


def test_score_high_on_corrupted_text():
    det = BornDigitalDetector()
    text = " ".join([_CORRUPT] * 6)
    assert det._encoding_corruption_ratio(text) > 0.05


def test_score_needs_enough_tokens():
    det = BornDigitalDetector()
    assert det._encoding_corruption_ratio("ofFinancial (/997)") == 0.0  # < 20 tokens


def test_pervasive_corruption_routes_to_ocr():
    det = BornDigitalDetector()
    _doc, page = _page([_CORRUPT] * 12)
    a = det._assess_page(page, 1)
    assert a.is_born_digital is False                 # not trusted -> OCR
    assert any("encoding corrupted" in n for n in a.notes)


def test_mild_corruption_is_flagged_not_escalated():
    det = BornDigitalDetector()
    # mostly clean body + a couple corrupted tokens (a broken header font)
    lines = [_CLEAN] * 11 + ["FrenchfJoumal ofFinancial Economics 43 (/997) /53-/93"]
    _doc, page = _page(lines)
    a = det._assess_page(page, 1)
    assert a.is_born_digital is True                  # body still trusted
    assert any("encoding suspect" in n for n in a.notes)  # but recorded


def test_clean_page_not_flagged():
    det = BornDigitalDetector()
    _doc, page = _page([_CLEAN] * 12)
    a = det._assess_page(page, 1)
    assert a.is_born_digital is True
    assert not any("encoding" in n for n in a.notes)
