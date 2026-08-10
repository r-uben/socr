"""GH-136: encoding corruption is tiered by SIGNATURE, not by score.

`_encoding_corruption_ratio` summed three unrelated signatures into one density
score. Two of them (mid-word capitals, run-on tokens) are lost spaces —
cosmetic, recoverable by any reader. The third is an eaten leading digit:
`(1997)` shipping as `(/997)`, a WRONG NUMBER, which this corpus's invariant
ranks as worse than a missing one.

Ratio gating fails that class twice over. It treats a wrong year as
interchangeable with a lost space, and page length dilutes it: the same two
destroyed citations score 3% on a references page and 0.75% on a long prose
page, and the second ships under the flag floor with no signal at all.

So: digit corruption is gated on ABSOLUTE COUNT and routed to OCR (which
recovers the true digit rather than merely confessing it was lost); the hygiene
class stays ratio-gated and ships SUCCESS with a durable audit event.
"""

from __future__ import annotations

import fitz
import pytest

from socr.core.born_digital import BornDigitalDetector, count_digit_corruption
from socr.core.result import PageOutput, PageStatus
from socr.core.state import DocumentState, PageState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CLEAN = "The estimated coefficient on schooling is 0.082 and significant at the 1 percent level"


def _page(text_lines: list[str]):
    doc = fitz.open()
    page = doc.new_page()
    y = 80
    for ln in text_lines:
        page.insert_text((60, y), ln, fontsize=9)
        y += 16
    return doc, page


# ---------------------------------------------------------------------------
# 1. The detector separates a wrong number from a lost space
# ---------------------------------------------------------------------------

# Text a financial/economics corpus genuinely contains. Every one of these
# matches the RATIO's broader `/\d|\d/|\(/` pattern or sits next to something
# that does — which is exactly why that pattern cannot drive an absolute gate.
LEGITIMATE = [
    "the ratio was 1/2 of the prior year",
    "fiscal year 2019/20 results",
    "dated 12/31/2024 in the appendix",
    "post-9/11 volatility regimes",
    "the P/E multiple compressed",
    "N/A in column 3",
    "see http://example.com/2024/report",
    "(see 1/4 of the sample)",
    "GDP growth 3/4 of a point",
    "vol. 12, no. 3/4, pp. 55-70",
    "AC/DC ratio of 5/8",
    "w/ 3 additional controls",
    "1/2/3 sequence labels",
    "equation (3)/(4) ratio",
]

CORRUPTED = [
    "Fama and French (/997)",  # eaten year digit after a paren
    "JournalofFinance 51, /55-84",  # eaten page-range digit
    "Shiller (/98/) argues",
    "(/2003) and (/1997) both",
    "pp. /23-/45",
    "Econometrica 47 (/979)",
]


@pytest.mark.parametrize("text", LEGITIMATE)
def test_legitimate_slash_usage_is_not_digit_corruption(text):
    """A false positive costs a needless OCR pass on a good page.

    The gate requires the slash to BEGIN a token; in every construct above the
    slash is preceded by a digit or letter.
    """
    assert count_digit_corruption(text) == 0


@pytest.mark.parametrize("text", CORRUPTED)
def test_eaten_leading_digit_is_detected(text):
    assert count_digit_corruption(text) >= 1


def test_count_is_absolute_not_diluted_by_page_length():
    """The invariant is per-token, so 5,000 clean words must not wash out a hit."""
    damage = "Fama and French (/997)"
    assert count_digit_corruption(damage) == 1
    assert count_digit_corruption(" ".join([_CLEAN] * 200) + " " + damage) == 1


# ---------------------------------------------------------------------------
# 2. Routing: a wrong number escalates, a lost space does not
# ---------------------------------------------------------------------------


def test_single_eaten_digit_routes_the_page_to_ocr():
    """One destroyed year is enough — and OCR can RECOVER it, not just flag it."""
    det = BornDigitalDetector()
    lines = [_CLEAN] * 11 + ["Fama, E. and French, K. (/997). Multifactor explanations."]
    _doc, page = _page(lines)

    a = det._assess_page(page, 1)

    assert a.is_born_digital is False
    assert any("digit corruption" in n for n in a.notes)


def test_the_dilution_case_that_ratio_gating_missed():
    """Long prose page, same damage: 0.75% is under the flag floor.

    This is the page that shipped SUCCESS with a wrong citation year and no
    signal anywhere — the concrete motivation for #136.
    """
    det = BornDigitalDetector()
    lines = [_CLEAN] * 40 + ["Fama, E. and French, K. (/997), 51, /55-84."]
    _doc, page = _page(lines)

    ratio = det._encoding_corruption_ratio(" ".join(lines))
    assert ratio < det.ENCODING_CORRUPTION_FLAG, "precondition: ratio gating stays silent here"

    a = det._assess_page(page, 1)
    assert a.is_born_digital is False  # the signature gate catches what density missed


def test_hygiene_corruption_still_ships_but_is_marked():
    """Lost spaces are recoverable; the page ships, and says so on a live field."""
    det = BornDigitalDetector()
    lines = [_CLEAN] * 11 + ["FrenchfJoumal ofFinancial Economics 43 volumeNumberFortyThree"]
    _doc, page = _page(lines)

    a = det._assess_page(page, 1)

    assert a.is_born_digital is True
    assert a.has_encoding_hygiene_suspect is True


def test_clean_page_is_neither_escalated_nor_marked():
    det = BornDigitalDetector()
    _doc, page = _page([_CLEAN] * 12)

    a = det._assess_page(page, 1)

    assert a.is_born_digital is True
    assert a.has_encoding_hygiene_suspect is False


# ---------------------------------------------------------------------------
# 3. The mark reaches something that ships
# ---------------------------------------------------------------------------


class _Assessment:
    def __init__(self, pages):
        self.pages = pages


def test_flag_propagates_from_assessment_to_page_state():
    """PageAssessment.notes is read by nothing in the pipeline; this field is."""
    det = BornDigitalDetector()
    lines = [_CLEAN] * 11 + ["FrenchfJoumal ofFinancial Economics 43 volumeNumberFortyThree"]
    _doc, page = _page(lines)
    pa = det._assess_page(page, 1)

    state = DocumentState.__new__(DocumentState)
    state.pages = {1: PageState(page_num=1)}
    DocumentState.apply_born_digital(state, _Assessment([pa]))

    assert state.pages[1].has_encoding_hygiene_suspect is True


def test_page_state_default_is_clean():
    assert PageState(page_num=1).has_encoding_hygiene_suspect is False


def test_audit_event_kind_is_stable():
    """Consumers gate on this string; pin it so a rename is a deliberate act."""
    from socr.core.audit_log import AuditEvent

    ev = AuditEvent(
        page_num=1,
        kind="native_encoding_hygiene_suspect",
        engine="native",
        detail="x",
    )
    assert ev.kind == "native_encoding_hygiene_suspect"
    assert PageOutput(page_num=1, text="t", status=PageStatus.SUCCESS).status is PageStatus.SUCCESS
