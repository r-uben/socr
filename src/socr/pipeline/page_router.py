"""Page-lane modality router — native PDF reading vs OCR LLM.

This is the first routing decision for every page in the progressive
(agentic) path:

  1. **Modality** (this module): NATIVE | CHART_ASSET | OCR
  2. **Provider ladder** (``route_page`` / ``route_ocr_provider``): which OCR
     engine to try when the lane is OCR.

OCR is the expensive exception. Trusted born-digital prose ships the PDF text
layer for free; chart pages keep that prose and attach a PNG; only scans,
corrupt layers, enhancement cases, and table pages climb the OCR ladder.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PageLane(str, Enum):
    """Extraction modality for one PDF page."""

    NATIVE = "native"
    CHART_ASSET = "chart_asset"
    OCR = "ocr"


# Stable reason codes for audit provenance / tests. Do not renumber casually —
# manifests and logs may already record them.
REASON_NO_NATIVE_FIRST = "no_native_first"
REASON_NOT_BORN_DIGITAL = "not_born_digital_or_empty_text"
REASON_NATIVE_ONLY = "native_only_policy"
REASON_NEEDS_ENHANCEMENT = "needs_ocr_enhancement"
REASON_HAS_TABLES = "has_tables"
REASON_CHART_MARKS = "chart_marks"
REASON_NATIVE_TRUSTED = "native_trusted_prose"


@dataclass(frozen=True)
class PageRouteDecision:
    """Result of the modality router for one page."""

    lane: PageLane
    reason: str


def decide_page_lane(
    *,
    native_first: bool,
    native_only: bool,
    is_born_digital: bool,
    native_text: str | None,
    needs_ocr_enhancement: bool,
    has_tables: bool,
    has_chart_marks: bool = False,
) -> PageRouteDecision:
    """Choose native PDF reading vs OCR LLM (vs chart-asset) for one page.

    Policy mirrors the historical ``_is_trusted_native_without_ocr`` +
    chart-lane predicates so this extraction is behavior-preserving.
    """
    if not native_first:
        return PageRouteDecision(PageLane.OCR, REASON_NO_NATIVE_FIRST)

    if not is_born_digital or not (native_text and native_text.strip()):
        return PageRouteDecision(PageLane.OCR, REASON_NOT_BORN_DIGITAL)

    if native_only:
        if has_chart_marks:
            return PageRouteDecision(PageLane.CHART_ASSET, REASON_CHART_MARKS)
        return PageRouteDecision(PageLane.NATIVE, REASON_NATIVE_ONLY)

    if needs_ocr_enhancement:
        return PageRouteDecision(PageLane.OCR, REASON_NEEDS_ENHANCEMENT)

    if has_tables:
        return PageRouteDecision(PageLane.OCR, REASON_HAS_TABLES)

    if has_chart_marks:
        return PageRouteDecision(PageLane.CHART_ASSET, REASON_CHART_MARKS)

    return PageRouteDecision(PageLane.NATIVE, REASON_NATIVE_TRUSTED)


def is_native_bypass_lane(decision: PageRouteDecision) -> bool:
    """True when the page skips the OCR provider ladder entirely."""
    return decision.lane in (PageLane.NATIVE, PageLane.CHART_ASSET)
