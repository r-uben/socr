"""Outcome-based accounting for detected math-glyph damage (#165, #140).

A born-digital page can carry PUA / weak-ToUnicode math glyphs: the prose
extracts cleanly, the mathematics does not. ``PageAssessment`` detects that
damage (``has_unmapped_math_glyphs``); this module answers the *separate*
question of whether the damage is still unresolved in the bytes the page
actually ships.

The distinction is the whole point of #165. The previous accounting suppressed
the ``native_math_unrecovered`` audit event whenever ``--detect-equations`` and
``--recover-clean-equations`` were both set -- a statement about the run's
CONFIGURATION, not about its OUTCOME. Enabling two flags silenced the only
durable record of lost math without recovering a single glyph.

What this module will accept as recovery is deliberately narrow. None of the
following establishes coverage on its own:

* a nonzero ``recovered_regions`` count -- some other region may have failed;
* an accepted model reading -- acceptance is a syntax gate, not a coverage
  proof, and the reading may still be discarded by later selection;
* the ABSENCE of PUA codepoints in the shipped text -- omission removes PUA
  exactly as effectively as recovery does.

Coverage is therefore asserted only for the corrupt-region lane, and only when
that lane enumerated at least one damaged region, resolved and aligned EVERY
region it enumerated, each of those replacements is still present verbatim in
the selected body, and no PUA codepoint survives in that body. Everything else
-- including the clean-region and legacy equation lanes, whose attachment
records do not map back to the damaged source spans -- stays unresolved. That
is a conservative answer, not a complete one: a lane gains the ability to clear
this signal by recording span-level coverage evidence, not by succeeding.

This module is pure. It opens no PDF, renders no crop and calls no provider,
because its callers run inside repeated page finalization.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The audit-event kind naming detected math-glyph damage that survived into the
#: shipped page. One per affected page.
UNRESOLVED_MATH_KIND = "native_math_unrecovered"

#: The only lane whose evidence can currently prove span coverage.
LANE_CORRUPT_REGION = "corrupt_math_region"


@dataclass(frozen=True)
class UnresolvedMathDetail:
    """Why a page's detected math-glyph damage is still unresolved."""

    reason: str
    lane: str
    regions_total: int
    regions_covered: int
    residual_pua: int

    @property
    def detail(self) -> str:
        """The audit-event / note prose. Deterministic: no counts of runs or time."""
        return (
            "born-digital native text shipped with unmapped math glyphs "
            "(private-use codepoints, weak ToUnicode) that no retained recovery "
            f"covers: {self.reason}"
        )

    def as_data(self) -> dict:
        return {
            "reason": self.reason,
            "lane": self.lane,
            "regions_total": self.regions_total,
            "regions_covered": self.regions_covered,
            "residual_pua_chars": self.residual_pua,
        }


def corrupt_region_evidence(regions: list) -> dict:
    """Sparse coverage evidence for one corrupt-region recovery attempt.

    Called AFTER ``splice_math``, which is what sets ``source_aligned``: before
    it, ``region.resolved`` cannot be read as "this replacement went into the
    body". Each covered region contributes its crop path, which is the witness
    :func:`unresolved_math_detail` later looks for in the selected text.
    """
    return {
        "lane": LANE_CORRUPT_REGION,
        "regions_total": len(regions),
        "covered_crops": [
            str(region.crop_path)
            for region in regions
            if region.resolved and region.source_aligned and region.crop_path
        ],
    }


def _resolved_witness(crop_path: str) -> str:
    """The exact bytes ``splice_math`` writes for a RESOLVED region's crop.

    Built from ``recover``'s own constants rather than restated here, so a change
    to the replacement markup cannot leave this recogniser silently matching
    nothing. The candidate header is load-bearing: an ALIGNED BUT UNRESOLVED
    region emits the same crop reference followed by the unresolved marker, so
    the crop path alone would read a refusal as a recovery.
    """
    from socr.math.recover import _CORRUPT_CANDIDATE_HEADER

    return f"![Corrupt equation crop]({crop_path})\n{_CORRUPT_CANDIDATE_HEADER}"


def missing_coverage_witnesses(evidence: dict | None, text: str) -> list[str]:
    """Which of ``evidence``'s recovered replacements are absent from ``text``.

    Split out from :func:`unresolved_math_detail` because the document-level
    reconciliation needs exactly this question and nothing else: run against the
    ASSEMBLED body, the residual-PUA term would read another page's surviving
    codepoints as this page's damage and blame the wrong page.
    """
    if not evidence or evidence.get("lane") != LANE_CORRUPT_REGION:
        return []
    body = text or ""
    return [
        str(crop)
        for crop in (evidence.get("covered_crops") or [])
        if _resolved_witness(str(crop)) not in body
    ]


def unresolved_math_detail(
    *,
    has_unmapped_math_glyphs: bool,
    evidence: dict | None,
    text: str,
) -> UnresolvedMathDetail | None:
    """Whether this page's damaged math glyphs remain unresolved in ``text``.

    ``text`` must be the SELECTED body -- the bytes that ship. A recovery that
    a later selection discarded has not covered anything.

    Returns ``None`` when the page carries no math-glyph damage signal, or when
    a retained recovery demonstrably covers all of it. Otherwise returns the
    detail naming what is still missing.
    """
    from socr.core.born_digital import count_pua_chars

    if not has_unmapped_math_glyphs:
        return None

    body = text or ""
    residual = count_pua_chars(body)

    if not evidence:
        return UnresolvedMathDetail(
            reason="no recovery evidence was retained for this page",
            lane="",
            regions_total=0,
            regions_covered=0,
            residual_pua=residual,
        )

    lane = str(evidence.get("lane") or "")
    regions_total = int(evidence.get("regions_total") or 0)
    covered_crops = [str(c) for c in (evidence.get("covered_crops") or [])]
    regions_covered = len(covered_crops)

    if lane != LANE_CORRUPT_REGION:
        return UnresolvedMathDetail(
            reason=(
                f"lane {lane!r} records no span-level coverage, so it cannot show "
                "which damaged glyphs were recovered"
            ),
            lane=lane,
            regions_total=regions_total,
            regions_covered=regions_covered,
            residual_pua=residual,
        )

    if regions_total == 0:
        return UnresolvedMathDetail(
            reason="recovery enumerated no region on a page detected as damaged",
            lane=lane,
            regions_total=0,
            regions_covered=0,
            residual_pua=residual,
        )

    if regions_covered != regions_total:
        return UnresolvedMathDetail(
            reason=(
                f"{regions_total - regions_covered} of {regions_total} damaged "
                "region(s) kept no aligned, validated replacement"
            ),
            lane=lane,
            regions_total=regions_total,
            regions_covered=regions_covered,
            residual_pua=residual,
        )

    missing = missing_coverage_witnesses(evidence, body)
    if missing:
        return UnresolvedMathDetail(
            reason=(
                f"{len(missing)} recovered replacement(s) are absent from the "
                "selected body, so the shipped page does not carry them"
            ),
            lane=lane,
            regions_total=regions_total,
            regions_covered=regions_covered,
            residual_pua=residual,
        )

    if residual:
        return UnresolvedMathDetail(
            reason=(
                f"{residual} private-use codepoint(s) survive in the selected body "
                "outside every recovered region"
            ),
            lane=lane,
            regions_total=regions_total,
            regions_covered=regions_covered,
            residual_pua=residual,
        )

    return None
