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
region it enumerated, those regions' sources account for EVERY damaged glyph in
the page's own source layer, each retained replacement is still present in the
selected body with its own occurrence, and no PUA codepoint survives in that
body. The last term is a corroboration and never the proof -- see above.
The replacement itself is the witness, not the crop reference beside it: an
image proves a crop was kept, not that the transcription survived. Everything else
-- including the clean-region and legacy equation lanes, whose attachment
records do not map back to the damaged source spans -- stays unresolved. That
is a conservative answer, not a complete one: a lane gains the ability to clear
this signal by recording span-level coverage evidence, not by succeeding.

This module is pure. It opens no PDF, renders no crop and calls no provider,
because its callers run inside repeated page finalization.
"""

from __future__ import annotations

from collections import Counter
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


def corrupt_region_evidence(regions: list, native_text: str) -> dict:
    """Sparse coverage evidence for one corrupt-region recovery attempt.

    Called AFTER ``splice_math``, which is what sets ``source_aligned``: before
    it, ``region.resolved`` cannot be read as "this replacement went into the
    body".

    Two things are recorded, and both are load-bearing.

    The RETAINED REPLACEMENT of each covered region, verbatim. An earlier
    version stored the crop path and looked for the crop's Markdown plus the
    fixed candidate header, which is a prefix EVERY successful replacement
    shares: deleting the ``$$ ... $$`` reading while leaving the image link
    behind still read as complete coverage. A crop preserves visual evidence
    and proves nothing about whether the transcription survived, so the witness
    is the reading itself. It is taken from ``_region_replacement`` -- the same
    function ``splice_math`` splices with -- rather than rebuilt here, because a
    re-derivation can drift from what was actually written and then match
    nothing at all.

    The DENOMINATOR, as the damaged glyph count of the page's own source. The
    region list is what recovery happened to enumerate, so "every region
    resolved" says nothing about a damaged span the detector never proposed.
    Counting private-use codepoints on both sides makes the claim checkable:
    coverage holds only when the covered regions' sources account for every
    damaged glyph the page had.
    """
    from socr.core.born_digital import count_pua_chars
    from socr.math.recover import _region_replacement

    covered = [r for r in regions if r.resolved and r.source_aligned and r.crop_path]
    return {
        "lane": LANE_CORRUPT_REGION,
        "regions_total": len(regions),
        "source_pua_chars": count_pua_chars(native_text or ""),
        "covered_pua_chars": sum(count_pua_chars(r.source_text or "") for r in covered),
        "covered_replacements": [_region_replacement(r) for r in covered],
    }


def missing_coverage_witnesses(evidence: dict | None, text: str) -> list[str]:
    """Which retained replacements in ``evidence`` are absent from ``text``.

    Occurrence identity, not mere membership. Two damaged spans with identical
    source text yield two regions and two identical replacements, and
    ``splice_math`` writes both; a single surviving copy covers one of them, so
    the check counts occurrences rather than asking whether the string appears.

    Split out from :func:`unresolved_math_detail` because the document-level
    reconciliation needs exactly this question and nothing else: run in full
    against the ASSEMBLED body, the residual-PUA term would read another page's
    surviving codepoints as this page's damage and blame the wrong page.
    """
    if not evidence or evidence.get("lane") != LANE_CORRUPT_REGION:
        return []
    body = text or ""
    missing: list[str] = []
    seen: Counter[str] = Counter()
    for replacement in evidence.get("covered_replacements") or []:
        rep = str(replacement)
        seen[rep] += 1
        if not rep or body.count(rep) < seen[rep]:
            missing.append(rep)
    return missing


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
    replacements = [str(c) for c in (evidence.get("covered_replacements") or [])]
    regions_covered = len(replacements)
    source_pua = int(evidence.get("source_pua_chars") or 0)
    covered_pua = int(evidence.get("covered_pua_chars") or 0)

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

    # The denominator the region list cannot supply. "Every enumerated region
    # resolved" is a statement about what the detector proposed; a damaged span
    # it never proposed leaves no trace in that count. Comparing damaged glyphs
    # in the covered regions' sources against damaged glyphs in the page's own
    # source is the check that closes it -- and it is the reason clearance does
    # NOT rest on the shipped body having no private-use codepoints left, which
    # omission achieves just as well as recovery.
    if source_pua <= 0 or covered_pua != source_pua:
        return UnresolvedMathDetail(
            reason=(
                f"the recovered regions account for {covered_pua} of {source_pua} "
                "damaged glyph(s) in the page source; the remainder is unlocalised"
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
                "selected body, so the shipped page does not carry the reading "
                "(a surviving crop link is evidence of a crop, not of a "
                "transcription)"
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
