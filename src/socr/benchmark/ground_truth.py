"""Ground truth extraction from born-digital PDFs using PyMuPDF.

Extracts native text per page and saves as individual text files for
benchmark comparison.

GH-39: the native text layer is NOT automatically truth. A broken font/ToUnicode
map yields valid characters in wrong positions — "(1997)" extracted as "(/997)"
(GH-136) — so a page whose text layer is corrupted cannot serve as ground truth.
Scoring an engine against it would count a CORRECT reading of the raster as an
error, penalising OCR exactly where OCR is most valuable and flattering the
native-trusting lane that agentic routing depends on.

Contaminated pages are therefore MARKED, not silently dropped: page numbers stay
aligned with engine output, and the exclusion count is a reportable figure rather
than an invisible narrowing of the corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


from socr.core.pdf import open_pdf

from socr.core.born_digital import BornDigitalDetector, count_digit_corruption


@dataclass
class PageGroundTruth:
    """Ground truth text for a single page."""

    page_num: int  # 1-indexed
    text: str
    word_count: int
    char_count: int
    #: False when the text layer is too corrupted to serve as truth (GH-39).
    usable: bool = True
    #: Human-readable reason; empty when ``usable``.
    exclusion_reason: str = ""


class GroundTruthExtractor:
    """Extract native text from born-digital PDFs as ground truth."""

    def __init__(self, detector: BornDigitalDetector | None = None) -> None:
        # Reuse the SHIPPED detector rather than restating what "corrupted"
        # means: a second definition here would drift from the one the pipeline
        # routes on, and the benchmark would then measure a different notion of
        # truth than the product enforces.
        self._detector = detector or BornDigitalDetector()

    def _assess_usability(self, text: str) -> tuple[bool, str]:
        """Whether ``text`` can serve as ground truth, and why not if it cannot.

        Two disqualifiers, both borrowed from the born-digital detector:

        - ANY eaten-leading-digit occurrence (GH-136). One is enough: a single
          "(/997)" is a wrong publication year, and an engine that reads the
          raster correctly would be scored wrong against it.
        - Pervasive hygiene corruption above ``MAX_ENCODING_CORRUPTION`` — the
          same threshold at which the pipeline stops trusting the layer and
          routes the page to OCR. If the product will not trust it, the
          benchmark must not score against it.

        The mild hygiene band (flag-level) is deliberately NOT excluded: lost
        spaces make the text ugly, not wrong, and dropping those pages would
        discard a large, legitimate slice of the corpus.
        """
        digit_hits = count_digit_corruption(text)
        if digit_hits >= self._detector.MAX_DIGIT_CORRUPTION_HITS:
            return False, (
                f"text layer has {digit_hits} eaten-digit occurrence(s) "
                "(e.g. '(/997)' for '(1997)'); numbers cannot be trusted as truth"
            )
        ratio = self._detector._encoding_corruption_ratio(text)
        if ratio > self._detector.MAX_ENCODING_CORRUPTION:
            return False, (
                f"text layer encoding corruption {ratio:.1%} exceeds "
                f"{self._detector.MAX_ENCODING_CORRUPTION:.0%}; layer is not trusted by the "
                "pipeline either"
            )
        return True, ""

    def extract(self, pdf_path: Path) -> list[PageGroundTruth]:
        """Extract native text per page using PyMuPDF.

        Every page is returned, including unusable ones (marked ``usable=False``)
        so page numbering stays aligned with engine output. Callers that need
        scoreable truth should use :func:`usable_truths`.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of PageGroundTruth, one per page.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pages: list[PageGroundTruth] = []
        with open_pdf(pdf_path) as doc:
            for page_idx in range(len(doc)):
                text = doc[page_idx].get_text("text").strip()
                usable, reason = self._assess_usability(text)
                pages.append(
                    PageGroundTruth(
                        page_num=page_idx + 1,
                        text=text,
                        word_count=len(text.split()) if text else 0,
                        char_count=len(text),
                        usable=usable,
                        exclusion_reason=reason,
                    )
                )

        return pages

    def save(self, truths: list[PageGroundTruth], output_dir: Path) -> None:
        """Save ground truth as per-page text files and a combined full text.

        Creates:
            output_dir/page_1.txt
            output_dir/page_2.txt
            ...
            output_dir/full.txt
            output_dir/exclusions.json

        Only USABLE pages are written as truth (GH-39) — an excluded page has no
        page file and contributes nothing to ``full.txt``, so a scorer reading
        this directory cannot accidentally score against a corrupted layer.
        ``exclusions.json`` records what was dropped and why: the exclusion count
        is a reported figure, never a silent narrowing of the corpus.

        Args:
            truths: List of PageGroundTruth to save.
            output_dir: Directory to write files into.
        """
        import json

        output_dir.mkdir(parents=True, exist_ok=True)

        full_parts: list[str] = []
        for page_gt in truths:
            if not page_gt.usable:
                continue
            page_file = output_dir / f"page_{page_gt.page_num}.txt"
            page_file.write_text(page_gt.text, encoding="utf-8")
            if page_gt.text:
                full_parts.append(page_gt.text)

        full_file = output_dir / "full.txt"
        full_file.write_text("\n\n".join(full_parts), encoding="utf-8")

        excluded = [t for t in truths if not t.usable]
        (output_dir / "exclusions.json").write_text(
            json.dumps(
                {
                    "pages_total": len(truths),
                    "pages_usable": len(truths) - len(excluded),
                    "pages_excluded": len(excluded),
                    "excluded": [
                        {"page_num": t.page_num, "reason": t.exclusion_reason} for t in excluded
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def extract_and_save(self, pdf_path: Path, output_dir: Path) -> list[PageGroundTruth]:
        """Extract ground truth and save to disk in one step.

        Args:
            pdf_path: Path to the PDF file.
            output_dir: Directory to write ground truth files.

        Returns:
            List of PageGroundTruth.
        """
        truths = self.extract(pdf_path)
        self.save(truths, output_dir)
        return truths


def usable_truths(truths: list[PageGroundTruth]) -> list[PageGroundTruth]:
    """The subset of ``truths`` that can serve as ground truth (GH-39)."""
    return [t for t in truths if t.usable]


def exclusion_report(
    truths: list[PageGroundTruth],
    page_types: dict[int, str] | None = None,
) -> dict:
    """Summarise what was excluded from ground truth, and where it came from.

    ``page_types`` maps page number -> the ``page_types.classify_page_type``
    label. Pass it whenever it is available: contaminated pages are NOT randomly
    distributed. Broken font maps cluster in older and denser documents, which
    are plausibly the same pages where escalation earns its keep — so excluding
    them makes the corpus easier for EVERY condition and can mask the very
    difference the benchmark exists to measure.

    A single scalar count hides that. The per-type breakdown is what lets a
    reader see whether the table stratum was quietly gutted, so any write-up
    that reports the total without the breakdown is under-reporting.
    """
    excluded = [t for t in truths if not t.usable]
    by_type: dict[str, int] = {}
    if page_types:
        for t in excluded:
            label = page_types.get(t.page_num, "untyped")
            by_type[label] = by_type.get(label, 0) + 1
    return {
        "pages_total": len(truths),
        "pages_usable": len(truths) - len(excluded),
        "pages_excluded": len(excluded),
        "excluded_fraction": (len(excluded) / len(truths)) if truths else 0.0,
        "excluded_by_page_type": by_type,
        "reasons": [{"page_num": t.page_num, "reason": t.exclusion_reason} for t in excluded],
    }
