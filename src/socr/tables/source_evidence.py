"""Fail-closed source-evidence gate for VLM-emitted markdown tables (GH-90).

On scanned pages (no PyMuPDF native words), a fluent hallucinated table passes
heuristic checks and self-reports success.  This module verifies table cell
tokens against LOCAL, NON-GENERATIVE page evidence only:

  - PyMuPDF ``get_text()`` / ``get_text("words")`` when present
  - ``locate_tables`` region crops rendered to raster
  - Optional classical OCR (pytesseract) over page / crop when installed

Born-digital pages with native words defer to ``NativeTableVerifierJudge``;
this gate is not invoked for them.

If no content evidence is available, or emitted tokens are unsupported, the
table is UNVERIFIABLE and must NOT be accepted.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from socr.tables.native_verifier import _numeric_multiset_from_tokens, is_numeric_token
from socr.tables.reconcile import find_table_blocks

logger = logging.getLogger(__name__)

# Minimum length for alphabetic content tokens extracted from table cells.
# Basis: tokens shorter than 3 characters are too ambiguous for reliable
# hallucination detection (articles, ordinals, single-letter stubs).
_MIN_CONTENT_TOKEN_LEN: int = 3

# Classical OCR raster resolution for crop evidence (PDF points -> pixels).
# Matches ``image_locate.DETECT_DPI``: enough to resolve printed labels without
# large images; the crop pass re-renders at this DPI before optional tesseract.
_EVIDENCE_OCR_DPI: int = 150

# Alphabetic runs in evidence text: at least _MIN_CONTENT_TOKEN_LEN chars.
_CONTENT_TOKEN_RE = re.compile(rf"[A-Za-z][A-Za-z0-9\-]{{{_MIN_CONTENT_TOKEN_LEN - 1},}}")

OcrImageFn = Callable[[object], str]


@dataclass(frozen=True)
class TableTokens:
    """Tokens collected from VLM-emitted markdown table cells."""

    numeric: Counter = field(default_factory=Counter)
    content: frozenset[str] = field(default_factory=frozenset)
    has_numeric: bool = False
    is_alpha_only: bool = False


@dataclass(frozen=True)
class SourceEvidenceBundle:
    """Local non-generative token evidence for a scanned page."""

    numeric: Counter = field(default_factory=Counter)
    content: frozenset[str] = field(default_factory=frozenset)
    has_content_evidence: bool = False
    table_regions_detected: bool = False


@dataclass(frozen=True)
class SourceEvidenceResult:
    """Verdict from the source-evidence gate."""

    verifiable: bool
    passed: bool
    reason: str
    deferred: bool = False


def page_has_native_words(page) -> bool:
    """True when PyMuPDF exposes any non-empty word on the page."""
    try:
        words = page.get_text("words")
    except Exception:
        return False
    return any(len(w) > 4 and str(w[4]).strip() for w in words)


def collect_table_tokens(markdown: str) -> TableTokens | None:
    """Extract numeric multiset and content-label tokens from markdown tables."""
    blocks = find_table_blocks(markdown)
    if not blocks:
        return None

    raw_numeric: list[str] = []
    raw_content: set[str] = set()
    for block in blocks:
        for row_idx, row in enumerate(block.grid):
            for cell in row:
                cell = cell.strip()
                if not cell or cell in ("---", "—"):
                    continue
                if is_numeric_token(cell):
                    raw_numeric.append(cell)
                # Content-label tokens come from data rows only (row 0 is the
                # header).  Generic header words ("Category", "Description") are
                # not reliable hallucination signals and rarely appear in OCR
                # evidence on scanned pages.
                if row_idx == 0:
                    continue
                for m in _CONTENT_TOKEN_RE.finditer(cell):
                    raw_content.add(m.group(0).lower())

    numeric = _numeric_multiset_from_tokens(raw_numeric)
    has_numeric = bool(numeric)
    is_alpha_only = not has_numeric and bool(raw_content)
    return TableTokens(
        numeric=numeric,
        content=frozenset(raw_content),
        has_numeric=has_numeric,
        is_alpha_only=is_alpha_only,
    )


def _tokens_from_plain_text(text: str) -> tuple[Counter, set[str]]:
    """Parse numeric multiset and content tokens from plain text."""
    if not text.strip():
        return Counter(), set()

    raw_numeric: list[str] = []
    content: set[str] = set()
    for word in re.findall(r"\S+", text):
        word = word.strip(".,;:!?()[]\"'")
        if not word:
            continue
        if is_numeric_token(word):
            raw_numeric.append(word)
        for m in _CONTENT_TOKEN_RE.finditer(word):
            content.add(m.group(0).lower())
    return _numeric_multiset_from_tokens(raw_numeric), content


def _merge_evidence(
    numeric: Counter,
    content: set[str],
    new_numeric: Counter,
    new_content: set[str],
) -> tuple[Counter, set[str]]:
    merged_numeric = numeric + new_numeric
    merged_content = content | new_content
    return merged_numeric, merged_content


def _render_crop_pixmap(page, bbox: tuple[float, float, float, float], dpi: int):
    import fitz

    from socr.core.born_digital import upright_rotation_for

    rect = fitz.Rect(bbox)
    # GH-304b: derive clip-local rotation; keep bbox and clip in page space, rotate only raster pixels.
    rotation = upright_rotation_for(page, clip=rect)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    if rotation != 0:
        mat.prerotate(rotation)
    return page.get_pixmap(matrix=mat, clip=rect)


def classical_ocr_pixmap(pix) -> str:
    """Optional classical OCR over a rendered pixmap; returns "" when unavailable."""
    try:
        import pytesseract
    except ImportError:
        return ""
    try:
        from PIL import Image

        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return pytesseract.image_to_string(img) or ""
    except Exception as exc:
        logger.debug("classical OCR failed: %s", exc)
        return ""


def build_scanned_evidence(
    page,
    *,
    ocr_image_fn: OcrImageFn | None = None,
    include_text_layer: bool = True,
) -> SourceEvidenceBundle:
    """Build local non-generative evidence for a scanned page.

    ``include_text_layer=False`` drops ``page.get_text()`` from the evidence.

    GH-163 review (cubic P1): the text layer is the FIRST source merged in, and
    the full-page raster branch below only fires when nothing else produced any
    evidence at all. So on a page whose text layer is not trusted, that layer
    was still the primary corroboration -- and a model table that agreed with a
    corrupt OCR layer verified against it. Excluding it leaves only readings
    taken from the pixels (per-table crops, then the full page), which is the
    independent evidence the scanned lane is supposed to provide.
    """
    ocr_fn = ocr_image_fn or classical_ocr_pixmap
    numeric: Counter = Counter()
    content: set[str] = set()

    if include_text_layer:
        try:
            plain = page.get_text() or ""
        except Exception:
            plain = ""
        n_plain, c_plain = _tokens_from_plain_text(plain)
        numeric, content = _merge_evidence(numeric, content, n_plain, c_plain)

    table_regions_detected = False
    try:
        from socr.tables.locate import locate_tables

        boxes = locate_tables(page)
        table_regions_detected = bool(boxes)
        for box in boxes:
            try:
                pix = _render_crop_pixmap(page, box.bbox, _EVIDENCE_OCR_DPI)
                crop_text = ocr_fn(pix)
            except Exception as exc:
                logger.debug("crop evidence render failed: %s", exc)
                continue
            n_crop, c_crop = _tokens_from_plain_text(crop_text)
            numeric, content = _merge_evidence(numeric, content, n_crop, c_crop)
    except Exception as exc:
        logger.debug("locate_tables evidence failed: %s", exc)

    if not numeric and not content:
        try:
            import fitz

            from socr.core.born_digital import upright_rotation_for

            # GH-304b: derive page-level rotation; rotate raster pixels only.
            rotation = upright_rotation_for(page)
            mat = fitz.Matrix(_EVIDENCE_OCR_DPI / 72, _EVIDENCE_OCR_DPI / 72)
            if rotation != 0:
                mat.prerotate(rotation)
            pix = page.get_pixmap(matrix=mat)
            page_text = ocr_fn(pix)
            n_page, c_page = _tokens_from_plain_text(page_text)
            numeric, content = _merge_evidence(numeric, content, n_page, c_page)
        except Exception as exc:
            logger.debug("full-page evidence OCR failed: %s", exc)

    has_content_evidence = bool(numeric) or bool(content)
    return SourceEvidenceBundle(
        numeric=numeric,
        content=frozenset(content),
        has_content_evidence=has_content_evidence,
        table_regions_detected=table_regions_detected,
    )


def _multiset_supported(output: Counter, evidence: Counter) -> bool:
    """True when every output token count is covered by evidence (N2-normalized)."""
    if not output:
        return True
    for tok, count in output.items():
        if evidence.get(tok, 0) < count:
            return False
    return True


def _content_supported(output: frozenset[str], evidence: frozenset[str]) -> bool:
    """True when every emitted content-label token appears in evidence."""
    if not output:
        return True
    return output.issubset(evidence)


def verify_table_tokens(
    bundle: SourceEvidenceBundle,
    tokens: TableTokens,
) -> SourceEvidenceResult:
    """Fail-closed check: emitted table tokens must be supported by evidence."""
    if not bundle.has_content_evidence:
        return SourceEvidenceResult(
            verifiable=False,
            passed=False,
            reason="no local content evidence available for scanned table",
        )

    if tokens.has_numeric and not _multiset_supported(tokens.numeric, bundle.numeric):
        missing = [
            tok for tok, count in tokens.numeric.items() if bundle.numeric.get(tok, 0) < count
        ]
        return SourceEvidenceResult(
            verifiable=True,
            passed=False,
            reason=f"numeric tokens unsupported by page evidence: {missing[:5]}",
        )

    if tokens.is_alpha_only and not _content_supported(tokens.content, bundle.content):
        missing = sorted(tokens.content - bundle.content)[:5]
        return SourceEvidenceResult(
            verifiable=True,
            passed=False,
            reason=f"content labels unsupported by page evidence: {missing}",
        )

    if tokens.has_numeric and tokens.content:
        if not _content_supported(tokens.content, bundle.content):
            missing = sorted(tokens.content - bundle.content)[:5]
            return SourceEvidenceResult(
                verifiable=True,
                passed=False,
                reason=f"content labels unsupported by page evidence: {missing}",
            )

    if not tokens.has_numeric and not tokens.is_alpha_only:
        return SourceEvidenceResult(
            verifiable=False,
            passed=False,
            reason="table cells contain no verifiable tokens",
        )

    return SourceEvidenceResult(
        verifiable=True,
        passed=True,
        reason="source evidence supports emitted table tokens",
    )


def verify_scanned_table(
    page,
    output_text: str,
    *,
    ocr_image_fn: OcrImageFn | None = None,
    native_trusted: bool | None = None,
) -> SourceEvidenceResult:
    """Full scanned-page pipeline: defer native, else verify or fail closed.

    ``native_trusted`` is the caller's born-digital classification for this page.

    GH-163: deferral used to hinge on ``page_has_native_words`` alone, and a
    scanned page with a baked-in or corrupt OCR layer has words. Such a page
    handed itself to the native verifier -- which checks the model's table
    against that same untrusted layer -- so the fail-closed raster/classical
    evidence check was skipped for exactly the pages that need it, and a
    hallucinated table could be corroborated by a hallucinated text layer.

    ``False`` means the caller classified the page as NOT trusted-native, and
    the evidence check runs however many words the layer contains -- and that
    layer is excluded from the evidence, so the table cannot be corroborated by
    the reading under suspicion. ``None`` means the caller cannot tell, and the
    pre-GH-163 word-presence behaviour is kept -- an unknown classification must
    not silently start failing pages closed.

    ``True`` still requires words before deferring. Reviewers ask why (cubic P2
    on #512): should not an explicit "trusted" select deferral on its own?  No
    -- the native verifier has nothing to check a table against on a page with
    no extractable words, so deferring there would skip verification entirely.
    A classification of trusted and a page with no words is a contradiction,
    and resolving it toward "run no check" is the fail-open direction this lane
    exists to prevent.
    """
    if native_trusted is not False and page_has_native_words(page):
        return SourceEvidenceResult(
            verifiable=True,
            passed=True,
            reason=(
                "native words present; defer to native verifier"
                if native_trusted is None
                else "trusted native page; defer to native verifier"
            ),
            deferred=True,
        )

    tokens = collect_table_tokens(output_text)
    if tokens is None:
        return SourceEvidenceResult(
            verifiable=True,
            passed=True,
            reason="no markdown table blocks",
            deferred=True,
        )

    # An explicitly UNTRUSTED layer must not corroborate the model's table
    # (cubic P1 on #512). Only pixel-derived readings count for such a page.
    bundle = build_scanned_evidence(
        page,
        ocr_image_fn=ocr_image_fn,
        include_text_layer=native_trusted is not False,
    )
    return verify_table_tokens(bundle, tokens)
