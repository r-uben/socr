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

import html
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


def _content_token(raw: str) -> str | None:
    """Fold a raw ``_CONTENT_TOKEN_RE`` match to a comparable label token.

    #659: the regex above deliberately allows internal dashes so a hyphenated
    word ("socio-economic") stays one token, but that same class lets a
    LEADING/TRAILING dash ride along too -- a VLM's ``Settlements--`` rule
    decoration was tokenised as ``settlements--`` and then never matched the
    bare ``settlements`` any OCR witness reads off the page. Presentation, not
    content, so it is stripped here rather than loosened out of the regex
    (which still must not swallow a genuinely hyphenated word's edges other
    than the decoration itself).
    """
    token = raw.strip("-").lower()
    return token if len(token) >= _MIN_CONTENT_TOKEN_LEN else None


#: Every Unicode dash/hyphen glyph a trailing decoration strip must fold.
#: ``&mdash;``/``&ndash;`` decode (via ``html.unescape``) to U+2014/U+2013,
#: not ASCII ``-``, so a strip that only recognised ASCII leaves the decoded
#: glyph trailing and still unmatched by ``is_numeric_token``'s anchored
#: regex -- half-closing the exact decoration case this exists for. Spelled
#: as explicit ``\N{...}`` code-point names, not bare literal glyphs or a
#: hand-typed string, so each member is legible and auditable at the
#: definition site.
_TRAILING_DASH_CHARS: str = (
    "\N{HYPHEN-MINUS}"  # U+002D, plain ASCII "-"
    "\N{HYPHEN}"  # U+2010
    "\N{NON-BREAKING HYPHEN}"  # U+2011
    "\N{FIGURE DASH}"  # U+2012
    "\N{EN DASH}"  # U+2013 -- what "&ndash;" decodes to
    "\N{EM DASH}"  # U+2014 -- what "&mdash;" decodes to
    "\N{HORIZONTAL BAR}"  # U+2015
    "\N{MINUS SIGN}"  # U+2212, the true math/typeset minus, distinct from hyphen
)


def _normalize_cell(raw: str) -> str:
    """Decode HTML entities and strip trailing dash-rule decoration.

    #679: ``collect_table_tokens`` ran ``is_numeric_token`` on the RAW cell
    and only decoded entities afterward, in the label branch -- so
    ``&nbsp;62.5`` never entered the numeric multiset even though the same
    row's label already got the #659 decode. A VLM's trailing rule
    decoration (``62.5--``) had the identical gap: ``is_numeric_token``
    anchors on ``_NUM_TOKEN_RE``, which does not accept a trailing ``-``, so
    the decorated value fails the check on either side of a decode.

    Called from BOTH ``collect_table_tokens`` (candidate) and
    ``_tokens_from_plain_text`` (evidence) so the two stay symmetric. That
    symmetry matters beyond tidiness: a scanned page's own printed
    decoration can end up verbatim in the OCR/text-layer evidence too, and
    normalizing only the candidate side would make a candidate that
    genuinely AGREES with the evidence (both carry ``62.5--``) look
    unsupported -- the candidate's dash-free ``62.5`` would no longer find a
    match in an evidence multiset still keyed on the raw, un-normalized
    ``62.5--`` token. Measured 2026-09-16 (docs/log/2026-09-16_679.md):
    applying the strip candidate-only turns a page the two sides already
    agree on into an active reject.

    Only a TRAILING dash run is stripped. A LEADING dash (ASCII or the
    Unicode minus/en/em-dash variants above) is a numeric sign (``-5.2``)
    and must never be removed here -- only presentation (``62.5--``), never
    value, is this function's job.
    """
    return html.unescape(raw).rstrip(_TRAILING_DASH_CHARS)


OcrImageFn = Callable[[object], str]

#: #658: the cause carried by a scanned-table verdict whose evidence bundle is
#: empty because NOTHING EVER READ THE PIXELS -- there is no witness to agree
#: or disagree with the model. Deliberately distinct from an empty bundle a
#: WORKING witness produced by reading the page and finding nothing: an absent
#: or broken witness is not evidence of fabrication, and the two endings need
#: different operator actions. Kept as a plain string so consumers outside this
#: module can gate on it without importing the dataclasses.
#:
#: One cause covers the whole family (missing package, missing binary, OCR
#: crash, render failure, never attempted) because every consumer's decision is
#: the same -- do not call this a hallucination. WHICH of the five it was rides
#: alongside in ``SourceEvidenceBundle.witness_state``, which is what an
#: operator needs, and is named in the reason string.
CAUSE_NO_WITNESS_BACKEND: str = "no_witness_backend"

#: The classical-OCR witness ran and returned text. Not a failure state.
WITNESS_READING: str = "reading"
#: The witness ran and returned nothing. This IS a reading of the page -- the
#: pixels were looked at -- so it does NOT excuse an unsupported table.
WITNESS_EMPTY_READING: str = "empty_reading"
#: ``import pytesseract`` failed. The package is not declared in
#: ``pyproject.toml`` at all, so a default install lands here.
WITNESS_PACKAGE_MISSING: str = "package_missing"
#: ``pytesseract`` imported but the ``tesseract`` executable is absent.
WITNESS_BINARY_MISSING: str = "binary_missing"
#: The witness was called and RAISED (a broken install, an image the binary
#: rejected, a caller-supplied reader that threw).
WITNESS_EXEC_ERROR: str = "exec_error"
#: The page or crop could not be rasterised, so the witness never saw an image.
WITNESS_RENDER_ERROR: str = "render_error"
#: No witness call was made at all on this page.
WITNESS_NOT_ATTEMPTED: str = "not_attempted"

#: The states in which no reading of the pixels was ever produced. An empty
#: bundle carrying one of these is an ABSENCE of evidence, never evidence
#: against the model. ``WITNESS_EMPTY_READING`` is deliberately excluded: a
#: witness that looked and saw nothing did its job.
NO_READING_STATES: frozenset[str] = frozenset(
    {
        WITNESS_PACKAGE_MISSING,
        WITNESS_BINARY_MISSING,
        WITNESS_EXEC_ERROR,
        WITNESS_RENDER_ERROR,
        WITNESS_NOT_ATTEMPTED,
    }
)

#: Operator-facing sentence per state. Each names what to fix, because these
#: five need five different actions and a single "no evidence" told the
#: operator none of them.
WITNESS_STATE_MESSAGES: dict[str, str] = {
    WITNESS_PACKAGE_MISSING: (
        "the pytesseract package is not installed; install it into socr's environment "
        "('uv pip install pytesseract') and install the tesseract executable it drives"
    ),
    WITNESS_BINARY_MISSING: (
        "the tesseract executable is not installed or not on PATH ('brew install tesseract' "
        "on macOS, 'apt install tesseract-ocr' on Debian/Ubuntu)"
    ),
    # Deliberately NOT an install instruction. The reader is present and
    # reachable; telling the operator to install it again is advice that cannot
    # work, and it hides the actual failure.
    WITNESS_EXEC_ERROR: (
        "the classical OCR reader is installed but failed while reading this page; "
        "installing it again will not help -- see the recorded error"
    ),
    WITNESS_RENDER_ERROR: (
        "the page could not be rasterised, so no image ever reached the OCR reader; "
        "the OCR install is not the problem here"
    ),
    WITNESS_NOT_ATTEMPTED: "no classical OCR read was attempted on this page",
}

#: Precedence when several failure states occur on one page: the most
#: actionable wins. A missing install outranks a crash, which outranks a render
#: failure, which outranks never having tried.
_WITNESS_STATE_RANK: dict[str, int] = {
    WITNESS_PACKAGE_MISSING: 0,
    WITNESS_BINARY_MISSING: 1,
    WITNESS_EXEC_ERROR: 2,
    WITNESS_RENDER_ERROR: 3,
    WITNESS_NOT_ATTEMPTED: 4,
}

#: #658a: witness states already reported LOUDLY (``logger.warning``, not
#: ``debug``) this process, keyed by ``witness_state`` so
#: ``WITNESS_PACKAGE_MISSING`` and ``WITNESS_BINARY_MISSING`` each get their
#: own first warning -- they name different fixes, and collapsing them under
#: one key would silence whichever one occurs second. Module-level and
#: process-lifetime by design: a 200-page scan with no tesseract install
#: would otherwise log the same "go install this" sentence 200 times, which
#: is its own defect (buries the one thing an operator needs to see).
_WARNED_WITNESS_STATES: set[str] = set()


def _warn_witness_state_once(state: str) -> None:
    """Emit ``WITNESS_STATE_MESSAGES[state]`` at WARNING level, once per state per run."""
    if state in _WARNED_WITNESS_STATES:
        return
    _WARNED_WITNESS_STATES.add(state)
    message = WITNESS_STATE_MESSAGES.get(state, state)
    logger.warning("scanned-table evidence unavailable: %s", message)


#: Audit event kind for the ending above. A kind of its own rather than more
#: prose inside ``source_evidence_table_reject`` so a consumer counting
#: unwitnessed pages does not have to parse a sentence, and so an operator can
#: grep one token to learn the run had no OCR witness at all.
NO_WITNESS_BACKEND_KIND: str = "source_evidence_no_witness_backend"

#: #659: audit event kind for a table that SHIPPED (numerics fully
#: corroborated) but carries a content-label token the page evidence did not
#: confirm. A constant, not a literal, so ``tables_trust.py`` and
#: ``orchestrator.py`` cannot spell it differently from the emitter (the same
#: reason ``NO_WITNESS_BACKEND_KIND`` is one).
LABEL_UNVERIFIED_KIND: str = "source_evidence_table_label_unverified"


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
    #: #658: what the classical-OCR witness actually did on this page -- one of
    #: the ``WITNESS_*`` states. Derived from the witness that RAN, never from
    #: the ambient environment: a caller that injects ``ocr_image_fn`` is the
    #: witness, and its behaviour (a reading, silence, or an exception) is what
    #: is recorded, regardless of whether this host has tesseract.
    witness_state: str = WITNESS_NOT_ATTEMPTED
    #: Free text from the failing witness (an exception type, a probe message),
    #: for the audit trail. Never parsed.
    witness_detail: str = ""

    @property
    def no_reading(self) -> bool:
        """True when nothing ever read this page's pixels.

        The distinction the whole ticket turns on: an empty bundle with
        ``no_reading`` is an absence of evidence; an empty bundle without it is
        a witness that looked and found nothing.
        """
        return self.witness_state in NO_READING_STATES


@dataclass(frozen=True)
class SourceEvidenceResult:
    """Verdict from the source-evidence gate."""

    verifiable: bool
    passed: bool
    reason: str
    deferred: bool = False
    #: #658: machine-readable cause for a fail-closed verdict; "" for every
    #: ending that predates the distinction. Currently only
    #: ``CAUSE_NO_WITNESS_BACKEND`` is emitted.
    cause: str = ""
    #: #658: WHICH no-reading state produced the cause above -- one of the
    #: ``WITNESS_*`` values. Carried separately from ``cause`` because every
    #: consumer's DECISION is the same (do not call this a hallucination) while
    #: the operator's ACTION is not: a missing install is fixed by installing,
    #: a crashed reader and an unrenderable page are not.
    witness_state: str = ""
    #: #659: set on an otherwise-PASSED verdict when every numeric token is
    #: supported but at least one content-label token is not. A label the
    #: page evidence does not confirm is weaker signal than an unsupported
    #: NUMBER (a table's values are the citable content; a stub word going
    #: unread by a noisy OCR witness is common and not on its own evidence of
    #: fabrication), so this does not reject -- but it must not go unsaid
    #: either, so the caller surfaces it as an audit note rather than
    #: silently shipping the page as fully corroborated. "" when every
    #: emitted content label was found, or when there was no numeric token to
    #: anchor the trade-off (see ``is_alpha_only`` below, which still rejects).
    content_unverified: str = ""


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
                # #679: decode + dash-strip BEFORE the numeric check, not
                # only in the label branch below -- see ``_normalize_cell``.
                normalized = _normalize_cell(cell)
                if is_numeric_token(normalized):
                    raw_numeric.append(normalized)
                # Content-label tokens come from data rows only (row 0 is the
                # header).  Generic header words ("Category", "Description") are
                # not reliable hallucination signals and rarely appear in OCR
                # evidence on scanned pages.
                if row_idx == 0:
                    continue
                for m in _CONTENT_TOKEN_RE.finditer(normalized):
                    token = _content_token(m.group(0))
                    if token is not None:
                        raw_content.add(token)

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
    decoded_text = html.unescape(text)
    for word in re.findall(r"\S+", decoded_text):
        word = word.strip(".,;:!?()[]\"'")
        if not word:
            continue
        # #679: same normalizer as the candidate side (``collect_table_tokens``)
        # -- a raw page's own printed decoration must get the identical
        # treatment or a candidate that genuinely agrees with it looks
        # unsupported. See ``_normalize_cell``.
        normalized = _normalize_cell(word)
        if is_numeric_token(normalized):
            raw_numeric.append(normalized)
        for m in _CONTENT_TOKEN_RE.finditer(normalized):
            token = _content_token(m.group(0))
            if token is not None:
                content.add(token)
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


def classical_ocr_with_state(pix) -> tuple[str, str, str]:
    """Classical OCR over a pixmap, reporting WHY it produced what it produced.

    Returns ``(text, witness_state, detail)``.

    #658. ``classical_ocr_pixmap`` collapses five outcomes into ``""``: the
    package is absent, the binary is absent, the reader raised, the reader ran
    and saw nothing, or it ran and saw text. The gate then acted on the last
    interpretation for all of them, so a host with no tesseract recorded
    perfectly good candidates as fabrications. Each outcome needs a different
    operator action, so each gets its own state.

    The absent-executable case is taken from ``TesseractNotFoundError``, which
    is the exception pytesseract raises for exactly that, rather than from a
    version probe. A probe that raises says only "this call failed": a present
    but BROKEN executable raises there too, and reading that as "not installed"
    tells the operator to install something they already have.
    """
    try:
        import pytesseract
    except ImportError:
        return "", WITNESS_PACKAGE_MISSING, WITNESS_STATE_MESSAGES[WITNESS_PACKAGE_MISSING]
    try:
        from PIL import Image

        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        text = pytesseract.image_to_string(img) or ""
    except Exception as exc:
        logger.debug("classical OCR failed: %s", exc)
        not_found = getattr(pytesseract, "TesseractNotFoundError", ())
        if not_found and isinstance(exc, not_found):
            return "", WITNESS_BINARY_MISSING, WITNESS_STATE_MESSAGES[WITNESS_BINARY_MISSING]
        # Installed and reachable, but it failed on this page. Not an install
        # problem, and it must never be reported as one.
        return "", WITNESS_EXEC_ERROR, f"{type(exc).__name__}: {exc}"
    if not text.strip():
        return "", WITNESS_EMPTY_READING, ""
    return text, WITNESS_READING, ""


def classical_ocr_pixmap(pix) -> str:
    """Optional classical OCR over a rendered pixmap; returns "" when unavailable.

    The public, text-only seam kept for callers that pass an ``ocr_image_fn``.
    ``classical_ocr_with_state`` is the one this module uses internally.
    """
    return classical_ocr_with_state(pix)[0]


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
    # #658: read through a state-reporting wrapper so the bundle can say WHY it
    # is empty. An injected ``ocr_image_fn`` is the caller's witness and is
    # judged on its own behaviour -- a reading, silence, or an exception --
    # never on whether this host happens to have tesseract installed.
    if ocr_image_fn is None:
        read_witness = classical_ocr_with_state
    else:

        def read_witness(pix) -> tuple[str, str, str]:
            try:
                text = ocr_image_fn(pix) or ""
            except Exception as exc:  # noqa: BLE001 - any reader failure is one state
                logger.debug("injected OCR reader failed: %s", exc)
                return "", WITNESS_EXEC_ERROR, f"{type(exc).__name__}: {exc}"
            if not text.strip():
                return "", WITNESS_EMPTY_READING, ""
            return text, WITNESS_READING, ""

    witness_states: list[str] = []
    witness_details: dict[str, str] = {}

    def record(state: str, detail: str) -> None:
        witness_states.append(state)
        if detail and state not in witness_details:
            witness_details[state] = detail

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
            except Exception as exc:
                # The image never existed, so the reader never saw one. Distinct
                # from a reader that ran and failed (#658 reviewer item 2).
                logger.debug("crop evidence render failed: %s", exc)
                record(WITNESS_RENDER_ERROR, f"{type(exc).__name__}: {exc}")
                continue
            crop_text, crop_state, crop_detail = read_witness(pix)
            record(crop_state, crop_detail)
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
            page_text, page_state, page_detail = read_witness(pix)
            record(page_state, page_detail)
            n_page, c_page = _tokens_from_plain_text(page_text)
            numeric, content = _merge_evidence(numeric, content, n_page, c_page)
        except Exception as exc:
            logger.debug("full-page evidence OCR failed: %s", exc)
            record(WITNESS_RENDER_ERROR, f"{type(exc).__name__}: {exc}")

    has_content_evidence = bool(numeric) or bool(content)
    witness_state = _resolve_witness_state(witness_states)
    return SourceEvidenceBundle(
        numeric=numeric,
        content=frozenset(content),
        has_content_evidence=has_content_evidence,
        table_regions_detected=table_regions_detected,
        witness_state=witness_state,
        witness_detail=witness_details.get(witness_state, ""),
    )


def _resolve_witness_state(states: list[str]) -> str:
    """Collapse this page's per-read outcomes into ONE state (#658).

    A reading anywhere means the witness worked, whatever else failed
    elsewhere on the page -- a crop that would not rasterise must not make a
    page that WAS read look unwitnessed. A blank reading likewise beats every
    failure state: the pixels were looked at. Only when neither happened does
    the most actionable failure win, by ``_WITNESS_STATE_RANK``.
    """
    if WITNESS_READING in states:
        return WITNESS_READING
    if WITNESS_EMPTY_READING in states:
        return WITNESS_EMPTY_READING
    failures = [s for s in states if s in _WITNESS_STATE_RANK]
    if not failures:
        return WITNESS_NOT_ATTEMPTED
    return min(failures, key=lambda s: _WITNESS_STATE_RANK[s])


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
        # #658: an empty bundle has two very different causes. Say which one.
        # The page fails closed either way -- an unwitnessed table is still
        # unverified -- but "we never looked" must not be reported as "the
        # evidence contradicts this", which is what a single reason did.
        if bundle.no_reading:
            why = WITNESS_STATE_MESSAGES.get(bundle.witness_state, bundle.witness_state)
            detail = f" [{bundle.witness_detail}]" if bundle.witness_detail else ""
            # #658a: loud and once, not once per page -- see
            # ``_warn_witness_state_once``.
            _warn_witness_state_once(bundle.witness_state)
            return SourceEvidenceResult(
                verifiable=False,
                passed=False,
                reason=f"no classical OCR witness read this page: {why}{detail}",
                cause=CAUSE_NO_WITNESS_BACKEND,
                witness_state=bundle.witness_state,
            )
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
        # No numeric token exists to anchor the #659 unverified-label
        # trade-off below -- a pure-label table's content IS the claim, so an
        # unsupported label here stays a reject, same as before #659.
        missing = sorted(tokens.content - bundle.content)[:5]
        return SourceEvidenceResult(
            verifiable=True,
            passed=False,
            reason=f"content labels unsupported by page evidence: {missing}",
        )

    content_unverified = ""
    if (
        tokens.has_numeric
        and tokens.content
        and not _content_supported(tokens.content, bundle.content)
    ):
        # #659: numerics are fully supported (the check above already passed)
        # and only a label is unconfirmed -- flag, do not reject.
        missing = sorted(tokens.content - bundle.content)[:5]
        content_unverified = f"content labels unverified by page evidence: {missing}"

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
        content_unverified=content_unverified,
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
    result = verify_table_tokens(bundle, tokens)

    # #658b: excluding the layer above is right for the "does this table
    # AGREE with the layer" check -- a hallucination must not be allowed to
    # corroborate itself against evidence nobody trusts. But when the pixel
    # evidence has NOTHING to say either (``has_content_evidence`` is False:
    # no crops, no page raster, no classical OCR reading), that same
    # exclusion also throws away a distrusted layer that may genuinely
    # REPRODUCE the model's rows. Measured (Fed swap-line pages,
    # docs/log/2026-09-07_D3-fed-table-lane-remeasure.md): 62/62, 67/67,
    # 66/66 of a page's numbers with zero extras, rejected outright.
    #
    # Row corroboration is a different, weaker check than merging the layer
    # into the evidence bundle: it asks only whether each candidate numeric
    # row appears, in order, as a contiguous run on ONE native printed line
    # (``row_corroboration.corroborate_rows``) -- never whether an individual
    # cell or column header is right. That tolerates the layer's own
    # corruption without ever trusting its content, so it can rescue a page
    # like this without promoting the layer to evidence.
    if native_trusted is False and result.passed is False and not bundle.has_content_evidence:
        rescued = _corroborate_via_distrusted_layer(page, output_text)
        if rescued is not None:
            return rescued

    return result


def _corroborate_via_distrusted_layer(page, output_text: str) -> SourceEvidenceResult | None:
    """#658b rescue: corroborate a candidate's rows against a DISTRUSTED text
    layer, for a page where pixel evidence produced nothing to judge by.

    Returns ``None`` when there is nothing to corroborate with (no native
    words at all -- criterion 3: a page with no text layer must still fail
    closed exactly as before) or when corroboration abstains or fails
    (criterion 4: a candidate that genuinely disagrees with the layer must
    still be rejected). The caller keeps its own reject unchanged in either
    case. Only ever returns a PASSED, flagged result -- this function never
    strengthens a rejection.
    """
    try:
        words = page.get_text("words") or []
    except Exception:
        words = []
    if not words:
        return None

    from socr.tables.row_corroboration import corroborate_rows

    # No detected table bbox is available at this layer (this gate runs
    # before/independent of ``locate_tables`` scoping used elsewhere); score
    # against the whole page, same abstention semantics as an empty region.
    rc = corroborate_rows(words, output_text, None)
    if rc.clears is not True:
        return None

    reason = (
        f"header_binding_unverified: distrusted text layer corroborates "
        f"{rc.bound}/{rc.total} candidate rows in order "
        f"({len(rc.extra_numbers)} extra number(s) beyond the layer); "
        "header binding was never independently verified"
    )
    return SourceEvidenceResult(
        verifiable=True,
        passed=True,
        reason=reason,
        # #658b criterion 1: this must read as a FLAG, never a clean SUCCESS.
        # ``content_unverified`` is the one channel this module already has
        # into the rest of the pipeline that is fully wired end to end
        # (``pipeline/agentic.py``'s ``SourceEvidenceTableJudge`` -> WARNING
        # status at finalization, ``table_label_unverified`` on the sidecar,
        # ``TABLE_DISTRUST_KINDS``, and the document-metadata / CLI note) --
        # reused here rather than inventing a second, unwired flag field.
        content_unverified=reason,
    )
