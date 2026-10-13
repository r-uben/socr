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

from socr.tables.image_locate import DETECT_DPI as _TEXT_LAYER_INK_DPI
from socr.tables.native_verifier import (
    _normalize_cell,
    _numeric_multiset_from_tokens,
    _unwrap_markdown_link,
    is_numeric_token,
)
from socr.tables.reconcile import _is_separator_row, _split_row, find_table_blocks

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


# #690: ``_normalize_cell`` moved to ``native_verifier`` (imported above) so
# ``binding.py`` can share the identical normalizer instead of growing a
# third copy. Its behaviour and docstring are unchanged; see
# ``native_verifier._normalize_cell`` for both.


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
    #: #1085: dash-led numbers the witness read but that support nothing. Counts
    #: as a reading (``has_content_evidence``), never as support.
    ambiguous_numeric: int = 0

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
    #: #1048: EVERY unsupported token of a verifiable-and-failed verdict, as
    #: bare strings (no cap: the reason text keeps its own 5-token excerpt), so the in-text "disputed" mark can name them. Empty otherwise.
    disputed_values: tuple[str, ...] = ()


def page_has_native_words(page) -> bool:
    """True when PyMuPDF exposes any non-empty word on the page."""
    try:
        words = page.get_text("words")
    except Exception:
        return False
    return any(len(w) > 4 and str(w[4]).strip() for w in words)


def _inline_dollar_positions(line: str) -> list[int]:
    """Offsets of single ``$`` on *line*; ``$$`` is skipped, not paired."""
    out: list[int] = []
    i = 0
    n = len(line)
    while i < n:
        if line.startswith("$$", i):
            i += 2
            continue
        if line[i] == "$" and (i == 0 or line[i - 1] != "\\"):
            out.append(i)
        i += 1
    return out


def _preceded_by_odd_backslashes(line: str, pos: int) -> bool:
    """True when *pos* is escaped: an odd run of ``\\`` immediately before it."""
    count = 0
    i = pos - 1
    while i >= 0 and line[i] == "\\":
        count += 1
        i -= 1
    return count % 2 == 1


def _dollar_interior_is_maths(interior: str) -> bool:
    """A ``$...$`` pair hides a pipe only when the interior is real maths.

    At least one letter, or a backslash command. Digits, spaces and punctuation
    only (currency such as ``5|10`` or ``5 | 10``) never hide a pipe.
    """
    return any(ch.isalpha() for ch in interior) or "\\" in interior


def _pipe_is_maths_local(line: str, pos: int) -> bool:
    """True when ``line[pos]`` is a ``|`` that is maths on this same line.

    Command forms (``\\|`` with odd backslash parity, ``\\left|``, ``\\right|``,
    ``\\vert``, ``\\mid``) or a ``$...$`` pair whose interior is maths (a
    letter or a backslash command). No multi-line spans, environments, or
    display tracking.
    """
    if pos < 0 or pos >= len(line) or line[pos] != "|":
        return False
    if pos >= 5 and line[pos - 5 : pos + 1] == "\\left|":
        return True
    if pos >= 6 and line[pos - 6 : pos + 1] == "\\right|":
        return True
    if pos >= 5 and line[pos - 5 : pos] == "\\vert":
        return True
    if pos >= 4 and line[pos - 4 : pos] == "\\mid":
        return True
    if _preceded_by_odd_backslashes(line, pos):
        return True
    dollars = _inline_dollar_positions(line)
    for a, b in zip(dollars[0::2], dollars[1::2]):
        if not (a < pos < b):
            continue
        return _dollar_interior_is_maths(line[a + 1 : b])
    return False


def _line_pipes_are_maths_local(line: str) -> bool:
    positions = [i for i, ch in enumerate(line) if ch == "|"]
    return bool(positions) and all(_pipe_is_maths_local(line, i) for i in positions)


def _block_has_separator(lines: list[str], start: int, end: int) -> bool:
    return any(_is_separator_row(_split_row(lines[idx])) for idx in range(start, end + 1))


def _is_phantom_maths_block(lines: list[str], start: int, end: int) -> bool:
    """A pipe run that is only maths, so the source-evidence check must skip it.

    Both: (a) no ``|---|`` separator, and (b) every ``|`` on every non-separator
    line is maths-local on that line. A separator keeps the block even when
    the data rows look like ``$P(A|B)$``.
    """
    if _block_has_separator(lines, start, end):
        return False
    data = [idx for idx in range(start, end + 1) if not _is_separator_row(_split_row(lines[idx]))]
    return bool(data) and all(_line_pipes_are_maths_local(lines[idx]) for idx in data)


def _evidence_table_blocks(markdown: str):
    """Table blocks the scanned-page source-evidence check is allowed to see.

    ``find_table_blocks`` runs on the original text (same as origin/main).
    Only a phantom maths block is dropped.
    """
    lines = markdown.splitlines()
    return [
        b for b in find_table_blocks(markdown) if not _is_phantom_maths_block(lines, b.start, b.end)
    ]


def evidence_readable_table_count(markdown: str) -> int:
    """How many tables ``collect_table_tokens`` / ``verify_scanned_table`` will read.

    ``find_table_blocks`` also counts the maths pipes #1098 excludes; this is
    the checker's count, and the one any wrapper must use.
    """
    return len(_evidence_table_blocks(markdown))


def page_has_page_covering_raster(page) -> bool:
    """True when embedded images cover at least ``RASTER_DOMINANCE_RATIO`` of the page.

    GH-1099 / GH-35-FU: the same signal ``BornDigitalDetector`` uses to treat a
    baked-in OCR text layer as untrusted (GH-163). Pages below the threshold are
    born-digital for evidence: the PDF's own characters are the witness.

    A coverage computation that raises is treated as image-dominant (fail closed),
    never as born-digital (#1099 r3).
    """
    from socr.core.born_digital import BornDigitalDetector

    try:
        coverage = BornDigitalDetector._raster_coverage(page)
    except Exception:
        logger.debug("raster coverage failed; treating page as scan for evidence", exc_info=True)
        return True
    return coverage >= BornDigitalDetector.RASTER_DOMINANCE_RATIO


def _text_dict_spans(page):
    """Yield text spans from ``get_text("dict")`` (text blocks only)."""
    try:
        page_dict = page.get_text("dict") or {}
    except Exception:
        return
    for block in page_dict.get("blocks", ()):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", ()):
            for span in line.get("spans", ()):
                if isinstance(span, dict):
                    yield span


def text_layer_span_is_visible(span: dict) -> bool:
    """True when a text-layer span is visibly rendered (#961 / #1099).

    Uses ``get_text("dict")`` span ``alpha`` (0 = invisible / render mode 3 on
    typical PDFs). ``get_texttrace()`` is avoided: repeated calls across many
    pages can trip a PyMuPDF refcount defect in ``find_tables`` later in-process.
    """
    return span.get("alpha", 255) != 0


def _invisible_text_layer_bboxes(page) -> list[tuple[float, float, float, float]]:
    """Bounding boxes of invisible text spans (#961); tests / diagnostics only."""
    boxes: list[tuple[float, float, float, float]] = []
    for span in _text_dict_spans(page):
        if text_layer_span_is_visible(span):
            continue
        bbox = span.get("bbox")
        if bbox and len(bbox) >= 4:
            boxes.append((bbox[0], bbox[1], bbox[2], bbox[3]))
    return boxes


def _text_layer_ink_matrix(page):
    """Render matrix for per-glyph ink clips (rotation-aware, GH-304b)."""
    import fitz

    from socr.core.born_digital import upright_rotation_for

    rotation = upright_rotation_for(page)
    mat = fitz.Matrix(_TEXT_LAYER_INK_DPI / 72, _TEXT_LAYER_INK_DPI / 72)
    if rotation != 0:
        mat.prerotate(rotation)
    return mat


def _pixmap_sample_has_ink(pix) -> bool:
    samples = pix.samples
    n = pix.n
    for i in range(0, len(samples), n):
        if samples[i] < 255:
            return True
        if n > 1 and samples[i + 1] < 255:
            return True
        if n > 2 and samples[i + 2] < 255:
            return True
    return False


def _glyph_bbox_has_ink(
    page,
    mat,
    bbox: tuple[float, float, float, float],
    *,
    page_pix=None,
) -> bool:
    """True when the glyph box contains a non-white pixel on a white-backed render."""
    import fitz

    rect = fitz.Rect(bbox)
    if rect.is_empty or rect.get_area() <= 0:
        return False
    if page_pix is not None:
        device = fitz.Rect(bbox) * mat
        x0 = max(0, int(device.x0))
        y0 = max(0, int(device.y0))
        x1 = min(page_pix.width, int(device.x1))
        y1 = min(page_pix.height, int(device.y1))
        if x1 <= x0 or y1 <= y0:
            return False
        samples = page_pix.samples
        width = page_pix.width
        n = page_pix.n
        for y in range(y0, y1):
            row = y * width * n
            for x in range(x0, x1):
                i = row + x * n
                if samples[i] < 255:
                    return True
                if n > 1 and samples[i + 1] < 255:
                    return True
                if n > 2 and samples[i + 2] < 255:
                    return True
        return False
    try:
        clip_pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    except Exception:
        logger.debug("glyph ink clip render failed", exc_info=True)
        return False
    return _pixmap_sample_has_ink(clip_pix)


def _plain_text_from_visible_chars(page, mat, page_pix=None) -> str:
    """Visible, ink-backed characters from ``rawdict``, grouped like ``get_text("words")``.

    Walk block → line → span → char in ``rawdict`` order.  Start a new word on a
    whitespace character or at each line boundary (same delimiters ``get_text("words")``
    uses for born-digital text: it does not split merely because the PDF put the
    next glyph in another span).  Invisible (``alpha==0``) or inkless characters
    are dropped without splitting the surrounding word.  No geometric gap rule.
    """
    try:
        raw = page.get_text("rawdict") or {}
    except Exception:
        return ""
    words: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            words.append("".join(current))
            current.clear()

    for block in raw.get("blocks", ()):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", ()):
            for span in line.get("spans", ()):
                if not isinstance(span, dict):
                    continue
                alpha = span.get("alpha", 255)
                for ch in span.get("chars", ()):
                    if not isinstance(ch, dict):
                        continue
                    bbox = ch.get("bbox")
                    char = ch.get("c")
                    if not bbox or len(bbox) < 4 or not char:
                        continue
                    if not text_layer_span_is_visible({"alpha": alpha}):
                        continue
                    if char.isspace():
                        flush()
                        continue
                    if not _glyph_bbox_has_ink(page, mat, bbox, page_pix=page_pix):
                        continue
                    current.append(char)
            flush()
    return " ".join(words)


def _tokens_from_page_text_layer(page) -> tuple[Counter, set[str], int]:
    """Tokenise the visibly rendered text layer for born-digital evidence (#1099).

    Character-level: drop ``rawdict`` chars with span ``alpha==0``, then drop
    chars with no non-white pixel in a ``DETECT_DPI`` colour render (white-on-white,
    text under a blank/white image). Re-group surviving chars into words per line.
    """
    try:
        mat = _text_layer_ink_matrix(page)
    except Exception:
        logger.debug("text-layer ink matrix failed", exc_info=True)
        return Counter(), set(), 0
    page_pix = None
    from socr.core.born_digital import upright_rotation_for

    if upright_rotation_for(page) == 0:
        try:
            import fitz

            page_pix = page.get_pixmap(matrix=mat, alpha=False)
        except Exception:
            logger.debug("text-layer ink page render failed", exc_info=True)
    try:
        plain = _plain_text_from_visible_chars(page, mat, page_pix)
    except Exception:
        logger.debug("visible char assembly failed", exc_info=True)
        plain = ""
    return _tokens_and_dash_led(plain)


def collect_table_tokens(markdown: str) -> TableTokens | None:
    """Extract numeric multiset and content-label tokens from markdown tables."""
    blocks = _evidence_table_blocks(markdown)
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


def _strip_word_edges(word: str) -> str:
    """Strip sentence punctuation and brackets from both ends of an evidence word.

    #1082: a leading ``.`` that starts a decimal (``.45``) is kept. The plain
    ``strip`` turned it into ``45``, which could SUPPORT a model ``45`` the
    page never printed; kept, it is ``0.45`` (``strip_presentation``).
    """
    word = word.rstrip(".,;:!?()[]\"'").lstrip(",;:!?()[]\"'")
    if word[:1] == "." and not word[1:2].isdigit():
        # #1085: punctuation can sit behind the dots (``...(0.025)``, ``.[45]``).
        word = word.lstrip(".").lstrip(",;:!?()[]\"'")
    return word


#: En/em dash: ``strip_presentation`` folds them to a minus, but on a printed page
#: they are as often an "n/a" or range dash as a sign.
_AMBIGUOUS_DASHES: str = "\N{EN DASH}\N{EM DASH}"


_C0_BEFORE_DIGIT_HEAD = re.compile(r"^[\x00-\x08\x0b\x0c\x0e-\x1f] ?")


def _is_control_byte_led(word: str) -> bool:
    """True when a C0 control byte sits before the first digit (#990 / #1099 r3).

    Like #1085 dash-led tokens: the witness read a number, but the byte cannot
    establish sign, so the token supports neither ``-x`` nor ``x``.
    """
    decoded = html.unescape(word)
    first_digit = next((i for i, ch in enumerate(decoded) if ch.isdigit()), len(decoded))
    if first_digit >= len(decoded):
        return False
    return bool(_C0_BEFORE_DIGIT_HEAD.match(decoded[:first_digit]))


def _is_dash_led(token: str) -> bool:
    """True when an en/em dash precedes the first digit of an evidence token.

    #1085: ``strip_presentation`` (shared with born-digital paths, left alone)
    turns ``\u20145`` into ``-5``, so a page printing an em dash before 5 would
    support a model ``-5`` it never printed as negative. Such a token supports
    nothing: neither ``-5`` nor ``5``, since a dash-then-number layout does not
    show which was printed. ASCII ``-`` and U+2212 keep their sign meaning.
    """
    # Decode to a fixed point: the shared numeric parser decodes again after
    # ``_normalize_cell``, so ``&amp;amp;mdash;5`` reaches it as ``&mdash;5``.
    # The same parser unwraps a whole-token markdown link first.
    while True:
        decoded = _unwrap_markdown_link(html.unescape(token))
        if decoded == token:
            break
        token = decoded
    first_digit = next((i for i, ch in enumerate(token) if ch.isdigit()), len(token))
    return any(ch in _AMBIGUOUS_DASHES for ch in token[:first_digit])


def _tokens_from_plain_text(text: str) -> tuple[Counter, set[str]]:
    """Parse numeric multiset and content tokens from plain text."""
    numeric, content, _ = _tokens_and_dash_led(text)
    return numeric, content


def _tokens_and_dash_led(text: str) -> tuple[Counter, set[str], int]:
    """As ``_tokens_from_plain_text``, plus how many dash-led numbers were dropped.

    #1085: a dropped dash-led number is still proof the witness READ a number,
    so the caller must not treat the page as unwitnessed.
    """
    if not text.strip():
        return Counter(), set(), 0
    dash_led = 0

    raw_numeric: list[str] = []
    content: set[str] = set()
    decoded_text = html.unescape(text)
    for word in re.findall(r"\S+", decoded_text):
        word = _strip_word_edges(word)
        if not word:
            continue
        # #679: same normalizer as the candidate side (``collect_table_tokens``)
        # -- a raw page's own printed decoration must get the identical
        # treatment or a candidate that genuinely agrees with it looks
        # unsupported. See ``_normalize_cell``.
        normalized = _normalize_cell(word)
        if is_numeric_token(normalized):
            if _is_dash_led(normalized) or _is_control_byte_led(word):
                dash_led += 1
            else:
                raw_numeric.append(normalized)
        for m in _CONTENT_TOKEN_RE.finditer(normalized):
            token = _content_token(m.group(0))
            if token is not None:
                content.add(token)
    return _numeric_multiset_from_tokens(raw_numeric), content, dash_led


def _merge_evidence(
    numeric: Counter,
    content: set[str],
    new_numeric: Counter,
    new_content: set[str],
) -> tuple[Counter, set[str]]:
    merged_numeric = numeric + new_numeric
    merged_content = content | new_content
    return merged_numeric, merged_content


def _max_merge_numeric(left: Counter, right: Counter) -> Counter:
    """Per-token maximum when two witnesses read the same page (#1099 r3).

    Summing would let one printed glyph support duplicate model cells (text layer
    plus OCR of the same glyph). Crop/full-page merges inside
    ``build_scanned_evidence`` still use ``_merge_evidence`` (additive).
    """
    merged = Counter(left)
    for tok, count in right.items():
        merged[tok] = max(merged.get(tok, 0), count)
    return merged


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


def classical_ocr_backend_state() -> tuple[str, str]:
    """Whether the classical-OCR witness can run at all, without reading a page.

    #1080. The startup counterpart of ``classical_ocr_with_state``: it returns
    ``(witness_state, detail)`` using the SAME state names and messages, so the
    preflight report and the per-page failure name a missing install the same
    way. ``WITNESS_READING`` means the package imports and the binary answers
    its version probe.

    Unlike the per-page path, a failing version probe is classified by the
    exception type only: ``TesseractNotFoundError`` is ``binary_missing``,
    anything else is ``exec_error`` (a present but broken executable).
    """
    try:
        import pytesseract
    except ImportError:
        return WITNESS_PACKAGE_MISSING, WITNESS_STATE_MESSAGES[WITNESS_PACKAGE_MISSING]
    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        not_found = getattr(pytesseract, "TesseractNotFoundError", ())
        if not_found and isinstance(exc, not_found):
            return WITNESS_BINARY_MISSING, WITNESS_STATE_MESSAGES[WITNESS_BINARY_MISSING]
        return WITNESS_EXEC_ERROR, f"{type(exc).__name__}: {exc}"
    return WITNESS_READING, ""


def classical_ocr_pixmap(pix) -> str:
    """Optional classical OCR over a rendered pixmap; returns "" when unavailable.

    The public, text-only seam kept for callers that pass an ``ocr_image_fn``.
    ``classical_ocr_with_state`` is the one this module uses internally.
    """
    return classical_ocr_with_state(pix)[0]


def _merge_source_bundles(
    text_layer: SourceEvidenceBundle,
    pixel_witness: SourceEvidenceBundle,
) -> SourceEvidenceBundle:
    """Union two evidence bundles (#1099 r2: born-digital text + crop/full-page OCR)."""
    numeric = _max_merge_numeric(text_layer.numeric, pixel_witness.numeric)
    content = set(text_layer.content) | set(pixel_witness.content)
    states = [s for s in (text_layer.witness_state, pixel_witness.witness_state) if s]
    witness_state = _resolve_witness_state(states) if states else WITNESS_NOT_ATTEMPTED
    detail = pixel_witness.witness_detail or text_layer.witness_detail
    ambiguous = text_layer.ambiguous_numeric + pixel_witness.ambiguous_numeric
    has_content = (
        bool(numeric)
        or bool(content)
        or bool(ambiguous)
        or text_layer.has_content_evidence
        or pixel_witness.has_content_evidence
    )
    return SourceEvidenceBundle(
        numeric=numeric,
        content=frozenset(content),
        has_content_evidence=has_content,
        table_regions_detected=text_layer.table_regions_detected
        or pixel_witness.table_regions_detected,
        witness_state=witness_state,
        witness_detail=detail,
        ambiguous_numeric=ambiguous,
    )


def build_born_digital_evidence(
    page,
    *,
    ocr_image_fn: OcrImageFn | None = None,
) -> SourceEvidenceBundle:
    """Born-digital table evidence: PDF text layer plus pixel/table OCR (#1099 r2).

    Merging ``include_text_layer=False`` pixel evidence avoids the GH-163 scan
    exclusion while still letting prose on the page satisfy ``has_content_evidence``
    and block spurious full-page OCR when the only gap is an image-embedded table.
    """
    n_text, c_text, amb_text = _tokens_from_page_text_layer(page)
    text_bundle = SourceEvidenceBundle(
        numeric=n_text,
        content=frozenset(c_text),
        has_content_evidence=bool(n_text) or bool(c_text) or bool(amb_text),
        table_regions_detected=False,
        witness_state=WITNESS_NOT_ATTEMPTED,
        ambiguous_numeric=amb_text,
    )
    pixel_bundle = build_scanned_evidence(
        page,
        ocr_image_fn=ocr_image_fn,
        include_text_layer=False,
    )
    return _merge_source_bundles(text_bundle, pixel_bundle)


def build_scanned_evidence(
    page,
    *,
    ocr_image_fn: OcrImageFn | None = None,
    include_text_layer: bool = True,
    text_layer_from_words: bool = False,
    allow_full_page_ocr: bool = True,
) -> SourceEvidenceBundle:
    """Build local non-generative evidence for a scanned page.

    ``include_text_layer=False`` drops the PDF text layer from the evidence.

    ``text_layer_from_words=True`` reads the layer via ``get_text("words")`` (for
    rotated born-digital pages) instead of a single ``get_text()`` blob.

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
    ambiguous = 0

    if include_text_layer:
        if text_layer_from_words:
            n_plain, c_plain, a_plain = _tokens_from_page_text_layer(page)
        else:
            try:
                plain = page.get_text() or ""
            except Exception:
                plain = ""
            n_plain, c_plain, a_plain = _tokens_and_dash_led(plain)
        ambiguous += a_plain
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
            n_crop, c_crop, a_crop = _tokens_and_dash_led(crop_text)
            ambiguous += a_crop
            numeric, content = _merge_evidence(numeric, content, n_crop, c_crop)
    except Exception as exc:
        logger.debug("locate_tables evidence failed: %s", exc)

    if allow_full_page_ocr and not numeric and not content and not ambiguous:
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
            n_page, c_page, a_page = _tokens_and_dash_led(page_text)
            ambiguous += a_page
            numeric, content = _merge_evidence(numeric, content, n_page, c_page)
        except Exception as exc:
            logger.debug("full-page evidence OCR failed: %s", exc)
            record(WITNESS_RENDER_ERROR, f"{type(exc).__name__}: {exc}")

    has_content_evidence = bool(numeric) or bool(content) or bool(ambiguous)
    witness_state = _resolve_witness_state(witness_states)
    return SourceEvidenceBundle(
        numeric=numeric,
        content=frozenset(content),
        has_content_evidence=has_content_evidence,
        table_regions_detected=table_regions_detected,
        witness_state=witness_state,
        witness_detail=witness_details.get(witness_state, ""),
        ambiguous_numeric=ambiguous,
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
            disputed_values=tuple(str(m) for m in missing),
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
            disputed_values=tuple(sorted(tokens.content - bundle.content)),
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


#: ``verify_scanned_table``'s reason when the text holds no markdown table to tokenise.
REASON_NO_TABLE_BLOCKS = "no markdown table blocks"
_HTML_TABLE_RE = re.compile(r"<table\b", re.IGNORECASE)


def html_table_for_check(text: str) -> tuple[str, bool]:
    """#1048: ``(text with HTML tables as markdown, an HTML table remains unparsed)``.

    The evidence check tokenises markdown pipe tables only, so a candidate that emitted its
    table as HTML (rung 2 after a disputed pipe table, say) used to skip the check and be
    accepted unverified. ``core.html_tables.convert_html_tables`` is the house parser; the
    converted text is what the check reads, and the SHIPPED text is never changed. A block
    that does not parse as rows is left in place by that function (it never deletes
    content), so a ``<table`` surviving the conversion means a table the check cannot read.
    """
    if not _HTML_TABLE_RE.search(text or ""):
        return text, False
    from socr.core.html_tables import convert_html_tables

    converted = convert_html_tables(text)
    return converted, bool(_HTML_TABLE_RE.search(converted))


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
    the evidence check runs however many words the layer contains. GH-163 still
    excludes that layer when a page-covering raster is present (scan + baked-in
    OCR); born-digital pages without such a raster use the PDF text layer as
    evidence (#1099). ``None`` means the caller cannot tell, and the
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
            reason=REASON_NO_TABLE_BLOCKS,
            deferred=True,
        )

    if native_trusted is False:
        # GH-163 for image-dominant scans; born-digital text + crop OCR (#1099 / r2).
        if page_has_page_covering_raster(page):
            bundle = build_scanned_evidence(
                page,
                ocr_image_fn=ocr_image_fn,
                include_text_layer=False,
            )
        else:
            bundle = build_born_digital_evidence(page, ocr_image_fn=ocr_image_fn)
    else:
        bundle = build_scanned_evidence(
            page,
            ocr_image_fn=ocr_image_fn,
            include_text_layer=native_trusted is not False,
            text_layer_from_words=False,
        )
    return verify_table_tokens(bundle, tokens)
