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
import string
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace

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

# Wall-clock bound for ONE pytesseract subprocess call (a page or a crop; a page
# with several crops can spend this bound once per call). pytesseract 0.3.13
# defaults ``timeout=0``, which is an unbounded ``proc.communicate()``; after
# #1135 moved the witness outside the page-judge deadline a wedged tesseract
# could hang the page. Derived 2026-10-13 from the 181 (doc, page) pairs in
# ``scratch/routing-baseline/witness_detail_out.json``, rendered at
# ``_EVIDENCE_OCR_DPI`` against ``witness-pdfs/``. All 181 pages timed
# (min 0.095 s, p50 0.752 s, p95 1.568 s, p99 1.789 s, max 2.326 s).
# Timeout is ``ceil(p99 * 2.0)`` — the 2.0 is the same headroom as
# ``tables.extract._CROP_WALL_CLOCK_MULTIPLIER`` — which is 4 s and already
# above ``ceil(max)``. Expire maps to ``WITNESS_EXEC_ERROR``.
TESSERACT_WITNESS_TIMEOUT_SEC: float = 4.0

# #1357: the further resolutions the scanned witness reads, in order, ONLY when
# the base-resolution reading leaves numeric table tokens unsupported. Measured
# 2026-11-16 with ``build_scanned_evidence(include_text_layer=False)`` on the
# 12 Fed sample pages, emitted tokens = the saved qwen table reading, unsupported
# numeric tokens per single reading at 150/200/300/400 dpi:
#   p007  20 numbers: 2 / 0 / 0 / 0
#   p009 134 numbers: 3 / 2 / 113 / 103     (300+ collapses)
#   p044  34 numbers: 31 / 19 / 13 / 10
#   p063 185 numbers: 4 / 1 / 123 / 124     (300+ collapses)
#   p097 162 numbers: 26 / 11 / 5 / 0
# A table passes at a rung only when ONE reading (the base or that rung) supports
# every token by itself; readings are never pooled (review round, #1357). Pooling
# different tokens across resolutions let one printed glyph count as two numbers:
# 55 of 5225 single-cell mutants of p097 (1.05%) made the whole table pass against
# 0.02% at base, while the single-reading rule passed 0 of 14106 mutants over the
# five pages and clears the same two pages (p007 at 200, p097 at 400).
# 300 dpi is not a rung: by the numbers above it clears no page, it is where p009
# and p063 collapse and drop decimal points, and a (200, 400) ladder gives the
# same verdict on all five pages with fewer coincidences. The 3 label-only pages
# (p027, p028, p051) have no numeric tokens and never trigger a rung. Ladder order
# is ascending cost, and the loop stops at the first rung that clears.
_EXTRA_EVIDENCE_OCR_DPIS: tuple[int, ...] = (200, 400)

# Wall-clock bound per tesseract call at each extra rung. The base bound is
# ``ceil(p99 * 2.0)`` at 150 dpi (p99 1.789 s, above). Per-call wall time
# measured 2026-11-16 on the 12 sample pages (median of 3 runs per call, 17
# calls) rose with resolution; the worst per-call ratio to the same call at
# 150 dpi was 1.82 (200) and 2.58 (400), and the slowest single call was
# 0.93 / 1.01 / 1.63 s at 150 / 200 / 400 dpi. Each bound is
# ``ceil(1.789 * worst_ratio * 2.0)``: 6.5 -> 7, 9.2 -> 10. Same 2.0 headroom as
# the base bound; the measured maxima sit well inside them.
_EXTRA_EVIDENCE_TIMEOUT_SEC: dict[int, float] = {200: 7.0, 400: 10.0}

# Alphabetic runs in evidence text: at least _MIN_CONTENT_TOKEN_LEN chars.
_CONTENT_TOKEN_RE = re.compile(rf"[A-Za-z][A-Za-z0-9\-]{{{_MIN_CONTENT_TOKEN_LEN - 1},}}")

# LaTeX control words (``\ldots``, ``\frac``) are markup, not table labels (#1246).
_LATEX_CMD_RE = re.compile(r"\\[A-Za-z]+(?:\{[^{}]*\})?")


def _strip_latex_markup_for_labels(text: str) -> str:
    """Remove LaTeX commands/delimiters before content-label tokenization."""
    if not text:
        return text
    text = text.replace("$", " ")
    text = _LATEX_CMD_RE.sub(" ", text)
    text = re.sub(r"[{}]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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
#: #1355: the audit event for numeric tokens the witness did not support itself but a second
#: reading by another engine corroborated (a lost sign or decimal point). One constant, so the
#: emitter and the resume-replay registry cannot spell it differently.
CORROBORATED_KIND: str = "source_evidence_table_corroborated"
#: #1357: audit event for a table that PASSED only because the witness read
#: further resolutions. ``data``: ``resolutions`` (dpi list), ``tokens_resolved``.
EXTRA_RESOLUTION_KIND: str = "source_evidence_table_extra_resolution"
#: #1399: audit event for a scanned table that passed with the stored text layer as its witness
#: but was not confirmed in full (ships WARNING). ``data``: ``values``.
LAYER_UNPROVEN_KIND: str = "source_evidence_table_layer_unproven"


@dataclass(frozen=True)
class TableHeader:
    """#1355: what the header rows of one table hold, kept apart from its data rows.

    ``bare``: numeric tokens of header cells that are a bare number (also in ``numeric``). The
    header region is row 0 plus the rows after it whose first cell is empty.
    ``label``: numbers inside header cells that are not (``2001 (Percent change<sup>1</sup>)``),
    tokenised by ``plain_text_tokens``. ``real``: the table has a header row at all (a markdown
    separator row after the first line; ``<th>`` for HTML, set by the caller): without one, row 0
    is data and neither counter may stand in for it.
    """

    bare: Counter = field(default_factory=Counter)
    label: Counter = field(default_factory=Counter)
    real: bool = False


@dataclass(frozen=True)
class TableTokens:
    """Tokens collected from VLM-emitted markdown table cells."""

    numeric: Counter = field(default_factory=Counter)
    content: frozenset[str] = field(default_factory=frozenset)
    has_numeric: bool = False
    is_alpha_only: bool = False
    #: #1355: one ``TableHeader`` per evidence-readable table, in page order. Used only by the
    #: winner-containment rule (``agentic.winner_accounts_for_marked_tables``); the witness check
    #: never reads it.
    headers: tuple[TableHeader, ...] = ()


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
    #: #1355: numeric tokens read from PIXELS (crop/full-page OCR) only. ``None`` on a bundle
    #: that does not separate its sources (the pixel-only bundle, where every token is a pixel
    #: token, and the scanned bundle that folds the text layer into ``numeric``); set by
    #: ``_merge_source_bundles``, where ``numeric`` is the per-token MAX of the text layer and the
    #: pixels and so cannot say which source a token came from.
    pixel_numeric: Counter | None = None
    #: #1355: numeric tokens of the exact text layer, kept apart from the pixels so a disputed
    #: token the layer states itself is never corroborated. Empty unless ``pixel_numeric`` is set.
    layer_numeric: Counter = field(default_factory=Counter)
    #: #1355: the digit key (``_digit_key``) of EVERY number-bearing word of the visible text layer,
    #: including the ones ``layer_numeric`` drops (dash-led, ``41bp``, ``x41``). Veto evidence only,
    #: never support.
    layer_digit_keys: frozenset[str] = frozenset()
    #: #1355: every numeric token the extra-resolution re-reads (#1357) saw, whether or not a
    #: re-read cleared the table. Veto evidence only, never support and never pooled.
    rung_tokens: frozenset[str] = frozenset()

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
    #: #1355: numeric tokens (one entry per occurrence) the witness did not support itself but a
    #: second reading by another engine corroborated; see ``corroborate_unsupported_numeric``.
    #: Only ever set on a PASSED verdict, and only when the witness evidence alone would have
    #: failed it.
    corroborated_values: tuple[str, ...] = ()
    #: #1355: the engine of that second reading ("" when ``corroborated_values`` is empty).
    corroborated_by: str = ""
    #: #1357: the extra resolutions (dpi) the witness read because the base
    #: reading left numeric tokens unsupported, in the order read. Empty when
    #: the base reading sufficed, which is the unchanged path.
    extra_resolutions: tuple[int, ...] = ()
    #: #1357: numeric tokens (multiset count) that were unsupported at the base
    #: resolution and supported after the extra readings.
    extra_resolution_tokens: int = 0
    #: #1399: non-empty on a PASSED verdict that ships marked (WARNING): the stored layer
    #: witness did not confirm the table in full. The text is the reason for the mark.
    unproven: str = ""
    #: #1399: the numbers behind ``unproven`` (layer tokens the reading lacks, or reading tokens
    #: the layer does not hold), one entry per occurrence, for the mark and the audit event.
    unproven_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class SecondReading:
    """#1355: the EARLIER disputed reading of the same page, as the witness may use it."""

    engine: str
    numeric: Counter = field(default_factory=Counter)


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
    typical PDFs). ``get_texttrace()`` is avoided because it has a refcount
    defect that ABORTS the interpreter; see
    :func:`text_layer_invisible_char_counts` for what that defect actually is.
    """
    return span.get("alpha", 255) != 0


def _text_layer_dict_flags() -> int:
    """``get_text("dict")`` flags with image payloads switched off.

    Not a tuning knob: this helper reads text spans and never looks at an image block,
    while ``TEXTFLAGS_DICT`` makes MuPDF carry every raster into the result -- copying a
    JPEG through, or re-encoding a flate raster to PNG. The pages #961 targets are
    precisely the ones with a full-page raster, so the default pays that on every page it
    is asked about. Measured on three Fed scans, 12 pages each, median per page:
    61.8 / 50.7 / 57.6 ms with the default against 3.9 / 3.6 / 3.7 ms with images off.
    At roughly 350,000 pages in the Fed re-OCR that is hours of CPU for bytes nothing
    reads. (#1189 review; the reviewer measured 105 ms -> 0.7 ms on a flate scan of
    theirs, same direction, larger gap.)
    """
    import fitz

    return fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES


_TEXT_LAYER_DICT_FLAGS = _text_layer_dict_flags()


def text_layer_invisible_char_counts(page) -> tuple[int, int]:
    """``(invisible chars, total chars)`` in *page*'s text layer (#1189).

    The one place that weighs a page's text layer by visibility. #961 asks whether
    most of it is invisible; ``native_contradiction`` asks the complement, whether
    enough of it is visible to be read as evidence. Both are ratios over the SAME
    two counts, so they live here together: a wrapper in front of one of them would
    blind the other.

    ``get_texttrace()`` is deliberately not used. In PyMuPDF 1.27.2.2, 1.27.2.3 and
    1.28.0 it decrements ``None``'s reference count once per span returned
    (measured: 309 spans -> -320, 515 -> -520). The count falls monotonically to
    zero and the interpreter dies with ``Fatal Python error: none_dealloc``. That is
    an ABORT, not an exception: no ``except``, no ``finally``, no fail-closed
    handling runs. Because the fatal decref can be any decref, the crash surfaces at
    whatever code happens to touch ``None`` next -- which is why #1189 was first
    reported as a ``find_tables`` defect, and why the comment here used to say so.
    Upstream fixed it in 1.28.2 (PyMuPDF #5042), but the Alliance cluster builds its
    own wheels and ran the affected ``1.27.2.2+computecanada``, so socr does not rely
    on having a fixed PyMuPDF.

    The ``alpha`` route agreed with the old one on every page of 506 pages of six
    documents -- five Fed scans that abort, plus a dense econ paper -- 315
    invisible-dominant and 191 not (``scratch/1189/alpha_equiv.jsonl``). It is NOT
    equivalent by construction, and two synthetic constructions diverge; both times
    ``alpha`` is the more faithful answer and both times it errs toward distrusting
    the text layer, which is the safe direction for a citation corpus:

    * **Render mode 7** (clip-only) text is invisible, and ``get_texttrace`` returns
      no spans at all for it, so the old route scored it ``0/0`` and called the page
      clean. ``alpha`` reports it as invisible, which it is.
    * **Render mode 0 with a fully transparent fill** is invisible ink in a visible
      mode. The old route saw type 0 and called it visible; ``alpha`` reports 0.

    One third shape diverges the OTHER way, and is accepted rather than fixed:
    **render mode 2 with a transparent fill and an opaque stroke** puts visible glyph
    outlines on the page, but the span's ``alpha`` comes from the fill pass, so this
    route calls it invisible. That is wrong about the ink and right about the risk:
    the page is sent for re-reading instead of being trusted, which costs OCR time
    and loses nothing. Outline-only text over a page-sized raster is not a shape any
    OCR tool produces.

    No page of the 506 was any of the three shapes, so the change is inert on
    everything measured. ``tests/test_gh1189_texttrace_abort.py`` pins all three as
    intended rather than leaving them to be rediscovered as regressions.

    Raises on a PyMuPDF failure, so a caller that must fail closed still can; an
    unreadable page is "unknown", never "clean". That includes a non-dict return:
    ``(page.get_text("dict") or {})`` would have turned a ``None`` into zero spans,
    i.e. a confident "no invisible text" on a page nothing could be read from. A
    dict carrying no ``blocks`` is different and is NOT an error -- that is a page
    with no text layer, which legitimately counts ``(0, 0)``.
    """
    page_dict = page.get_text("dict", flags=_TEXT_LAYER_DICT_FLAGS)
    if not isinstance(page_dict, dict):
        raise TypeError(
            f'page.get_text("dict") returned {type(page_dict).__name__}, not a dict; '
            "the page's text layer could not be read"
        )
    invisible = total = 0
    for block in page_dict.get("blocks", ()):
        if block.get("type") != 0:  # 0 = text; images carry no spans
            continue
        for line in block.get("lines", ()):
            for span in line.get("spans", ()):
                if not isinstance(span, dict):
                    continue
                if "alpha" not in span:
                    # A missing key is not evidence of visible ink. PyMuPDF below
                    # 1.25.3 emits no "alpha" at all, and the 255 default in
                    # text_layer_span_is_visible would then read a scan's invisible OCR
                    # layer as fully visible and ship it as trusted native -- silently,
                    # the one outcome #961 exists to prevent. pyproject's floor rules
                    # those versions out; this raise is what makes that floor
                    # load-bearing rather than advisory, for a wheel built outside it
                    # (the Alliance cluster builds its own). The callers fail closed.
                    raise ValueError(
                        'get_text("dict") span carries no "alpha" key; PyMuPDF >= 1.25.3 '
                        "is required to judge text-layer visibility (#1189)"
                    )
                count = len(span.get("text") or "")
                total += count
                if not text_layer_span_is_visible(span):
                    invisible += count
    return invisible, total


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
    """True when the glyph box contains a non-white pixel on a white-backed render.

    *bbox* is a ``rawdict`` char box, which MuPDF reports in the page's UNROTATED
    coordinate space. ``get_pixmap(clip=...)`` and a full-page render both work in the
    ROTATED space (``page.rect``, with ``/Rotate`` already applied). On a page with
    ``/Rotate`` 90 or 270 the raw box therefore lands on the wrong part of the render
    and most glyphs read as inkless (#1350). ``page.rotation_matrix`` maps one space
    to the other; it is the identity at rotation 0 and is skipped there.
    """
    import fitz

    rect = fitz.Rect(bbox)
    if rect.is_empty or rect.get_area() <= 0:
        return False
    if page.rotation:
        rect = rect * page.rotation_matrix
    if page_pix is not None:
        device = rect * mat
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


def _digit_keys_of_text(text: str) -> frozenset[str]:
    """#1355: the digit key of every whitespace-delimited word of *text* that holds a digit, plus
    each digit run on its own. A veto set: it errs toward more keys (``41bp``, ``x41``, ``-41`` and
    ``4.1`` all give ``41``; ``2001-2002`` gives ``20012002``, ``2001`` and ``2002``)."""
    keys: set[str] = set()
    for word in re.findall(r"\S+", html.unescape(text or "")):
        digits = "".join(ch for ch in word if ch in string.digits)
        if digits:
            keys.add(digits.lstrip("0") or "0")
        for run in re.findall(r"[0-9]+", word):
            keys.add(run.lstrip("0") or "0")
    return frozenset(keys)


def _visible_layer_plain(page) -> str:
    """The visibly rendered text layer as plain text ("" when it cannot be read)."""
    try:
        mat = _text_layer_ink_matrix(page)
    except Exception:
        logger.debug("text-layer ink matrix failed", exc_info=True)
        return ""
    page_pix = None
    from socr.core.born_digital import upright_rotation_for

    if upright_rotation_for(page) == 0:
        try:
            import fitz

            page_pix = page.get_pixmap(matrix=mat, alpha=False)
        except Exception:
            logger.debug("text-layer ink page render failed", exc_info=True)
    try:
        return _plain_text_from_visible_chars(page, mat, page_pix)
    except Exception:
        logger.debug("visible char assembly failed", exc_info=True)
        return ""


def _tokens_from_page_text_layer(page) -> tuple[Counter, set[str], int]:
    """Tokenise the visibly rendered text layer for born-digital evidence (#1099).

    Character-level: drop ``rawdict`` chars with span ``alpha==0``, then drop
    chars with no non-white pixel in a ``DETECT_DPI`` colour render (white-on-white,
    text under a blank/white image). Re-group surviving chars into words per line.
    """
    return _tokens_and_dash_led(_visible_layer_plain(page))


def _block_has_header_separator(lines: list[str], block) -> bool:
    """True when the line after the block's first is a markdown separator row."""
    nxt = block.start + 1
    return nxt <= block.end and _is_separator_row(_split_row(lines[nxt]))


def collect_table_tokens(markdown: str, *, headers_real: bool = True) -> TableTokens | None:
    """Extract numeric multiset and content-label tokens from markdown tables.

    ``headers_real=False`` says the caller knows the tables' first rows are not headers (an HTML
    table without ``<th>``, converted to markdown with a synthetic separator row).
    """
    blocks = _evidence_table_blocks(markdown)
    if not blocks:
        return None
    lines = markdown.splitlines()

    raw_numeric: list[str] = []
    raw_content: set[str] = set()
    headers: list[TableHeader] = []
    for block in blocks:
        h_bare: Counter = Counter()
        h_label: Counter = Counter()
        in_header = True
        for row_idx, row in enumerate(block.grid):
            # #1355: the header region is row 0 plus the rows right after it whose first (stub)
            # cell is empty: a spanning header written over two rows. The first row with a stub
            # label is data.
            if row_idx and in_header and row and row[0].strip():
                in_header = False
            for cell in row:
                cell = cell.strip()
                if not cell or cell in ("---", "—"):
                    continue
                # #679: decode + dash-strip BEFORE the numeric check, not
                # only in the label branch below -- see ``_normalize_cell``.
                normalized = _normalize_cell(cell)
                if is_numeric_token(normalized):
                    raw_numeric.append(normalized)
                    if in_header:
                        h_bare.update(_numeric_multiset_from_tokens([normalized]))
                elif in_header:
                    h_label.update(plain_text_tokens(cell)[0])
                # Content-label tokens come from data rows only (row 0 is the
                # header).  Generic header words ("Category", "Description") are
                # not reliable hallucination signals and rarely appear in OCR
                # evidence on scanned pages.
                if row_idx == 0:
                    continue
                label_text = _strip_latex_markup_for_labels(normalized)
                for m in _CONTENT_TOKEN_RE.finditer(label_text):
                    token = _content_token(m.group(0))
                    if token is not None:
                        raw_content.add(token)

        headers.append(
            TableHeader(
                bare=h_bare,
                label=h_label,
                real=headers_real and _block_has_header_separator(lines, block),
            )
        )

    numeric = _numeric_multiset_from_tokens(raw_numeric)
    has_numeric = bool(numeric)
    is_alpha_only = not has_numeric and bool(raw_content)
    return TableTokens(
        numeric=numeric,
        content=frozenset(raw_content),
        has_numeric=has_numeric,
        is_alpha_only=is_alpha_only,
        headers=tuple(headers),
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


def plain_text_tokens(text: str) -> tuple[Counter, set[str]]:
    """Public seam over ``_tokens_from_plain_text`` (#1050 step 2).

    ``socr.core.text_coverage`` asks whether a native word inside a candidate
    figure is a CONTENT word, and must use the same definition the table
    witness does rather than a second one that drifts. Cross-package imports of
    private names are forbidden (``test_package_layering``), so the one
    definition is exported here rather than copied.
    """
    return _tokens_from_plain_text(text)


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


def classical_ocr_with_state(
    pix, timeout_sec: float = TESSERACT_WITNESS_TIMEOUT_SEC
) -> tuple[str, str, str]:
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
        text = pytesseract.image_to_string(img, timeout=timeout_sec) or ""
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
        pixel_numeric=Counter(pixel_witness.numeric),
        layer_numeric=Counter(text_layer.numeric),
        layer_digit_keys=text_layer.layer_digit_keys,
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
    plain_layer = _visible_layer_plain(page)
    n_text, c_text, amb_text = _tokens_and_dash_led(plain_layer)
    text_bundle = SourceEvidenceBundle(
        numeric=n_text,
        content=frozenset(c_text),
        has_content_evidence=bool(n_text) or bool(c_text) or bool(amb_text),
        table_regions_detected=False,
        witness_state=WITNESS_NOT_ATTEMPTED,
        ambiguous_numeric=amb_text,
        layer_digit_keys=_digit_keys_of_text(plain_layer),
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
    dpi: int = _EVIDENCE_OCR_DPI,
) -> SourceEvidenceBundle:
    """Build local non-generative evidence for a scanned page.

    ``include_text_layer=False`` drops the PDF text layer from the evidence.

    ``dpi`` is the raster resolution of the pixel readings (#1357: the extra
    rungs of ``widen_scanned_evidence`` pass the ladder's resolutions).

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
        if dpi == _EVIDENCE_OCR_DPI:
            read_witness = classical_ocr_with_state
        else:
            # An extra rung: the bound derived for its resolution (#1357).
            try:
                rung_timeout = _EXTRA_EVIDENCE_TIMEOUT_SEC[dpi]
            except KeyError:
                raise ValueError(
                    f"no derived timeout for {dpi} dpi; supported: "
                    f"{_EVIDENCE_OCR_DPI} or {sorted(_EXTRA_EVIDENCE_TIMEOUT_SEC)}"
                ) from None

            def read_witness(pix) -> tuple[str, str, str]:
                return classical_ocr_with_state(pix, timeout_sec=rung_timeout)

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
                pix = _render_crop_pixmap(page, box.bbox, dpi)
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
            mat = fitz.Matrix(dpi / 72, dpi / 72)
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


def _unsupported_numeric_count(output: Counter, evidence: Counter) -> int:
    """Multiset count of emitted numeric tokens the evidence does not cover."""
    return sum(max(0, count - evidence.get(tok, 0)) for tok, count in output.items())


def widen_scanned_evidence(
    page,
    bundle: SourceEvidenceBundle,
    tokens: TableTokens,
    *,
    ocr_image_fn: OcrImageFn | None = None,
) -> tuple[SourceEvidenceBundle, tuple[int, ...], int]:
    """#1357: read further resolutions when the base reading leaves numbers unsupported.

    Returns ``(bundle, resolutions_read, tokens_resolved)``. When the base
    bundle already supports every emitted numeric token, or holds no reading to
    judge against, the input bundle is returned untouched with no extra read.

    Otherwise each rung of ``_EXTRA_EVIDENCE_OCR_DPIS`` re-reads the same
    regions as pixels only (never the text layer: ``include_text_layer=False``,
    so on the born-digital arm too the rungs add pixel evidence only). The
    table passes at a rung only when that ONE reading supports every emitted
    numeric token by itself; readings are never pooled across resolutions,
    because different tokens from different readings let one printed glyph
    count as two numbers (#1357 review). The loop stops at the first rung that
    clears. A rung that reads nothing (timeout, exec error, blank) contributes
    nothing and is not listed in ``resolutions_read``; the base bundle's witness
    state and detail are kept. On failure the base bundle is returned.

    Content-label evidence stays the base reading's. Labels are a set (no
    counts), so a larger pool can only add coincidental matches; they never
    reject a table that has numbers (#659 flags them) and an alpha-only table
    has no numeric token to trigger a rung, so the label verdict of a table is
    the same as if the base reading had sufficed.
    """
    if (
        not tokens.has_numeric
        or not bundle.has_content_evidence
        or _multiset_supported(tokens.numeric, bundle.numeric)
    ):
        return bundle, (), 0
    base_missing = _unsupported_numeric_count(tokens.numeric, bundle.numeric)
    read: list[int] = []
    seen: set[str] = set()
    for dpi in _EXTRA_EVIDENCE_OCR_DPIS:
        rung = build_scanned_evidence(
            page,
            ocr_image_fn=ocr_image_fn,
            include_text_layer=False,
            # Same fallback as the base pass: the full page is read only when
            # the table crops yielded no reading at all (no region located, or
            # every crop read empty), never in addition to crops that read.
            allow_full_page_ocr=True,
            dpi=dpi,
        )
        if not rung.has_content_evidence:
            # Nothing was read at this resolution (timeout, exec error, blank):
            # it contributes nothing and is not recorded as a resolution read.
            continue
        read.append(dpi)
        seen.update(rung.numeric)
        if _multiset_supported(tokens.numeric, rung.numeric):
            # One reading supports every token on its own. The rung's numeric
            # evidence replaces the base's; its labels and witness state do not.
            return (
                replace(
                    bundle,
                    numeric=Counter(rung.numeric),
                    pixel_numeric=(None if bundle.pixel_numeric is None else Counter(rung.numeric)),
                ),
                tuple(read),
                base_missing,
            )
    return replace(bundle, rung_tokens=frozenset(seen)), tuple(read), 0


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


def _digit_key(token: str) -> str:
    """The digits of a normalised numeric token, sign and decimal point dropped (#1355).

    Leading zeros are dropped as well, because the tokenisers WRITE one: ``.7`` is normalised
    to ``0.7`` (``strip_presentation``, #1082/GH-206) while a witness that lost the point reads
    ``7``. Without folding it the two would never meet. Trailing zeros are kept: ``12.0`` and
    ``120`` share a key (a lost point) but ``12.0`` and ``12`` do not, since a missing trailing
    digit is not a lost sign or point.
    """
    digits = "".join(ch for ch in token if ch in string.digits)
    return digits.lstrip("0") or "0"


def _marks(token: str) -> frozenset[str]:
    """The sign and decimal point a normalised numeric token carries (#1355 review)."""
    return frozenset(ch for ch in token if ch in "-.")


def _witness_may_stand_for(witness_token: str, emitted_token: str) -> bool:
    """True when *witness_token* is *emitted_token* with a sign and/or point DELETED.

    String-level: the witness token must be a subsequence of the emitted one and every deleted
    character must be a ``-`` or ``.``. A point in another place (``1.25`` for ``12.5``), a
    missing digit, or a mark the emitted token lacks (witness ``-4.1``, emitted ``4.1``: a witness
    that read the cell more completely than the readers did, ``-.7`` shipped as ``-7`` being a
    documented model error) is never a stand-in.

    Leading zeros are digits, not marks, so ``7`` does not stand for ``0.07`` or ``0.7`` (a
    conservative refusal: a witness that lost both the zero and the point is not a lost mark).
    ``07`` does stand for ``0.7``; the tokenisers write ``.7`` as ``0.7``, so ``.7`` meets ``0.7``
    directly.
    """
    if witness_token == emitted_token:
        return True
    j = 0
    for ch in emitted_token:
        if j < len(witness_token) and witness_token[j] == ch:
            j += 1
        elif ch not in "-.":
            return False
    return j == len(witness_token)


def corroborate_unsupported_numeric(
    emitted: Counter,
    evidence: Counter,
    second: Counter,
    *,
    standin: Counter | None = None,
    exact_keys: frozenset[str] = frozenset(),
    finer: frozenset[str] = frozenset(),
) -> tuple[Counter, Counter]:
    """#1355: split the emitted tokens the witness leaves unsupported into
    ``(corroborated, still_unsupported)``. Both are multisets of tokens.

    The witness reads a 150 dpi raster and drops minus signs and decimal points, so a correct
    ``-4.1`` can meet the witness as ``41``. An unsupported emitted token counts as corroborated
    only when BOTH hold:

    - ``second`` (the earlier disputed reading's tokens) carries that token at least as many
      times as it was emitted;
    - the witness evidence holds a token that is the emitted token with a sign and/or point
      removed (``_witness_may_stand_for``: same digits, witness marks a subset of the emitted
      marks) and is still unconsumed, after every directly supported emitted token and every
      earlier corroboration has taken its own. One witness ``41`` can therefore stand behind
      one ``-4.1``, never two.

    A token whose digits the witness does not hold stays unsupported, and so does one the
    witness holds WITH a mark the emitted token lacks (witness ``-4.1``, emitted ``4.1``): that
    is a disagreement about the number, not a lost sign or point.

    On a bundle that merges the exact text layer with pixel readings, ``standin`` is the
    pixel-read tokens and ``exact_keys`` the digit keys of every number in the text layer. Only a
    ``standin`` token may stand behind an emitted one, and an emitted token whose digits the text
    layer holds in ANY form is never corroborated: the layer is an exact source, so it already
    made its statement about that number. ``finer`` is every token the higher-resolution re-reads
    (#1357) saw: one that has the emitted token's digits but is not the emitted token or a
    mark-deleted form of it (a sign the emitted token lacks, a point elsewhere) vetoes it. Directly supported tokens still consume from ``evidence`` (the merged multiset).
    Both default to the pure-pixel case: every evidence token stands in, nothing vetoes.
    """
    spare: Counter = Counter(evidence)
    unsupported: Counter = Counter()
    for tok, count in emitted.items():
        direct = min(count, spare.get(tok, 0))
        spare[tok] -= direct
        if count > direct:
            unsupported[tok] = count - direct
    corroborated: Counter = Counter()
    still: Counter = Counter()
    # Most marks first: a token with more marks fits fewer witness tokens, so it picks first, and
    # among compatible witness tokens the most-marked is taken, leaving the least-marked (which
    # stands for the most emitted tokens) for the rest.
    for tok in sorted(unsupported, key=lambda t: (-len(_marks(t)), t)):
        for _ in range(unsupported[tok]):
            candidates = [
                w
                for w, n in spare.items()
                if n > 0
                and (standin is None or standin.get(w, 0) > 0)
                and _witness_may_stand_for(w, tok)
            ]
            if (
                second.get(tok, 0) >= emitted[tok]
                and candidates
                and _digit_key(tok) not in exact_keys
                and not any(
                    _digit_key(r) == _digit_key(tok) and not _witness_may_stand_for(r, tok)
                    for r in finer
                )
            ):
                spare[max(candidates, key=lambda w: (len(_marks(w)), w))] -= 1
                corroborated[tok] += 1
            else:
                still[tok] += 1
    return corroborated, still


def _content_supported(output: frozenset[str], evidence: frozenset[str]) -> bool:
    """True when every emitted content-label token appears in evidence."""
    if not output:
        return True
    return output.issubset(evidence)


def verify_table_tokens(
    bundle: SourceEvidenceBundle,
    tokens: TableTokens,
    *,
    second_reading: SecondReading | None = None,
    engine: str = "",
) -> SourceEvidenceResult:
    """Fail-closed check: emitted table tokens must be supported by evidence.

    ``second_reading`` (#1355) is passed only for a paid reading run after a witness dispute.
    It can corroborate a numeric token the witness lost a sign or point on
    (``corroborate_unsupported_numeric``), and only when BOTH engines are known and differ:
    ``engine`` is the engine of the reading under check. A content-label dispute is never touched.
    """
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

    corroborated: Counter = Counter()
    if tokens.has_numeric and not _multiset_supported(tokens.numeric, bundle.numeric):
        missing = [
            tok for tok, count in tokens.numeric.items() if bundle.numeric.get(tok, 0) < count
        ]
        still: Counter = Counter(tokens.numeric)
        if (
            second_reading is not None
            and second_reading.engine
            and engine
            and second_reading.engine != engine
        ):
            corroborated, still = corroborate_unsupported_numeric(
                tokens.numeric,
                bundle.numeric,
                second_reading.numeric,
                standin=bundle.pixel_numeric,
                exact_keys=bundle.layer_digit_keys
                | frozenset(_digit_key(t) for t in bundle.layer_numeric),
                finer=bundle.rung_tokens,
            )
        if still:
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

    if corroborated:
        return SourceEvidenceResult(
            verifiable=True,
            passed=True,
            reason=(
                "source evidence supports emitted table tokens, except numeric tokens its own "
                "reading lost a sign or decimal point on, which a second reading corroborates"
            ),
            content_unverified=content_unverified,
            corroborated_values=tuple(sorted(corroborated.elements())),
            corroborated_by=second_reading.engine if second_reading is not None else "",
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
    layer_is_evidence: bool = True,
    second_reading: SecondReading | None = None,
    engine: str = "",
    layer_witness: bool = False,
    chart_page: bool | None = None,
) -> SourceEvidenceResult:
    """Full scanned-page pipeline: defer native, else verify or fail closed.

    #1399: ``layer_witness=True`` lets the stored layer of a page whose layer is NOT evidence
    (``layer_is_evidence=False``, the suspect union) serve as the witness of this table check
    alone: see ``socr.tables.layer_witness``. Off by default; a page with no layer numbers, or
    where the layer is evidence, is unchanged. ``chart_page`` is the page's chart/figure
    classification (a pre-gate); ``None`` detects it with the existing chart and caption
    detectors (``layer_witness.page_is_chart``).

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

    #1205 round 1 (Argon P0 1.1, Grok P1): ``layer_is_evidence=False`` says this
    page's text layer may not serve as evidence at all --
    ``state.layer_is_admissible_evidence``, the union, so a FAILED detection scan is
    included. ``native_trusted=False`` is NOT that question: it is also False for an
    ordinary non-born-digital page, where #1099's born-digital bundle is wanted. The
    distinction is load-bearing and was measured on a flagged page with a VISIBLE layer
    and no page-covering raster (a scan-failed page, a cover under 50%, an
    ``get_image_info`` error): ``build_born_digital_evidence`` unions
    ``page.get_text("rawdict")`` into the numeric evidence, so with no usable pixel
    witness the layer's own ``350`` PASSED and a correct pixel ``-350`` was REFUSED as
    unsupported. With the layer excluded the evidence is pixel-only -- the same bundle
    the page-covering-raster branch already builds, and the same reason #1207 dropped
    ``witness_lines`` from it -- and a page with no pixel witness falls to this lane's
    own fail-closed ending rather than to the layer.

    Round 2 (Argon P1-1) corrects a mistake round 1 made here. Round 1 removed the
    guard on the deferral below, reasoning that the orchestrator derives
    ``native_trusted`` from the SAME predicate, so a suspect page always arrives with
    ``native_trusted=False`` and the deferral is unreachable. That is true only of the
    orchestrator's non-raising path: ``SourceEvidenceTableJudge`` sets ``trusted=None``
    when its own trust source RAISES or is absent, and ``None`` is "not False" -- so the
    page deferred to the native verifier with its suspect layer intact, which is the one
    ending this function exists to deny it. Measured: ``native_trusted=None`` plus
    ``layer_is_evidence=False`` returned ``passed=True, deferred=True`` for the layer's
    own reading. Every arm that consults the layer now reads this flag: the deferral,
    the born-digital bundle, and ``include_text_layer`` on the scanned bundle.
    """
    if native_trusted is not False and layer_is_evidence and page_has_native_words(page):
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
        # #1205: a suspect layer is excluded on BOTH arms -- the raster arm already
        # excluded it, and the born-digital arm is the hole Argon found.
        if page_has_page_covering_raster(page) or not layer_is_evidence:
            bundle = build_scanned_evidence(
                page,
                ocr_image_fn=ocr_image_fn,
                include_text_layer=False,
            )
            pixel_only = True
        else:
            bundle = build_born_digital_evidence(page, ocr_image_fn=ocr_image_fn)
            pixel_only = False
    else:
        pixel_only = not (native_trusted is not False and layer_is_evidence)
        bundle = build_scanned_evidence(
            page,
            ocr_image_fn=ocr_image_fn,
            include_text_layer=native_trusted is not False and layer_is_evidence,
            text_layer_from_words=False,
        )
    # #1355: a second reading only stands in for a LOSSY RASTER witness. The pixel-only bundle is
    # one throughout; the born-digital bundle is one PER TOKEN (its ``pixel_numeric`` /
    # ``layer_numeric`` split); the scanned bundle that folds the text layer into ``numeric`` is
    # not one.
    corroborates = pixel_only or bundle.pixel_numeric is not None
    bundle, extra_dpis, resolved = widen_scanned_evidence(
        page, bundle, tokens, ocr_image_fn=ocr_image_fn
    )
    result = verify_table_tokens(
        bundle,
        tokens,
        second_reading=second_reading if corroborates else None,
        engine=engine,
    )
    if extra_dpis:
        result = replace(result, extra_resolutions=extra_dpis, extra_resolution_tokens=resolved)
    if layer_witness and pixel_only and not layer_is_evidence:
        result = _apply_layer_witness(
            page,
            output_text,
            tokens,
            bundle,
            result,
            chart_page=chart_page,
        )
    return result


def _apply_layer_witness(
    page,
    output_text: str,
    tokens: TableTokens,
    bundle: SourceEvidenceBundle,
    result: SourceEvidenceResult,
    *,
    chart_page: bool | None,
) -> SourceEvidenceResult:
    """#1399: decide a table on a scanned page whose layer is excluded from the evidence, using
    that layer as the table's witness. *result* is what the tesseract-only check said.

    Pre-gates (a cut-off reading, a chart page) and B or C (a mark conflict, a format outlier)
    fail the table whatever tesseract said. Otherwise:

    * A passes, D empty, a crop located: SUCCESS (nothing unproven) when tesseract's own check
      also passed. Where it did not, the layer alone confirms only under
      ``layer_witness.LAYER_CONFIRMS_ALONE`` (the open owner ruling, default off): else the table
      ships marked.
    * A passes but D is non-empty, or no crop was located: ships marked (``unproven``).
    * A fails: tesseract's own verdict stands; when it passed the table, the layer does not
      confirm it, so it ships marked instead of unmarked (never SUCCESS).

    A page whose layer holds no numbers is not touched, and neither is one the image check never
    read (``bundle.has_content_evidence`` False: no backend, a crashed reader, an unrenderable
    page). That ending stays fail-closed as before (#658): the layer was never allowed to stand in
    for a missing reader.
    """
    if not bundle.has_content_evidence:
        return result
    from socr.tables.layer_witness import judge_against_layer, page_is_chart, table_layer_witness

    witness = table_layer_witness(page)
    if witness is None or not tokens.has_numeric:
        return result
    if chart_page is None:
        chart_page = page_is_chart(page)
    check_text = output_text
    verdict = judge_against_layer(
        check_text,
        tokens,
        witness,
        tesseract_tokens=frozenset(bundle.numeric) | bundle.rung_tokens,
        chart_page=chart_page,
    )
    if verdict.pregate:
        return SourceEvidenceResult(
            verifiable=False,
            passed=False,
            reason=verdict.pregate,
            extra_resolutions=result.extra_resolutions,
            extra_resolution_tokens=result.extra_resolution_tokens,
        )
    if verdict.disputed:
        return SourceEvidenceResult(
            verifiable=True,
            passed=False,
            reason=(
                "table numbers disputed by the stored text layer or a format outlier: "
                f"{list(verdict.disputed)[:5]}"
            ),
            disputed_values=verdict.disputed,
            extra_resolutions=result.extra_resolutions,
            extra_resolution_tokens=result.extra_resolution_tokens,
        )
    if not verdict.a_passes:
        if not (result.verifiable and result.passed):
            return result
        missing = tuple(sorted(verdict.a_missing.elements()))
        return replace(
            result,
            unproven=(
                f"the stored text layer does not hold these table numbers: {list(missing)[:5]}"
            ),
            unproven_values=missing,
        )
    # A passes. The labels were never the layer's to confirm; keep tesseract's word on them.
    content_unverified = result.content_unverified
    if not (result.verifiable and result.passed):
        if tokens.content and not _content_supported(tokens.content, bundle.content):
            missing_labels = sorted(tokens.content - bundle.content)[:5]
            content_unverified = f"content labels unverified by page evidence: {missing_labels}"
    base = SourceEvidenceResult(
        verifiable=True,
        passed=True,
        reason="the stored text layer confirms every table number",
        content_unverified=content_unverified,
        extra_resolutions=result.extra_resolutions,
        extra_resolution_tokens=result.extra_resolution_tokens,
    )
    from socr.tables import layer_witness as _lw

    if verdict.crops_found and not verdict.d:
        if (result.verifiable and result.passed) or _lw.LAYER_CONFIRMS_ALONE:
            return base
        return replace(
            base,
            unproven=(
                "the stored text layer agrees with every table number but the image check "
                "could not confirm them: " + result.reason
            ),
        )
    if not verdict.crops_found:
        why = "no table crop was located, so the stored layer's numbers could not be scoped"
        missing_layer: tuple[str, ...] = ()
    else:
        missing_layer = tuple(sorted(verdict.d.elements()))
        why = (
            "the stored text layer holds numbers in the table crop the reading lacks: "
            f"{list(missing_layer)[:5]}"
        )
    return replace(base, unproven=why, unproven_values=missing_layer)
