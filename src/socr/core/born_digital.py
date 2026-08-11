"""Born-digital PDF detection and native text extraction.

Detects pages with genuine native text layers (born-digital) vs scanned pages
that may have low-quality baked-in OCR. Born-digital pages can skip OCR entirely
and use the extracted text directly.

Heuristics for distinguishing born-digital from baked-in OCR:
  - Text density: born-digital pages have consistent, dense text relative to
    page area; garbage OCR layers tend to be sparse or garbled.
  - Character quality: genuine text has normal word-length distributions and
    low ratios of non-ASCII/garbage characters.
  - Font consistency: born-digital PDFs embed proper fonts; scanned PDFs with
    OCR layers often have no real font info or use a single "invisible" font.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import fitz

logger = logging.getLogger(__name__)

# Font families used exclusively for typeset mathematics in LaTeX/LuaTeX/XeTeX.
# These appear in basefont names regardless of whether PyMuPDF's text extraction
# still looks like math — which it often doesn't (subscripts flattened, Greek
# dropped, reading order broken).  A single match means the page has real math.
#
# Uses re.search (not re.match) to handle subset-prefixed names like
# "ABCDEF+CMMI10".
_MATH_FONT_RE = re.compile(
    r"(?i)(CMMI|CMSY|CMEX|MSAM|MSBM|"  # Computer Modern + AMS math
    r"STIXMath|STIXSize|STIXNonUnicode|XITSMath|LatinModernMath|LMMath|"  # STIX/OpenType math
    r"AsanaMath|LibertinusMath|CambriaMath|NewCMMath|"  # other OTF math families
    r"Euler|rsfs)"  # Euler script, RSFS (calligraphic)
)

# Math typeset with a broken font/ToUnicode map decodes operators and delimiters
# as Latin-1 letters and fraction glyphs: '=' -> '¼', '(' -> 'ð', ')' -> 'Þ',
# '+' -> 'þ', '-' -> a C0 control char. These glyphs do not occur in clean
# English or math prose, so even a few per line are a near-perfect signal of
# corrupted math — which the prose-oriented `_encoding_corruption_ratio`
# (mid-caps / slash-digits / run-on words) misses entirely. Greek (σ, ρ) and real
# typography (en-dash, U+2212 minus, ﬁ ligature, curly quotes) are deliberately
# NOT in this set: they are legitimate and must never trip math-corruption.
_MATH_MOJIBAKE_CHARS = frozenset("ðÞþÐ¼½¾")
# C0 control characters (excluding tab/newline/CR) — '-' and other operators
# decode to these on this font class.
_C0_CONTROL_CHARS = frozenset(chr(c) for c in range(0x20) if c not in (0x09, 0x0A, 0x0D))
_MATH_CORRUPTION_GLYPHS = _MATH_MOJIBAKE_CHARS | _C0_CONTROL_CHARS


def line_has_corrupt_math(line: str) -> bool:
    """True if a line contains font-map mojibake standing in for math symbols."""
    return any(ch in _MATH_CORRUPTION_GLYPHS for ch in line)


#: Left-to-right text. Exempt from relegation on every page — see
#: ``block_is_page_furniture``.
_HORIZONTAL = (1.0, 0.0)

#: Decimal places used when comparing PyMuPDF line directions. They are unit
#: vectors, so three places distinguish the axis-aligned cases (1,0) / (0,1) /
#: (0,-1) while absorbing float noise in text set at a slight angle.
_DIR_PRECISION = 3


def dominant_text_direction(blocks: list) -> tuple[float, float]:
    """The page's prevailing writing direction, from its own line dirs.

    Derived from the page, never assumed to be horizontal: a rotated scan or a
    CJK vertical layout has a different prevailing direction, and hard-coding
    ``(1, 0)`` would classify every line on such a page as furniture.
    """
    tally: dict[tuple[float, float], int] = {}
    for block in blocks:
        for line in block.get("lines", []) or []:
            d = line.get("dir", (1.0, 0.0))
            key = (round(float(d[0]), _DIR_PRECISION), round(float(d[1]), _DIR_PRECISION))
            tally[key] = tally.get(key, 0) + 1
    if not tally:
        return (1.0, 0.0)
    return max(tally.items(), key=lambda kv: kv[1])[0]


def block_is_page_furniture(block: dict, dominant: tuple[float, float]) -> bool:
    """True when a block runs against the page's prevailing text direction.

    Publisher stamps ride the page margin rotated 90 degrees — "Downloaded from
    https://academic.oup.com/..." on every page of a JSTOR/OUP download. Spliced
    into prose they wreck its reading order.

    Callers RELEGATE such blocks to the end of the page; they must not delete
    them. Rotated text is not proof of worthlessness — a chart's axis labels and a
    rotated table header are also counter-directional, and are content. Direction
    is evidence about layout only (#145 review).

    Direction is the discriminator because it is a fact about the block rather
    than a guess about its words: matching on "Downloaded from" would be a
    publisher-specific string list that rots, while a stamp set at 90 degrees to
    the body text is structurally distinguishable on any page, in any language.

    A block with no directional lines is NOT furniture — absence of evidence must
    not delete text (#145).

    HORIZONTAL TEXT IS NEVER FURNITURE, even when it is the minority direction on
    its page. A page whose table is set landscape has a ROTATED dominant
    direction, and without this exemption the rule deletes the one horizontal
    block on it — which on the reference document is the running head carrying
    the page number, and a page number is citable content. Suppressing a stray
    publisher stamp is worth little; deleting a page number is precisely the loss
    this module exists to prevent, so the asymmetry is deliberate.
    """
    dirs = [
        (
            round(float(line.get("dir", dominant)[0]), _DIR_PRECISION),
            round(float(line.get("dir", dominant)[1]), _DIR_PRECISION),
        )
        for line in block.get("lines", []) or []
    ]
    if not dirs:
        return False
    return all(d != dominant and d != _HORIZONTAL for d in dirs)


#: Word tokens used to decide whether a region's markdown already contains a
#: line. Letters and numbers only: punctuation and the markdown table scaffolding
#: (pipes, dashes) must not count as content.
_WORD_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+(?:\.\d+)?")

#: Fraction of a text block that must lie inside a table region before the
#: region is treated as already representing it. Set at the midpoint because that
#: is what "the region contains this block" means — most of the block is inside
#: it. Not tuned: anything above 0.5 is a block the region genuinely covers, and
#: anything below is a block that merely brushes the region's edge. The old
#: behaviour was effectively a threshold of zero, which deleted a paragraph over
#: a single point of contact (#145).
_REGION_COVERAGE_DROP = 0.5


def _region_token_index(regions) -> Counter:
    """Multiset of word tokens across every region's markdown replacement."""
    idx: Counter = Counter()
    for _rect, md in regions:
        idx.update(_WORD_TOKEN_RE.findall(md or ""))
    return idx


def _line_is_in_region_text(line: dict, region_tokens: Counter) -> bool:
    """True when a region's markdown already contains this line's words.

    The test for "already represented", replacing the geometric one. A line whose
    every token appears in the replacement text is safe to drop; a line carrying
    even one token the replacement lacks is content that would be lost, and is
    kept — duplicated at worst.

    A line with no word tokens (pure punctuation, a rule) is droppable: it
    carries nothing to lose.
    """
    text = "".join(s.get("text", "") for s in line.get("spans", []) or [])
    tokens = _WORD_TOKEN_RE.findall(text)
    if not tokens:
        return True
    return all(region_tokens.get(t, 0) > 0 for t in tokens)


def _rect_coverage(inner, outer) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer`` (0.0-1.0)."""
    area = inner.get_area()
    if area <= 0:
        return 0.0
    overlap = inner & outer
    if not overlap.is_valid or overlap.is_empty:
        return 0.0
    return overlap.get_area() / area


# A slash that BEGINS a token and is immediately followed by a digit: "(/997)",
# "/55-84", "pp. /23". This is the eaten-leading-digit signature — a stroke glyph
# decoded as '/' where a digit belongs, so "(1997)" ships as "(/997)".
#
# Deliberately narrower than the ratio's `/\d|\d/|\(/`. That broader pattern is
# correct for a DENSITY score, where a few false hits wash out, but it fires on
# text this corpus is full of — "1/2", "2019/20", "12/31/2024", "9/11",
# "example.com/2024" — and so cannot drive an absolute-count gate. Requiring the
# slash to start the token excludes every one of those (the slash there is
# preceded by a digit or letter) while still catching both the eaten year and the
# eaten page range.
_DIGIT_CORRUPTION_RE = re.compile(r"(?<![0-9A-Za-z])/[0-9]")


def count_digit_corruption(text: str) -> int:
    """Number of eaten-leading-digit occurrences in ``text``.

    An ABSOLUTE count, never a ratio: the corpus invariant is per-token ("a wrong
    number is worse than a missing one"), so page length must not dilute it. Two
    destroyed citation years are two destroyed citation years whether the page
    carries 500 words or 5,000 (#136).
    """
    return len(_DIGIT_CORRUPTION_RE.findall(text or ""))


# Semantically-empty characters that publishers inject into the text layer for
# justification / line-breaking. They carry no content but corrupt downstream use:
# the soft hyphen splits words so search misses them ("hetero\xadgeneous" != the
# query), and zero-width spaces wedge themselves around sub/superscripts so math
# reads as "c​i,t". Native text bypasses ``OutputNormalizer`` (it is set
# straight from the PDF text layer), so it must be cleaned HERE, at the extraction
# boundary, or the junk ships unaltered.
_ZERO_WIDTH_STRIP = frozenset({chr(0x200B), chr(0x00AD)})  # ZWSP, soft hyphen
# Non-breaking / figure / thin / hair / narrow spaces. Collapsed to U+0020 (not
# stripped): they ARE spaces, just typographic variants that break search and
# token splitting. Plain ASCII space is the portable form for Markdown prose.
_EXOTIC_SPACES = frozenset(
    chr(c) for c in (0x00A0, 0x2002, 0x2003, 0x2007, 0x2008, 0x2009, 0x200A, 0x202F)
)


def clean_native_text(text: str) -> tuple[str, int, int]:
    """Strip semantically-empty invisibles and normalize exotic spaces.

    Returns ``(cleaned, n_stripped, n_spaces_normalized)``. Lossless for content:
    only removes zero-width chars (ZWSP, soft hyphen) and collapses typographic
    spaces to U+0020. Does NOT touch Private Use Area glyphs — those are a content
    signal (see :func:`count_pua_chars`) and are flagged, never silently altered.
    Deliberately not NFKC: that would fold math alphanumerics (𝒯, superscripts)
    into ASCII and create a NEW silent math corruption.
    """
    stripped = 0
    spaced = 0
    out: list[str] = []
    for ch in text:
        if ch in _ZERO_WIDTH_STRIP:
            stripped += 1
        elif ch in _EXOTIC_SPACES:
            spaced += 1
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out), stripped, spaced


def count_pua_chars(text: str) -> int:
    """Count Unicode Private Use Area codepoints in a text layer.

    A born-digital text layer should never contain PUA characters: they are
    font-private glyphs (math symbols the font maps without a real ToUnicode
    entry — e.g. UniMath U+F766 = script-T, STIXNonUnicode U+E14B). They render
    correctly ONLY inside the embedding font and become boxes or vanish anywhere
    else, so their presence is a direct fingerprint of weak/broken ToUnicode and
    hence silent math-glyph loss. Covers the BMP PUA and both supplementary PUA-A/B
    planes.
    """
    return sum(
        1
        for ch in text
        if 0xE000 <= ord(ch) <= 0xF8FF
        or 0xF0000 <= ord(ch) <= 0xFFFFD
        or 0x100000 <= ord(ch) <= 0x10FFFD
    )


@dataclass
class PageAssessment:
    """Per-page born-digital assessment."""

    page_num: int  # 1-indexed
    is_born_digital: bool
    native_text: str
    confidence: float  # 0.0 to 1.0
    char_count: int = 0
    word_count: int = 0
    font_count: int = 0
    has_images: bool = False
    has_tables: bool = False  # page contains table-like structures
    has_figures: bool = False  # page contains embedded images (alias for has_images)
    has_equations: bool = False  # page contains math/equations
    needs_ocr_enhancement: bool = False  # native layer has a known deficiency
    has_corrupt_math: bool = False  # font-map mojibake in math (needs region OCR -> LaTeX)
    has_unmapped_math_glyphs: bool = False  # PUA glyphs (weak ToUnicode) -> silent math-glyph loss
    has_unverifiable_table_region: bool = False  # TR-3: per-region geometry hard-fail
    #: #136: mid-band encoding corruption of the COSMETIC class (lost spaces, fused
    #: words) — the page is trustworthy for content but its text layer is suspect.
    #: Propagated to PageState so the agentic native lane can emit a durable audit
    #: event; the historical ``notes`` entry alone reached nothing that ships.
    has_encoding_hygiene_suspect: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class DocumentAssessment:
    """Document-level born-digital assessment."""

    path: Path
    pages: list[PageAssessment]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def born_digital_count(self) -> int:
        return sum(1 for p in self.pages if p.is_born_digital)

    @property
    def scanned_count(self) -> int:
        return self.page_count - self.born_digital_count

    @property
    def is_fully_born_digital(self) -> bool:
        return all(p.is_born_digital for p in self.pages)

    @property
    def is_fully_scanned(self) -> bool:
        return not any(p.is_born_digital for p in self.pages)

    @property
    def is_mixed(self) -> bool:
        return not self.is_fully_born_digital and not self.is_fully_scanned

    def born_digital_pages(self) -> list[int]:
        """Return 1-indexed page numbers of born-digital pages."""
        return [p.page_num for p in self.pages if p.is_born_digital]

    def scanned_pages(self) -> list[int]:
        """Return 1-indexed page numbers of scanned pages."""
        return [p.page_num for p in self.pages if not p.is_born_digital]


def _is_lane_stacked(table: object) -> bool:
    """True when a ``find_tables()`` result looks lane-stacked (whitespace-gutter collapse).

    A whitespace-gutter table (no ruling lines, CE forecaster-grid style) may be
    returned by ``find_tables()`` with multiple values newline-stacked inside a
    single cell — the "lines" strategy collapses the whitespace-separated columns.
    Detecting this lets ``extract_structured`` route the region to the
    word-geometry rowizer instead of the passthrough ``_table_to_markdown`` path.

    The check is purely structural: any cell containing an embedded newline
    signals that the extractor concatenated what should be separate column values.
    Never raises.
    """
    try:
        rows = table.extract()
    except Exception:
        return False
    for row in rows or []:
        for cell in row or []:
            if cell and isinstance(cell, str) and "\n" in cell:
                return True
    return False


class BornDigitalDetector:
    """Detect born-digital pages and extract native text from PDFs.

    A page is considered born-digital if it has a text layer that looks like
    genuine authored text rather than a baked-in OCR layer. The detector uses
    multiple signals: text density, character quality, font diversity, and
    the presence/absence of embedded images.
    """

    # Minimum characters for a page to be considered as having meaningful text.
    # A typical academic paper page has 2000-4000 chars. Very short text layers
    # are likely artifacts or watermarks rather than genuine content.
    MIN_CHARS_FOR_TEXT_LAYER = 50

    # Minimum words per page for born-digital classification when text quality is
    # *unknown* (i.e., before quality checks run).  Pages with fewer words than
    # this AND poor quality signals are classified as scanned.
    #
    # For *clean* text (no CID artifacts, low garbage ratio, normal word lengths,
    # at least one proper font), this threshold is NOT enforced — a sparse born-
    # digital page (chapter divider, figure caption, section heading) should be
    # classified as born-digital even with fewer words.  Only the absolute floor
    # MIN_WORDS_SPARSE is enforced for clean text.
    #
    # Basis: 15 was historically calibrated for detecting baked-in OCR layers
    # (garbage OCR tends to produce many single-char "words", so a low absolute
    # count was a useful proxy).  That signal is now captured directly by the
    # quality checks (garbage ratio, word-length distribution, CID artifacts).
    MIN_WORDS_PER_PAGE = 15

    # Absolute minimum words to trust as a real native text layer, even when
    # all quality checks pass.  Below this, the text is too thin to extract
    # meaningful content (single page numbers, watermarks, stray characters).
    #
    # Basis: a figure caption or section heading is "Figure 1." or "Chapter 3"
    # at minimum — at least 2 tokens.  We use 3 to require at minimum a brief
    # phrase (number + noun + adjective) rather than a lone label or watermark.
    # Three words is a loose floor; the quality checks do the real discrimination.
    MIN_WORDS_SPARSE = 3

    # Fraction of page area covered by embedded raster images above which a page
    # is considered image-dominant.  Used to gate the clean-short-text exception
    # (GH-35-FU, consilium decision id 20260615T104828Z-1577):
    #
    # A scanned page + baked-in OCR layer looks indistinguishable from a
    # born-digital figure page with a caption when the text layer is short and
    # clean.  But on a scanned page the raster image covers the ENTIRE page,
    # while on a genuine born-digital figure page the chart/figure occupies a
    # substantial but clearly sub-total fraction (surrounding white-space, caption
    # area, header region all reduce image coverage well below 1.0).
    #
    # The panel converged on ~0.90–0.95.  We use 0.90 as the threshold:
    #   - A full-page scan rendered as a single raster hits ≥ 0.98 (image fills
    #     essentially the whole printable area after margins).
    #   - A born-digital figure page with a chart in the body and a caption at
    #     the top/bottom typically reaches 0.50–0.80 coverage, well below 0.90.
    #   - The margin of 0.10 absorbs scanner over-crop and image-rect rounding
    #     without encroaching on the legitimate figure-page population.
    #
    # Over-correcting (routing to OCR) is the safer direction here: an extra OCR
    # call on a true born-digital page is suboptimal but recoverable; silently
    # skipping OCR on a real scan causes permanent content loss.
    RASTER_DOMINANCE_RATIO = 0.90

    # Maximum ratio of garbage/non-printable characters. Born-digital text is
    # clean; OCR layers on scanned PDFs often contain (cid:XX) references,
    # replacement chars, and control characters.
    MAX_GARBAGE_RATIO = 0.05

    # Maximum ratio of characters that are just spaces. Baked-in OCR on scanned
    # pages often produces text with excessive spacing (one char per glyph with
    # spaces between).
    MAX_SPACE_RATIO = 0.60

    # Minimum average word length. Garbage OCR produces many single-char "words".
    MIN_AVG_WORD_LENGTH = 2.5

    # Maximum average word length. Garbled text can fuse characters into long
    # non-word strings.
    MAX_AVG_WORD_LENGTH = 20.0

    # Encoding-corruption ratio: a born-digital PDF can have a broken font/ToUnicode
    # map that yields VALID characters in wrong positions ("Journal"->"Joumal",
    # "1997"->"(/997)") — invisible to the garbage-char check. Signals: mid-word
    # capitals ("ofFinancial"), slash-for-digit ("(/997)"), run-on words. Calibrated
    # on real pages: clean prose ~0.000, a header-only glitch (Fama-French) ~0.019,
    # pervasive data corruption ~0.095.
    # FLAG threshold: record the page as suspect (visibility) but still trust native.
    ENCODING_CORRUPTION_FLAG = 0.01
    # ESCALATE threshold: native is untrustworthy page-wide -> route to OCR (read the
    # image, not the broken encoding). Set well above the header-only case so a clean
    # body is never sent to a slower/lossier VLM over a few bad header tokens.
    MAX_ENCODING_CORRUPTION = 0.05
    # #136: eaten-leading-digit occurrences that route the page to OCR, as an
    # ABSOLUTE count rather than a density. One is enough: a single "(/997)" is a
    # wrong publication year, which the corpus invariant ranks as worse than a
    # missing one, and no amount of surrounding clean prose makes it less wrong.
    MAX_DIGIT_CORRUPTION_HITS = 1

    def __init__(
        self,
        min_chars: int | None = None,
        min_words: int | None = None,
        max_garbage_ratio: float | None = None,
    ) -> None:
        if min_chars is not None:
            self.MIN_CHARS_FOR_TEXT_LAYER = min_chars
        if min_words is not None:
            self.MIN_WORDS_PER_PAGE = min_words
        if max_garbage_ratio is not None:
            self.MAX_GARBAGE_RATIO = max_garbage_ratio

    def detect(self, pdf_path: Path | str) -> DocumentAssessment:
        """Analyze all pages of a PDF for born-digital content.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            DocumentAssessment with per-page results.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pages: list[PageAssessment] = []
        with fitz.open(pdf_path) as doc:
            for page_idx in range(len(doc)):
                assessment = self._assess_page(doc[page_idx], page_idx + 1)
                pages.append(assessment)

        return DocumentAssessment(path=pdf_path, pages=pages)

    def detect_page(self, pdf_path: Path | str, page_num: int) -> PageAssessment:
        """Assess a single page (1-indexed).

        Args:
            pdf_path: Path to the PDF file.
            page_num: 1-indexed page number.

        Returns:
            PageAssessment for the requested page.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        with fitz.open(pdf_path) as doc:
            if page_num < 1 or page_num > len(doc):
                raise ValueError(f"Page {page_num} out of range (document has {len(doc)} pages)")
            return self._assess_page(doc[page_num - 1], page_num)

    def _assess_page(self, page: fitz.Page, page_num: int) -> PageAssessment:
        """Assess whether a single page is born-digital.

        Uses multiple signals to distinguish genuine born-digital text from
        low-quality baked-in OCR layers.
        """
        notes: list[str] = []

        # Extract raw text from the PDF text layer, then clean it at the boundary:
        # native text never passes through OutputNormalizer, so publisher-injected
        # zero-width spaces / soft hyphens / exotic spaces must be removed here or
        # they ship verbatim (broken search, unreadable math). PUA glyphs survive
        # the clean and are detected separately as a math-glyph-loss signal.
        raw_text = page.get_text("text")
        raw_text, _zw_stripped, _spaces_normalized = clean_native_text(raw_text)
        pua_count = count_pua_chars(raw_text)
        has_unmapped_math_glyphs = pua_count > 0
        char_count = len(raw_text)
        words = raw_text.split()
        word_count = len(words)

        # Count distinct fonts used on the page
        font_count = self._count_fonts(page)

        # Check for embedded images (raster content)
        has_images = self._has_images(page)

        # Detect structured content types
        has_tables = self._detect_tables(page)
        has_figures = has_images  # figures = embedded raster images
        has_corrupt_math = self._detect_corrupt_math(raw_text)
        has_equations = (
            self._detect_math_fonts(page) or self._detect_equations(raw_text) or has_corrupt_math
        )

        # --- Decision logic ---

        # No text layer at all: definitely scanned.  This is a hard gate — fewer
        # than MIN_CHARS_FOR_TEXT_LAYER characters means there is not enough
        # content to be useful even if the few chars are perfectly clean.
        if char_count < self.MIN_CHARS_FOR_TEXT_LAYER:
            notes.append(
                f"insufficient text layer ({char_count} chars < {self.MIN_CHARS_FOR_TEXT_LAYER})"
            )
            return PageAssessment(
                page_num=page_num,
                is_born_digital=False,
                native_text="",
                confidence=0.95,
                char_count=char_count,
                word_count=word_count,
                font_count=font_count,
                has_images=has_images,
                has_tables=has_tables,
                has_figures=has_figures,
                has_equations=has_equations,
                notes=notes,
            )

        # Compute text quality signals before the word-count check so that
        # "short but clean" (sparse born-digital pages: figure captions, chapter
        # headings, section dividers) can be rescued from the word-count gate.
        # All quality signals below are still checked regardless of word count.
        garbage_ratio = self._garbage_ratio(raw_text)
        space_ratio = raw_text.count(" ") / max(len(raw_text), 1)
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
        has_cid = bool(re.search(r"\(cid:\d+\)", raw_text))

        # A text layer is "clean" when ALL of the following hold:
        #   - no CID font-mapping artifacts (definitive sign of broken font map)
        #   - low garbage-character ratio (baked-in OCR on scans tends to be noisy)
        #   - normal average word length (garbled text fuses or fragments tokens)
        #   - at least one embedded font (pages with no fonts have no real text layer)
        #
        # Clean short text on a page with proper fonts is a genuine native layer
        # (figure caption, section heading, sparse table).  Such pages must NOT be
        # classified as scanned solely because they have fewer than MIN_WORDS_PER_PAGE
        # words — that would route them to OCR and potentially degrade the output.
        text_layer_is_clean = (
            not has_cid
            and garbage_ratio <= self.MAX_GARBAGE_RATIO
            and self.MIN_AVG_WORD_LENGTH <= avg_word_len <= self.MAX_AVG_WORD_LENGTH
            and font_count > 0
        )

        # Word-count gate.  Enforced unconditionally only when the text layer is
        # dirty (quality signals indicate baked-in OCR or no real layer).  When the
        # text is clean, only the absolute floor MIN_WORDS_SPARSE is applied — this
        # catches lone watermarks or page-number stubs that would extract as a single
        # token, not sparse-but-real content.
        if word_count < self.MIN_WORDS_PER_PAGE:
            if not text_layer_is_clean or word_count < self.MIN_WORDS_SPARSE:
                notes.append(
                    f"too few words ({word_count} < {self.MIN_WORDS_PER_PAGE})"
                    + ("" if text_layer_is_clean else "; dirty text layer")
                )
                return PageAssessment(
                    page_num=page_num,
                    is_born_digital=False,
                    native_text="",
                    confidence=0.85,
                    char_count=char_count,
                    word_count=word_count,
                    font_count=font_count,
                    has_images=has_images,
                    has_tables=has_tables,
                    has_figures=has_figures,
                    has_equations=has_equations,
                    notes=notes,
                )

            # Clean short text — but gate the pass-through by raster coverage.
            # A scanned page with a baked-in OCR caption is INDISTINGUISHABLE from
            # a born-digital figure page via text quality alone.  However, a scan
            # always fills nearly the full page with a raster image, while a genuine
            # born-digital figure page has the chart occupying a sub-total fraction
            # of the page area (caption, header, and white-space reduce coverage).
            # If the page is image-dominant (raster coverage >= RASTER_DOMINANCE_RATIO),
            # route to OCR to prevent permanent content loss on real scans.
            # Non-image-dominant sparse-but-clean pages still classify born-digital.
            # GH-35-FU | consilium decision id 20260615T104828Z-1577
            raster_cov = self._raster_coverage(page)
            if raster_cov >= self.RASTER_DOMINANCE_RATIO:
                notes.append(
                    f"sparse native layer ({word_count} words); image-dominant page "
                    f"({raster_cov:.1%} raster coverage >= {self.RASTER_DOMINANCE_RATIO:.0%} "
                    "threshold) — routing to OCR to avoid baked-in-OCR false-positive (GH-35-FU)"
                )
                return PageAssessment(
                    page_num=page_num,
                    is_born_digital=False,
                    native_text="",
                    confidence=0.85,
                    char_count=char_count,
                    word_count=word_count,
                    font_count=font_count,
                    has_images=has_images,
                    has_tables=has_tables,
                    has_figures=has_figures,
                    has_equations=has_equations,
                    notes=notes,
                )

            # Clean short text on a non-image-dominant page: pass through to
            # born-digital path with a note so the audit log can surface it.
            notes.append(
                f"sparse native layer ({word_count} words); clean text, "
                f"raster coverage {raster_cov:.1%} < {self.RASTER_DOMINANCE_RATIO:.0%}, "
                "classifying as born-digital"
            )

        # CID artifacts: definitive sign of broken font mapping on scanned PDF
        if has_cid:
            notes.append("CID font mapping artifacts detected")
            return PageAssessment(
                page_num=page_num,
                is_born_digital=False,
                native_text="",
                confidence=0.95,
                char_count=char_count,
                word_count=word_count,
                font_count=font_count,
                has_images=has_images,
                has_tables=has_tables,
                has_figures=has_figures,
                has_equations=has_equations,
                notes=notes,
            )

        # High garbage ratio: likely baked-in OCR with garbled output
        if garbage_ratio > self.MAX_GARBAGE_RATIO:
            notes.append(f"high garbage ratio ({garbage_ratio:.1%})")
            return PageAssessment(
                page_num=page_num,
                is_born_digital=False,
                native_text="",
                confidence=0.80,
                char_count=char_count,
                word_count=word_count,
                font_count=font_count,
                has_images=has_images,
                has_tables=has_tables,
                has_figures=has_figures,
                has_equations=has_equations,
                notes=notes,
            )

        # Excessive spacing: baked-in OCR often spaces out individual chars
        if space_ratio > self.MAX_SPACE_RATIO:
            notes.append(f"excessive spacing ({space_ratio:.1%})")
            return PageAssessment(
                page_num=page_num,
                is_born_digital=False,
                native_text="",
                confidence=0.75,
                char_count=char_count,
                word_count=word_count,
                font_count=font_count,
                has_images=has_images,
                has_tables=has_tables,
                has_figures=has_figures,
                has_equations=has_equations,
                notes=notes,
            )

        # Abnormal word lengths: garbled text fuses or fragments words
        if avg_word_len < self.MIN_AVG_WORD_LENGTH:
            notes.append(f"avg word length too short ({avg_word_len:.1f})")
            return PageAssessment(
                page_num=page_num,
                is_born_digital=False,
                native_text="",
                confidence=0.70,
                char_count=char_count,
                word_count=word_count,
                font_count=font_count,
                has_images=has_images,
                has_tables=has_tables,
                has_figures=has_figures,
                has_equations=has_equations,
                notes=notes,
            )

        if avg_word_len > self.MAX_AVG_WORD_LENGTH:
            notes.append(f"avg word length too long ({avg_word_len:.1f})")
            return PageAssessment(
                page_num=page_num,
                is_born_digital=False,
                native_text="",
                confidence=0.70,
                char_count=char_count,
                word_count=word_count,
                font_count=font_count,
                has_images=has_images,
                has_tables=has_tables,
                has_figures=has_figures,
                has_equations=has_equations,
                notes=notes,
            )

        # Encoding corruption: a broken font/ToUnicode map yields valid characters
        # in wrong positions, invisible to the garbage-char check. Pervasive
        # corruption means the text layer can't be trusted page-wide -> read the
        # image instead. (Mild, header-only glitches are flagged below, not escalated.)
        encoding_corruption = self._encoding_corruption_ratio(raw_text)
        # #136: eaten-leading-digit corruption is a WRONG NUMBER, not a hygiene
        # defect, so it is gated on absolute count and routed to OCR at the first
        # occurrence — the same remedy as pervasive corruption below, for the same
        # reason: OCR RECOVERS the true digit instead of merely confessing that one
        # was lost. Ratio gating cannot serve this class; the identical damage
        # scores 3% on a references page and 0.75% on a long prose page, and the
        # second ships under the flag floor with no signal at all.
        #
        # A false positive costs one extra OCR pass on a page (spend); a false
        # negative ships a wrong citation year as SUCCESS. For a citation corpus
        # that asymmetry justifies erring hot.
        digit_corruption = count_digit_corruption(raw_text)
        if digit_corruption >= self.MAX_DIGIT_CORRUPTION_HITS:
            notes.append(
                f"text-layer digit corruption ({digit_corruption} eaten-digit "
                "occurrence(s), e.g. '(/997)' for '(1997)') -> OCR"
            )
            return PageAssessment(
                page_num=page_num,
                is_born_digital=False,
                native_text="",
                confidence=0.75,
                char_count=char_count,
                word_count=word_count,
                font_count=font_count,
                has_images=has_images,
                has_tables=has_tables,
                has_figures=has_figures,
                has_equations=has_equations,
                notes=notes,
            )
        if encoding_corruption > self.MAX_ENCODING_CORRUPTION:
            notes.append(f"text-layer encoding corrupted ({encoding_corruption:.1%}) -> OCR")
            return PageAssessment(
                page_num=page_num,
                is_born_digital=False,
                native_text="",
                confidence=0.75,
                char_count=char_count,
                word_count=word_count,
                font_count=font_count,
                has_images=has_images,
                has_tables=has_tables,
                has_figures=has_figures,
                has_equations=has_equations,
                notes=notes,
            )

        # --- All checks passed: page is born-digital ---

        # Compute confidence based on signal strength
        confidence = self._compute_confidence(
            char_count=char_count,
            word_count=word_count,
            garbage_ratio=garbage_ratio,
            space_ratio=space_ratio,
            avg_word_len=avg_word_len,
            font_count=font_count,
            has_images=has_images,
        )

        # Complex content is metadata, not a routing decision. Clean native
        # tables are extracted from the PDF text layer, figures are handled by
        # the figure pass, and clean equations should not force whole-page OCR.
        # Only evidence that the native layer itself is deficient should request
        # enhancement.
        has_complex_content = has_tables or has_figures or has_equations
        needs_ocr_enhancement = has_corrupt_math

        # Flag mild encoding corruption (e.g. a broken header font) for visibility
        # without escalating: the body is still trustworthy, but the page is marked
        # suspect so it is never silently relied on.
        #
        # #136: "marked suspect" used to mean ONLY this note, and nothing in the
        # pipeline reads PageAssessment.notes — so the page shipped SUCCESS with no
        # surface anywhere. The flag below is what carries the mark into PageState,
        # the sidecar and the audit log. The digit class never reaches here; it was
        # routed to OCR above.
        encoding_hygiene_suspect = encoding_corruption > self.ENCODING_CORRUPTION_FLAG
        if encoding_hygiene_suspect:
            notes.append(f"text-layer encoding suspect ({encoding_corruption:.1%})")

        # TR-3: reset the per-extraction unverifiable flag so _assess_page can
        # read it after calling extract_structured.  Initialise to False so pages
        # without tables (which never call _verify_regions) are not flagged.
        self._last_extraction_had_unverifiable: bool = False

        if has_tables:
            # Use structured extraction that renders tables as markdown. Clean it
            # too: extract_structured builds its own text from the layer, so it
            # carries the same invisibles as raw_text and bypasses the boundary
            # clean above.
            native_text, _, _ = clean_native_text(self.extract_structured(page))
            notes.append("born-digital: structured extraction (tables detected)")
        else:
            native_text = raw_text.strip()
            notes.append("born-digital: clean text layer detected")

        if _zw_stripped or _spaces_normalized:
            notes.append(
                f"native layer cleaned: stripped {_zw_stripped} zero-width char(s), "
                f"normalized {_spaces_normalized} exotic space(s)"
            )

        # TR-3: read the per-region geometry hard-fail flag written by
        # extract_structured → _verify_regions during table extraction above.
        has_unverifiable_table_region = self._last_extraction_had_unverifiable

        if has_complex_content:
            content_types = []
            if has_tables:
                content_types.append("tables")
            if has_figures:
                content_types.append("figures")
            if has_equations:
                content_types.append("equations")
            notes.append(f"complex content detected ({', '.join(content_types)})")

        if has_corrupt_math:
            notes.append(
                f"corrupt math text layer ({self._count_math_corruption(raw_text)} mojibake "
                "glyphs); equation regions need image-OCR -> LaTeX"
            )

        # Unmapped math glyphs (PUA): the native prose is trustworthy, but math
        # symbols are font-private and lost outside the embedding font. We do NOT
        # set needs_ocr_enhancement here — that would route the whole page to OCR,
        # the lane empirically shown to make these pages worse (it falls back to the
        # same broken native layer). The flag surfaces via the audit log (and, when
        # equation recovery is enabled, the region-crop lane), never silently.
        if has_unmapped_math_glyphs:
            notes.append(
                f"unmapped math glyphs ({pua_count} private-use codepoint(s)); native prose "
                "trusted, math symbols are font-private (weak ToUnicode) -> need region OCR"
            )

        return PageAssessment(
            page_num=page_num,
            is_born_digital=True,
            native_text=native_text,
            confidence=confidence,
            char_count=char_count,
            word_count=word_count,
            font_count=font_count,
            has_images=has_images,
            has_tables=has_tables,
            has_figures=has_figures,
            has_equations=has_equations,
            needs_ocr_enhancement=needs_ocr_enhancement,
            has_corrupt_math=has_corrupt_math,
            has_unmapped_math_glyphs=has_unmapped_math_glyphs,
            has_unverifiable_table_region=has_unverifiable_table_region,
            has_encoding_hygiene_suspect=encoding_hygiene_suspect,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Content type detection
    # ------------------------------------------------------------------

    def _detect_tables(self, page: fitz.Page) -> bool:
        """Check if the page contains table-like structures.

        Two passes:
        1. PyMuPDF's built-in table detector (catches ruled/bordered tables).
        2. Lane-cooccupancy gate (``has_numeric_columns``) — catches borderless
           regression tables common in academic PDFs.  Replaces the old
           single-token-line-ratio heuristic (``_detect_columnar_numbers``)
           that false-fired on chart-axis labels and CE front-matter where many
           lines each carry a single x/y tick value.

        The lane-cooccupancy gate requires numeric tokens to form a genuine
        row × column grid (multiple distinct x-lanes co-occupied per data row),
        which chart axes never satisfy: tick labels run down a single x position,
        not across several independent lanes simultaneously.
        """
        try:
            tables = page.find_tables()
            if len(tables.tables) > 0:
                return True
        except Exception:
            pass

        from socr.tables.reconstruct import has_numeric_columns

        return has_numeric_columns(page)

    @staticmethod
    def _detect_math_fonts(page: fitz.Page) -> bool:
        """Detect mathematical content via PDF font metadata (source-side).

        PyMuPDF's text extraction silently mangles math typeset with Computer
        Modern, STIX, or similar math fonts: subscripts flatten, Greek letters
        drop, reading order breaks around equations.  The *extracted text*
        looks like normal prose to any string-based checker — so we must read
        the font names embedded in the PDF before extraction ever runs.

        A single match anywhere on the page is sufficient: even one inline math
        span (e.g., ``β̂`` mid-sentence) is enough to corrupt that passage.

        Uses ``page.get_fonts()`` only — it lists all fonts used on the page
        and is O(font_count), far cheaper than a full dict extraction.  Font
        subsets are handled transparently because we use ``re.search`` rather
        than ``re.match``, so ``ABCDEF+CMMI10`` still matches ``CMMI``.
        """
        try:
            for font in page.get_fonts():
                # font tuple: (xref, ext, type, basefont, name, encoding, ...)
                basefont = font[3] if len(font) > 3 else ""
                if basefont and _MATH_FONT_RE.search(basefont):
                    return True
        except Exception:
            pass
        return False

    # Minimum corrupt-math glyphs on a page before it is flagged. A couple of
    # stray control chars can occur incidentally; systematic math corruption
    # produces dozens (observed 5-145 per affected page on the test corpus).
    MIN_MATH_CORRUPTION_GLYPHS = 4

    @staticmethod
    def _count_math_corruption(text: str) -> int:
        """Count font-map mojibake glyphs standing in for math symbols."""
        return sum(1 for ch in text if ch in _MATH_CORRUPTION_GLYPHS)

    def _detect_corrupt_math(self, text: str) -> bool:
        """True if the page's math is font-map corrupted (operators/delimiters
        decoded as Latin-1 letters / fraction glyphs / control chars).

        Distinct from `_encoding_corruption_ratio`, which targets *prose*
        corruption (mid-caps, slash-digits, run-ons) and is blind to these math
        substitutions. The text layer is unrecoverable here, so a flagged page
        routes its equation regions to image-OCR -> LaTeX rather than being
        trusted or whole-page OCR'd.
        """
        return self._count_math_corruption(text) >= self.MIN_MATH_CORRUPTION_GLYPHS

    def _detect_equations(self, text: str) -> bool:
        """Check if text contains raw LaTeX math markup.

        This is a secondary fallback for PDFs that embed literal LaTeX strings
        (rare in typeset documents, but possible in some preprint formats).
        Font-based detection (_detect_math_fonts) is the primary signal and
        runs first.

        Inline dollar signs are common in non-math contexts (currency), so we
        require paired delimiters or explicit LaTeX commands.
        """
        if not text:
            return False

        # LaTeX math commands (high confidence)
        latex_commands = re.compile(
            r"\\(?:frac|sum|int|prod|lim|infty|partial|nabla|alpha|beta|gamma"
            r"|delta|epsilon|theta|lambda|sigma|omega|begin\{(?:equation|align"
            r"|gather|math|displaymath)\})"
        )
        if latex_commands.search(text):
            return True

        # Display math delimiters: $$ ... $$ or \[ ... \]
        if re.search(r"\$\$.+?\$\$", text, re.DOTALL):
            return True
        if re.search(r"\\\[.+?\\\]", text, re.DOTALL):
            return True

        return False

    # ------------------------------------------------------------------
    # Structured text extraction
    # ------------------------------------------------------------------

    def extract_structured(self, page: fitz.Page) -> str:
        """Extract text with tables rendered as markdown.

        For pages with tables, replaces table regions with markdown table
        representations while keeping surrounding prose as plain text.
        For pages without tables, returns plain text (same as get_text()).
        """
        try:
            tables_result = page.find_tables()
        except Exception:
            return page.get_text("text").strip()

        # Collect table bounding boxes and their markdown representations.
        # For each table returned by find_tables(), detect lane-stacking: a
        # whitespace-gutter table (no ruling lines, CE-style) may come back with
        # embedded newlines inside cells because the "lines" strategy collapses
        # whitespace-separated columns into one cell.  Such a region is
        # lane-stacked — route it to the word-geometry rowizer rather than the
        # passthrough _table_to_markdown path that preserves the embedded newlines.
        table_regions: list[tuple[fitz.Rect, str]] = []
        lane_stacked_regions: list[tuple[fitz.Rect, str]] = []
        for table in tables_result.tables:
            if _is_lane_stacked(table):
                # Route to word-geometry rowizer scoped to this region's words.
                # A multi-schema merge is TR-2's job: if rowize_from_word_list
                # cannot produce a valid single-schema grid from these words it
                # returns [] and we fall through to the whole-page fallbacks.
                from socr.tables.reconstruct import rowize_from_word_list

                bbox = fitz.Rect(table.bbox)
                region_words = [
                    w for w in page.get_text("words") if bbox.contains(fitz.Point(w[0], w[1]))
                ]
                rowized = rowize_from_word_list(region_words)
                for rect, md in rowized:
                    lane_stacked_regions.append((rect, md))
            else:
                md = self._table_to_markdown(table)
                if md:
                    table_regions.append((fitz.Rect(table.bbox), md))

        # Merge: lane-stacked replacements take priority; non-stacked tables keep
        # their passthrough markdown.  The rowizer may produce more or fewer
        # regions than the original find_tables() result — that is expected.
        table_regions.extend(lane_stacked_regions)

        # Born-digital booktabs tables (top/mid/bottom rules only) make the
        # default lines strategy return nothing, so the table would otherwise be
        # dumped as a flat token stream. Recover the grid from text alignment
        # (char-exact native values, no model). reconstruct_table_regions self-
        # gates on numeric-column structure, so it is safe to call on any page.
        if not table_regions:
            from socr.tables.reconstruct import reconstruct_table_regions

            table_regions = reconstruct_table_regions(page)

        # Word-geometry rowizer fallback (TR-1/TR-2): when find_tables() returns
        # nothing AND the text-strategy reconstruct also fails (e.g. a multi-region
        # page where the text strategy over-merges chart + table + prose into one
        # spanning grid that fails _looks_tabular), try segmenting the page by
        # vertical gaps derived from the page's own row-height distribution and
        # rowizing each segment independently.  This is the correct path for
        # CE-style whitespace-gutter tables on pages with multiple content regions.
        #
        # TR-2 chart-clip: use the chart-aware variant so chart-drawing clusters
        # are excluded from the rowizer and returned as image-ref placeholders.
        # The chart's tick-label rows (all single-column) previously diluted the
        # historical table's data_row_frac below _looks_tabular's 0.5 threshold,
        # causing it to fall back to flat text.  Clipping the chart words first
        # fixes this without any magic threshold.
        if not table_regions:
            from socr.tables.reconstruct import rowize_from_words_chart_aware

            page_num = getattr(page, "number", 0) + 1
            table_regions = rowize_from_words_chart_aware(page, page_num=page_num)

        if not table_regions:
            return page.get_text("text").strip()

        # Sort regions top-to-bottom by their y0 coordinate (reading order).
        # Column-aware ordering (x-band then y) would be needed for multi-column
        # layouts; for single-column pages (CE p.4 style) y0 alone is correct.
        # This is documented as a known limitation in the design note §1b.
        table_regions.sort(key=lambda tr: tr[0].y0)

        # TR-2/TR-3 per-region verifier scoping: verify each table region
        # independently against its own native numeric lanes.  The whole-page
        # verifier (verify_native_table) combines lanes from ALL tables on the
        # page, which causes false geometry_impossible_collapse hard-fails when
        # two tables with different schemas sit on the same page (e.g. a 4-col
        # forecaster grid + a 3-col historical table → 7 combined lanes but each
        # table only has 3-4).  Calling verify_native_table_region per region
        # fixes this by clipping get_text("words") to the region bbox first.
        # TR-3: _verify_regions now returns True when any region hard-fails so
        # _assess_page can flag the page for D3 fail-closed routing.
        had_unverifiable = self._verify_regions(page, table_regions)
        # Store on the instance so _assess_page can read it after the call to
        # extract_structured — the PageAssessment is built there, not here.
        self._last_extraction_had_unverifiable = had_unverifiable

        # TR-2 token-coverage post-check: every native numeric token must land in
        # exactly one region (no orphaned / double-counted token).  This is a
        # deterministic safety net — not a gate that suppresses output, just a
        # debug log so operators can trace lost tokens without re-running.
        self._check_token_coverage(page, table_regions)

        # Collect all region bboxes for the overlap-suppress check below.
        all_region_rects = [r for r, _ in table_regions]

        # Build output by interleaving prose text and region content.
        # Use text blocks from get_text("dict") to get position-aware text.
        try:
            page_dict = page.get_text("dict")
        except Exception:
            return page.get_text("text").strip()

        blocks = page_dict.get("blocks", [])
        output_parts: list[str] = []
        relegated_parts: list[str] = []
        region_idx = 0
        dominant = dominant_text_direction(blocks)

        for block in blocks:
            # Skip image blocks (type 1)
            if block.get("type", 0) == 1:
                continue

            # Step 0: text running against the page's prevailing direction is
            # RELEGATED to the end of the page, not deleted.
            #
            # The problem being solved is reading order, not the existence of the
            # text: a publisher stamp spliced into the middle of a paragraph
            # corrupts the prose around it. Relegation fixes that completely.
            #
            # Deleting it would ALSO discard legitimately rotated content — a
            # chart's axis labels, a rotated table header, a marginal note — and
            # this module must not delete text it cannot prove is worthless.
            # Direction is evidence about layout, never about value (#145 review).
            if block_is_page_furniture(block, dominant):
                for line in block.get("lines", []) or []:
                    text = "".join(s.get("text", "") for s in line.get("spans", []) or [])
                    if text.strip():
                        relegated_parts.append(text.strip())
                continue

            block_rect = fitz.Rect(block["bbox"])

            # Check if we need to insert a region before this block
            while region_idx < len(table_regions):
                region_rect, region_content = table_regions[region_idx]
                if region_rect.y0 <= block_rect.y0:
                    output_parts.append(f"\n{region_content}\n")
                    region_idx += 1
                else:
                    break

            # How much of this block does a table region already represent?
            #
            # This was `any(block_rect.intersects(r))` — true on ONE POINT of
            # contact, dropping the whole block. A table region routinely ends a
            # few points below its last rule, clipping the top line of the notes
            # paragraph beneath it; the entire paragraph then vanished from a page
            # that still shipped SUCCESS. On the reference page the notes block was
            # 23% inside the region and 77% outside, taking with it the sample
            # sizes and "robust standard errors are in parentheses" (#145).
            #
            # Coverage is measured PER REGION and the content test is run against
            # the tokens of the regions this block actually overlaps. A combined
            # index across every region on the page would let a line be deleted
            # because its words appear in a DIFFERENT table — which does not make
            # this region's replacement contain it.
            covering = [
                (r, md)
                for (r, md) in table_regions
                if _rect_coverage(block_rect, r) >= _REGION_COVERAGE_DROP
            ]

            lines = block.get("lines", []) or []
            if covering:
                # Mostly inside a region — but geometry alone must NOT decide. A
                # region's rectangle routinely spans text its markdown never
                # captured: the table's own title and column headers sit inside the
                # bbox, and an over-long region swallows the first lines of the
                # notes paragraph beneath the table. Dropping on position deleted
                # all of those.
                #
                # So drop a line only when the covering region's markdown
                # DEMONSTRABLY already contains it. Absence of the text in the
                # replacement is proof that dropping would lose it (#145).
                #
                # A block less than half inside any region keeps every line
                # untouched: the region does not represent it, so nothing about it
                # is a candidate for suppression.
                covering_tokens = _region_token_index(covering)
                lines = [ln for ln in lines if not _line_is_in_region_text(ln, covering_tokens)]

            for line in lines:
                spans = line.get("spans", [])
                line_text = "".join(s.get("text", "") for s in spans)
                if line_text.strip():
                    output_parts.append(line_text.strip())

        # Append any remaining regions that come after all text blocks
        while region_idx < len(table_regions):
            _, region_content = table_regions[region_idx]
            output_parts.append(f"\n{region_content}\n")
            region_idx += 1

        # Counter-directional text last, so it never interrupts the reading order
        # it was removed from — and never disappears either.
        output_parts.extend(relegated_parts)

        return "\n".join(output_parts).strip()

    def _verify_regions(
        self,
        page: fitz.Page,
        regions: list[tuple[fitz.Rect, str]],
    ) -> bool:
        """TR-2/TR-3 per-region native verifier.

        For each table region, clips get_text("words") to the region bbox and
        runs the two-tier geometry check against the region's own markdown text.
        This avoids false hard-fails that arise when whole-page lane counting
        merges columns from multiple tables with different schemas.

        Only regions whose text contains a markdown table separator are checked
        (image-asset placeholders like chart refs have no numeric lanes).

        Returns
        -------
        bool
            True if ANY table region produced a geometry hard-fail
            (``geometry_impossible_collapse``).  The caller stores this flag on
            the PageAssessment so the D3 fail-closed floor in
            ``_winning_page_output`` can route an unverifiable page to the
            image-asset lane instead of silently shipping the collapsed native.
        """
        from socr.tables.native_verifier import verify_native_table_region

        any_hard_fail = False
        for region_rect, region_text in regions:
            # Skip non-table regions (chart image refs, prose blocks)
            if "| --- |" not in region_text and "| --- " not in region_text:
                continue
            try:
                vr = verify_native_table_region(page, region_text, region_rect)
            except Exception as exc:
                logger.debug("per-region verifier failed for region %s: %s", region_rect, exc)
                continue

            if vr.hard_fail:
                any_hard_fail = True
                logger.warning(
                    "per-region verifier: geometry_impossible_collapse in region y=%.0f..%.0f — %s",
                    region_rect.y0,
                    region_rect.y1,
                    vr.reason,
                )
            elif vr.warn:
                logger.debug(
                    "per-region verifier: warn for region y=%.0f..%.0f — %s",
                    region_rect.y0,
                    region_rect.y1,
                    vr.reason,
                )
        return any_hard_fail

    def _check_token_coverage(
        self,
        page: fitz.Page,
        regions: list[tuple[fitz.Rect, str]],
    ) -> None:
        """TR-2 token-coverage post-check (diagnostic, not a gate).

        Verifies that every native numeric token on the page falls inside at
        most one table/chart region.  Orphaned tokens (in no region) and
        double-counted tokens (in two or more overlapping regions) are both
        logged at DEBUG level so operators can trace lost or duplicated values
        without re-running the pipeline.

        This is a deterministic safety net — it never suppresses output.  A
        future escalation path (TR-5 VLM confirm/split) can use these counts
        to decide when geometry-led segmentation has failed.
        """
        from socr.tables.reconstruct import _NUM_TOKEN_RE, _NUMERIC_RE

        try:
            words = page.get_text("words")
        except Exception:
            return

        # Collect all region bboxes.
        region_rects = [r for r, _ in regions]

        orphaned: list[str] = []
        double_counted: list[str] = []

        for w in words:
            x0, y0, _, _, text = w[0], w[1], w[2], w[3], w[4]
            if not (_NUM_TOKEN_RE.match(text) and _NUMERIC_RE.search(text)):
                continue

            hits = sum(1 for r in region_rects if r.x0 <= x0 <= r.x1 and r.y0 <= y0 <= r.y1)
            if hits == 0:
                orphaned.append(f"{text}@({x0:.0f},{y0:.0f})")
            elif hits > 1:
                double_counted.append(f"{text}@({x0:.0f},{y0:.0f})")

        if orphaned:
            logger.debug(
                "token-coverage: %d orphaned numeric token(s) not in any region: %s",
                len(orphaned),
                orphaned[:10],
            )
        if double_counted:
            logger.debug(
                "token-coverage: %d double-counted numeric token(s) in >1 region: %s",
                len(double_counted),
                double_counted[:10],
            )

    def _table_to_markdown(self, table: object) -> str:
        """Convert a PyMuPDF Table object to a markdown table string.

        Args:
            table: A table object from page.find_tables() with an
                   extract() method returning a list of rows (lists of cells).

        Returns:
            Markdown table string, or empty string if table is empty/invalid.
        """
        try:
            rows = table.extract()
        except Exception:
            return ""

        if not rows:
            return ""

        # Clean cell values: replace None with empty string, strip whitespace
        cleaned: list[list[str]] = []
        for row in rows:
            cleaned.append([(cell.strip() if isinstance(cell, str) else "") for cell in row])

        if not cleaned:
            return ""

        # Build markdown table
        col_count = max(len(row) for row in cleaned)
        # Pad rows to uniform column count
        for row in cleaned:
            while len(row) < col_count:
                row.append("")

        lines: list[str] = []

        # Header row
        header = cleaned[0]
        lines.append("| " + " | ".join(header) + " |")

        # Separator
        lines.append("| " + " | ".join("---" for _ in header) + " |")

        # Data rows
        for row in cleaned[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Low-level page analysis helpers
    # ------------------------------------------------------------------

    def _count_fonts(self, page: fitz.Page) -> int:
        """Count distinct fonts used on a page.

        Born-digital pages typically use 2-6 fonts (body, bold, italic, math).
        Scanned pages with OCR layers often use 0 or 1 font.
        """
        fonts = page.get_fonts()
        # Each font entry is (xref, ext, type, basefont, name, encoding)
        unique_names = {f[3] for f in fonts if f[3]}
        return len(unique_names)

    def _has_images(self, page: fitz.Page) -> bool:
        """Check if the page has embedded raster images.

        Scanned pages are essentially large images. Born-digital pages may
        have images too (figures), but combined with a rich text layer.
        """
        images = page.get_images()
        return len(images) > 0

    @staticmethod
    def _raster_coverage(page: fitz.Page) -> float:
        """Fraction of page area covered by embedded raster images (0.0 – 1.0).

        Uses ``page.get_image_info()`` which returns per-image bounding boxes in
        PDF user-space coordinates (same coordinate system as ``page.rect``).

        Area is computed as the SUM of individual image bbox areas, clamped to
        the page area, rather than a union bounding box.  Union would overcount
        for non-overlapping images (e.g. two small charts placed side-by-side
        would give a bounding box that covers the gap between them).  Summing
        individual areas slightly overcounts for overlapping images (rare in
        practice), but never exceeds 1.0 after clamping — still safe for a
        dominance threshold comparison.

        Returns 0.0 on any exception so that a buggy or unavailable API never
        wrongly gates an otherwise clean classification.
        """
        try:
            page_area = page.rect.get_area()
            if page_area <= 0:
                return 0.0
            image_area_sum = sum(
                fitz.Rect(info["bbox"]).get_area()
                for info in page.get_image_info()
                if info.get("bbox")
            )
            return min(image_area_sum / page_area, 1.0)
        except Exception:
            return 0.0

    def _encoding_corruption_ratio(self, text: str) -> float:
        """Fraction of word tokens that show font/ToUnicode encoding corruption.

        Catches the failure mode the garbage-char check misses: VALID characters
        in wrong positions from a broken font map. Three structural signals, none
        of which clean prose produces in quantity:

        - mid-word capital ("ofFinancial", "FrenchfJoumal") — a capital after a
          lowercase inside a token, from dropped inter-word spaces;
        - slash-for-digit ("(/997)", "/53") — a stroke glyph decoded as '/';
        - run-on tokens (>= 16 chars) — fused words.

        Returns 0.0 for fewer than 20 alpha tokens (too little to judge).
        """
        tokens = text.split()
        alpha = [t for t in tokens if any(c.isalpha() for c in t)]
        if len(alpha) < 20:
            return 0.0
        midcap = sum(1 for t in alpha if re.search(r"[a-z][A-Z]", t))
        slash_digit = sum(1 for t in tokens if re.search(r"/\d|\d/|\(/", t))
        run_on = sum(1 for t in alpha if len(t) >= 16)
        return (midcap + slash_digit + run_on) / len(alpha)

    def _garbage_ratio(self, text: str) -> float:
        """Ratio of garbage characters to total characters.

        Garbage = control chars, replacement chars, private-use-area chars,
        and other non-printable characters that shouldn't appear in real text.
        """
        if not text:
            return 0.0

        garbage_count = 0
        for ch in text:
            cp = ord(ch)
            # Control chars (except newline, tab, carriage return)
            if cp < 0x20 and cp not in (0x09, 0x0A, 0x0D):
                garbage_count += 1
            # Replacement character
            elif cp == 0xFFFD:
                garbage_count += 1
            # Private use area
            elif 0xE000 <= cp <= 0xF8FF:
                garbage_count += 1
            # Surrogates (should not appear in valid text)
            elif 0xD800 <= cp <= 0xDFFF:
                garbage_count += 1

        return garbage_count / len(text)

    def _compute_confidence(
        self,
        char_count: int,
        word_count: int,
        garbage_ratio: float,
        space_ratio: float,
        avg_word_len: float,
        font_count: int,
        has_images: bool,
    ) -> float:
        """Compute confidence score for born-digital classification.

        Starts at a base confidence and adjusts based on signal strength.
        More text, more fonts, lower garbage = higher confidence.
        """
        confidence = 0.80

        # More text = more confident
        if word_count > 100:
            confidence += 0.05
        if word_count > 300:
            confidence += 0.05

        # Very clean text = more confident
        if garbage_ratio < 0.01:
            confidence += 0.03

        # Multiple fonts = clearly authored, not OCR
        if font_count >= 2:
            confidence += 0.03
        if font_count >= 4:
            confidence += 0.02

        # Normal word lengths
        if 3.5 <= avg_word_len <= 7.0:
            confidence += 0.02

        return min(confidence, 1.0)
