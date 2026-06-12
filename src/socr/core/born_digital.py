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

import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz

# Font families used exclusively for typeset mathematics in LaTeX/LuaTeX/XeTeX.
# These appear in basefont names regardless of whether PyMuPDF's text extraction
# still looks like math — which it often doesn't (subscripts flattened, Greek
# dropped, reading order broken).  A single match means the page has real math.
#
# Uses re.search (not re.match) to handle subset-prefixed names like
# "ABCDEF+CMMI10".
_MATH_FONT_RE = re.compile(
    r"(?i)(CMMI|CMSY|CMEX|MSAM|MSBM|"  # Computer Modern + AMS math
    r"STIXMath|XITSMath|LatinModernMath|LMMath|"  # OpenType math (modern LaTeX)
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
    needs_ocr_enhancement: bool = False  # OCR preferred over native text for this page
    has_corrupt_math: bool = False  # font-map mojibake in math (needs region OCR -> LaTeX)
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

    # Minimum words per page for born-digital classification.
    # Single words or short phrases are likely headers/footers on scanned pages.
    MIN_WORDS_PER_PAGE = 15

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

        # Extract raw text from the PDF text layer
        raw_text = page.get_text("text")
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

        # No text layer at all: definitely scanned
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

        # Too few words: likely just headers, footers, or page numbers
        if word_count < self.MIN_WORDS_PER_PAGE:
            notes.append(f"too few words ({word_count} < {self.MIN_WORDS_PER_PAGE})")
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

        # Check text quality signals
        garbage_ratio = self._garbage_ratio(raw_text)
        space_ratio = raw_text.count(" ") / max(len(raw_text), 1)
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
        has_cid = bool(re.search(r"\(cid:\d+\)", raw_text))

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

        # For born-digital pages with complex content, use structured
        # extraction (markdown tables) instead of plain get_text().
        # Also flag that OCR is preferred for these pages.
        has_complex_content = has_tables or has_figures or has_equations
        needs_ocr_enhancement = has_complex_content

        # Flag mild encoding corruption (e.g. a broken header font) for visibility
        # without escalating: the body is still trustworthy, but the page is marked
        # suspect so it is never silently relied on.
        if encoding_corruption > self.ENCODING_CORRUPTION_FLAG:
            notes.append(f"text-layer encoding suspect ({encoding_corruption:.1%})")

        if has_tables:
            # Use structured extraction that renders tables as markdown
            native_text = self.extract_structured(page)
            notes.append("born-digital: structured extraction (tables detected)")
        else:
            native_text = raw_text.strip()
            notes.append("born-digital: clean text layer detected")

        if has_complex_content:
            content_types = []
            if has_tables:
                content_types.append("tables")
            if has_figures:
                content_types.append("figures")
            if has_equations:
                content_types.append("equations")
            notes.append(
                f"complex content detected ({', '.join(content_types)}); OCR enhancement preferred"
            )

        if has_corrupt_math:
            notes.append(
                f"corrupt math text layer ({self._count_math_corruption(raw_text)} mojibake "
                "glyphs); equation regions need image-OCR -> LaTeX"
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
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Content type detection
    # ------------------------------------------------------------------

    def _detect_tables(self, page: fitz.Page) -> bool:
        """Check if the page contains table-like structures.

        Two passes:
        1. PyMuPDF's built-in table detector (catches ruled/bordered tables).
        2. Columnar-numeric heuristic (catches borderless regression tables
           common in academic PDFs — whitespace-aligned columns of numbers
           that find_tables() cannot see).
        """
        try:
            tables = page.find_tables()
            if len(tables.tables) > 0:
                return True
        except Exception:
            pass

        return self._detect_columnar_numbers(page)

    @staticmethod
    def _detect_columnar_numbers(page: fitz.Page) -> bool:
        """Detect borderless tables via single-token line ratio.

        When PyMuPDF extracts text from a PDF table whose columns are
        whitespace-aligned (no drawn borders), each table cell lands on
        its own line as a single token.  Prose pages never exhibit this:
        a justified paragraph produces multi-word lines.

        Heuristic: if >50% of non-empty lines are single-token AND there
        are at least 15 such lines, the page is almost certainly tabular.
        The count floor avoids false-positives on short pages with headers
        or bullet lists.
        """
        lines = page.get_text("text").splitlines()
        nonempty = [l.strip() for l in lines if l.strip()]
        if not nonempty:
            return False
        single_token = sum(1 for l in nonempty if len(l.split()) == 1)
        return single_token >= 15 and single_token / len(nonempty) > 0.50

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

        # Collect table bounding boxes and their markdown representations
        table_regions: list[tuple[fitz.Rect, str]] = []
        for table in tables_result.tables:
            md = self._table_to_markdown(table)
            if md:
                table_regions.append((fitz.Rect(table.bbox), md))

        # Born-digital booktabs tables (top/mid/bottom rules only) make the
        # default lines strategy return nothing, so the table would otherwise be
        # dumped as a flat token stream. Recover the grid from text alignment
        # (char-exact native values, no model). reconstruct_table_regions self-
        # gates on numeric-column structure, so it is safe to call on any page.
        if not table_regions:
            from socr.tables.reconstruct import reconstruct_table_regions

            table_regions = reconstruct_table_regions(page)

        if not table_regions:
            return page.get_text("text").strip()

        # Sort table regions top-to-bottom by their y0 coordinate
        table_regions.sort(key=lambda tr: tr[0].y0)

        # Build output by interleaving prose text and markdown tables.
        # Use text blocks from get_text("dict") to get position-aware text.
        try:
            page_dict = page.get_text("dict")
        except Exception:
            return page.get_text("text").strip()

        blocks = page_dict.get("blocks", [])
        output_parts: list[str] = []
        table_idx = 0

        for block in blocks:
            # Skip image blocks (type 1)
            if block.get("type", 0) == 1:
                continue

            block_rect = fitz.Rect(block["bbox"])

            # Check if we need to insert a table before this block
            while table_idx < len(table_regions):
                table_rect, table_md = table_regions[table_idx]
                if table_rect.y0 <= block_rect.y0:
                    # Check if this text block overlaps the table region;
                    # if so, skip the text block (table markdown replaces it)
                    output_parts.append(f"\n{table_md}\n")
                    table_idx += 1
                else:
                    break

            # Check if this block overlaps any remaining table region
            overlaps_table = False
            for t_rect, _ in table_regions:
                if block_rect.intersects(t_rect):
                    overlaps_table = True
                    break

            if not overlaps_table:
                # Extract text from this block
                lines = block.get("lines", [])
                for line in lines:
                    spans = line.get("spans", [])
                    line_text = "".join(s.get("text", "") for s in spans)
                    if line_text.strip():
                        output_parts.append(line_text.strip())

        # Append any remaining tables that come after all text blocks
        while table_idx < len(table_regions):
            _, table_md = table_regions[table_idx]
            output_parts.append(f"\n{table_md}\n")
            table_idx += 1

        return "\n".join(output_parts).strip()

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
