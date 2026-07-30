"""Output normalizer for OCR engine markdown.

Strips engine-specific artifacts and normalizes markdown structure so
downstream processing gets consistent input regardless of which engine
produced the text.

Designed as a standalone module so it can be used by both BaseEngine
(CLI mode) and the HPC pipeline.
"""

import re
import unicodedata
from pathlib import Path

from socr.core.html_tables import clean_residual_html


class OutputNormalizer:
    """Normalize OCR engine output to consistent markdown.

    Usage::

        normalizer = OutputNormalizer()
        cleaned = normalizer.normalize(text, engine="deepseek")

    Engine-specific cleanup runs first, then generic normalization.
    """

    # --- engine-specific patterns (compiled once) ---

    # DeepSeek / GLM grounding tags
    _RE_REF_TAG = re.compile(r"<\|ref\|>.*?<\|/ref\|>")
    _RE_DET_TAG = re.compile(r"<\|det\|>\[\[.*?\]\]<\|/det\|>")
    _RE_SPECIAL_TOKEN = re.compile(r"<\|[^|]+\|>")
    _RE_BBOX = re.compile(r"\[\[\d+,\s*\d+,\s*\d+,\s*\d+\]\]")

    # Nougat LaTeX preamble
    _RE_LATEX_PREAMBLE = re.compile(
        r"^\\documentclass(?:\[.*?\])?\{.*?\}"
        r"(?:.*?\\begin\{document\})?",
        re.DOTALL,
    )
    _RE_LATEX_END = re.compile(r"\\end\{document\}\s*$")

    # Marker missing-page markers
    _RE_MISSING_PAGE = re.compile(r"\[MISSING_PAGE_POST(?::[\w]+)?\]")
    # Marker also emits [MISSING_PAGE_EMPTY:N]
    _RE_MISSING_PAGE_EMPTY = re.compile(r"\[MISSING_PAGE_EMPTY(?::[\w]+)?\]")

    # Mistral standalone header (without the metadata lines, which
    # BaseEngine._clean_output already handles when they're present)
    _RE_MISTRAL_HEADER = re.compile(r"^#\s*OCR Results\s*\n+")
    _RE_MISTRAL_META = re.compile(
        r"^\*\*(?:Original File|Full Path|Processed|Processing Time):\*\*[^\n]*\n?",
        re.MULTILINE,
    )

    # --- generic patterns ---
    # NOTE: the old blanket ``<[^>]+>`` tag-strip lived here. It flattened
    # DeepSeek/GLM HTML tables into fused digit-streams and deleted
    # inequality spans; HTML handling now lives in socr.core.html_tables.
    _RE_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)
    _RE_EXCESS_BLANK = re.compile(r"\n{3,}")

    # Markdown image references: ![alt](path)
    _RE_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]+\)")

    # VLM OCR engines emit generic placeholder paths when they "see" a figure
    # but have no real asset.  These must never survive into saved markdown —
    # a real extracted figure is always ``figures/figure_<N>_page<P>.png``.
    _VLM_PLACEHOLDER_BASENAMES: frozenset[str] = frozenset(
        {
            "image.png",
            "image.jpeg",
            "image.jpg",
            "image.gif",
            "image.webp",
            "image-url",
            "image_url",
            "imageurl",
            "img.png",
            "img.jpeg",
            "img.jpg",
            "placeholder.png",
            "figure.png",
        }
    )

    # Markdown fences wrapping entire output (VLMs often wrap in ```markdown...```)
    _RE_MD_FENCE = re.compile(
        r"^```(?:markdown|md|text|ocr)?\s*\n(.*?)```\s*$",
        re.DOTALL,
    )

    # Degenerate output patterns: repetition loops where the same line
    # repeats 5+ times consecutively (partial degenerate, can be trimmed)
    _RE_LINE_REPEAT = re.compile(r"^(.{20,})\n(?:\1\n){4,}", re.MULTILINE)

    # Unicode replacements: smart quotes, ligatures, etc.
    _UNICODE_MAP = {
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote
        "\u201c": '"',  # left double quote
        "\u201d": '"',  # right double quote
        "\u2013": "-",  # en dash
        "\u2014": "--",  # em dash
        "\u2026": "...",  # ellipsis
        "\ufb01": "fi",  # fi ligature
        "\ufb02": "fl",  # fl ligature
        "\ufb03": "ffi",  # ffi ligature
        "\ufb04": "ffl",  # ffl ligature
        "\ufb00": "ff",  # ff ligature
    }

    def normalize(self, text: str, engine: str = "") -> str:
        """Normalize OCR output text.

        Args:
            text: Raw OCR markdown output.
            engine: Engine name (e.g. "deepseek", "mistral"). Controls
                which engine-specific cleanups run. Empty string skips
                engine-specific passes.

        Returns:
            Cleaned, normalized markdown text.
        """
        if not text:
            return text

        # Engine-specific cleanup first
        engine_lower = engine.lower() if engine else ""
        if engine_lower in ("deepseek", "deepseek-vllm", "glm"):
            text = self._clean_deepseek_glm(text)
        elif engine_lower == "mistral":
            text = self._clean_mistral(text)
        elif engine_lower == "nougat":
            text = self._clean_nougat(text)
        elif engine_lower == "marker":
            text = self._clean_marker(text)

        # Generic normalization
        text = self._normalize_generic(text)
        return text

    # --- engine-specific cleaners ---

    def _clean_deepseek_glm(self, text: str) -> str:
        """Strip grounding tags/bboxes, then convert HTML structure (DeepSeek / GLM).

        DeepSeek-OCR emits tables as HTML. Tables are converted to markdown
        BEFORE any tag stripping (a blanket strip fuses adjacent cell digits
        into fabricated numbers), and only tag-shaped, known-HTML markup is
        removed — inequalities and literal ``<EOS>``-style tokens survive.
        """
        text = self._RE_REF_TAG.sub("", text)
        text = self._RE_DET_TAG.sub("", text)
        text = self._RE_SPECIAL_TOKEN.sub("", text)
        text = self._RE_BBOX.sub("", text)
        return clean_residual_html(text)

    def _clean_mistral(self, text: str) -> str:
        """Strip Mistral-specific header/meta lines."""
        text = self._RE_MISTRAL_HEADER.sub("", text)
        text = self._RE_MISTRAL_META.sub("", text)
        return text

    def _clean_nougat(self, text: str) -> str:
        """Strip LaTeX document preamble/postamble from Nougat output."""
        text = self._RE_LATEX_PREAMBLE.sub("", text)
        text = self._RE_LATEX_END.sub("", text)
        return text

    def _clean_marker(self, text: str) -> str:
        """Strip Marker [MISSING_PAGE_*] markers."""
        text = self._RE_MISSING_PAGE.sub("", text)
        text = self._RE_MISSING_PAGE_EMPTY.sub("", text)
        return text

    # --- phantom image stripping ---

    @classmethod
    def is_vlm_placeholder_path(cls, path_str: str) -> bool:
        """True when *path_str* is a known VLM sentinel, not a real asset."""
        cleaned = path_str.strip().strip("<>").split("?", 1)[0]
        basename = Path(cleaned).name.lower()
        return basename in cls._VLM_PLACEHOLDER_BASENAMES

    def text_has_vlm_image_placeholders(self, text: str) -> bool:
        """Return True if *text* contains a markdown image ref to a VLM sentinel."""
        if "![" not in text:
            return False
        for match in self._RE_MD_IMAGE.finditer(text):
            full = match.group(0)
            paren_start = full.rfind("(")
            path_str = full[paren_start + 1 : -1].strip()
            if self.is_vlm_placeholder_path(path_str):
                return True
        return False

    def strip_phantom_images(self, text: str, output_dir: Path | None = None) -> str:
        """Remove markdown image references that point to nonexistent files.

        OCR engines (Gemini, Mistral, etc.) emit ``![alt](path)`` references
        to images they "saw" during OCR but never saved to disk.  These
        phantom references clutter the output and confuse downstream tools.

        Args:
            text: Markdown text potentially containing image references.
            output_dir: If given, an image ref is kept only when the path
                resolves to an existing file relative to *output_dir*.
                When ``None``, **all** relative image refs are stripped
                (they are virtually always phantoms).

        Returns:
            Text with phantom image references removed.
        """
        if "![" not in text:
            return text

        def _should_strip(match: re.Match) -> bool:
            # Extract the path from ![alt](path)
            full = match.group(0)
            paren_start = full.rfind("(")
            path_str = full[paren_start + 1 : -1].strip()

            # Absolute URLs are never phantom (external images)
            if path_str.startswith(("http://", "https://", "data:")):
                return False

            # VLM sentinels are never real assets — strip even if a namesake file exists.
            if self.is_vlm_placeholder_path(path_str):
                return True

            # Absolute local paths — check existence directly
            p = Path(path_str)
            if p.is_absolute():
                return not p.exists()

            # Relative path — resolve against output_dir if available
            if output_dir is not None:
                return not (output_dir / p).exists()

            # No output_dir: strip all relative refs (phantom by default)
            return True

        def _replace(match: re.Match) -> str:
            if _should_strip(match):
                return ""
            return match.group(0)

        text = self._RE_MD_IMAGE.sub(_replace, text)
        # Clean up blank lines left behind
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    # --- generic normalization ---

    # Codepoints NFKC would fold into semantically different ASCII. These are
    # MEANING-BEARING in OCR'd math and must survive normalization:
    # superscripts/subscripts (x², ⁻¹), vulgar fractions (½), letterlike
    # symbols (ℝ, ℏ), and the math-alphanumeric block (𝛽, 𝒙).
    @staticmethod
    def _is_math_preserved(ch: str) -> bool:
        cp = ord(ch)
        return (
            cp in (0x00B2, 0x00B3, 0x00B9)  # ² ³ ¹
            or 0x00BC <= cp <= 0x00BE  # ¼ ½ ¾
            or 0x2070 <= cp <= 0x209F  # superscripts and subscripts block
            or 0x2100 <= cp <= 0x214F  # letterlike symbols (ℝ, ℵ, ℏ, ...)
            or 0x2150 <= cp <= 0x215E  # additional vulgar fractions
            or 0x1D400 <= cp <= 0x1D7FF  # mathematical alphanumeric symbols
        )

    @classmethod
    def _math_safe_nfkc(cls, text: str) -> str:
        """NFKC-normalize everything except math-meaning-bearing codepoints."""
        if not any(cls._is_math_preserved(ch) for ch in text):
            return unicodedata.normalize("NFKC", text)
        out: list[str] = []
        segment: list[str] = []
        for ch in text:
            if cls._is_math_preserved(ch):
                if segment:
                    out.append(unicodedata.normalize("NFKC", "".join(segment)))
                    segment.clear()
                out.append(ch)
            else:
                segment.append(ch)
        if segment:
            out.append(unicodedata.normalize("NFKC", "".join(segment)))
        return "".join(out)

    def _normalize_generic(self, text: str) -> str:
        """Apply generic markdown normalization."""
        # CRLF -> LF
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Strip markdown fences wrapping the entire output.
        # VLMs (Gemini, DeepSeek, etc.) often wrap output in ```markdown...```.
        # Noah's socOCRbench found a bug where "entire response stripped instead
        # of just the fence markers" — we only strip if the fence wraps everything.
        fence_match = self._RE_MD_FENCE.match(text.strip())
        if fence_match:
            text = fence_match.group(1)

        # Repair repetition loops: if the same line repeats 5+ times,
        # keep only one copy. This is a partial degenerate pattern that
        # can be salvaged rather than discarded entirely.
        text = self._RE_LINE_REPEAT.sub(r"\1\n", text)

        # Unicode NFKC normalization (handles compatibility chars), but
        # math-preserving: blanket NFKC folds superscripts (x² -> x2),
        # vulgar fractions (½ -> 1/2) and math-alphanumeric symbols into
        # ASCII, silently corrupting equations in EVERY engine's output.
        text = self._math_safe_nfkc(text)

        # Smart quotes / ligatures -> ASCII equivalents
        for src, dst in self._UNICODE_MAP.items():
            text = text.replace(src, dst)

        # Strip trailing whitespace per line
        text = self._RE_TRAILING_WS.sub("", text)

        # Collapse 3+ blank lines -> 2
        text = self._RE_EXCESS_BLANK.sub("\n\n", text)

        return text.strip()


# ----------------------------------------------------------------------
# GH-97: degenerate table-row repetition
# ----------------------------------------------------------------------
#
# ``OutputNormalizer._RE_LINE_REPEAT`` above already collapses a line that
# repeats 5+ times, but it is not sufficient for table rows, for two reasons:
#
# 1. It only matches lines of 20+ characters, so a narrow table's empty row
#    (``"| | | | | | | |"``, 15 chars) can never match it.
# 2. ``normalize()`` runs at the *engine* boundary (``engines/base.py``), before
#    dual-pass, header repair and table reconstruction. Degenerate rows those
#    stages introduce or preserve are never re-checked.
#
# The function below covers exactly that gap: table rows only, no width floor,
# applied at a post-assembly seam.
#
# Bound derivation (no magic number): measured over the OBR EFO November 2022
# reference document (68 pages, 15 carrying tables). 13 of those pages have a
# longest run of consecutive byte-identical table rows of exactly 1. The only
# two exceptions are both VLM degeneration, not content - a run of 15 (p48, a
# wholly contentless table) and 864 (p62, real content followed by 865 blank
# rows). Legitimate tables in that corpus never repeat a row at all, so a run of
# two is already anomalous. Two is kept as slack for genuine spacer rows in
# layouts outside this corpus; longer runs are treated as a runaway.
MAX_CONSECUTIVE_IDENTICAL_TABLE_ROWS = 2

_RE_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")


def collapse_repeated_table_rows(
    text: str,
    *,
    max_run: int = MAX_CONSECUTIVE_IDENTICAL_TABLE_ROWS,
) -> tuple[str, int]:
    """Truncate runaway repetition of byte-identical markdown table rows.

    Returns ``(text, rows_removed)``.

    Only *consecutive* identical table rows are collapsed, and only down to
    ``max_run`` surviving copies. Non-table lines are never touched - prose
    repetition remains ``OutputNormalizer``'s generic rule's business.

    Content-preserving by construction: every removed line is byte-identical to
    a line that is kept, so no cell value can be lost. That matters because this
    runs on a citation corpus where a dropped number is worse than a missing
    one.
    """
    if not text:
        return text, 0

    out: list[str] = []
    removed = 0
    prev: str | None = None
    kept_run = 0

    for line in text.split("\n"):
        is_row = bool(_RE_TABLE_ROW.match(line))
        if is_row and line == prev:
            kept_run += 1
            if kept_run > max_run:
                removed += 1
                continue
        else:
            kept_run = 1 if is_row else 0
            prev = line if is_row else None
        out.append(line)

    return "\n".join(out), removed
