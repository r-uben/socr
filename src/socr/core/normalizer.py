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
    _RE_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)
    _RE_EXCESS_BLANK = re.compile(r"\n{3,}")
    _RE_HTML_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
    _RE_HTML_TAG = re.compile(r"<[^>]+>")

    # Markdown image references: ![alt](path)
    _RE_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]+\)")

    # Unicode replacements: smart quotes, ligatures, etc.
    _UNICODE_MAP = {
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2013": "-",   # en dash
        "\u2014": "--",  # em dash
        "\u2026": "...", # ellipsis
        "\ufb01": "fi",  # fi ligature
        "\ufb02": "fl",  # fl ligature
        "\ufb03": "ffi", # ffi ligature
        "\ufb04": "ffl", # ffl ligature
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
        """Strip grounding tags and bounding boxes (DeepSeek / GLM)."""
        text = self._RE_REF_TAG.sub("", text)
        text = self._RE_DET_TAG.sub("", text)
        text = self._RE_SPECIAL_TOKEN.sub("", text)
        text = self._RE_BBOX.sub("", text)
        text = self._RE_HTML_BR.sub("\n", text)
        text = self._RE_HTML_TAG.sub("", text)
        return text

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

    def strip_phantom_images(
        self, text: str, output_dir: Path | None = None
    ) -> str:
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

    def _normalize_generic(self, text: str) -> str:
        """Apply generic markdown normalization."""
        # CRLF -> LF
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Unicode NFKC normalization (handles compatibility chars)
        text = unicodedata.normalize("NFKC", text)

        # Smart quotes / ligatures -> ASCII equivalents
        for src, dst in self._UNICODE_MAP.items():
            text = text.replace(src, dst)

        # Strip trailing whitespace per line
        text = self._RE_TRAILING_WS.sub("", text)

        # Collapse 3+ blank lines -> 2
        text = self._RE_EXCESS_BLANK.sub("\n\n", text)

        return text.strip()
