"""Heuristic quality checks for OCR results."""

import re
from dataclasses import dataclass, field


@dataclass
class AuditMetric:
    """A single audit metric result."""

    name: str
    value: str | float
    threshold: str | float | None = None
    passed: bool = True
    severity: str = "info"  # info, warning, error


@dataclass
class HeuristicsResult:
    """Result of heuristics-based quality check."""

    passed: bool = True
    metrics: list[AuditMetric] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_metric(self, metric: AuditMetric) -> None:
        """Add a metric and update overall pass status."""
        self.metrics.append(metric)
        if not metric.passed:
            if metric.severity == "error":
                self.passed = False
                self.errors.append(f"{metric.name}: {metric.value}")
            else:
                self.warnings.append(f"{metric.name}: {metric.value}")


class HeuristicsChecker:
    """Fast heuristic checks for OCR quality."""

    NATIVE_TABLE_STRUCTURE_METRIC = "Native table structure"

    # Markdown/HTML table structure is already usable downstream.
    _MARKDOWN_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
    _MARKDOWN_TABLE_SEPARATOR_RE = re.compile(
        r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
    )
    _HTML_TABLE_RE = re.compile(r"<\s*table\b", re.IGNORECASE)

    # Native table failures from PyMuPDF usually arrive as one cell per line:
    # header, year, year, row-label, number, number... A small list or glossary
    # can look similar, so the flat-stream gate also requires numeric content.
    _MIN_FLAT_TABLE_LINES = 12
    _MIN_FLAT_TABLE_NUMERIC_LINES = 6
    _NUMERIC_CELL_RE = re.compile(r"^[\(\[]?-?\d[\d,]*(?:\.\d+)?[\)\]%]?$")
    _NUMERIC_TOKEN_RE = re.compile(r"(?<![A-Za-z])-?\d[\d,]*(?:\.\d+)?%?")

    # Markdown table sanity checks: these ratios tolerate occasional section
    # labels, spanning notes, and footnotes while failing grids whose data rows
    # mostly collapsed into one or two cells.
    _MIN_MARKDOWN_DATA_ROWS_FOR_RATIO = 3
    _WIDE_MARKDOWN_TABLE_COLS = 4
    _GRID_STABILITY_COLS = 3
    _MAX_COLLAPSED_ROW_SHARE = 0.50
    _MAX_GLUED_NUMERIC_ROW_SHARE = 0.25
    _MIN_ALIGNED_ROW_SHARE = 0.40

    # LLM refusal patterns (case-insensitive)
    REFUSAL_PATTERNS = [
        r"I cannot read",
        r"I am sorry",
        r"I'm sorry",
        r"As an AI",
        r"I'm unable to",
        r"cannot process this image",
        r"I cannot assist",
        r"I can't read",
        r"unable to extract",
        r"cannot extract text",
    ]

    # Formatting instruction hallucination patterns.
    # DeepSeek-OCR hallucinates these when given generic prompts.
    HALLUCINATION_PATTERNS = [
        r"Use a standard font",
        r"print on \d+\.?\d*\s*[x×]\s*\d+",
        r"Include (?:all )?(?:figures|tables|links|references)",
        r"Include links to other resources",
        r"Include page numbers",
        r"Include captions",
        r"Proofread your work",
        r"double[- ]spaced",
        r"single[- ]spaced",
        r"Times New Roman",
        r"formatting guidelines",
        r"submission guidelines",
        r"page margins",
    ]

    def __init__(
        self,
        min_word_count: int = 50,
        max_garbage_ratio: float = 0.15,
        min_avg_word_length: float = 2.0,
        max_avg_word_length: float = 15.0,
    ) -> None:
        self.min_word_count = min_word_count
        self.max_garbage_ratio = max_garbage_ratio
        self.min_avg_word_length = min_avg_word_length
        self.max_avg_word_length = max_avg_word_length

    def check(
        self, text: str, expected_pages: int = 0, sparse_ok: bool = False
    ) -> HeuristicsResult:
        """Run all heuristic checks on OCR output.

        Args:
            text: The OCR output text to check.
            expected_pages: If > 0, check for truncation based on expected page count.
            sparse_ok: The page is legitimately sparse (figure-dominated, or
                the source page itself carries fewer words than the minimum).
                Low word count is then a warning, not an error — a correct
                24-word figure-caption page must not hard-fail the gate and
                trigger paid escalation.
        """
        result = HeuristicsResult()

        if not text or not text.strip():
            result.add_metric(
                AuditMetric(
                    name="Empty output",
                    value="No text extracted",
                    passed=False,
                    severity="error",
                )
            )
            return result

        # LLM refusal detection (critical failure)
        if self._check_llm_refusal(text):
            result.add_metric(
                AuditMetric(
                    name="LLM refusal",
                    value="Model refused to process image",
                    passed=False,
                    severity="error",
                )
            )
            return result  # Early exit - no point checking further

        # CID artifact detection (PDF font mapping failures)
        if self._check_cid_artifacts(text):
            result.add_metric(
                AuditMetric(
                    name="CID artifacts",
                    value="PDF font mapping failures detected",
                    passed=False,
                    severity="error",
                )
            )

        # Hallucination loop detection
        if self._check_hallucination_loops(text):
            result.add_metric(
                AuditMetric(
                    name="Hallucination loops",
                    value="Repeated sentence patterns detected",
                    passed=False,
                    severity="error",
                )
            )

        # Formatting instruction hallucination detection
        halluc_count = self._check_formatting_hallucination(text)
        if halluc_count >= 2:
            result.add_metric(
                AuditMetric(
                    name="Formatting hallucination",
                    value=f"{halluc_count} formatting instruction patterns detected",
                    passed=False,
                    severity="error",
                )
            )

        # Word count check
        words = text.split()
        word_count = len(words)
        if word_count < self.min_word_count:
            # On a sparse page the low count is expected, not evidence of a
            # bad extraction — record it as a warning for visibility only.
            wc_severity = "warning" if sparse_ok else "error"
        else:
            wc_severity = "info"
        result.add_metric(
            AuditMetric(
                name="Word count",
                value=word_count,
                threshold=self.min_word_count,
                passed=word_count >= self.min_word_count,
                severity=wc_severity,
            )
        )

        # Truncation detection: if we know the page count, check that
        # word count is plausible. Academic papers average ~300 words/page;
        # anything below 100 words/page on a multi-page doc is truncated.
        if expected_pages > 5 and word_count > 0:
            words_per_page = word_count / expected_pages
            is_truncated = words_per_page < 100
            result.add_metric(
                AuditMetric(
                    name="Truncation check",
                    value=(
                        f"{words_per_page:.0f} words/page "
                        f"({word_count} words / {expected_pages} pages)"
                    ),
                    threshold=">50 words/page",
                    passed=not is_truncated,
                    severity="error" if is_truncated else "info",
                )
            )

        # Average word length check
        if words:
            avg_word_len = sum(len(w) for w in words) / len(words)
            result.add_metric(
                AuditMetric(
                    name="Avg word length",
                    value=f"{avg_word_len:.1f}",
                    threshold=f"{self.min_avg_word_length}-{self.max_avg_word_length}",
                    passed=self.min_avg_word_length <= avg_word_len <= self.max_avg_word_length,
                    severity="warning",
                )
            )

        # Check if content is math-dense (high LaTeX tokens)
        is_math_dense = self._is_math_dense(text)

        # Garbage character ratio - exception for math-dense content
        garbage_ratio = self._calculate_garbage_ratio(text)
        garbage_passed = garbage_ratio <= self.max_garbage_ratio or is_math_dense
        result.add_metric(
            AuditMetric(
                name="Garbage ratio",
                value=f"{garbage_ratio:.1%}" + (" (math-dense)" if is_math_dense else ""),
                threshold=f"<{self.max_garbage_ratio:.0%}",
                passed=garbage_passed,
                severity="info" if garbage_passed else "error",
            )
        )

        # Unicode issues check
        unicode_issues = self._check_unicode_issues(text)
        if unicode_issues:
            result.add_metric(
                AuditMetric(
                    name="Unicode issues",
                    value=", ".join(unicode_issues),
                    passed=False,
                    severity="warning",
                )
            )

        # Repeated character sequences (OCR artifacts)
        repeated = self._check_repeated_patterns(text)
        if repeated:
            result.add_metric(
                AuditMetric(
                    name="Repeated patterns",
                    value=f"{len(repeated)} suspicious patterns",
                    passed=False,
                    severity="warning",
                )
            )

        # Structure check
        has_structure = self._check_structure(text)
        result.add_metric(
            AuditMetric(
                name="Has structure",
                value="Yes" if has_structure else "No",
                passed=True,  # Informational only
                severity="info",
            )
        )

        return result

    def check_native_table_structure(self, text: str) -> HeuristicsResult:
        """Check whether native text preserved a usable table structure.

        This is intentionally narrower than the general OCR audit. It only
        answers: "a source page known to contain a table came through native
        extraction; did the Markdown still preserve a grid?" Clean prose metrics
        such as word count are irrelevant here and would recreate the old
        over-escalation behavior for sparse but correct born-digital pages.
        """
        result = HeuristicsResult()
        failure_reason = self._native_table_structure_failure_reason(text)
        failed = failure_reason is not None
        result.add_metric(
            AuditMetric(
                name=self.NATIVE_TABLE_STRUCTURE_METRIC,
                value=failure_reason or "usable",
                passed=not failed,
                severity="error" if failed else "info",
            )
        )
        return result

    def _check_formatting_hallucination(self, text: str) -> int:
        """Count formatting instruction hallucination patterns.

        DeepSeek-OCR hallucinates formatting instructions (font size,
        page margins, submission guidelines) when given generic prompts.
        Returns the number of distinct patterns matched; 2+ is a failure.
        """
        count = 0
        for pattern in self.HALLUCINATION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                count += 1
        return count

    def _native_table_structure_failed(self, text: str) -> bool:
        return self._native_table_structure_failure_reason(text) is not None

    def _native_table_structure_failure_reason(self, text: str) -> str | None:
        """Why a table page's native extraction lost the grid, if it did.

        Accepts three usable structures:
        - GitHub-style Markdown tables.
        - HTML tables.
        - Plain aligned rows with multiple whitespace-separated cells.

        Flags only the high-confidence bad shape: many single-cell lines,
        with enough numeric cells to look like a data table rather than a
        glossary, TOC, or bullet list.
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None

        markdown_quality = self._markdown_table_structure_quality(text, lines)
        if markdown_quality == "malformed":
            return "malformed Markdown table structure"
        if markdown_quality == "valid":
            return None

        if len(lines) < self._MIN_FLAT_TABLE_LINES:
            return None

        single_cell_lines = [line for line in lines if len(line.split()) == 1]
        numeric_single_cell_lines = [
            line for line in single_cell_lines if self._NUMERIC_CELL_RE.match(line)
        ]
        if (
            len(single_cell_lines) > len(lines) / 2
            and len(numeric_single_cell_lines) >= self._MIN_FLAT_TABLE_NUMERIC_LINES
        ):
            return "flat cell stream without Markdown/HTML/column structure"
        return None

    def _markdown_table_structure_quality(self, text: str, lines: list[str]) -> str:
        """Return ``valid``, ``malformed``, or ``none`` for table-like markup."""
        if self._HTML_TABLE_RE.search(text):
            return "valid"

        context_quality = self._markdown_separator_context_quality(lines)
        if context_quality != "none":
            return context_quality

        saw_valid = False
        for block in self._pipe_line_blocks(lines):
            quality = self._markdown_pipe_block_quality(block)
            if quality == "malformed":
                return "malformed"
            if quality == "valid":
                saw_valid = True
        if saw_valid:
            return "valid"
        return "none"

    def _markdown_separator_context_quality(self, lines: list[str]) -> str:
        """Evaluate rows around Markdown separators, including broken rows.

        PyMuPDF can emit a syntactically valid header+separator, then put the
        data rows on later pipe-containing lines that are not valid Markdown
        rows because cell bodies spilled across intervening lines. Treat those
        later pipe-lines as the separator's data context so a valid-looking
        header cannot mask a collapsed table body.
        """
        saw_valid = False
        for sep_idx, line in enumerate(lines):
            if not self._MARKDOWN_TABLE_SEPARATOR_RE.match(line):
                continue
            if sep_idx == 0 or "|" not in lines[sep_idx - 1]:
                continue

            header_cols = len(self._markdown_cells(lines[sep_idx - 1]))
            separator_cols = len(self._markdown_cells(line))
            if header_cols < 2 or separator_cols < 2:
                continue
            if abs(header_cols - separator_cols) > 1:
                return "malformed"

            data_rows: list[list[str]] = []
            for data_line in lines[sep_idx + 1 :]:
                if self._MARKDOWN_TABLE_SEPARATOR_RE.match(data_line):
                    break
                if "|" in data_line:
                    data_rows.append(self._markdown_cells(data_line))

            if not data_rows:
                continue
            rows_quality = self._markdown_data_rows_quality(separator_cols, data_rows)
            if rows_quality == "malformed":
                return "malformed"
            if rows_quality == "valid":
                saw_valid = True

        if saw_valid:
            return "valid"
        return "none"

    @staticmethod
    def _pipe_line_blocks(lines: list[str]) -> list[list[str]]:
        """Group adjacent non-empty pipe lines; non-pipe text ends a block."""
        blocks: list[list[str]] = []
        current: list[str] = []
        for line in lines:
            if "|" in line:
                current.append(line)
            elif current:
                blocks.append(current)
                current = []
        if current:
            blocks.append(current)
        return blocks

    def _markdown_pipe_block_quality(self, block: list[str]) -> str:
        saw_valid = False
        for sep_idx, line in enumerate(block):
            if not self._MARKDOWN_TABLE_SEPARATOR_RE.match(line):
                continue
            if sep_idx == 0 or not self._MARKDOWN_TABLE_ROW_RE.match(block[sep_idx - 1]):
                continue

            header_cols = len(self._markdown_cells(block[sep_idx - 1]))
            separator_cols = len(self._markdown_cells(line))
            expected_cols = separator_cols
            if header_cols < 2 or separator_cols < 2:
                continue
            if abs(header_cols - separator_cols) > 1:
                return "malformed"

            data_rows: list[list[str]] = []
            for data_line in block[sep_idx + 1 :]:
                if self._MARKDOWN_TABLE_SEPARATOR_RE.match(data_line):
                    break
                if self._MARKDOWN_TABLE_ROW_RE.match(data_line):
                    data_rows.append(self._markdown_cells(data_line))

            if not data_rows:
                continue
            rows_quality = self._markdown_data_rows_quality(expected_cols, data_rows)
            if rows_quality == "malformed":
                return "malformed"
            if rows_quality == "valid":
                saw_valid = True

        if saw_valid:
            return "valid"
        return "none"

    @staticmethod
    def _markdown_cells(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    def _markdown_data_rows_quality(self, expected_cols: int, rows: list[list[str]]) -> str:
        row_count = len(rows)
        aligned_rows = sum(1 for cells in rows if abs(len(cells) - expected_cols) <= 1)

        if row_count >= self._MIN_MARKDOWN_DATA_ROWS_FOR_RATIO:
            collapsed_limit = max(1, expected_cols // 2)
            collapsed_rows = sum(1 for cells in rows if len(cells) <= collapsed_limit)
            glued_numeric_rows = sum(
                1 for cells in rows if any(self._cell_has_multiple_numbers(cell) for cell in cells)
            )
            aligned_share = aligned_rows / row_count

            if (
                expected_cols >= self._WIDE_MARKDOWN_TABLE_COLS
                and collapsed_rows / row_count >= self._MAX_COLLAPSED_ROW_SHARE
            ):
                return "malformed"
            if (
                expected_cols >= self._WIDE_MARKDOWN_TABLE_COLS
                and glued_numeric_rows / row_count >= self._MAX_GLUED_NUMERIC_ROW_SHARE
            ):
                return "malformed"
            if (
                expected_cols >= self._GRID_STABILITY_COLS
                and aligned_share < self._MIN_ALIGNED_ROW_SHARE
            ):
                return "malformed"

        if aligned_rows:
            return "valid"
        return "none"

    def _cell_has_multiple_numbers(self, cell: str) -> bool:
        return len(self._NUMERIC_TOKEN_RE.findall(cell)) >= 2

    def _check_llm_refusal(self, text: str) -> bool:
        """Detect LLM refusal patterns indicating model couldn't process image.

        Real refusals appear at the start of short output. A phrase like
        "I am sorry" buried in page 30 of a long paper is legitimate text,
        not a refusal. Only flag if text is short or the match is near the top.
        """
        words = text.split()
        is_short = len(words) < 200

        # For short text, scan the whole thing
        search_text = text if is_short else text[:500]

        for pattern in self.REFUSAL_PATTERNS:
            if re.search(pattern, search_text, re.IGNORECASE):
                return True
        return False

    def _check_cid_artifacts(self, text: str) -> bool:
        """Detect PDF font mapping failures (CID references)."""
        # (cid:XX) patterns indicate failed character mapping
        return bool(re.search(r"\(cid:\d+\)", text))

    def _is_math_dense(self, text: str) -> bool:
        """Check if high 'garbage' characters are actually LaTeX.

        Math-heavy pages have many backslashes, braces, underscores, carets.
        If >30% of characters are LaTeX tokens, don't penalize as garbage.
        """
        if not text:
            return False
        latex_chars = sum(1 for c in text if c in r"\{}^_$")
        return latex_chars / len(text) > 0.30

    def _check_hallucination_loops(self, text: str) -> bool:
        """Detect exact sentence repetition (hallucination loops).

        If the same sentence appears 3+ times consecutively, it's likely
        a model hallucination rather than legitimate content.
        """
        # Split into sentences
        sentences = re.split(r"[.!?]\s+", text)
        if len(sentences) < 6:
            return False

        # Check for consecutive repetition
        for i in range(len(sentences) - 2):
            s = sentences[i].strip()
            if len(s) < 20:  # Skip very short "sentences"
                continue
            if s == sentences[i + 1].strip() == sentences[i + 2].strip():
                return True
        return False

    def _calculate_garbage_ratio(self, text: str) -> float:
        """Calculate ratio of garbage characters to total characters."""
        if not text:
            return 0.0

        # Characters that are typically garbage in OCR output
        garbage_pattern = r"[^\w\s.,!?;:\'\"()\[\]{}<>@#$%&*+=/\\-]"
        garbage_chars = len(re.findall(garbage_pattern, text))

        # Also count excessive whitespace as potential garbage
        excessive_ws = len(re.findall(r"\s{4,}", text))

        total_garbage = garbage_chars + excessive_ws
        return total_garbage / len(text)

    def _check_unicode_issues(self, text: str) -> list[str]:
        """Check for common Unicode issues."""
        issues = []

        # Replacement characters
        if "\ufffd" in text:
            issues.append("replacement chars (�)")

        # Private use area characters
        if re.search(r"[\ue000-\uf8ff]", text):
            issues.append("private use chars")

        # Control characters (except newlines/tabs)
        if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text):
            issues.append("control chars")

        # Mixed scripts that shouldn't appear together
        # (simplified check)
        has_latin = bool(re.search(r"[a-zA-Z]", text))
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", text))
        has_arabic = bool(re.search(r"[\u0600-\u06ff]", text))

        script_count = sum([has_latin, has_cjk, has_arabic])
        if script_count > 1:
            # Could be legitimate, just flag it
            pass

        return issues

    def _check_repeated_patterns(self, text: str) -> list[str]:
        """Check for suspicious repeated patterns (OCR artifacts)."""
        issues = []

        # Same character repeated 5+ times
        if re.search(r"(.)\1{4,}", text):
            issues.append("repeated chars")

        # Same word repeated 3+ times consecutively
        if re.search(r"\b(\w+)\s+\1\s+\1\b", text, re.IGNORECASE):
            issues.append("repeated words")

        # Alternating character patterns (e.g., "ababab")
        if re.search(r"(..)\1{3,}", text):
            issues.append("alternating patterns")

        return issues

    def _check_structure(self, text: str) -> bool:
        """Check if text has recognizable structure."""
        # Look for markdown headers
        if re.search(r"^#+\s+\w", text, re.MULTILINE):
            return True

        # Look for numbered lists
        if re.search(r"^\d+\.\s+\w", text, re.MULTILINE):
            return True

        # Look for bullet points
        if re.search(r"^[-*•]\s+\w", text, re.MULTILINE):
            return True

        # Look for paragraph breaks
        paragraphs = text.split("\n\n")
        if len(paragraphs) > 2:
            return True

        return False
