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

    def check(self, text: str, expected_pages: int = 0) -> HeuristicsResult:
        """Run all heuristic checks on OCR output.

        Args:
            text: The OCR output text to check.
            expected_pages: If > 0, check for truncation based on expected page count.
        """
        result = HeuristicsResult()

        if not text or not text.strip():
            result.add_metric(AuditMetric(
                name="Empty output",
                value="No text extracted",
                passed=False,
                severity="error",
            ))
            return result

        # LLM refusal detection (critical failure)
        if self._check_llm_refusal(text):
            result.add_metric(AuditMetric(
                name="LLM refusal",
                value="Model refused to process image",
                passed=False,
                severity="error",
            ))
            return result  # Early exit - no point checking further

        # CID artifact detection (PDF font mapping failures)
        if self._check_cid_artifacts(text):
            result.add_metric(AuditMetric(
                name="CID artifacts",
                value="PDF font mapping failures detected",
                passed=False,
                severity="error",
            ))

        # Hallucination loop detection
        if self._check_hallucination_loops(text):
            result.add_metric(AuditMetric(
                name="Hallucination loops",
                value="Repeated sentence patterns detected",
                passed=False,
                severity="error",
            ))

        # Formatting instruction hallucination detection
        halluc_count = self._check_formatting_hallucination(text)
        if halluc_count >= 2:
            result.add_metric(AuditMetric(
                name="Formatting hallucination",
                value=f"{halluc_count} formatting instruction patterns detected",
                passed=False,
                severity="error",
            ))

        # Word count check
        words = text.split()
        word_count = len(words)
        result.add_metric(AuditMetric(
            name="Word count",
            value=word_count,
            threshold=self.min_word_count,
            passed=word_count >= self.min_word_count,
            # Low word count almost always indicates a bad extraction; treat as failing.
            severity="error" if word_count < self.min_word_count else "info",
        ))

        # Truncation detection: if we know the page count, check that
        # word count is plausible. Academic papers average ~300 words/page;
        # anything below 100 words/page on a multi-page doc is truncated.
        if expected_pages > 5 and word_count > 0:
            words_per_page = word_count / expected_pages
            is_truncated = words_per_page < 100
            result.add_metric(AuditMetric(
                name="Truncation check",
                value=f"{words_per_page:.0f} words/page ({word_count} words / {expected_pages} pages)",
                threshold=">50 words/page",
                passed=not is_truncated,
                severity="error" if is_truncated else "info",
            ))

        # Average word length check
        if words:
            avg_word_len = sum(len(w) for w in words) / len(words)
            result.add_metric(AuditMetric(
                name="Avg word length",
                value=f"{avg_word_len:.1f}",
                threshold=f"{self.min_avg_word_length}-{self.max_avg_word_length}",
                passed=self.min_avg_word_length <= avg_word_len <= self.max_avg_word_length,
                severity="warning",
            ))

        # Check if content is math-dense (high LaTeX tokens)
        is_math_dense = self._is_math_dense(text)

        # Garbage character ratio - exception for math-dense content
        garbage_ratio = self._calculate_garbage_ratio(text)
        garbage_passed = garbage_ratio <= self.max_garbage_ratio or is_math_dense
        result.add_metric(AuditMetric(
            name="Garbage ratio",
            value=f"{garbage_ratio:.1%}" + (" (math-dense)" if is_math_dense else ""),
            threshold=f"<{self.max_garbage_ratio:.0%}",
            passed=garbage_passed,
            severity="info" if garbage_passed else "error",
        ))

        # Unicode issues check
        unicode_issues = self._check_unicode_issues(text)
        if unicode_issues:
            result.add_metric(AuditMetric(
                name="Unicode issues",
                value=", ".join(unicode_issues),
                passed=False,
                severity="warning",
            ))

        # Repeated character sequences (OCR artifacts)
        repeated = self._check_repeated_patterns(text)
        if repeated:
            result.add_metric(AuditMetric(
                name="Repeated patterns",
                value=f"{len(repeated)} suspicious patterns",
                passed=False,
                severity="warning",
            ))

        # Structure check
        has_structure = self._check_structure(text)
        result.add_metric(AuditMetric(
            name="Has structure",
            value="Yes" if has_structure else "No",
            passed=True,  # Informational only
            severity="info",
        ))

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
        return bool(re.search(r'\(cid:\d+\)', text))

    def _is_math_dense(self, text: str) -> bool:
        """Check if high 'garbage' characters are actually LaTeX.

        Math-heavy pages have many backslashes, braces, underscores, carets.
        If >30% of characters are LaTeX tokens, don't penalize as garbage.
        """
        if not text:
            return False
        latex_chars = sum(1 for c in text if c in r'\{}^_$')
        return latex_chars / len(text) > 0.30

    def _check_hallucination_loops(self, text: str) -> bool:
        """Detect exact sentence repetition (hallucination loops).

        If the same sentence appears 3+ times consecutively, it's likely
        a model hallucination rather than legitimate content.
        """
        # Split into sentences
        sentences = re.split(r'[.!?]\s+', text)
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
        garbage_pattern = r'[^\w\s.,!?;:\'\"()\[\]{}<>@#$%&*+=/\\-]'
        garbage_chars = len(re.findall(garbage_pattern, text))

        # Also count excessive whitespace as potential garbage
        excessive_ws = len(re.findall(r'\s{4,}', text))

        total_garbage = garbage_chars + excessive_ws
        return total_garbage / len(text)

    def _check_unicode_issues(self, text: str) -> list[str]:
        """Check for common Unicode issues."""
        issues = []

        # Replacement characters
        if '\ufffd' in text:
            issues.append("replacement chars (�)")

        # Private use area characters
        if re.search(r'[\ue000-\uf8ff]', text):
            issues.append("private use chars")

        # Control characters (except newlines/tabs)
        if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', text):
            issues.append("control chars")

        # Mixed scripts that shouldn't appear together
        # (simplified check)
        has_latin = bool(re.search(r'[a-zA-Z]', text))
        has_cjk = bool(re.search(r'[\u4e00-\u9fff]', text))
        has_arabic = bool(re.search(r'[\u0600-\u06ff]', text))

        script_count = sum([has_latin, has_cjk, has_arabic])
        if script_count > 1:
            # Could be legitimate, just flag it
            pass

        return issues

    def _check_repeated_patterns(self, text: str) -> list[str]:
        """Check for suspicious repeated patterns (OCR artifacts)."""
        issues = []

        # Same character repeated 5+ times
        if re.search(r'(.)\1{4,}', text):
            issues.append("repeated chars")

        # Same word repeated 3+ times consecutively
        if re.search(r'\b(\w+)\s+\1\s+\1\b', text, re.IGNORECASE):
            issues.append("repeated words")

        # Alternating character patterns (e.g., "ababab")
        if re.search(r'(..)\1{3,}', text):
            issues.append("alternating patterns")

        return issues

    def _check_structure(self, text: str) -> bool:
        """Check if text has recognizable structure."""
        # Look for markdown headers
        if re.search(r'^#+\s+\w', text, re.MULTILINE):
            return True

        # Look for numbered lists
        if re.search(r'^\d+\.\s+\w', text, re.MULTILINE):
            return True

        # Look for bullet points
        if re.search(r'^[-*•]\s+\w', text, re.MULTILINE):
            return True

        # Look for paragraph breaks
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 2:
            return True

        return False
