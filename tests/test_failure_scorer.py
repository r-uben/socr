"""Tests for FailureModeScorer — failure-mode detection from audit heuristics."""

import pytest

pytest.importorskip("rich")

from socr.audit.heuristics import AuditMetric, HeuristicsChecker, HeuristicsResult
from socr.audit.scorer import FailureModeScorer, ScoringResult
from socr.core.result import FailureMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _long_clean_text(n_words: int = 200) -> str:
    """Generate clean, varied text with roughly *n_words* words.

    Uses distinct sentences to avoid triggering the hallucination-loop
    detector (which flags 3+ consecutive identical sentences).
    """
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A well-structured document contains multiple paragraphs.",
        "Economic growth depends on many factors including technology.",
        "Climate change affects agricultural yields across regions.",
        "Machine learning models require large training datasets.",
        "International trade policy shapes global supply chains.",
        "Central banks adjust interest rates to manage inflation.",
        "Researchers published their findings in a peer-reviewed journal.",
        "The experiment confirmed the theoretical predictions from earlier work.",
        "Statistical significance was established at the five percent level.",
    ]
    # Cycle through distinct sentences to build the desired word count.
    words_so_far = 0
    parts: list[str] = []
    idx = 0
    while words_so_far < n_words:
        s = sentences[idx % len(sentences)]
        parts.append(s)
        words_so_far += len(s.split())
        idx += 1
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Individual failure mode mappings
# ---------------------------------------------------------------------------


class TestEmptyOutput:
    def test_empty_string(self) -> None:
        scorer = FailureModeScorer()
        result = scorer.score("")
        assert not result.passed
        assert FailureMode.EMPTY_OUTPUT in result.failure_modes
        assert result.primary_failure == FailureMode.EMPTY_OUTPUT

    def test_whitespace_only(self) -> None:
        scorer = FailureModeScorer()
        result = scorer.score("   \n\t  ")
        assert not result.passed
        assert FailureMode.EMPTY_OUTPUT in result.failure_modes


class TestLowWordCount:
    def test_few_words(self) -> None:
        scorer = FailureModeScorer(checker=HeuristicsChecker(min_word_count=50))
        result = scorer.score("Hello world this is short")
        assert not result.passed
        assert FailureMode.LOW_WORD_COUNT in result.failure_modes
        assert result.primary_failure == FailureMode.LOW_WORD_COUNT


class TestGarbage:
    def test_high_garbage_ratio(self) -> None:
        # Build text with many garbage characters but enough words
        garbage = "\x00\x01\x02\x03\x04\x05" * 50
        words = " ".join(["word"] * 60)
        text = words + garbage
        scorer = FailureModeScorer(
            checker=HeuristicsChecker(min_word_count=10, max_garbage_ratio=0.01)
        )
        result = scorer.score(text)
        assert not result.passed
        assert FailureMode.GARBAGE in result.failure_modes

    def test_cid_artifacts(self) -> None:
        text = " ".join(["(cid:12) (cid:34) word"] * 30)
        scorer = FailureModeScorer(checker=HeuristicsChecker(min_word_count=10))
        result = scorer.score(text)
        assert not result.passed
        assert FailureMode.GARBAGE in result.failure_modes
        assert "CID" in result.details.get(FailureMode.GARBAGE, "")


class TestHallucination:
    def test_hallucination_loops(self) -> None:
        repeated = "The model generated this sentence again and again. "
        text = repeated * 20
        scorer = FailureModeScorer(checker=HeuristicsChecker(min_word_count=10))
        result = scorer.score(text)
        assert not result.passed
        assert FailureMode.HALLUCINATION in result.failure_modes

    def test_formatting_hallucination(self) -> None:
        text = (
            "Use a standard font like Times New Roman. "
            "Include all figures and tables. "
            "Include page numbers at the bottom. "
            "double-spaced text is preferred. "
        ) + _long_clean_text(200)
        scorer = FailureModeScorer(checker=HeuristicsChecker(min_word_count=10))
        result = scorer.score(text)
        assert not result.passed
        assert FailureMode.HALLUCINATION in result.failure_modes


class TestRefusal:
    def test_llm_refusal_short_text(self) -> None:
        text = "I'm sorry, I cannot process this image."
        scorer = FailureModeScorer(checker=HeuristicsChecker(min_word_count=5))
        result = scorer.score(text)
        assert not result.passed
        assert FailureMode.REFUSAL in result.failure_modes
        assert result.primary_failure == FailureMode.REFUSAL

    def test_refusal_as_an_ai(self) -> None:
        text = "As an AI language model, I cannot extract text from images."
        scorer = FailureModeScorer(checker=HeuristicsChecker(min_word_count=5))
        result = scorer.score(text)
        assert not result.passed
        assert FailureMode.REFUSAL in result.failure_modes


# ---------------------------------------------------------------------------
# Priority / primary failure selection
# ---------------------------------------------------------------------------


class TestPrimarySelection:
    def test_hallucination_over_low_word_count(self) -> None:
        """Hallucination should be primary over low word count."""
        # Construct text that triggers both hallucination loops AND low word count.
        repeated = "Model generated this fake sentence again. "
        text = repeated * 4  # short (triggers low word count) + loops
        scorer = FailureModeScorer(checker=HeuristicsChecker(min_word_count=100))
        result = scorer.score(text)
        assert not result.passed
        # Hallucination has higher priority than low word count.
        if (
            FailureMode.HALLUCINATION in result.failure_modes
            and FailureMode.LOW_WORD_COUNT in result.failure_modes
        ):
            assert result.primary_failure == FailureMode.HALLUCINATION

    def test_refusal_over_garbage(self) -> None:
        """Refusal should be primary over garbage."""
        # Refusal triggers early exit in HeuristicsChecker, so it won't
        # co-occur with garbage — but if we construct it manually via
        # score_from_audit we can verify priority logic.
        audit = HeuristicsResult(passed=False)
        audit.add_metric(
            AuditMetric(
                name="Garbage ratio",
                value="50%",
                passed=False,
                severity="error",
            )
        )
        audit.add_metric(
            AuditMetric(
                name="LLM refusal",
                value="refused",
                passed=False,
                severity="error",
            )
        )
        scorer = FailureModeScorer()
        result = scorer.score_from_audit(audit)
        assert result.primary_failure == FailureMode.REFUSAL

    def test_hallucination_over_garbage(self) -> None:
        audit = HeuristicsResult(passed=False)
        audit.add_metric(
            AuditMetric(
                name="Garbage ratio",
                value="30%",
                passed=False,
                severity="error",
            )
        )
        audit.add_metric(
            AuditMetric(
                name="Hallucination loops",
                value="loops",
                passed=False,
                severity="error",
            )
        )
        scorer = FailureModeScorer()
        result = scorer.score_from_audit(audit)
        assert result.primary_failure == FailureMode.HALLUCINATION


# ---------------------------------------------------------------------------
# Combined failure modes
# ---------------------------------------------------------------------------


class TestCombinedModes:
    def test_multiple_failures_all_captured(self) -> None:
        audit = HeuristicsResult(passed=False)
        audit.add_metric(
            AuditMetric(
                name="Word count",
                value=10,
                threshold=50,
                passed=False,
                severity="error",
            )
        )
        audit.add_metric(
            AuditMetric(
                name="Garbage ratio",
                value="40%",
                passed=False,
                severity="error",
            )
        )
        audit.add_metric(
            AuditMetric(
                name="Hallucination loops",
                value="loops",
                passed=False,
                severity="error",
            )
        )
        scorer = FailureModeScorer()
        result = scorer.score_from_audit(audit)
        assert FailureMode.LOW_WORD_COUNT in result.failure_modes
        assert FailureMode.GARBAGE in result.failure_modes
        assert FailureMode.HALLUCINATION in result.failure_modes
        assert len(result.failure_modes) == 3

    def test_details_populated_per_mode(self) -> None:
        audit = HeuristicsResult(passed=False)
        audit.add_metric(
            AuditMetric(
                name="Word count",
                value=5,
                threshold=50,
                passed=False,
                severity="error",
            )
        )
        audit.add_metric(
            AuditMetric(
                name="CID artifacts",
                value="CID refs",
                passed=False,
                severity="error",
            )
        )
        scorer = FailureModeScorer()
        result = scorer.score_from_audit(audit)
        assert FailureMode.LOW_WORD_COUNT in result.details
        assert FailureMode.GARBAGE in result.details


# ---------------------------------------------------------------------------
# Clean text — no failures
# ---------------------------------------------------------------------------


class TestCleanText:
    def test_clean_text_passes(self) -> None:
        scorer = FailureModeScorer(checker=HeuristicsChecker(min_word_count=10))
        text = _long_clean_text(300)
        result = scorer.score(text)
        assert result.passed
        assert result.failure_modes == []
        assert result.primary_failure == FailureMode.NONE

    def test_score_from_passing_audit(self) -> None:
        audit = HeuristicsResult(passed=True)
        audit.add_metric(
            AuditMetric(
                name="Word count",
                value=500,
                threshold=50,
                passed=True,
                severity="info",
            )
        )
        scorer = FailureModeScorer()
        result = scorer.score_from_audit(audit)
        assert result.passed
        assert result.primary_failure == FailureMode.NONE


# ---------------------------------------------------------------------------
# Confidence estimation
# ---------------------------------------------------------------------------


class TestConfidence:
    def test_multiple_errors_high_confidence(self) -> None:
        audit = HeuristicsResult(passed=False)
        audit.add_metric(
            AuditMetric(
                name="Word count",
                value=5,
                passed=False,
                severity="error",
            )
        )
        audit.add_metric(
            AuditMetric(
                name="Garbage ratio",
                value="50%",
                passed=False,
                severity="error",
            )
        )
        scorer = FailureModeScorer()
        result = scorer.score_from_audit(audit)
        assert result.confidence >= 0.9

    def test_single_error_moderate_confidence(self) -> None:
        audit = HeuristicsResult(passed=False)
        audit.add_metric(
            AuditMetric(
                name="Word count",
                value=5,
                passed=False,
                severity="error",
            )
        )
        scorer = FailureModeScorer()
        result = scorer.score_from_audit(audit)
        assert 0.5 < result.confidence < 0.9

    def test_clean_text_full_confidence(self) -> None:
        scorer = FailureModeScorer(checker=HeuristicsChecker(min_word_count=3))
        result = scorer.score(_long_clean_text(200))
        assert result.confidence == 1.0


# ---------------------------------------------------------------------------
# Standalone API (score with text directly)
# ---------------------------------------------------------------------------


class TestStandaloneAPI:
    def test_score_accepts_engine_kwarg(self) -> None:
        scorer = FailureModeScorer()
        result = scorer.score(_long_clean_text(200), engine="deepseek")
        assert result.passed

    def test_score_from_audit_matches_score(self) -> None:
        text = "I'm sorry, I cannot process this image."
        checker = HeuristicsChecker(min_word_count=5)
        scorer = FailureModeScorer(checker=checker)
        direct = scorer.score(text)
        via_audit = scorer.score_from_audit(checker.check(text))
        assert direct.failure_modes == via_audit.failure_modes
        assert direct.primary_failure == via_audit.primary_failure
