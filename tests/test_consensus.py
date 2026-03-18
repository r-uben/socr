"""Tests for multi-engine consensus selection."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from socr.core.document import DocumentHandle
from socr.core.result import (
    DocumentStatus,
    EngineResult,
    PageOutput,
    PageStatus,
)
from socr.core.state import DocumentState, PageState
from socr.pipeline.consensus import (
    ConsensusEngine,
    ConsensusResult,
    _agreement_score,
    _compute_wer,
    _count_structure,
    _levenshtein,
    _pairwise_agreement,
    _score_attempt,
    _score_attempt_grounded,
    _score_attempt_ungrounded,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handle(page_count: int = 3) -> DocumentHandle:
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        h = DocumentHandle(path=Path("/tmp/fake.pdf"), page_count=page_count)
    return h


def _make_page_output(
    page_num: int,
    text: str = "some text",
    audit_passed: bool = True,
    engine: str = "deepseek",
    confidence: float = 0.0,
    status: PageStatus = PageStatus.SUCCESS,
) -> PageOutput:
    return PageOutput(
        page_num=page_num,
        text=text,
        status=status,
        audit_passed=audit_passed,
        engine=engine,
        confidence=confidence,
    )


def _make_engine_result(
    pages: list[PageOutput],
    engine: str = "deepseek",
) -> EngineResult:
    return EngineResult(
        document_path=Path("/tmp/fake.pdf"),
        engine=engine,
        status=DocumentStatus.SUCCESS,
        pages=pages,
    )


# ---------------------------------------------------------------------------
# Edit-distance / WER helpers
# ---------------------------------------------------------------------------


class TestLevenshtein:
    def test_identical(self) -> None:
        assert _levenshtein(["a", "b", "c"], ["a", "b", "c"]) == 0

    def test_insertion(self) -> None:
        assert _levenshtein(["a", "b"], ["a", "x", "b"]) == 1

    def test_deletion(self) -> None:
        assert _levenshtein(["a", "x", "b"], ["a", "b"]) == 1

    def test_substitution(self) -> None:
        assert _levenshtein(["a", "b"], ["a", "c"]) == 1

    def test_empty_both(self) -> None:
        assert _levenshtein([], []) == 0

    def test_empty_one(self) -> None:
        assert _levenshtein([], ["a", "b"]) == 2
        assert _levenshtein(["a", "b"], []) == 2


class TestComputeWer:
    def test_identical(self) -> None:
        assert _compute_wer("hello world", "hello world") == 0.0

    def test_completely_different(self) -> None:
        assert _compute_wer("alpha bravo", "charlie delta") == 1.0

    def test_both_empty(self) -> None:
        assert _compute_wer("", "") == 0.0

    def test_hypothesis_empty(self) -> None:
        assert _compute_wer("", "hello world") == 1.0

    def test_reference_empty(self) -> None:
        assert _compute_wer("hello world", "") == 1.0

    def test_partial_match(self) -> None:
        # ref = 4 words, hyp = 4 words, 2 substitutions -> WER = 0.5
        wer = _compute_wer("hello world foo bar", "hello world baz qux")
        assert abs(wer - 0.5) < 1e-9

    def test_case_insensitive(self) -> None:
        assert _compute_wer("Hello World", "hello world") == 0.0


# ---------------------------------------------------------------------------
# Agreement helpers
# ---------------------------------------------------------------------------


class TestAgreementScore:
    def test_identical_texts(self) -> None:
        assert _agreement_score("hello world", "hello world") == 1.0

    def test_completely_different(self) -> None:
        assert _agreement_score("alpha bravo charlie", "delta echo foxtrot") == 0.0

    def test_partial_overlap(self) -> None:
        score = _agreement_score("hello world foo", "hello world bar")
        assert 0.0 < score < 1.0

    def test_word_order_matters(self) -> None:
        """Unlike Jaccard, reversed word order should not score 1.0."""
        score = _agreement_score(
            "the dog bit the man", "the man bit the dog"
        )
        # Same word set but different order -> WER > 0 -> agreement < 1.0
        assert score < 1.0

    def test_clamped_to_zero(self) -> None:
        """WER can exceed 1.0; agreement should clamp at 0.0."""
        # Very short reference, very long hypothesis
        score = _agreement_score(
            "a b c d e f g h i j k l m n", "x"
        )
        assert score >= 0.0


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


class TestCountStructure:
    def test_headers(self) -> None:
        text = "# Title\n\nSome text\n\n## Subtitle\n\nMore text"
        assert _count_structure(text) == 2

    def test_tables(self) -> None:
        text = "| A | B |\n| 1 | 2 |"
        assert _count_structure(text) == 2

    def test_lists(self) -> None:
        text = "- item1\n- item2\n1. item3"
        assert _count_structure(text) == 3

    def test_no_structure(self) -> None:
        assert _count_structure("plain text") == 0


class TestScoreAttemptUngrounded:
    """Tests for the ungrounded (no reference) scoring path."""

    def test_longer_text_scores_higher(self) -> None:
        short = _make_page_output(1, text="word " * 10)
        long = _make_page_output(1, text="word " * 100)
        assert _score_attempt(long) > _score_attempt(short)

    def test_structured_text_scores_higher(self) -> None:
        plain = _make_page_output(1, text="word " * 50)
        structured = _make_page_output(
            1, text="# Title\n\n" + "word " * 50 + "\n\n## Section\n\n" + "word " * 50
        )
        assert _score_attempt(structured) > _score_attempt(plain)

    def test_audit_passed_gets_bonus(self) -> None:
        passed = _make_page_output(1, text="word " * 50, audit_passed=True)
        failed = _make_page_output(1, text="word " * 50, audit_passed=False)
        assert _score_attempt(passed) > _score_attempt(failed)

    def test_higher_confidence_scores_higher(self) -> None:
        low = _make_page_output(1, text="word " * 50, confidence=0.2)
        high = _make_page_output(1, text="word " * 50, confidence=0.9)
        assert _score_attempt(high) > _score_attempt(low)

    def test_no_structure_cap(self) -> None:
        """Structure score should not be capped at 20 anymore."""
        # 30 list items: old code capped at 20, new code uses log-scale
        items = "\n".join(f"- item {i}" for i in range(30))
        many_struct = _make_page_output(1, text=items)
        # 20 list items
        fewer_items = "\n".join(f"- item {i}" for i in range(20))
        fewer_struct = _make_page_output(1, text=fewer_items)
        assert _score_attempt_ungrounded(many_struct) > _score_attempt_ungrounded(fewer_struct)

    def test_dispatches_to_ungrounded_when_no_reference(self) -> None:
        """_score_attempt with empty reference should use ungrounded path."""
        attempt = _make_page_output(1, text="word " * 50)
        assert _score_attempt(attempt) == _score_attempt(attempt, reference_text="")
        assert _score_attempt(attempt) == _score_attempt_ungrounded(attempt)


class TestScoreAttemptGrounded:
    """Tests for the grounded (reference text) scoring path."""

    def test_closer_to_reference_wins(self) -> None:
        """Output closest to reference should score highest, even if shorter."""
        reference = "The quick brown fox jumps over the lazy dog"
        close = _make_page_output(1, text="The quick brown fox jumps over the lazy dog")
        farther = _make_page_output(
            1, text="The fast brown fox leaps over the sleepy dog and cat and bird"
        )
        assert _score_attempt(close, reference) > _score_attempt(farther, reference)

    def test_shorter_but_faithful_beats_longer_hallucination(self) -> None:
        """A shorter output that matches the reference should beat a longer
        hallucinating output."""
        reference = "alpha bravo charlie delta echo"
        faithful = _make_page_output(1, text="alpha bravo charlie delta echo")
        hallucinated = _make_page_output(
            1, text="alpha bravo charlie delta echo " + "garbage " * 50
        )
        assert _score_attempt(faithful, reference) > _score_attempt(hallucinated, reference)

    def test_hallucination_penalty_applied(self) -> None:
        """Output with >150% of reference word count should be penalised."""
        reference = "word " * 20  # 20 words
        normal = _make_page_output(1, text="word " * 20)
        bloated = _make_page_output(1, text="word " * 40)  # 200% = hallucination
        score_normal = _score_attempt_grounded(normal, reference)
        score_bloated = _score_attempt_grounded(bloated, reference)
        # Bloated gets -20 hallucination penalty
        assert score_normal > score_bloated

    def test_no_hallucination_penalty_within_threshold(self) -> None:
        """Output within 150% of reference word count should not be penalised."""
        reference = "word " * 20  # 20 words
        slightly_longer = _make_page_output(1, text="word " * 28)  # 140% -- OK
        # Should not get the -20 penalty
        score = _score_attempt_grounded(slightly_longer, reference)
        # Check that score is reasonable (WER is 0 for identical words + some
        # insertions, but fidelity should be positive)
        assert score > 0

    def test_audit_bonus_in_grounded(self) -> None:
        reference = "hello world"
        passed = _make_page_output(1, text="hello world", audit_passed=True)
        failed = _make_page_output(1, text="hello world", audit_passed=False)
        assert _score_attempt_grounded(passed, reference) > _score_attempt_grounded(failed, reference)

    def test_dispatches_to_grounded_when_reference_provided(self) -> None:
        """_score_attempt with reference should use grounded path."""
        attempt = _make_page_output(1, text="hello world")
        reference = "hello world"
        assert _score_attempt(attempt, reference) == _score_attempt_grounded(attempt, reference)


# ---------------------------------------------------------------------------
# Pairwise agreement (WER-based)
# ---------------------------------------------------------------------------


class TestPairwiseAgreement:
    def test_single_attempt(self) -> None:
        assert _pairwise_agreement([_make_page_output(1, "hello world")]) == 1.0

    def test_identical_attempts(self) -> None:
        a = _make_page_output(1, "hello world")
        b = _make_page_output(1, "hello world", engine="gemini")
        assert _pairwise_agreement([a, b]) == 1.0

    def test_different_attempts(self) -> None:
        a = _make_page_output(1, "alpha bravo charlie")
        b = _make_page_output(1, "delta echo foxtrot", engine="gemini")
        assert _pairwise_agreement([a, b]) == 0.0

    def test_partial_overlap(self) -> None:
        a = _make_page_output(1, "hello world foo")
        b = _make_page_output(1, "hello world bar", engine="gemini")
        score = _pairwise_agreement([a, b])
        assert 0.0 < score < 1.0

    def test_order_sensitive(self) -> None:
        """Pairwise agreement should detect word-order differences."""
        a = _make_page_output(1, "the dog bit the man")
        b = _make_page_output(1, "the man bit the dog", engine="gemini")
        score = _pairwise_agreement([a, b])
        assert score < 1.0


# ---------------------------------------------------------------------------
# ConsensusEngine.select_best (heuristic)
# ---------------------------------------------------------------------------


class TestSelectBest:
    def test_no_attempts(self) -> None:
        engine = ConsensusEngine()
        result = engine.select_best([])
        assert result.selected_engine == "none"
        assert result.merged_text == ""
        assert result.agreement_score == 0.0

    def test_single_attempt(self) -> None:
        engine = ConsensusEngine()
        attempt = _make_page_output(1, "hello world")
        result = engine.select_best([attempt])
        assert result.selected_engine == "deepseek"
        assert result.merged_text == "hello world"
        assert result.agreement_score == 1.0
        assert result.page_num == 1

    def test_longer_text_wins_ungrounded(self) -> None:
        engine = ConsensusEngine()
        short = _make_page_output(1, "short text", engine="engine-a")
        long = _make_page_output(
            1, "a much longer text with many more words in it", engine="engine-b"
        )
        result = engine.select_best([short, long])
        assert result.selected_engine == "engine-b"

    def test_structured_text_wins(self) -> None:
        engine = ConsensusEngine()
        plain = _make_page_output(
            1, "word " * 50, engine="engine-a"
        )
        structured = _make_page_output(
            1,
            "# Title\n\n" + "word " * 45 + "\n\n## Section\n\n- item\n- item",
            engine="engine-b",
        )
        result = engine.select_best([plain, structured])
        assert result.selected_engine == "engine-b"

    def test_audit_passed_wins_over_failed(self) -> None:
        engine = ConsensusEngine()
        # Make both texts the same length so audit_passed is the tiebreaker
        text = "word " * 50
        failed = _make_page_output(1, text, audit_passed=False, engine="engine-a")
        passed = _make_page_output(1, text, audit_passed=True, engine="engine-b")
        result = engine.select_best([failed, passed])
        assert result.selected_engine == "engine-b"

    def test_all_failed_attempts(self) -> None:
        engine = ConsensusEngine()
        a = _make_page_output(
            1, "", engine="engine-a", status=PageStatus.ERROR
        )
        b = _make_page_output(
            1, "", engine="engine-b", status=PageStatus.ERROR
        )
        result = engine.select_best([a, b])
        # Falls back to first attempt
        assert result.selected_engine == "engine-a"
        assert result.agreement_score == 0.0
        assert len(result.discrepancies) > 0

    def test_agreement_score_calculated(self) -> None:
        engine = ConsensusEngine()
        a = _make_page_output(1, "hello world foo bar", engine="engine-a")
        b = _make_page_output(1, "hello world baz qux", engine="engine-b")
        result = engine.select_best([a, b])
        assert 0.0 < result.agreement_score < 1.0

    def test_discrepancies_on_word_count_divergence(self) -> None:
        engine = ConsensusEngine()
        short = _make_page_output(1, "short", engine="engine-a")
        long = _make_page_output(
            1, "a much longer text with many more words", engine="engine-b"
        )
        result = engine.select_best([short, long])
        assert any("Word count" in d for d in result.discrepancies)

    def test_discrepancies_on_audit_divergence(self) -> None:
        engine = ConsensusEngine()
        text = "word " * 50
        passed = _make_page_output(1, text, audit_passed=True, engine="engine-a")
        failed = _make_page_output(1, text, audit_passed=False, engine="engine-b")
        result = engine.select_best([passed, failed])
        assert any("Audit divergence" in d for d in result.discrepancies)

    def test_filters_empty_text(self) -> None:
        engine = ConsensusEngine()
        empty = _make_page_output(1, "   ", engine="engine-a")
        good = _make_page_output(1, "some real text here", engine="engine-b")
        result = engine.select_best([empty, good])
        assert result.selected_engine == "engine-b"


# ---------------------------------------------------------------------------
# Grounded select_best (with reference_text)
# ---------------------------------------------------------------------------


class TestSelectBestGrounded:
    def test_closest_to_reference_wins(self) -> None:
        """Output closest to native text should win, even if shorter."""
        engine = ConsensusEngine()
        reference = "The quick brown fox jumps over the lazy dog"
        close = _make_page_output(1, text=reference, engine="engine-a")
        farther = _make_page_output(
            1,
            text="The fast brown fox leaps over the sleepy dog and cat and bird",
            engine="engine-b",
        )
        result = engine.select_best([close, farther], reference_text=reference)
        assert result.selected_engine == "engine-a"

    def test_hallucinating_engine_loses(self) -> None:
        """Output with 2x reference word count should be penalised."""
        engine = ConsensusEngine()
        reference = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
        faithful = _make_page_output(1, text=reference, engine="engine-good")
        hallucinated = _make_page_output(
            1,
            text=reference + " " + "hallucinated " * 20,
            engine="engine-bad",
        )
        result = engine.select_best(
            [faithful, hallucinated], reference_text=reference
        )
        assert result.selected_engine == "engine-good"

    def test_falls_back_to_ungrounded_with_empty_reference(self) -> None:
        """With empty reference text, should behave like ungrounded."""
        engine = ConsensusEngine()
        short = _make_page_output(1, "short text", engine="engine-a")
        long = _make_page_output(
            1, "a much longer text with many more words in it", engine="engine-b"
        )
        result = engine.select_best([short, long], reference_text="")
        # Ungrounded: longer text wins
        assert result.selected_engine == "engine-b"


# ---------------------------------------------------------------------------
# ConsensusEngine.select_best_with_llm
# ---------------------------------------------------------------------------


class TestSelectBestWithLLM:
    def test_falls_back_without_model(self) -> None:
        engine = ConsensusEngine(use_llm=True, ollama_model="")
        a = _make_page_output(1, "hello world foo", engine="engine-a")
        b = _make_page_output(1, "hello world bar", engine="engine-b")
        result = engine.select_best_with_llm([a, b])
        # Should fall back to heuristic (no model specified)
        assert result.selected_engine in ("engine-a", "engine-b")

    def test_falls_back_when_single_attempt(self) -> None:
        engine = ConsensusEngine(use_llm=True, ollama_model="llama3")
        a = _make_page_output(1, "only one attempt")
        result = engine.select_best_with_llm([a])
        assert result.selected_engine == "deepseek"
        assert result.merged_text == "only one attempt"

    def test_uses_llm_response(self) -> None:
        engine = ConsensusEngine(use_llm=True, ollama_model="llama3")
        a = _make_page_output(1, "text from engine a", engine="engine-a")
        b = _make_page_output(1, "text from engine b", engine="engine-b")

        llm_response = json.dumps({
            "selected": 2,
            "text": "text from engine b",
        })

        with patch(
            "socr.pipeline.consensus._call_ollama", return_value=llm_response
        ):
            result = engine.select_best_with_llm([a, b])

        assert result.selected_engine == "engine-b"
        assert result.merged_text == "text from engine b"

    def test_llm_merge_response(self) -> None:
        engine = ConsensusEngine(use_llm=True, ollama_model="llama3")
        a = _make_page_output(1, "text from engine a", engine="engine-a")
        b = _make_page_output(1, "text from engine b", engine="engine-b")

        llm_response = json.dumps({
            "selected": 0,
            "text": "merged best of both",
        })

        with patch(
            "socr.pipeline.consensus._call_ollama", return_value=llm_response
        ):
            result = engine.select_best_with_llm([a, b])

        assert result.selected_engine == "llm-merged"
        assert result.merged_text == "merged best of both"

    def test_falls_back_on_ollama_failure(self) -> None:
        engine = ConsensusEngine(use_llm=True, ollama_model="llama3")
        a = _make_page_output(1, "text from engine a" + " word" * 50, engine="engine-a")
        b = _make_page_output(1, "short", engine="engine-b")

        with patch(
            "socr.pipeline.consensus._call_ollama", return_value=None
        ):
            result = engine.select_best_with_llm([a, b])

        # Falls back to heuristic -- engine-a has more words
        assert result.selected_engine == "engine-a"

    def test_falls_back_on_bad_json(self) -> None:
        engine = ConsensusEngine(use_llm=True, ollama_model="llama3")
        a = _make_page_output(1, "text from engine a" + " word" * 50, engine="engine-a")
        b = _make_page_output(1, "short", engine="engine-b")

        with patch(
            "socr.pipeline.consensus._call_ollama",
            return_value="this is not valid json at all",
        ):
            result = engine.select_best_with_llm([a, b])

        assert result.selected_engine == "engine-a"

    def test_falls_back_on_empty_text_in_json(self) -> None:
        engine = ConsensusEngine(use_llm=True, ollama_model="llama3")
        a = _make_page_output(1, "text from engine a" + " word" * 50, engine="engine-a")
        b = _make_page_output(1, "short", engine="engine-b")

        llm_response = json.dumps({"selected": 1, "text": ""})

        with patch(
            "socr.pipeline.consensus._call_ollama", return_value=llm_response
        ):
            result = engine.select_best_with_llm([a, b])

        # Empty text -> parse returns None -> falls back to heuristic
        assert result.selected_engine == "engine-a"


# ---------------------------------------------------------------------------
# ConsensusEngine.reconcile_document
# ---------------------------------------------------------------------------


class TestReconcileDocument:
    def test_skips_pages_with_single_attempt(self) -> None:
        state = DocumentState(handle=_make_handle(2))
        state.apply_result(
            _make_engine_result(
                [_make_page_output(1, "page 1 text")], engine="deepseek"
            )
        )
        state.apply_result(
            _make_engine_result(
                [_make_page_output(2, "page 2 text")], engine="deepseek"
            )
        )

        engine = ConsensusEngine()
        results = engine.reconcile_document(state)
        assert len(results) == 0

    def test_processes_multi_attempt_pages(self) -> None:
        state = DocumentState(handle=_make_handle(2))
        # Page 1 gets two attempts
        state.apply_result(
            _make_engine_result(
                [
                    _make_page_output(1, "page 1 first", engine="engine-a"),
                    _make_page_output(2, "page 2 first", engine="engine-a"),
                ],
                engine="engine-a",
            )
        )
        state.apply_result(
            _make_engine_result(
                [
                    _make_page_output(
                        1, "page 1 second with more words and detail", engine="engine-b"
                    ),
                ],
                engine="engine-b",
            )
        )

        engine = ConsensusEngine()
        results = engine.reconcile_document(state)

        # Only page 1 has multiple attempts
        assert len(results) == 1
        assert results[0].page_num == 1

    def test_updates_best_output(self) -> None:
        state = DocumentState(handle=_make_handle(1))
        a = _make_page_output(1, "short", engine="engine-a")
        b = _make_page_output(
            1, "a longer version with more useful text and detail", engine="engine-b"
        )
        state.apply_result(_make_engine_result([a], engine="engine-a"))
        state.apply_result(_make_engine_result([b], engine="engine-b"))

        engine = ConsensusEngine()
        engine.reconcile_document(state)

        best = state.pages[1].best_output
        assert best is not None
        assert best.engine.startswith("consensus(")
        assert best.audit_passed is True

    def test_skips_born_digital_pages(self) -> None:
        state = DocumentState(handle=_make_handle(1))
        state.pages[1].is_born_digital = True
        state.pages[1].native_text = "native text"

        # Even if we add multiple attempts, born-digital should be skipped
        a = _make_page_output(1, "ocr text a", engine="engine-a")
        b = _make_page_output(1, "ocr text b", engine="engine-b")
        state.pages[1].attempts = [a, b]

        engine = ConsensusEngine()
        results = engine.reconcile_document(state)
        assert len(results) == 0

    def test_uses_llm_when_configured(self) -> None:
        state = DocumentState(handle=_make_handle(1))
        a = _make_page_output(1, "text a", engine="engine-a")
        b = _make_page_output(1, "text b", engine="engine-b")
        state.apply_result(_make_engine_result([a], engine="engine-a"))
        state.apply_result(_make_engine_result([b], engine="engine-b"))

        llm_response = json.dumps({
            "selected": 1,
            "text": "text a",
        })

        engine = ConsensusEngine(use_llm=True, ollama_model="llama3")
        with patch(
            "socr.pipeline.consensus._call_ollama", return_value=llm_response
        ):
            results = engine.reconcile_document(state)

        assert len(results) == 1
        assert results[0].selected_engine == "engine-a"

    def test_empty_document(self) -> None:
        state = DocumentState(handle=_make_handle(0))
        engine = ConsensusEngine()
        results = engine.reconcile_document(state)
        assert results == []

    def test_whole_doc_consensus_with_native_text(self) -> None:
        """Whole-doc consensus should assemble native text from all pages."""
        state = DocumentState(handle=_make_handle(2))
        # Set native text on both pages
        state.pages[1].native_text = "page one native text"
        state.pages[2].native_text = "page two native text"

        native_combined = "page one native text\n\npage two native text"

        # Two whole-doc attempts: one faithful, one hallucinated
        faithful = _make_page_output(
            0, text=native_combined, engine="engine-good"
        )
        hallucinated = _make_page_output(
            0,
            text=native_combined + " " + "garbage " * 30,
            engine="engine-bad",
        )
        state.whole_doc_attempts = [faithful, hallucinated]

        engine = ConsensusEngine()
        results = engine.reconcile_document(state)

        # Should have a whole-doc consensus result
        whole_doc_results = [r for r in results if r.page_num == 0]
        assert len(whole_doc_results) == 1
        assert whole_doc_results[0].selected_engine == "engine-good"

    def test_per_page_passes_native_text_as_reference(self) -> None:
        """Per-page consensus should pass native_text to select_best."""
        state = DocumentState(handle=_make_handle(1))
        # Page is NOT born-digital (no skip), but has native_text set
        # (this can happen for pages that need OCR enhancement)
        state.pages[1].is_born_digital = False
        state.pages[1].native_text = "alpha bravo charlie"

        # Two attempts: one close to native text, one hallucinated
        close = _make_page_output(1, text="alpha bravo charlie", engine="engine-close")
        far = _make_page_output(
            1,
            text="alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima " * 3,
            engine="engine-far",
        )
        state.pages[1].attempts = [close, far]

        engine = ConsensusEngine()
        results = engine.reconcile_document(state)

        assert len(results) == 1
        assert results[0].selected_engine == "engine-close"


# ---------------------------------------------------------------------------
# ConsensusResult dataclass
# ---------------------------------------------------------------------------


class TestConsensusResult:
    def test_defaults(self) -> None:
        r = ConsensusResult(
            page_num=1,
            selected_engine="deepseek",
            merged_text="hello",
            agreement_score=0.95,
        )
        assert r.discrepancies == []
        assert r.page_num == 1
        assert r.agreement_score == 0.95
