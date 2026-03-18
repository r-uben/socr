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
    _count_structure,
    _jaccard,
    _pairwise_agreement,
    _score_attempt,
    _word_set,
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
# Scoring helpers
# ---------------------------------------------------------------------------


class TestWordSet:
    def test_basic(self) -> None:
        assert _word_set("Hello World") == {"hello", "world"}

    def test_empty(self) -> None:
        assert _word_set("") == set()


class TestJaccard:
    def test_identical(self) -> None:
        assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint(self) -> None:
        assert _jaccard({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial_overlap(self) -> None:
        # {a,b} & {b,c} = {b}, union = {a,b,c}, similarity = 1/3
        assert abs(_jaccard({"a", "b"}, {"b", "c"}) - 1 / 3) < 1e-9

    def test_both_empty(self) -> None:
        assert _jaccard(set(), set()) == 1.0


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


class TestScoreAttempt:
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

    def test_longer_text_wins(self) -> None:
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

        # Falls back to heuristic — engine-a has more words
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
