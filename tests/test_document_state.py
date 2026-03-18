"""Tests for DocumentState blackboard and PageState."""

from pathlib import Path
from unittest.mock import patch

from socr.core.born_digital import DocumentAssessment, PageAssessment
from socr.core.document import DocumentHandle
from socr.core.result import (
    DocumentStatus,
    EngineResult,
    PageOutput,
    PageStatus,
)
from socr.core.state import DocumentState, PageState


# ---------------------------------------------------------------------------
# Helpers — build a DocumentHandle without touching the filesystem
# ---------------------------------------------------------------------------

def _make_handle(page_count: int = 3) -> DocumentHandle:
    """Create a DocumentHandle with a fake path and preset page count."""
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        h = DocumentHandle(path=Path("/tmp/fake.pdf"), page_count=page_count)
    return h


def _make_page_output(
    page_num: int,
    text: str = "some text",
    audit_passed: bool = True,
    engine: str = "deepseek",
) -> PageOutput:
    return PageOutput(
        page_num=page_num,
        text=text,
        status=PageStatus.SUCCESS,
        audit_passed=audit_passed,
        engine=engine,
    )


def _make_engine_result(
    pages: list[PageOutput],
    engine: str = "deepseek",
    cost: float = 0.0,
) -> EngineResult:
    return EngineResult(
        document_path=Path("/tmp/fake.pdf"),
        engine=engine,
        status=DocumentStatus.SUCCESS,
        pages=pages,
        cost=cost,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_creates_page_states_from_handle(self) -> None:
        handle = _make_handle(page_count=5)
        state = DocumentState(handle=handle)

        assert len(state.pages) == 5
        assert set(state.pages.keys()) == {1, 2, 3, 4, 5}
        for i in range(1, 6):
            assert state.pages[i].page_num == i
            assert not state.pages[i].is_born_digital
            assert state.pages[i].native_text is None
            assert state.pages[i].attempts == []
            assert state.pages[i].best_output is None

    def test_default_status_is_pending(self) -> None:
        state = DocumentState(handle=_make_handle())
        assert state.status == DocumentStatus.PENDING

    def test_empty_engine_runs_and_whole_doc(self) -> None:
        state = DocumentState(handle=_make_handle())
        assert state.engine_runs == []
        assert state.whole_doc_attempts == []

    def test_zero_page_document(self) -> None:
        handle = _make_handle(page_count=0)
        state = DocumentState(handle=handle)
        assert len(state.pages) == 0

    def test_preserves_existing_pages(self) -> None:
        """If pages dict is pre-populated, __post_init__ doesn't overwrite."""
        handle = _make_handle(page_count=2)
        custom = PageState(page_num=1, is_born_digital=True, native_text="hi")
        state = DocumentState(handle=handle, pages={1: custom})
        assert state.pages[1].is_born_digital is True
        assert state.pages[1].native_text == "hi"
        # page 2 should still be auto-created
        assert 2 in state.pages
        assert not state.pages[2].is_born_digital


# ---------------------------------------------------------------------------
# apply_result — per-page outputs
# ---------------------------------------------------------------------------

class TestApplyResultPerPage:
    def test_stores_attempts_on_correct_page(self) -> None:
        state = DocumentState(handle=_make_handle(3))
        p1 = _make_page_output(1, "page 1 text")
        p2 = _make_page_output(2, "page 2 text")
        result = _make_engine_result([p1, p2])

        state.apply_result(result)

        assert len(state.pages[1].attempts) == 1
        assert state.pages[1].attempts[0] is p1
        assert len(state.pages[2].attempts) == 1
        assert state.pages[2].attempts[0] is p2
        # page 3 untouched
        assert state.pages[3].attempts == []

    def test_sets_best_output_on_first_passing_attempt(self) -> None:
        state = DocumentState(handle=_make_handle(2))
        p1 = _make_page_output(1, "good", audit_passed=True)
        state.apply_result(_make_engine_result([p1]))

        assert state.pages[1].best_output is p1

    def test_does_not_replace_best_output_with_later_passing(self) -> None:
        state = DocumentState(handle=_make_handle(1))
        first = _make_page_output(1, "first", audit_passed=True, engine="engine-a")
        second = _make_page_output(1, "second", audit_passed=True, engine="engine-b")

        state.apply_result(_make_engine_result([first], engine="engine-a"))
        state.apply_result(_make_engine_result([second], engine="engine-b"))

        assert state.pages[1].best_output is first
        assert len(state.pages[1].attempts) == 2

    def test_skips_failed_audit_for_best_output(self) -> None:
        state = DocumentState(handle=_make_handle(1))
        bad = _make_page_output(1, "bad", audit_passed=False)
        good = _make_page_output(1, "good", audit_passed=True)

        state.apply_result(_make_engine_result([bad]))
        assert state.pages[1].best_output is None

        state.apply_result(_make_engine_result([good]))
        assert state.pages[1].best_output is good

    def test_ignores_unknown_page_numbers(self) -> None:
        state = DocumentState(handle=_make_handle(2))
        out = _make_page_output(99, "ghost page")
        state.apply_result(_make_engine_result([out]))

        # page 99 does not exist in pages dict — no crash, no side effects
        assert 99 not in state.pages

    def test_records_engine_run(self) -> None:
        state = DocumentState(handle=_make_handle(1))
        result = _make_engine_result([_make_page_output(1)], engine="gemini", cost=0.01)
        state.apply_result(result)

        assert len(state.engine_runs) == 1
        assert state.engine_runs[0] is result


# ---------------------------------------------------------------------------
# apply_result — whole-doc outputs (page_num=0)
# ---------------------------------------------------------------------------

class TestApplyResultWholeDoc:
    def test_whole_doc_output_goes_to_whole_doc_attempts(self) -> None:
        state = DocumentState(handle=_make_handle(2))
        whole = _make_page_output(0, "whole doc text")
        state.apply_result(_make_engine_result([whole]))

        assert len(state.whole_doc_attempts) == 1
        assert state.whole_doc_attempts[0] is whole
        # should NOT appear in per-page attempts
        assert state.pages[1].attempts == []
        assert state.pages[2].attempts == []

    def test_mixed_whole_doc_and_per_page(self) -> None:
        state = DocumentState(handle=_make_handle(2))
        whole = _make_page_output(0, "whole")
        p1 = _make_page_output(1, "page 1")
        state.apply_result(_make_engine_result([whole, p1]))

        assert len(state.whole_doc_attempts) == 1
        assert len(state.pages[1].attempts) == 1


# ---------------------------------------------------------------------------
# apply_born_digital
# ---------------------------------------------------------------------------

class TestApplyBornDigital:
    def test_marks_born_digital_pages(self) -> None:
        state = DocumentState(handle=_make_handle(3))
        assessment = DocumentAssessment(
            path=Path("/tmp/fake.pdf"),
            pages=[
                PageAssessment(page_num=1, is_born_digital=True, native_text="native p1", confidence=0.9),
                PageAssessment(page_num=2, is_born_digital=False, native_text="", confidence=0.95),
                PageAssessment(page_num=3, is_born_digital=True, native_text="native p3", confidence=0.85),
            ],
        )
        state.apply_born_digital(assessment)

        assert state.pages[1].is_born_digital is True
        assert state.pages[1].native_text == "native p1"
        assert state.pages[2].is_born_digital is False
        assert state.pages[2].native_text is None  # not set for scanned
        assert state.pages[3].is_born_digital is True
        assert state.pages[3].native_text == "native p3"

    def test_ignores_assessment_for_unknown_pages(self) -> None:
        state = DocumentState(handle=_make_handle(1))
        assessment = DocumentAssessment(
            path=Path("/tmp/fake.pdf"),
            pages=[
                PageAssessment(page_num=1, is_born_digital=True, native_text="ok", confidence=0.9),
                PageAssessment(page_num=5, is_born_digital=True, native_text="ghost", confidence=0.9),
            ],
        )
        state.apply_born_digital(assessment)

        assert state.pages[1].is_born_digital is True
        assert 5 not in state.pages


# ---------------------------------------------------------------------------
# needs_repair
# ---------------------------------------------------------------------------

class TestNeedsRepair:
    def test_fresh_page_needs_repair(self) -> None:
        ps = PageState(page_num=1)
        assert ps.needs_repair

    def test_born_digital_with_text_does_not_need_repair(self) -> None:
        ps = PageState(page_num=1, is_born_digital=True, native_text="good text")
        assert not ps.needs_repair

    def test_born_digital_without_text_needs_repair(self) -> None:
        ps = PageState(page_num=1, is_born_digital=True, native_text=None)
        assert ps.needs_repair

    def test_best_output_passing_audit_does_not_need_repair(self) -> None:
        ps = PageState(
            page_num=1,
            best_output=_make_page_output(1, "ok", audit_passed=True),
        )
        assert not ps.needs_repair

    def test_best_output_failing_audit_needs_repair(self) -> None:
        ps = PageState(
            page_num=1,
            best_output=_make_page_output(1, "bad", audit_passed=False),
        )
        assert ps.needs_repair

    def test_ocr_enhancement_needs_repair_when_no_attempts(self) -> None:
        """Born-digital page with complex content needs repair if no OCR attempted."""
        ps = PageState(
            page_num=1,
            is_born_digital=True,
            native_text="native text with table",
            needs_ocr_enhancement=True,
        )
        assert ps.needs_repair

    def test_ocr_enhancement_no_repair_after_passing_ocr(self) -> None:
        """Born-digital page with complex content is done after passing OCR."""
        ps = PageState(
            page_num=1,
            is_born_digital=True,
            native_text="native text with table",
            needs_ocr_enhancement=True,
            best_output=_make_page_output(1, "ocr text", audit_passed=True),
        )
        assert not ps.needs_repair

    def test_ocr_enhancement_no_repair_after_failed_ocr(self) -> None:
        """Born-digital page falls back to native text when OCR fails."""
        failed = _make_page_output(1, "bad ocr", audit_passed=False)
        ps = PageState(
            page_num=1,
            is_born_digital=True,
            native_text="native fallback",
            needs_ocr_enhancement=True,
            attempts=[failed],
        )
        assert not ps.needs_repair

    def test_ocr_enhancement_propagated_via_apply_born_digital(self) -> None:
        """apply_born_digital propagates needs_ocr_enhancement to PageState."""
        state = DocumentState(handle=_make_handle(2))
        assessment = DocumentAssessment(
            path=Path("/tmp/fake.pdf"),
            pages=[
                PageAssessment(
                    page_num=1,
                    is_born_digital=True,
                    native_text="table page",
                    confidence=0.9,
                    needs_ocr_enhancement=True,
                ),
                PageAssessment(
                    page_num=2,
                    is_born_digital=True,
                    native_text="prose page",
                    confidence=0.9,
                    needs_ocr_enhancement=False,
                ),
            ],
        )
        state.apply_born_digital(assessment)

        assert state.pages[1].needs_ocr_enhancement is True
        assert state.pages[2].needs_ocr_enhancement is False
        # Page 1 needs repair (OCR not attempted), page 2 does not
        assert state.pages[1].needs_repair
        assert not state.pages[2].needs_repair


# ---------------------------------------------------------------------------
# text assembly
# ---------------------------------------------------------------------------

class TestTextAssembly:
    def test_per_page_text(self) -> None:
        state = DocumentState(handle=_make_handle(3))
        state.apply_result(_make_engine_result([
            _make_page_output(1, "alpha"),
            _make_page_output(2, "beta"),
            _make_page_output(3, "gamma"),
        ]))
        assert state.text == "alpha\n\n---\n\nbeta\n\n---\n\ngamma"

    def test_whole_doc_fallback(self) -> None:
        """When no per-page best outputs exist, use the last whole-doc attempt."""
        state = DocumentState(handle=_make_handle(2))
        state.apply_result(_make_engine_result([
            _make_page_output(0, "full doc text"),
        ]))
        assert state.text == "full doc text"

    def test_whole_doc_uses_last_attempt(self) -> None:
        state = DocumentState(handle=_make_handle(1))
        state.apply_result(_make_engine_result([
            _make_page_output(0, "attempt 1"),
        ], engine="engine-a"))
        state.apply_result(_make_engine_result([
            _make_page_output(0, "attempt 2"),
        ], engine="engine-b"))

        assert state.text == "attempt 2"

    def test_born_digital_pages_use_native_text(self) -> None:
        state = DocumentState(handle=_make_handle(2))
        # Mark page 1 as born-digital
        assessment = DocumentAssessment(
            path=Path("/tmp/fake.pdf"),
            pages=[
                PageAssessment(page_num=1, is_born_digital=True, native_text="native text", confidence=0.9),
                PageAssessment(page_num=2, is_born_digital=False, native_text="", confidence=0.9),
            ],
        )
        state.apply_born_digital(assessment)
        # Provide OCR output only for page 2
        state.apply_result(_make_engine_result([
            _make_page_output(2, "ocr text"),
        ]))

        assert state.text == "native text\n\n---\n\nocr text"

    def test_empty_text_when_no_outputs(self) -> None:
        state = DocumentState(handle=_make_handle(2))
        assert state.text == ""

    def test_pages_without_output_are_skipped(self) -> None:
        """Pages with no best_output and not born-digital are excluded."""
        state = DocumentState(handle=_make_handle(3))
        state.apply_result(_make_engine_result([
            _make_page_output(1, "alpha"),
            # page 2 has no output
            _make_page_output(3, "gamma"),
        ]))
        assert state.text == "alpha\n\n---\n\ngamma"

    def test_per_page_takes_priority_over_whole_doc(self) -> None:
        """If any per-page best_output exists, whole-doc fallback is not used."""
        state = DocumentState(handle=_make_handle(2))
        state.apply_result(_make_engine_result([
            _make_page_output(0, "whole doc text"),
            _make_page_output(1, "page 1 text"),
        ]))
        # page 1 has best_output, so whole-doc is not used
        assert state.text == "page 1 text"

    def test_ocr_preferred_over_native_text_when_both_exist(self) -> None:
        """When a born-digital page also has passing OCR, OCR wins."""
        state = DocumentState(handle=_make_handle(1))
        # Mark as born-digital with native text
        assessment = DocumentAssessment(
            path=Path("/tmp/fake.pdf"),
            pages=[
                PageAssessment(
                    page_num=1,
                    is_born_digital=True,
                    native_text="native text",
                    confidence=0.9,
                    needs_ocr_enhancement=True,
                ),
            ],
        )
        state.apply_born_digital(assessment)
        # OCR also succeeds
        state.apply_result(_make_engine_result([
            _make_page_output(1, "ocr text with table", audit_passed=True),
        ]))
        # OCR output should be preferred
        assert state.text == "ocr text with table"

    def test_native_text_used_when_ocr_fails(self) -> None:
        """When OCR fails, fall back to native text."""
        state = DocumentState(handle=_make_handle(1))
        assessment = DocumentAssessment(
            path=Path("/tmp/fake.pdf"),
            pages=[
                PageAssessment(
                    page_num=1,
                    is_born_digital=True,
                    native_text="native fallback",
                    confidence=0.9,
                    needs_ocr_enhancement=True,
                ),
            ],
        )
        state.apply_born_digital(assessment)
        # OCR fails
        state.apply_result(_make_engine_result([
            _make_page_output(1, "bad ocr", audit_passed=False),
        ]))
        # Native text should be the fallback
        assert state.text == "native fallback"


# ---------------------------------------------------------------------------
# pages_needing_repair
# ---------------------------------------------------------------------------

class TestPagesNeedingRepair:
    def test_all_pages_need_repair_initially(self) -> None:
        state = DocumentState(handle=_make_handle(3))
        assert state.pages_needing_repair == [1, 2, 3]

    def test_repaired_pages_removed(self) -> None:
        state = DocumentState(handle=_make_handle(3))
        state.apply_result(_make_engine_result([
            _make_page_output(1, "good", audit_passed=True),
            _make_page_output(2, "bad", audit_passed=False),
        ]))
        # page 1 is good, page 2 failed audit, page 3 has no output
        assert state.pages_needing_repair == [2, 3]

    def test_born_digital_not_needing_repair(self) -> None:
        state = DocumentState(handle=_make_handle(2))
        assessment = DocumentAssessment(
            path=Path("/tmp/fake.pdf"),
            pages=[
                PageAssessment(page_num=1, is_born_digital=True, native_text="native", confidence=0.9),
                PageAssessment(page_num=2, is_born_digital=False, native_text="", confidence=0.9),
            ],
        )
        state.apply_born_digital(assessment)
        # page 1 is born-digital with text, page 2 is scanned with no output
        assert state.pages_needing_repair == [2]

    def test_empty_when_all_done(self) -> None:
        state = DocumentState(handle=_make_handle(2))
        state.apply_result(_make_engine_result([
            _make_page_output(1, "ok"),
            _make_page_output(2, "ok"),
        ]))
        assert state.pages_needing_repair == []


# ---------------------------------------------------------------------------
# total_cost and engines_used
# ---------------------------------------------------------------------------

class TestTelemetry:
    def test_total_cost_sums_all_runs(self) -> None:
        state = DocumentState(handle=_make_handle(1))
        state.apply_result(_make_engine_result(
            [_make_page_output(1, "a")], engine="deepseek", cost=0.002,
        ))
        state.apply_result(_make_engine_result(
            [_make_page_output(1, "b")], engine="gemini", cost=0.005,
        ))
        assert abs(state.total_cost - 0.007) < 1e-9

    def test_total_cost_zero_initially(self) -> None:
        state = DocumentState(handle=_make_handle(1))
        assert state.total_cost == 0.0

    def test_engines_used_preserves_order_and_deduplicates(self) -> None:
        state = DocumentState(handle=_make_handle(1))
        state.apply_result(_make_engine_result([], engine="deepseek", cost=0.0))
        state.apply_result(_make_engine_result([], engine="gemini", cost=0.0))
        state.apply_result(_make_engine_result([], engine="deepseek", cost=0.0))

        assert state.engines_used == ["deepseek", "gemini"]

    def test_engines_used_empty_initially(self) -> None:
        state = DocumentState(handle=_make_handle(1))
        assert state.engines_used == []


# ---------------------------------------------------------------------------
# Import from socr.core
# ---------------------------------------------------------------------------

class TestExports:
    def test_importable_from_core_package(self) -> None:
        from socr.core import DocumentState as DS, PageState as PS
        assert DS is DocumentState
        assert PS is PageState
