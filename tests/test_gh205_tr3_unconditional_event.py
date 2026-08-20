"""GH-205: the TR-3 per-region geometry hard-fail must be surfaced, always.

``has_unverifiable_table_region`` is computed on every native table page, but it
only ever reached a surface IN CONJUNCTION with something else — ``--native-only``
for the analyze-time ``table_structure_failed`` event, a failed OCR ladder for the
assemble-time D3-floor event, ``native_table_structure_failed`` for
``_winning_page_output``.  Measured over 32 papers / 245 native table pages, 62
pages (25.3%) carry the hard-fail; on the ones where no conjunction holds the
verdict reached no page status, no document status, no metadata field and no CLI
line.

SCOPE IS SURFACING ONLY.  25.3% is a firing rate, not a measured defect rate, and
the issue blocks any status or routing decision on hand-judging that 62-page set
first.  The second test below is a scope guard, not a nicety: it fails if this
change ever starts keying page status on the flag.

Hermetic: drives ``_phase_analyze`` with a stubbed detector — no PDF rendering, no
provider ladder, no ``_phase_agentic``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from socr.core.born_digital import DocumentAssessment, PageAssessment
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline


def _make_pipeline(**overrides) -> UnifiedPipeline:
    cfg = PipelineConfig(
        agentic=True,
        judge_backend="heuristic",
        enabled_engines=[EngineType.GEMINI],
        primary_engine=EngineType.DEEPSEEK,
        save_figures=False,
        dual_pass_tables=False,
        detect_equations=False,
        recover_clean_equations=False,
        quiet=True,
        audit_enabled=False,
        write_manifest=False,
        **overrides,
    )
    return UnifiedPipeline(cfg)


def _page(page_num: int, *, unverifiable: bool) -> PageAssessment:
    return PageAssessment(
        page_num=page_num,
        is_born_digital=True,
        native_text=f"native table text for page {page_num} " * 5,
        confidence=0.9,
        has_tables=True,
        has_unverifiable_table_region=unverifiable,
    )


def _run_analyze(pipeline: UnifiedPipeline, pages: list[PageAssessment]) -> DocumentState:
    pdf_path = Path("/tmp/gh205.pdf")
    assessment = DocumentAssessment(path=pdf_path, pages=pages)
    pipeline.bd_detector = MagicMock()
    pipeline.bd_detector.detect.return_value = assessment
    state = DocumentState(handle=DocumentHandle(path=pdf_path, page_count=len(pages)))
    pipeline._phase_analyze(state)
    return state


def test_tr3_hard_fail_is_surfaced_without_native_only(tmp_path: Path) -> None:
    """The conjunction-free case: an ordinary run, not ``--native-only``.

    This is the 62-page shape from the issue. At main_sha the flag is set on the
    assessment and nothing anywhere records it.
    """
    pipeline = _make_pipeline(native_only=False)
    state = _run_analyze(pipeline, [_page(1, unverifiable=True), _page(2, unverifiable=False)])

    tr3 = [e for e in state.events if e.kind == "table_region_unverifiable"]
    assert [e.page_num for e in tr3] == [1], (
        "The TR-3 geometry hard-fail on page 1 reached no surface at all; "
        f"recorded events were {[(e.page_num, e.kind) for e in state.events]}"
    )
    # The detection must reach the consumer-facing trust file, not just the log.
    from socr.core.tables_trust import TABLE_DISTRUST_KINDS

    assert "table_region_unverifiable" in TABLE_DISTRUST_KINDS


def test_surfacing_does_not_key_page_status_scope_guard(tmp_path: Path) -> None:
    """Scope guard: detection ONLY — no status, no routing, no demotion.

    The issue forbids acting on the hard-fail before its 62-page set is
    hand-judged: 25.3% is a firing rate, TR-3 shares the ``is_numeric_token``
    machinery whose notation gaps plausibly inflate it, and GH-151 B1 already cost
    three review rounds by treating a firing rate as a defect rate. Routing on it
    unattended could delete good tables.

    So on a page carrying ONLY the TR-3 flag, outside ``--native-only``:
      * no ``table_structure_failed`` event (that is the demotion event, and its
        detail asserts the page "can no longer pass as a trusted native page");
      * the page keeps its native-lane PageState untouched.
    """
    pipeline = _make_pipeline(native_only=False)
    state = _run_analyze(pipeline, [_page(1, unverifiable=True)])

    assert not [e for e in state.events if e.kind == "table_structure_failed"], (
        "Surfacing must not have become a demotion: a bare TR-3 hard-fail outside "
        "--native-only emitted the structure-failed event, which claims the page is "
        "no longer a trusted native page."
    )
    ps = state.pages[1]
    assert ps.is_born_digital is True
    assert ps.needs_ocr_enhancement is False, (
        "The TR-3 flag must not route the page to OCR — that is step 3, blocked on "
        "the hand-judgement."
    )
    assert ps.native_table_structure_failed is False
    assert ps.native_table_structure_defective is False


def test_clean_table_page_stays_quiet(tmp_path: Path) -> None:
    """Reverse regression: no flag, no event. A clean run leaves no noise."""
    pipeline = _make_pipeline(native_only=False)
    state = _run_analyze(pipeline, [_page(1, unverifiable=False), _page(2, unverifiable=False)])

    assert not [e for e in state.events if e.kind == "table_region_unverifiable"], state.events
