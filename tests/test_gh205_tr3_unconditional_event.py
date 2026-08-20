"""GH-205: the TR-3 per-region geometry hard-fail must be surfaced, always.

``has_unverifiable_table_region`` is computed on every native table page, but it
only ever reached a surface IN CONJUNCTION with something else — ``--native-only``
for the analyze-time ``table_structure_failed`` event, a failed OCR ladder for the
assemble-time D3-floor event, ``native_table_structure_failed`` for
``_winning_page_output``.  On a page where no conjunction holds the verdict
reached no page status, no document status, no metadata field and no CLI line.

SCOPE IS SURFACING ONLY.  The flag's firing rate is not a measured defect rate,
and the issue blocks any status or routing decision on hand-judging the firing
set first.  The scope guard below is not a nicety: it drives the whole of
``process()`` — analyze, agentic routing AND assemble — and fails if the
detection ever starts moving a status or a route.

The detection has its OWN kind, ``table_region_geometry_hard_fail``, distinct
from the D3 fail-closed ``table_region_unverifiable``: one says "detected, and
nothing acted on it", the other says "acted on it, the region went to the
image-asset lane".  A consumer of ``tables_trust.json`` must be able to tell
those apart, and a D3 page must not read as one kind counted twice.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import fitz
import pytest

from socr.core.born_digital import DocumentAssessment, PageAssessment
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import DocumentStatus, FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline

DETECTION_KIND = "table_region_geometry_hard_fail"
DISPOSITION_KIND = "table_region_unverifiable"


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

    At main_sha the flag is set on the assessment and nothing anywhere records it.
    """
    pipeline = _make_pipeline(native_only=False)
    state = _run_analyze(pipeline, [_page(1, unverifiable=True), _page(2, unverifiable=False)])

    tr3 = [e for e in state.events if e.kind == DETECTION_KIND]
    assert [e.page_num for e in tr3] == [1], (
        "The TR-3 geometry hard-fail on page 1 reached no surface at all; "
        f"recorded events were {[(e.page_num, e.kind) for e in state.events]}"
    )
    # The detection must reach the consumer-facing trust file, not just the log.
    from socr.core.tables_trust import TABLE_DISTRUST_KINDS

    assert DETECTION_KIND in TABLE_DISTRUST_KINDS


def test_detection_kind_is_distinct_from_the_d3_disposition_kind() -> None:
    """The two TR-3 kinds mean different things and must stay different kinds.

    ``table_region_unverifiable`` is emitted at assemble and means the OCR ladder
    also failed, so the region was routed to the image-asset lane. The analyze-time
    detection means only that TR-3 fired and nothing acted on it — the page's
    native text may still ship. Collapsing them makes ``tables_trust.json``
    unable to distinguish a routed page from an unrouted one, and makes a D3 page
    carry the same kind twice.
    """
    from socr.core.audit_log import build_run_audit
    from socr.core.tables_trust import TABLE_DISTRUST_KINDS

    assert DETECTION_KIND != DISPOSITION_KIND
    assert {DETECTION_KIND, DISPOSITION_KIND} <= TABLE_DISTRUST_KINDS

    pipeline = _make_pipeline(native_only=False)
    state = _run_analyze(pipeline, [_page(1, unverifiable=True)])
    kinds = [e.kind for e in state.events]
    assert kinds.count(DETECTION_KIND) == 1
    assert DISPOSITION_KIND not in kinds, (
        "analyze-time detection must not claim the D3 fail-closed disposition: "
        "nothing has been routed to the image-asset lane at this point"
    )

    # Both kinds are ranked, so a page carrying the detection AND the later D3
    # disposition reads top-to-bottom as "detected here, acted on there".
    audit = build_run_audit(state)
    ranked = [e.kind for e in audit.events if e.page_num == 1]
    assert ranked == [DETECTION_KIND]


def test_clean_table_page_stays_quiet(tmp_path: Path) -> None:
    """Reverse regression: no flag, no event. A clean run leaves no noise."""
    pipeline = _make_pipeline(native_only=False)
    state = _run_analyze(pipeline, [_page(1, unverifiable=False), _page(2, unverifiable=False)])

    assert not [e for e in state.events if e.kind == DETECTION_KIND], state.events


# ----------------------------------------------------------------------
# End-to-end: the detection must reach every surface, and move nothing.
#
# The two tests below drive the whole of ``process()`` — ``_phase_analyze``,
# ``_phase_agentic`` and ``_phase_assemble`` — because page status is decided in
# assemble and routing in agentic, and a guard that stops at analyze cannot see
# either. ``_available_engines_for_agentic`` is patched because CI has no ollama
# and no provider: without it the ladder is empty and the loop bails before
# routing, so the test would pass here and fail in CI.
# ----------------------------------------------------------------------


class _FixedDetector:
    def __init__(self, assessment: DocumentAssessment) -> None:
        self._assessment = assessment

    def detect(self, path: Path) -> DocumentAssessment:
        assert path == self._assessment.path
        return self._assessment


_FIXTURE_MARKDOWN = "\n".join(
    ["| Coefficient | Estimate | Standard error |", "| --- | --- | --- |"]
    + [f"| Coefficient {row} | {row}.12 | {row}.34 |" for row in range(10)]
)


def _native_table_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 70), "Estimate Standard error", fontsize=9)
    for row in range(10):
        y = 90 + row * 16
        page.insert_text((60, y), f"Coefficient {row}", fontsize=9)
        page.insert_text((260, y), f"{row}.12", fontsize=9)
    page.draw_line(fitz.Point(50, 60), fitz.Point(360, 60))
    page.draw_line(fitz.Point(50, 260), fitz.Point(360, 260))
    doc.save(path)
    doc.close()


def _run_process(tmp_path: Path, *, agentic: bool, unverifiable: bool):
    """Full ``process()`` run, NOT ``--native-only``, with the OCR ladder watched."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf_path = tmp_path / "paper.pdf"
    _native_table_pdf(pdf_path)
    native_text = _FIXTURE_MARKDOWN
    assessment = DocumentAssessment(
        path=pdf_path,
        pages=[
            PageAssessment(
                page_num=1,
                is_born_digital=True,
                native_text=native_text,
                confidence=1.0,
                has_tables=True,
                has_unverifiable_table_region=unverifiable,
            )
        ],
    )
    pipeline = UnifiedPipeline(
        PipelineConfig(
            agentic=agentic,
            native_first=True,
            native_only=False,
            primary_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            tiered=False,
            dual_pass_tables=False,
            detect_equations=False,
            save_figures=False,
            write_manifest=True,
            quiet=False,
        )
    )
    pipeline.bd_detector = _FixedDetector(assessment)
    # CI has no ollama and no provider: without this the agentic ladder is empty
    # and the loop bails before routing, so the run would never exercise routing.
    pipeline._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL]
    pipeline._surface_table_scoring = lambda *args, **kwargs: None

    # The OCR ladder is recorded, not forbidden: this run is NOT --native-only, so
    # a table page legitimately enters it. What #205 forbids is the TR-3 flag
    # CHANGING that decision, which is why the guard below compares a flagged run
    # against an unflagged one rather than asserting an absolute route.
    ocr_calls: list[tuple[int, ...]] = []

    def _record_ocr(_state, page_nums, _fallback, engine, _phase, *args, **kwargs):
        ocr_calls.append((tuple(page_nums), engine.value))
        return [
            PageOutput(
                page_num=n,
                text=native_text,
                status=PageStatus.SUCCESS,
                engine=engine.value,
                confidence=0.95,
                audit_passed=True,
            )
            for n in page_nums
        ]

    pipeline._run_engine_on_pages = _record_ocr

    captured: dict = {}
    assemble = pipeline._phase_assemble

    def _capture_state(state, output_dir):
        captured["state"] = state
        return assemble(state, output_dir)

    pipeline._phase_assemble = _capture_state
    output_dir = tmp_path / "out"
    result = pipeline.process(pdf_path, output_dir)
    return result, captured["state"], output_dir, ocr_calls


def _only(path: Path, pattern: str) -> Path:
    matches = list(path.rglob(pattern))
    assert len(matches) == 1, f"expected exactly one {pattern} under {path}, got {matches}"
    return matches[0]


@pytest.mark.parametrize("agentic", [False, True])
def test_tr3_detection_reaches_every_surface_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], agentic: bool
) -> None:
    """Cardinal rule: the detection must surface at page, document, metadata AND CLI.

    At the branch's base commit this page's TR-3 hard-fail reaches none of them in
    the non-agentic cell (no distrust event at all, hence no trust sidecar, no
    metadata note, no CLI line) and reaches them only under the D3 disposition's
    own name in the agentic cell.
    """
    capsys.readouterr()
    _result, state, output_dir, _ocr = _run_process(tmp_path, agentic=agentic, unverifiable=True)

    # 1. page level: the event is recorded against the page, exactly once
    detections = [e for e in state.events if e.kind == DETECTION_KIND]
    assert [e.page_num for e in detections] == [1], [(e.page_num, e.kind) for e in state.events]

    # 2. audit log
    audit = json.loads(_only(output_dir, "audit_log.json").read_text(encoding="utf-8"))
    assert audit["counts"][DETECTION_KIND] == 1

    # 3. trust sidecar, under the DETECTION kind's own name
    trust = json.loads(_only(output_dir, "tables_trust.json").read_text(encoding="utf-8"))
    assert trust["untrusted_pages"] == [1]
    assert DETECTION_KIND in trust["pages"]["1"]["reasons"], trust["pages"]["1"]["reasons"]

    # 4. document metadata
    doc_metadata = json.loads(
        _only(output_dir / "paper", "metadata.json").read_text(encoding="utf-8")
    )
    assert "untrusted tables" in (doc_metadata.get("error") or "")

    # 5. CLI
    cli_output = capsys.readouterr().out
    assert "untrusted tables" in cli_output


def test_d3_page_carries_detection_and_disposition_once_each(tmp_path: Path) -> None:
    """A D3 fail-closed page reads as "detected here, acted on there".

    Before this change both emissions used ``table_region_unverifiable``, so the
    same page carried one kind twice and a consumer could not tell a routed
    region from an unrouted detection. They are now two kinds, one each.
    """
    _result, state, output_dir, _ocr = _run_process(tmp_path, agentic=True, unverifiable=True)

    kinds = [e.kind for e in state.events if e.page_num == 1]
    assert kinds.count(DETECTION_KIND) == 1, kinds
    assert kinds.count(DISPOSITION_KIND) == 1, kinds

    audit = json.loads(_only(output_dir, "audit_log.json").read_text(encoding="utf-8"))
    assert audit["counts"][DETECTION_KIND] == 1
    assert audit["counts"][DISPOSITION_KIND] == 1
    # Ranked detection-before-disposition, so the page's story reads in order.
    page_kinds = [e["kind"] for e in audit["events"] if e["page_num"] == 1]
    assert page_kinds.index(DETECTION_KIND) < page_kinds.index(DISPOSITION_KIND), page_kinds


# Measured at the branch's base commit (7696719, i.e. the pipeline WITHOUT the
# GH-205 surfacing) for all four cells below. #205 forbids the surfacing keying
# page status, document status or routing on the TR-3 flag until its firing set
# is hand-judged, so these outcomes must not move. They are pinned rather than
# compared flagged-vs-clean because the flag already influences some of them
# through pre-existing paths (the D3 floor, native_fallback) that #205 does not
# touch; a flagged-vs-clean equality test would assert something untrue and a
# guard that stops at ``_phase_analyze`` would see none of it.
_PINNED_OUTCOMES = {
    # (agentic, unverifiable): (page status, audit_passed, failure mode, doc status)
    (False, True): (
        PageStatus.SUCCESS,
        False,
        FailureMode.AUDIT_FAILED,
        DocumentStatus.AUDIT_FAILED,
    ),
    (False, False): (PageStatus.SUCCESS, False, FailureMode.AUDIT_FAILED, DocumentStatus.SUCCESS),
    (True, True): (PageStatus.SUCCESS, False, FailureMode.NONE, DocumentStatus.ERROR),
    (True, False): (PageStatus.SUCCESS, False, FailureMode.NONE, DocumentStatus.AUDIT_FAILED),
}


@pytest.mark.parametrize("agentic", [False, True])
@pytest.mark.parametrize("unverifiable", [True, False])
def test_surfacing_does_not_key_page_status_or_routing_scope_guard(
    tmp_path: Path, agentic: bool, unverifiable: bool
) -> None:
    """Scope guard over the WHOLE pipeline: detection only — no status, no routing.

    #205 forbids acting on the hard-fail before its firing set is hand-judged: the
    firing rate is not a measured defect rate, TR-3 shares the ``is_numeric_token``
    machinery whose notation gaps plausibly inflate it, and GH-151 B1 already cost
    three review rounds by reading a firing rate as a defect rate. Routing on it
    unattended could delete good tables.

    A guard confined to ``_phase_analyze`` cannot enforce that: page status is
    decided in ``_phase_assemble`` and routing in ``_phase_agentic``. This drives
    the whole of ``process()`` and pins every status and routing outcome to its
    pre-change value, so any downstream branch that starts keying on the new
    event fails here.
    """
    result, state, _output_dir, ocr_calls = _run_process(
        tmp_path, agentic=agentic, unverifiable=unverifiable
    )
    expected = _PINNED_OUTCOMES[(agentic, unverifiable)]
    page = state.pages[1]
    assert page.attempts
    got = (
        page.attempts[0].status,
        page.attempts[0].audit_passed,
        page.attempts[0].failure_mode,
        result.status,
    )
    assert got == expected, (
        "the TR-3 surfacing moved a status the issue forbids it to move "
        f"(agentic={agentic}, flagged={unverifiable}): {got} != {expected}"
    )
    # Routing: the same single OCR call on the same page, flagged or not.
    assert ocr_calls == [((1,), "qwen")], ocr_calls
    # The --native-only demotion event must not have leaked out of --native-only.
    assert not [e for e in state.events if e.kind == "table_structure_failed"]
