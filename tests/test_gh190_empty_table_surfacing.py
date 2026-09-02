"""GH-190: Surfacing layers for empty-table structural rejection.

CLAUDE.md governing rule: no silent content loss. Failures must surface at every level:
- Judge seam: NativeTableVerifierJudge._apply_structural_gate rejects empty/dash tables and emits
  table_structure_failed AuditEvent with detail='table_content_empty'.
- Trust seam: build_tables_trust records the page in untrusted_pages, counts the defect kind,
  and trust_note produces the document-level error pointer for metadata.json.
- Manifest read-only seam: kept_table_grid_defect identifies the defect without mutating
  manifest.py.
- End-to-end pipeline run: hermetic paired execution showing that changing only one body cell
  from empty/placeholder to populated moves the document from AUDIT_FAILED with table distrust
  to clean SUCCESS.
- CLI seam: Click process command displays 'Completed with warnings' when audit fails.

Hermetic:
- No live provider/model calls; CI has no ollama and no model API keys.
- _available_engines_for_agentic is patched whenever pipeline execution is involved.
- All assertions compare concrete objects and strings; no bare MagicMock negative assertions.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from socr.core.audit_log import AuditEvent
from socr.core.born_digital import BornDigitalDetector
from socr.core.config import EngineType, PipelineConfig
from socr.core.manifest import kept_table_grid_defect
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import (
    DocumentStatus,
    EngineResult,
    FailureMode,
    PageOutput,
    PageStatus,
)
from socr.core.tables_trust import (
    TRUST_NOTE_PREFIX,
    build_tables_trust,
    trust_note,
)
from socr.pipeline.agentic import (
    AcceptDecision,
    NativeTableVerifierJudge,
    PageDecision,
    ProviderAttempt,
)
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.structure_check import DEFECT_TABLE_CONTENT_EMPTY

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

EMPTY_TABLE_FIXTURE = (
    "| | | | | | | | | | |\n|---|---|---|---|---|---|---|---|---|---|\n| | | | | | | | | | |\n"
)

PAIRED_POPULATED_TABLE_FIXTURE = (
    "| | | | | | | | | | |\n|---|---|---|---|---|---|---|---|---|---|\n| 1 | | | | | | | | | |\n"
)

DASH_PLACEHOLDER_TABLE_FIXTURE = "| Col A | Col B |\n| --- | --- |\n| - | — |\n"


class _StubAcceptingInnerJudge:
    """Concrete inner judge stub that unconditionally accepts."""

    def assess(self, output: PageOutput, state: object) -> AcceptDecision:
        return AcceptDecision(accept=True, reason="inner stub accepted", confidence=1.0)


class _PipelineStub:
    """Concrete process-command seam returning a pre-built result."""

    def __init__(self, result: EngineResult) -> None:
        self._result = result

    def process(self, pdf_path: Path, output_dir: Path | None = None) -> EngineResult:
        return self._result


def _persisted_document_metadata(output_dir: Path) -> dict:
    """Load the per-document metadata object, excluding the root index."""
    for path in output_dir.rglob("metadata.json"):
        payload = json.loads(path.read_text())
        if "status" in payload:
            return payload
    raise AssertionError(f"no per-document metadata.json under {output_dir}")


# --------------------------------------------------------------------------
# 1. Judge Seam: NativeTableVerifierJudge._apply_structural_gate
# --------------------------------------------------------------------------


def test_judge_structural_gate_rejects_empty_table_with_audit_event() -> None:
    """An empty or dash-placeholder table changes accept=True to False with an audit event."""
    assert DEFECT_TABLE_CONTENT_EMPTY == "table_content_empty"
    events: list[AuditEvent] = []

    judge = NativeTableVerifierJudge(
        inner=_StubAcceptingInnerJudge(),
        get_fitz_page=lambda pn: None,
        is_table_page=lambda pn: True,
        record_event=events.append,
    )

    output = PageOutput(
        page_num=1,
        text=EMPTY_TABLE_FIXTURE,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    decision_in = AcceptDecision(accept=True, reason="inner stub accepted", confidence=1.0)

    decision_out = judge._apply_structural_gate(
        decision_in, output, page_num=1, words=None, rules=None
    )

    # Decision must be rejected
    assert decision_out.accept is False
    assert decision_out.reason == f"table_structure_failed: {DEFECT_TABLE_CONTENT_EMPTY}"

    # Exactly one AuditEvent emitted with table_structure_failed and defect detail
    assert events == [
        AuditEvent(
            page_num=1,
            kind="table_structure_failed",
            engine="qwen",
            detail=DEFECT_TABLE_CONTENT_EMPTY,
            data={"defect": DEFECT_TABLE_CONTENT_EMPTY},
        )
    ]


def test_judge_structural_gate_rejects_dash_placeholder_table() -> None:
    """A dash/hyphen placeholder table is also rejected by the structural gate."""
    events: list[AuditEvent] = []

    judge = NativeTableVerifierJudge(
        inner=_StubAcceptingInnerJudge(),
        get_fitz_page=lambda pn: None,
        is_table_page=lambda pn: True,
        record_event=events.append,
    )

    output = PageOutput(
        page_num=1,
        text=DASH_PLACEHOLDER_TABLE_FIXTURE,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    decision_in = AcceptDecision(accept=True, reason="inner stub accepted")

    decision_out = judge._apply_structural_gate(
        decision_in, output, page_num=1, words=None, rules=None
    )

    assert decision_out.accept is False
    assert decision_out.reason == f"table_structure_failed: {DEFECT_TABLE_CONTENT_EMPTY}"
    assert events == [
        AuditEvent(
            page_num=1,
            kind="table_structure_failed",
            engine="qwen",
            detail=DEFECT_TABLE_CONTENT_EMPTY,
            data={"defect": DEFECT_TABLE_CONTENT_EMPTY},
        )
    ]


def test_judge_structural_gate_passes_populated_table_by_identity() -> None:
    """A paired populated table passes through the gate by identity and emits no event."""
    events: list[AuditEvent] = []

    judge = NativeTableVerifierJudge(
        inner=_StubAcceptingInnerJudge(),
        get_fitz_page=lambda pn: None,
        is_table_page=lambda pn: True,
        record_event=events.append,
    )

    output = PageOutput(
        page_num=1,
        text=PAIRED_POPULATED_TABLE_FIXTURE,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    decision_in = AcceptDecision(accept=True, reason="inner stub accepted", confidence=1.0)

    decision_out = judge._apply_structural_gate(
        decision_in, output, page_num=1, words=None, rules=None
    )

    # Returns the exact same decision object by identity
    assert decision_out is decision_in
    assert len(events) == 0


def test_judge_structural_gate_leaves_pre_rejected_decision_unchanged() -> None:
    """A decision that was already rejected by the inner judge is returned unchanged."""
    events: list[AuditEvent] = []

    judge = NativeTableVerifierJudge(
        inner=_StubAcceptingInnerJudge(),
        get_fitz_page=lambda pn: None,
        is_table_page=lambda pn: True,
        record_event=events.append,
    )

    output = PageOutput(
        page_num=1,
        text=EMPTY_TABLE_FIXTURE,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=False,
    )
    decision_in = AcceptDecision(
        accept=False, reason="already rejected by verifier", confidence=0.0
    )

    decision_out = judge._apply_structural_gate(
        decision_in, output, page_num=1, words=None, rules=None
    )

    assert decision_out is decision_in
    assert len(events) == 0


# --------------------------------------------------------------------------
# 2. Trust Seam: build_tables_trust & trust_note
# --------------------------------------------------------------------------


def test_tables_trust_captures_table_structure_failed_content_empty() -> None:
    """build_tables_trust marks page untrusted and trust_note formats document error."""
    event = AuditEvent(
        page_num=1,
        kind="table_structure_failed",
        engine="qwen",
        detail=DEFECT_TABLE_CONTENT_EMPTY,
        data={"defect": DEFECT_TABLE_CONTENT_EMPTY},
    )

    trust = build_tables_trust("paper.pdf", [event])

    assert trust.untrusted_pages == [1]
    assert trust.counts_by_kind().get("table_structure_failed") == 1
    assert trust.flag_count == 1

    note = trust_note(trust)
    assert note is not None
    assert TRUST_NOTE_PREFIX in note
    assert "1 page(s)" in note
    assert "see tables_trust.json" in note


# --------------------------------------------------------------------------
# 3. Manifest Read-Only Seam: kept_table_grid_defect
# --------------------------------------------------------------------------


def test_manifest_kept_table_grid_defect_detects_empty_table() -> None:
    """kept_table_grid_defect identifies the content defect without editing manifest.py."""
    defect_empty = kept_table_grid_defect(EMPTY_TABLE_FIXTURE)
    defect_pop = kept_table_grid_defect(PAIRED_POPULATED_TABLE_FIXTURE)

    assert defect_empty == DEFECT_TABLE_CONTENT_EMPTY
    assert defect_pop == ""


# --------------------------------------------------------------------------
# 4. Detector Seam: BornDigitalDetector native-table assessment (GH-190 cold review)
# --------------------------------------------------------------------------


def test_detector_native_table_structure_defective_paired_empty_vs_populated(
    tmp_path: Path,
) -> None:
    """Paired detector-level regression for born-digital native-lane empty-table gate.

    BornDigitalDetector computes native_table_structure_defective from raw emission,
    raw content, and parsed shape defects. For a born-digital page with structured
    extraction:
    - Empty fixture -> native_table_structure_defective is True
    - Populated fixture -> native_table_structure_defective is False
    """
    fitz = pytest.importorskip("fitz")

    # Create a 1-page PDF fixture with clean prose longer than MIN_CHARS_FOR_TEXT_LAYER (50)
    pdf_path = tmp_path / "detector_doc.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (54, 72),
        "This is clean born-digital text that is definitely longer than fifty characters in length.",
    )
    doc.save(str(pdf_path))
    doc.close()

    # Precondition assertion: text layer exceeds threshold
    doc_check = fitz.open(str(pdf_path))
    assert len(doc_check[0].get_text().strip()) > BornDigitalDetector.MIN_CHARS_FOR_TEXT_LAYER
    doc_check.close()

    detector = BornDigitalDetector()

    # 1. Run detector with EMPTY_TABLE_FIXTURE
    with (
        patch.object(BornDigitalDetector, "_detect_tables", return_value=True),
        patch.object(
            BornDigitalDetector,
            "extract_structured",
            return_value=EMPTY_TABLE_FIXTURE,
        ) as mock_extract_empty,
    ):
        assessment_empty = detector.detect(pdf_path)

    # 2. Run detector with PAIRED_POPULATED_TABLE_FIXTURE
    with (
        patch.object(BornDigitalDetector, "_detect_tables", return_value=True),
        patch.object(
            BornDigitalDetector,
            "extract_structured",
            return_value=PAIRED_POPULATED_TABLE_FIXTURE,
        ) as mock_extract_pop,
    ):
        assessment_pop = detector.detect(pdf_path)

    # Precondition assertions on setup: structured extraction seam was called
    assert mock_extract_empty.called, "extract_structured must be called for empty fixture"
    assert mock_extract_pop.called, "extract_structured must be called for populated fixture"

    page_empty = assessment_empty.pages[0]
    page_pop = assessment_pop.pages[0]

    assert assessment_empty.is_fully_born_digital is True
    assert page_empty.is_born_digital is True
    assert page_empty.has_tables is True

    assert assessment_pop.is_fully_born_digital is True
    assert page_pop.is_born_digital is True
    assert page_pop.has_tables is True

    # Paired difference assertion
    assert page_empty.native_table_structure_defective is True
    assert page_pop.native_table_structure_defective is False


# --------------------------------------------------------------------------
# 5. End-to-End Paired Pipeline Run (Hermetic UnifiedPipeline)
# --------------------------------------------------------------------------


def test_paired_pipeline_run_empty_vs_populated_differs_at_all_surfaces(
    tmp_path: Path,
) -> None:
    """End-to-end paired run: changing exactly one body cell flips all validation surfaces.

    Both runs execute the full pipeline under identical conditions except the text of
    one cell:
    - Empty run: gains table_structure_failed, DocumentStatus.AUDIT_FAILED,
      PageStatus.WARNING sidecar, tables_trust.json untrusted entry, and metadata note.
    - Populated run: achieves DocumentStatus.SUCCESS, PageStatus.SUCCESS, no trust flags.
    """
    fitz = pytest.importorskip("fitz")

    # Create a 1-page PDF fixture with clean prose longer than MIN_CHARS_FOR_TEXT_LAYER (50)
    pdf_path = tmp_path / "table_doc.pdf"
    doc = fitz.open()
    page = doc.new_page()
    prose = (
        "Table 1. Experimental Summary of treatment outcomes across all patient cohorts "
        "and control groups evaluated over the thirty-day observation period."
    )
    page.insert_text((54, 72), prose)
    doc.save(str(pdf_path))
    doc.close()

    # Preconditions: fixture text exceeds threshold and is classified born-digital
    doc_check = fitz.open(str(pdf_path))
    extracted_text = doc_check[0].get_text()
    doc_check.close()
    assert len(extracted_text.strip()) > BornDigitalDetector.MIN_CHARS_FOR_TEXT_LAYER

    pre_assessment = BornDigitalDetector().detect(pdf_path)
    assert pre_assessment.is_fully_born_digital is True
    assert pre_assessment.pages[0].is_born_digital is True

    def _make_route_fn(table_text: str):
        def _route(page_num, ladder, run_provider, judge, **kwargs):
            out = PageOutput(
                page_num=page_num,
                text=table_text,
                status=PageStatus.SUCCESS,
                engine="qwen",
                audit_passed=True,
            )
            prof = ladder[0]
            decision = judge.assess(out, prof)
            att = ProviderAttempt(
                engine=prof.engine,
                output=out,
                cost_usd=0.0,
                accepted=decision.accept,
                reason=decision.reason,
                provider_id=prof.id,
                model=prof.model,
                backend=prof.backend,
            )
            return PageDecision(
                page_num=page_num,
                final_output=out,
                attempts=[att],
                accepted=decision.accept,
            )

        return _route

    config = PipelineConfig(
        primary_engine=EngineType.QWEN,
        agentic=True,
        judge_backend="heuristic",
        enabled_engines=[EngineType.QWEN],
        quiet=True,
        save_figures=False,
        write_manifest=True,
        native_first=False,
        dual_pass_tables=False,
        escalate_ambiguous_tables=False,
    )

    # 1. Run with empty table
    pipeline_empty = UnifiedPipeline(config)
    out_empty = tmp_path / "out_empty"
    with (
        patch.object(
            pipeline_empty,
            "_available_engines_for_agentic",
            return_value=[PROFILE_QWEN_LOCAL],
        ),
        patch(
            "socr.pipeline.orchestrator.route_page",
            side_effect=_make_route_fn(EMPTY_TABLE_FIXTURE),
        ) as mock_route_empty,
        patch(
            "socr.pipeline.agentic.HeuristicPageJudge.assess",
            return_value=AcceptDecision(accept=True, reason="heuristics passed"),
        ),
        patch("socr.pipeline.orchestrator.probe_ollama_idle", return_value=True),
    ):
        result_empty = pipeline_empty.process(pdf_path, out_empty)

    assert mock_route_empty.call_count == 1, "route_page must be called for empty table run"

    # 2. Run with populated table (control)
    pipeline_pop = UnifiedPipeline(config)
    out_pop = tmp_path / "out_pop"
    with (
        patch.object(
            pipeline_pop,
            "_available_engines_for_agentic",
            return_value=[PROFILE_QWEN_LOCAL],
        ),
        patch(
            "socr.pipeline.orchestrator.route_page",
            side_effect=_make_route_fn(PAIRED_POPULATED_TABLE_FIXTURE),
        ) as mock_route_pop,
        patch(
            "socr.pipeline.agentic.HeuristicPageJudge.assess",
            return_value=AcceptDecision(accept=True, reason="heuristics passed"),
        ),
        patch("socr.pipeline.orchestrator.probe_ollama_idle", return_value=True),
    ):
        result_pop = pipeline_pop.process(pdf_path, out_pop)

    assert mock_route_pop.call_count == 1, "route_page must be called for populated table run"

    # --- DIFFERENCE ASSERTIONS ---

    # Document-level status
    assert result_empty.status is DocumentStatus.AUDIT_FAILED
    assert result_empty.audit_passed is False
    assert result_pop.status is DocumentStatus.SUCCESS
    assert result_pop.audit_passed is True

    # Page-level sidecar JSON
    sidecar_empty = json.loads(next(out_empty.rglob("pages/00001.json")).read_text())
    sidecar_pop = json.loads(next(out_pop.rglob("pages/00001.json")).read_text())

    assert sidecar_empty["winning_output"]["audit_passed"] is False
    assert sidecar_empty["winning_output"]["status"] == PageStatus.WARNING.value
    assert (
        sidecar_empty["winning_output"]["failure_mode"]
        == FailureMode.NATIVE_TABLE_STRUCTURE_FAILED.value
    )

    assert sidecar_pop["winning_output"]["audit_passed"] is True
    assert sidecar_pop["winning_output"]["status"] == PageStatus.SUCCESS.value

    # Audit log JSON events: only the rejected run needs an audit artifact.
    empty_audit_paths = list(out_empty.rglob("audit_log.json"))
    populated_audit_paths = list(out_pop.rglob("audit_log.json"))
    assert len(empty_audit_paths) == 1
    assert populated_audit_paths == []
    audit_empty = json.loads(empty_audit_paths[0].read_text())

    kinds_empty = [e.get("kind") for e in audit_empty.get("events", [])]
    assert "table_structure_failed" in kinds_empty

    defect_event = next(
        e for e in audit_empty["events"] if e.get("kind") == "table_structure_failed"
    )
    assert defect_event["data"]["defect"] == DEFECT_TABLE_CONTENT_EMPTY

    # Tables trust sidecar JSON: only the rejected run needs a trust artifact.
    empty_trust_paths = list(out_empty.rglob("tables_trust.json"))
    populated_trust_paths = list(out_pop.rglob("tables_trust.json"))
    assert len(empty_trust_paths) == 1
    assert populated_trust_paths == []
    trust_empty = json.loads(empty_trust_paths[0].read_text())

    assert 1 in trust_empty["untrusted_pages"]
    assert trust_empty["counts_by_kind"]["table_structure_failed"] == 1

    # Persisted metadata error field / trust note, not merely the in-memory result.
    empty_metadata = _persisted_document_metadata(out_empty)
    populated_metadata = _persisted_document_metadata(out_pop)
    assert TRUST_NOTE_PREFIX in (empty_metadata.get("error") or "")
    assert TRUST_NOTE_PREFIX not in (populated_metadata.get("error") or "")


# --------------------------------------------------------------------------
# 6. CLI Seam: Click process command
# --------------------------------------------------------------------------


def test_cli_process_surfaces_completed_with_warnings_for_empty_table_run(
    tmp_path: Path,
) -> None:
    """Click process command prints 'Completed with warnings' on AUDIT_FAILED."""
    from click.testing import CliRunner

    from socr.cli import cli as cli_group

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    # AUDIT_FAILED result (from empty table run)
    res_empty = EngineResult(
        document_path=pdf,
        engine="qwen",
        status=DocumentStatus.AUDIT_FAILED,
        error="untrusted tables on 1 page(s), 1 flag(s)",
    )

    fake_pipeline_empty = _PipelineStub(res_empty)
    with (
        patch("socr.pipeline.orchestrator.UnifiedPipeline", return_value=fake_pipeline_empty),
    ):
        result_empty_cli = CliRunner().invoke(
            cli_group,
            ["process", str(pdf), "--primary", "qwen", "-o", str(tmp_path / "out1"), "-q"],
        )

    # GH-177: a partial document now exits NONZERO on the single-file path too,
    # matching batch and the contract's uniform policy. That makes this pair a
    # stronger difference than it was -- the two runs differ in exit code as
    # well as in message.
    assert result_empty_cli.exit_code != 0
    assert "Completed with warnings" in result_empty_cli.output

    # SUCCESS result (from populated table run)
    res_pop = EngineResult(
        document_path=pdf,
        engine="qwen",
        status=DocumentStatus.SUCCESS,
        error=None,
    )

    fake_pipeline_pop = _PipelineStub(res_pop)
    with (
        patch("socr.pipeline.orchestrator.UnifiedPipeline", return_value=fake_pipeline_pop),
    ):
        result_pop_cli = CliRunner().invoke(
            cli_group,
            ["process", str(pdf), "--primary", "qwen", "-o", str(tmp_path / "out2"), "-q"],
        )

    assert result_pop_cli.exit_code == 0
    assert "Completed with warnings" not in result_pop_cli.output
