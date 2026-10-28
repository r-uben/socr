"""Harvest follow-up: typed table refusals, withheld-table manifest, witness gating."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.audit_log import AuditEvent
from socr.core.born_digital import BornDigitalDetector, DocumentAssessment, PageAssessment
from socr.core.manifest import is_page_failed_marker, structure_class_floor_text
from socr.core.result import (
    JUDGE_OUTCOME_COMPLETED,
    JUDGE_OUTCOME_NATIVE_TABLE_VERIFIER,
    JUDGE_OUTCOME_SOURCE_EVIDENCE,
    JUDGE_OUTCOME_TIMEOUT,
    PageOutput,
    PageStatus,
)
from socr.core.state import DocumentState, PageState, layer_is_admissible_evidence
from socr.core.table_verification_floor import (
    TABLE_VERIFICATION_PROSE_RECOVERY_KIND,
    page_failed_only_table_verification,
    record_table_prose_recovery_audit,
    withheld_table_manifest_from_page,
)
from socr.core.document import DocumentHandle
from socr.core.result import FailureMode
from socr.pipeline.agentic import AcceptDecision, NativeTableVerifierJudge, SourceEvidenceTableJudge
from socr.tables.source_evidence import CAUSE_NO_WITNESS_BACKEND, WITNESS_PACKAGE_MISSING

_PHYS_COL_GAP: float = 60.0


def _make_handle(page_count: int = 1) -> DocumentHandle:
    from pathlib import Path
    from unittest.mock import patch

    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        return DocumentHandle(path=Path("/tmp/fake.pdf"), page_count=page_count)


def _make_fitz_page_with_words(rows: list[list[tuple[float, str]]]) -> fitz.Page:
    doc = fitz.open()
    page = doc.new_page(width=700, height=900)
    for row_idx, cells in enumerate(rows):
        y = 100.0 + row_idx * 30
        for x, word in cells:
            page.insert_text((x, y), word, fontsize=9)
    return page


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    sep = "| " + " | ".join(["---"] * len(header)) + " |"
    lines = ["| " + " | ".join(header) + " |", sep]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def test_native_table_verifier_hard_fail_carries_typed_outcome() -> None:
    native_rows = [
        [
            (100.0, "0.1"),
            (100.0 + _PHYS_COL_GAP, "0.2"),
            (100.0 + 2 * _PHYS_COL_GAP, "0.3"),
        ],
        [
            (100.0, "1.1"),
            (100.0 + _PHYS_COL_GAP, "1.2"),
            (100.0 + 2 * _PHYS_COL_GAP, "1.3"),
        ],
    ]
    fitz_page = _make_fitz_page_with_words(native_rows)
    output_text = _md_table(
        ["label", "vals"],
        [
            ["row1", "0.1"],
            ["row2", "1.1 1.2 1.3"],
        ],
    )
    output = PageOutput(page_num=1, text=output_text, status=PageStatus.SUCCESS, confidence=0.9)
    inner = MagicMock()
    judge = NativeTableVerifierJudge(
        inner=inner,
        get_fitz_page=lambda _pn: fitz_page,
        is_table_page=lambda _pn: True,
        record_event=lambda _e: None,
    )
    decision = judge.assess(output, MagicMock())
    assert decision.accept is False
    assert decision.judge_outcome == JUDGE_OUTCOME_NATIVE_TABLE_VERIFIER
    inner.assess.assert_not_called()


def test_table_only_classification_uses_typed_outcomes_not_skip_reason() -> None:
    ps = PageState(page_num=1)
    ps.attempts.append(
        PageOutput(
            page_num=1,
            text="x",
            status=PageStatus.ERROR,
            engine="qwen",
            audit_passed=False,
            judge_outcome=JUDGE_OUTCOME_NATIVE_TABLE_VERIFIER,
        )
    )
    assert page_failed_only_table_verification(ps)

    ps.attempts.append(
        PageOutput(
            page_num=1,
            text="y",
            status=PageStatus.ERROR,
            engine="qwen",
            audit_passed=False,
            judge_outcome=JUDGE_OUTCOME_COMPLETED,
            judge_reason="page judge refused prose cells",
        )
    )
    assert not page_failed_only_table_verification(ps)

    ps2 = PageState(page_num=2)
    ps2.attempts.append(
        PageOutput(
            page_num=2,
            text="z",
            status=PageStatus.ERROR,
            engine="qwen",
            audit_passed=False,
            judge_outcome=JUDGE_OUTCOME_TIMEOUT,
            skip_reason="paid escalation blocked (source_evidence_table)",
        )
    )
    ps2.attempts.append(
        PageOutput(
            page_num=2,
            text="t",
            status=PageStatus.ERROR,
            engine="qwen",
            audit_passed=False,
            judge_outcome=JUDGE_OUTCOME_NATIVE_TABLE_VERIFIER,
        )
    )
    assert not page_failed_only_table_verification(ps2)


def test_withheld_manifest_records_regions_not_attempt_reasons() -> None:
    ps = PageState(
        page_num=1,
        detected_table_count=2,
        detected_table_bboxes=[[10.0, 20.0, 100.0, 200.0], [110.0, 20.0, 200.0, 200.0]],
    )
    ps.attempts.append(
        PageOutput(
            page_num=1,
            text="",
            status=PageStatus.ERROR,
            engine="qwen",
            audit_passed=False,
            judge_reason="native_table_verifier: numbers not backed by page text",
            judge_outcome=JUDGE_OUTCOME_NATIVE_TABLE_VERIFIER,
        )
    )
    records = withheld_table_manifest_from_page(ps)
    assert len(records) == 2
    assert records[0]["bbox"] == [10.0, 20.0, 100.0, 200.0]
    assert "reason" not in records[0] or records[0]["reason"] == "withheld_unverified_table"
    assert all("engine" not in r for r in records)


def test_prose_recovery_audit_lands_in_state_events() -> None:
    state = DocumentState(handle=_make_handle(3))
    ps = PageState(
        page_num=3,
        detected_table_count=1,
        detected_table_bboxes=[[0.0, 0.0, 400.0, 300.0]],
    )
    record_table_prose_recovery_audit(state, 3, ps, engine="native")
    assert ps.withheld_table_manifest
    assert any(ev.kind == TABLE_VERIFICATION_PROSE_RECOVERY_KIND for ev in state.events)


def test_table_detection_miss_fails_closed_whole_page_marker() -> None:
    """Borderless Yes/No rows with no detector box must not ship as loose prose."""
    ps = PageState(page_num=1, detected_table_count=0, detected_table_bboxes=[])
    ps.native_text = "Scope 3 narrative\n\nYes No Yes\n"
    ps.native_words = [
        (72.0, 380.0, 120.0, 392.0, "Scope", 0, 0, 0),
        (122.0, 380.0, 140.0, 392.0, "3", 0, 0, 0),
        (80.0, 230.0, 110.0, 242.0, "Yes", 0, 0, 0),
        (400.0, 230.0, 420.0, 242.0, "No", 0, 0, 0),
    ]
    floor = structure_class_floor_text(ps, 1)
    assert is_page_failed_marker(floor)
    assert "Yes" not in floor
    assert "No" not in floor


def test_apply_born_digital_propagates_witness_unusable(tmp_path) -> None:
    path = tmp_path / "garbled.pdf"
    doc = fitz.open()
    page = doc.new_page()
    cid_line = " ".join(f"(cid:{n})" for n in range(24))
    page.insert_text((72, 72), cid_line + " " + cid_line, fontsize=10)
    doc.save(str(path))
    doc.close()

    assessment = BornDigitalDetector().detect_page(path, 1)
    assert assessment.text_layer_witness_unusable is True
    state = DocumentState(handle=_make_handle(1))
    state.apply_born_digital(DocumentAssessment(path=path, pages=[assessment]))
    assert state.pages[1].text_layer_witness_unusable is True
    assert layer_is_admissible_evidence(state.pages[1]) is False


def test_reverting_witness_flag_breaks_admissibility_gate(tmp_path) -> None:
    """Fails if quality-rejection pages stop marking the witness unusable."""
    path = tmp_path / "garbled.pdf"
    doc = fitz.open()
    page = doc.new_page()
    cid_line = " ".join(f"(cid:{n})" for n in range(24))
    page.insert_text((72, 72), cid_line + " " + cid_line, fontsize=10)
    doc.save(str(path))
    doc.close()

    assessment = BornDigitalDetector().detect_page(path, 1)
    assert assessment.text_layer_witness_unusable is True
    broken = PageAssessment(
        page_num=1,
        is_born_digital=assessment.is_born_digital,
        native_text=assessment.native_text,
        confidence=assessment.confidence,
        char_count=assessment.char_count,
        word_count=assessment.word_count,
        font_count=assessment.font_count,
        has_images=assessment.has_images,
        has_tables=assessment.has_tables,
        has_figures=assessment.has_figures,
        has_equations=assessment.has_equations,
        has_math_font_typesetting=assessment.has_math_font_typesetting,
        text_layer_witness_unusable=False,
        notes=assessment.notes,
    )
    state = DocumentState(handle=_make_handle(1))
    state.apply_born_digital(DocumentAssessment(path=path, pages=[broken]))
    assert layer_is_admissible_evidence(state.pages[1]) is True


def test_source_evidence_judge_real_chain_no_witness_not_hallucination() -> None:
    from unittest.mock import patch

    doc = fitz.open()
    page = doc.new_page(width=500, height=700)
    table = _md_table(["a", "b"], [["x", "1"], ["y", "2"]])
    output = PageOutput(
        page_num=1,
        text=table,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    inner = MagicMock()
    judge = SourceEvidenceTableJudge(
        inner=inner,
        get_fitz_page=lambda _pn: page,
        record_event=lambda _e: None,
        native_trusted=lambda _pn: False,
    )
    witness = lambda _pix: ("", WITNESS_PACKAGE_MISSING, "no tesseract")
    with patch("socr.tables.source_evidence.classical_ocr_with_state", witness):
        decision = judge.assess(output, MagicMock())
    inner.assess.assert_not_called()
    assert decision.accept is False
    assert decision.judge_outcome == JUDGE_OUTCOME_SOURCE_EVIDENCE
    assert output.failure_mode == FailureMode.NO_WITNESS_BACKEND
    assert CAUSE_NO_WITNESS_BACKEND in decision.reason


def test_mild_encoding_suspect_keeps_witness_admissible() -> None:
    _CLEAN = (
        "The quick brown fox jumps over the lazy dog. "
        "Revenue increased by twelve percent year over year."
    )

    def _page(lines: list[str]) -> tuple[fitz.Document, fitz.Page]:
        doc = fitz.open()
        page = doc.new_page()
        y = 72
        for line in lines:
            page.insert_text((72, y), line, fontsize=10)
            y += 14
        return doc, page

    det = BornDigitalDetector()
    lines = [_CLEAN] * 11 + ["FrenchfJoumal ofFinancial Economics 43 volumeNumberFortyThree"]
    _doc, page = _page(lines)
    a = det._assess_page(page, 1)
    assert a.is_born_digital is True
    assert a.text_layer_witness_unusable is False
    ps = PageState(page_num=1, text_layer_witness_unusable=False)
    assert layer_is_admissible_evidence(ps) is True


def test_source_evidence_outcome_counts_as_table_only() -> None:
    ps = PageState(page_num=1)
    ps.attempts.append(
        PageOutput(
            page_num=1,
            text="",
            status=PageStatus.ERROR,
            engine="qwen",
            audit_passed=False,
            judge_outcome=JUDGE_OUTCOME_SOURCE_EVIDENCE,
            table_unverifiable="witness empty",
        )
    )
    assert page_failed_only_table_verification(ps)
