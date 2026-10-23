"""Follow-ups: advisory native-cell outcomes must not read as document distrust."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fitz")

from socr.core.audit_log import AuditEvent
from socr.core.config import EngineType, PipelineConfig
from socr.core.manifest import _credential_admission_refusal
from socr.core.result import DocumentStatus, FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState, PageState
from socr.core.tables_trust import (
    NATIVE_CELL_ADVISORY_DISTRUST_KINDS,
    build_tables_trust,
    trust_note,
)
from socr.review import html as review_html

import test_resf_native_cell_pipeline as pipe


def test_advisory_native_cell_in_separate_trust_section() -> None:
    events = [
        AuditEvent(page_num=1, kind="native_cell_check_ambiguous", data={"table_id": "t0"}),
    ]
    trust = build_tables_trust("doc.pdf", events, native_cell_advisory=True)
    assert not trust.untrusted_pages
    assert 1 in trust.advisory_native_cell_pages
    payload = trust.to_dict()
    assert "native_cell_advisory" in payload
    assert payload["untrusted_pages"] == []
    assert "advisory only" in payload["native_cell_advisory"]["note"].lower()


def test_gating_native_cell_stays_in_main_trust_section() -> None:
    events = [
        AuditEvent(page_num=1, kind="native_cell_check_contradicted", data={"table_id": "t0"}),
    ]
    trust = build_tables_trust("doc.pdf", events, native_cell_advisory=False)
    assert 1 in trust.pages
    assert not trust.advisory_native_cell_pages


def test_advisory_native_cell_does_not_produce_trust_note() -> None:
    events = [
        AuditEvent(page_num=1, kind="native_cell_check_skipped", data={"table_id": "t0"}),
    ]
    trust = build_tables_trust("doc.pdf", events, native_cell_advisory=True)
    assert trust_note(trust) is None


def test_plfss_p38_verified_table_count_with_advisory_ambiguous(tmp_path: Path) -> None:
    """PLFSS p.38: ambiguous advisory binding must not zero verified_text."""
    md = (pipe.FIXTURES / "p38_tableau18.md").read_text(encoding="utf-8")
    _pipe, result, state, out_dir = _run_page(tmp_path, "plfss_p38.pdf", md)
    assert result.status in (DocumentStatus.SUCCESS, DocumentStatus.AUDIT_FAILED)
    counts = state.table_counts or {}
    assert counts.get("verified_text", 0) >= 1
    assert state.pages[1].native_cell_binding


def test_review_html_success_page_not_suspicious_from_advisory_native_cell(tmp_path: Path) -> None:
    md = (pipe.FIXTURES / "p59_tableau1.md").read_text(encoding="utf-8")
    bad = md.replace("Solde public | -5,5 | -6,1", "Solde public | -6,1 | -5,5")
    _pipe, _result, _state, out_dir = pipe._run_page(tmp_path, "resf_p59.pdf", bad)
    pdf = pipe.FIXTURES / "resf_p59.pdf"
    doc_dir = out_dir / pdf.stem
    report = review_html.collect_pages(doc_dir, pdf)
    page1 = next(p for p in report.pages if p.page_num == 1)
    assert page1.reported_status == "success"
    assert not page1.suspect
    assert not any(k.startswith("native_cell_check_") for k in page1.signals)


def test_advisory_matches_check_disabled_byte_identical_outcomes(tmp_path: Path) -> None:
    md = (pipe.FIXTURES / "p67_tableau8.md").read_text(encoding="utf-8")
    _a, _r, _s, out_adv = pipe._run_page(tmp_path / "a", "resf_p67.pdf", md)
    _b, _r, _s, out_off = pipe._run_page(
        tmp_path / "b", "resf_p67.pdf", md, skip_native_cell_check=True
    )
    side_adv = pipe._page_sidecar(out_adv, 1)
    side_off = pipe._page_sidecar(out_off, 1)
    win_adv = side_adv["winning_output"]
    win_off = side_off["winning_output"]
    assert win_adv["text"] == win_off["text"]
    assert win_adv["status"] == win_off["status"]
    assert win_adv.get("failure_mode") == win_off.get("failure_mode")


def test_gating_blocks_credential_via_ladder_disposition_not_native_cell_event() -> None:
    """#713: native_cell audit kinds are non-blocking; disposition blocks."""
    from socr.core.manifest import CREDENTIAL_NON_BLOCKING_EVENT_KINDS

    for kind in NATIVE_CELL_ADVISORY_DISTRUST_KINDS:
        assert kind in CREDENTIAL_NON_BLOCKING_EVENT_KINDS

    from socr.core.document import DocumentHandle

    pdf = Path("/tmp/native_cell_cred_test.pdf")
    handle = DocumentHandle(path=pdf, page_count=2, page_count_known=True)
    state = DocumentState(handle=handle)
    ps = state.pages[1]
    ps.table_ladder_disposition = FailureMode.TABLE_UNVERIFIED
    out = PageOutput(page_num=1, text="| a | b |\n|---|---|\n| 1 | 2 |", status=PageStatus.WARNING)
    state.events.append(
        AuditEvent(page_num=1, kind="native_cell_check_ambiguous", data={"table_id": "t0"})
    )
    assert _credential_admission_refusal(state, ps, 1, out) == "adverse table-ladder disposition"

    state2 = DocumentState(handle=handle)
    ps2 = state2.pages[2]
    state2.events.append(
        AuditEvent(page_num=2, kind="native_cell_check_ambiguous", data={"table_id": "t0"})
    )
    assert _credential_admission_refusal(state2, ps2, 2, out) == ""


def _run_page(tmp_path: Path, pdf_name: str, md: str):
    return pipe._run_page(tmp_path, pdf_name, md)
