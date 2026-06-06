"""Durable per-run audit log: derived escalations + source-appended events."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from socr.core.audit_log import AuditEvent, RunAudit, build_run_audit
from socr.core.document import DocumentHandle
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState


def _handle(pages: int = 1) -> DocumentHandle:
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        return DocumentHandle(path=Path("/tmp/paper.pdf"), page_count=pages)


def _attempt(engine, mode=FailureMode.NONE, passed=True):
    return PageOutput(
        page_num=1,
        text="x",
        status=PageStatus.SUCCESS,
        engine=engine,
        failure_mode=mode,
        audit_passed=passed,
    )


# --------------------------------------------------------------------------
# Derived escalations
# --------------------------------------------------------------------------


def test_recitation_escalation_is_derived_when_recovered():
    state = DocumentState(handle=_handle(1))
    state.pages[1].attempts = [
        _attempt("gemini", FailureMode.RECITATION, passed=False),
        _attempt("qwen"),  # took over
    ]
    audit = build_run_audit(state)
    kinds = [e.kind for e in audit.events]
    assert kinds == ["recitation_escalation"]
    assert audit.events[0].data["recovered_by"] == "qwen"


def test_failed_attempt_with_no_successor_is_not_an_escalation():
    # A page that failed and nothing recovered it is a failure, not an escalation
    # (no handoff happened); we don't want to over-report.
    state = DocumentState(handle=_handle(1))
    state.pages[1].attempts = [_attempt("gemini", FailureMode.RECITATION, passed=False)]
    assert build_run_audit(state).events == []


def test_clean_run_produces_no_events():
    state = DocumentState(handle=_handle(1))
    state.pages[1].attempts = [_attempt("qwen")]
    audit = build_run_audit(state)
    assert audit.events == [] and audit.summary_line() == ""


# --------------------------------------------------------------------------
# Source-appended events merge + ordering
# --------------------------------------------------------------------------


def test_source_events_merge_and_order_by_page_then_phase():
    state = DocumentState(handle=_handle(2))
    # page 1: an escalation (derived) + a dual-pass patch (appended)
    state.pages[1].attempts = [
        _attempt("gemini", FailureMode.TRUNCATED, passed=False),
        _attempt("qwen"),
    ]
    state.events.append(
        AuditEvent(
            page_num=1,
            kind="dualpass_patch",
            engine="qwen",
            detail="table 0: '(0.0l0)' -> '(0.010)'",
        )
    )
    # page 2: a judge reject (appended)
    state.pages[2].attempts = [_attempt("gemini")]
    state.events.append(
        AuditEvent(page_num=2, kind="judge_reject", engine="gemini", detail="wrong digits")
    )

    audit = build_run_audit(state)
    seq = [(e.page_num, e.kind) for e in audit.events]
    # page 1 escalation before page-1 dual-pass; page 2 after page 1.
    assert seq == [(1, "escalation"), (1, "dualpass_patch"), (2, "judge_reject")]


def test_counts_and_summary_line():
    audit = RunAudit(
        pdf_filename="p.pdf",
        events=[
            AuditEvent(1, "dualpass_patch"),
            AuditEvent(2, "dualpass_patch"),
            AuditEvent(3, "recitation_escalation"),
        ],
    )
    assert audit.counts() == {"dualpass_patch": 2, "recitation_escalation": 1}
    assert "2 dualpass_patch" in audit.summary_line()


def test_save_writes_structured_json(tmp_path):
    audit = RunAudit(
        pdf_filename="p.pdf",
        events=[
            AuditEvent(
                1,
                "dualpass_patch",
                engine="qwen",
                detail="x",
                data={
                    "changed_cells": [{"row": 3, "col": 2, "page": "(0.0l0)", "crop": "(0.010)"}]
                },
            ),
        ],
    )
    out = tmp_path / "audit_log.json"
    audit.save(out)
    loaded = json.loads(out.read_text())
    assert loaded["event_count"] == 1
    assert loaded["counts"] == {"dualpass_patch": 1}
    assert loaded["events"][0]["data"]["changed_cells"][0]["crop"] == "(0.010)"
