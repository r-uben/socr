"""GH-162: a table verifier exception must not fail open into an accepting inner judge.

Both table judges run a deterministic verifier BEFORE the inner (VLM/heuristic)
judge. When that verifier raises, the deterministic evidence is unavailable --
so the table is *unverified*, not *verified good*. Delegating to an inner judge
that accepts ships an unverified generated table under a clean status, which the
class docstrings explicitly promise not to do.

Hermetic: no ollama, no GPU, no provider ladder.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import fitz

from socr.core.audit_log import AuditEvent
from socr.core.manifest import D3_SUPERSEDING_REJECTIONS
from socr.core.result import (
    REJECTION_VERIFIER_ERROR,
    PageOutput,
    PageStatus,
)
from socr.pipeline.agentic import (
    AcceptDecision,
    NativeTableVerifierJudge,
    SourceEvidenceTableJudge,
)

_RECTANGULAR_TABLE = (
    "| label | c1 | c2 |\n| --- | --- | --- |\n| row1 | 0.1 | 0.2 |\n| row2 | 0.3 | 0.4 |\n"
)


def _empty_page() -> fitz.Page:
    doc = fitz.open()
    return doc.new_page(width=500, height=700)


def _output(text: str = _RECTANGULAR_TABLE) -> PageOutput:
    return PageOutput(
        page_num=1,
        text=text,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )


def _accepting_inner() -> MagicMock:
    """An inner judge that always accepts, returning a REAL AcceptDecision.

    Deliberately not a bare MagicMock return: an auto-created attribute would
    make ``decision.accept is False`` vacuously informative.
    """
    inner = MagicMock()
    inner.assess.return_value = AcceptDecision(accept=True, reason="inner ok")
    return inner


def test_source_evidence_verifier_exception_does_not_accept() -> None:
    """Scanned lane: verify_scanned_table raising must not yield an accept."""
    events: list[AuditEvent] = []
    inner = _accepting_inner()
    judge = SourceEvidenceTableJudge(
        inner=inner,
        get_fitz_page=lambda pn: _empty_page(),
        record_event=events.append,
        ocr_image_fn=lambda pix: "",
    )
    output = _output()

    with patch(
        "socr.tables.source_evidence.verify_scanned_table",
        side_effect=RuntimeError("geometry exploded"),
    ):
        decision = judge.assess(output, MagicMock())

    assert decision.accept is False, "verifier exception fell open into the accepting inner judge"


def test_native_table_verifier_exception_does_not_accept() -> None:
    """Born-digital lane: verify_native_table raising must not yield an accept.

    The table text is rectangular, so the string-only structural gate (GH-200)
    does not fire -- the verifier exception is the only thing between this
    output and acceptance.
    """
    events: list[AuditEvent] = []
    inner = _accepting_inner()
    judge = NativeTableVerifierJudge(
        inner=inner,
        get_fitz_page=lambda pn: _empty_page(),
        is_table_page=lambda pn: True,
        record_event=events.append,
    )
    output = _output()

    with patch(
        "socr.tables.native_verifier.verify_native_table",
        side_effect=RuntimeError("geometry exploded"),
    ):
        decision = judge.assess(output, MagicMock())

    assert decision.accept is False, "verifier exception fell open into the accepting inner judge"


def test_rejection_records_durable_audit_event_and_class() -> None:
    """Acceptance: a durable event names the verifier type and the failure reason."""
    events: list[AuditEvent] = []
    judge = SourceEvidenceTableJudge(
        inner=_accepting_inner(),
        get_fitz_page=lambda pn: _empty_page(),
        record_event=events.append,
        ocr_image_fn=lambda pix: "",
    )
    output = _output()

    with patch(
        "socr.tables.source_evidence.verify_scanned_table",
        side_effect=RuntimeError("geometry exploded"),
    ):
        judge.assess(output, MagicMock())

    errs = [e for e in events if e.kind == "table_verifier_error"]
    assert len(errs) == 1
    assert errs[0].data["judge"] == "SourceEvidenceTableJudge"
    assert errs[0].data["exception_type"] == "RuntimeError"
    assert "geometry exploded" in errs[0].detail
    assert output.rejection_class == REJECTION_VERIFIER_ERROR


def test_verifier_error_cannot_supersede_the_fail_closed_floor() -> None:
    """A crashed verifier is not a positively-SOFT refusal.

    ``D3_SUPERSEDING_REJECTIONS`` is an allowlist of dispositions on which socr
    can say a deterministic gate did NOT refute the reading. A verifier that
    raised establishes nothing, so admitting it would let an unverified table
    ship over the fail-closed floor -- reintroducing GH-162 one layer down.
    """
    assert REJECTION_VERIFIER_ERROR not in D3_SUPERSEDING_REJECTIONS


def test_rejection_does_not_touch_audit_passed() -> None:
    """``audit_passed`` is winner-SELECTION; clearing it discards page text (#252)."""
    judge = SourceEvidenceTableJudge(
        inner=_accepting_inner(),
        get_fitz_page=lambda pn: _empty_page(),
        record_event=None,
        ocr_image_fn=lambda pix: "",
    )
    output = _output()
    assert output.audit_passed is True

    with patch(
        "socr.tables.source_evidence.verify_scanned_table",
        side_effect=RuntimeError("geometry exploded"),
    ):
        judge.assess(output, MagicMock())

    assert output.audit_passed is True
    assert output.status == PageStatus.SUCCESS


def test_fatal_process_errors_propagate() -> None:
    """A wedged/exhausted interpreter must surface, not become a quiet rejection."""
    judge = SourceEvidenceTableJudge(
        inner=_accepting_inner(),
        get_fitz_page=lambda pn: _empty_page(),
        record_event=None,
        ocr_image_fn=lambda pix: "",
    )
    for fatal in (MemoryError, RecursionError):
        with patch(
            "socr.tables.source_evidence.verify_scanned_table",
            side_effect=fatal("exhausted"),
        ):
            try:
                judge.assess(_output(), MagicMock())
            except fatal:
                continue
            raise AssertionError(f"{fatal.__name__} was swallowed into a page rejection")
