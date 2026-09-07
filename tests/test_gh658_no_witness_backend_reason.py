"""#658: an unwitnessed scanned table must not be reported as a fabrication.

``SourceEvidenceTableJudge`` fails a scanned table closed when the local
evidence bundle is empty. Two very different situations produced that empty
bundle and both collapsed into ``FailureMode.HALLUCINATION``:

1. a classical OCR witness ran, read the pixels, and its reading does not
   contain the model's numbers -- evidence AGAINST the table; and
2. no classical OCR backend is installed at all, so nothing ever read the
   pixels -- NO evidence either way.

On the Fed swap-line minutes (1977/1982/1990 p3) case 2 marked candidates
carrying 62/62, 67/67 and 66/66 of the page's numbers with zero extras as
hallucinations, purely because the host had no ``tesseract``. This ticket
keeps both endings fail-closed and makes case 2 say what it is, at page,
document and CLI level.

Hermetic: no ollama, no provider ladder, no real tesseract. The backend probe
(``classical_ocr_backend_missing``) and the default witness
(``classical_ocr_pixmap``) are both patched, so the test result does not depend
on whether the host running it happens to have tesseract installed -- which is
the whole point of the ticket.

Every assertion here pins the DIFFERENCE between the two cases rather than an
absolute tuple: the two runs share one candidate, one page and one judge, and
differ only in what the witness probe reports.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import pytest

from socr.core.audit_log import _ESCALATION_MODES, AuditEvent
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.core.tables_trust import TABLE_DISTRUST_KINDS, build_tables_trust
from socr.pipeline.agentic import AcceptDecision, SourceEvidenceTableJudge
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.source_evidence import (
    CAUSE_NO_WITNESS_BACKEND,
    NO_WITNESS_BACKEND_KIND,
    WITNESS_BINARY_MISSING,
    WITNESS_EMPTY_READING,
    WITNESS_EXEC_ERROR,
    WITNESS_PACKAGE_MISSING,
    WITNESS_READING,
    WITNESS_RENDER_ERROR,
    build_scanned_evidence,
    classical_ocr_pixmap,
)

# A swap-line-shaped table: numeric body plus row labels, the shape the Fed
# pages emit. Small enough to keep the fixture readable.
CANDIDATE_TABLE = (
    "| Counterparty | Amount | Drawn |\n"
    "| --- | --- | --- |\n"
    "| Bundesbank | 62.5 | 12.5 |\n"
    "| Bank of Japan | 67.0 | 15.0 |\n"
)

# A witness reading that positively CONTRADICTS the candidate: it is a real
# reading of a page (non-empty), and none of the candidate's numbers are in it.
CONTRADICTING_WITNESS_TEXT = "Counterparty Amount Drawn Bundesbank 999.9 888.8"


def _witness(state: str, text: str = "", detail: str = ""):
    """A stand-in for ``classical_ocr_with_state`` pinned to one outcome."""
    return lambda pix: (text, state, detail)


def _scanned_page() -> fitz.Page:
    """A page with no native words -- the scanned lane's precondition."""
    doc = fitz.open()
    return doc.new_page(width=500, height=700)


def _candidate_output() -> PageOutput:
    return PageOutput(
        page_num=3,
        text=CANDIDATE_TABLE,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
        confidence=0.9,
    )


def _assess(*, witness) -> tuple[AcceptDecision, PageOutput, list]:
    """Run the judge once with the witness environment fully controlled."""
    events: list[AuditEvent] = []
    inner = MagicMock()
    inner.assess.return_value = AcceptDecision(accept=True, reason="inner ok")
    page = _scanned_page()
    judge = SourceEvidenceTableJudge(
        inner=inner,
        get_fitz_page=lambda pn: page,
        record_event=events.append,
        # ocr_image_fn deliberately left None: the production default is the
        # code path whose availability this ticket is about. Patching the
        # module attribute below controls what it reads.
        native_trusted=lambda pn: False,
    )
    output = _candidate_output()
    with patch("socr.tables.source_evidence.classical_ocr_with_state", witness):
        decision = judge.assess(output, MagicMock())
    inner.assess.assert_not_called()
    return decision, output, events


@pytest.fixture(scope="module")
def absent_backend():
    return _assess(witness=_witness(WITNESS_BINARY_MISSING))


@pytest.fixture(scope="module")
def contradicting_backend():
    return _assess(
        witness=_witness(WITNESS_READING, text=CONTRADICTING_WITNESS_TEXT),
    )


# --------------------------------------------------------------------------
# 0. Preconditions: without these the difference assertions would be vacuous.
# --------------------------------------------------------------------------


def test_absent_backend_really_produces_an_empty_bundle_and_names_the_gap() -> None:
    page = _scanned_page()
    with patch(
        "socr.tables.source_evidence.classical_ocr_with_state",
        _witness(WITNESS_BINARY_MISSING),
    ):
        bundle = build_scanned_evidence(page, include_text_layer=False)
    assert not bundle.has_content_evidence
    assert bundle.no_reading
    assert bundle.witness_state == WITNESS_BINARY_MISSING


@pytest.mark.parametrize(
    ("state", "text", "expect_no_reading"),
    [
        (WITNESS_PACKAGE_MISSING, "", True),
        (WITNESS_BINARY_MISSING, "", True),
        (WITNESS_EXEC_ERROR, "", True),
        (WITNESS_EMPTY_READING, "", False),
        (WITNESS_READING, "###", False),
    ],
)
def test_five_witness_outcomes_are_distinguished(state, text, expect_no_reading) -> None:
    """Reviewer item 2: ``classical_ocr_pixmap`` mapped a missing package, a
    missing executable, a crashed reader, a blank reading and a token-free
    reading all to ``""``. Each must now reach the bundle as itself.

    ``WITNESS_READING`` with un-tokenisable text is the case that proves the
    split is not just "empty bundle == no witness": the bundle IS empty and the
    page WAS read, so it must not take the no-witness ending.
    """
    page = _scanned_page()
    with patch("socr.tables.source_evidence.classical_ocr_with_state", _witness(state, text=text)):
        bundle = build_scanned_evidence(page, include_text_layer=False)
    assert not bundle.has_content_evidence
    assert bundle.witness_state == state
    assert bundle.no_reading is expect_no_reading


def test_render_failure_is_its_own_state_not_a_missing_backend() -> None:
    """A page that cannot be rasterised never handed the reader an image. The
    reader is fine; the render is not, and the two need different fixes.
    """
    page = _scanned_page()
    with (
        patch(
            "socr.tables.source_evidence.classical_ocr_with_state",
            _witness(WITNESS_PACKAGE_MISSING),
        ),
        patch.object(type(page), "get_pixmap", side_effect=RuntimeError("boom")),
    ):
        bundle = build_scanned_evidence(page, include_text_layer=False)
    assert bundle.witness_state == WITNESS_RENDER_ERROR
    assert bundle.no_reading


def test_injected_witness_is_judged_on_its_own_behaviour_not_ambient_path() -> None:
    """Reviewer item 2: a caller may pass a WORKING ``ocr_image_fn`` on a host
    with no tesseract. The bundle must describe the reader that actually ran.

    Both directions are pinned under a patched-away ambient backend: an
    injected reader that returns text is a reading, and one that returns
    nothing is a blank reading -- never ``package_missing``. Without this the
    existing GH-90 fixtures (which inject ``lambda pix: ""``) would flip to the
    new ending and the ticket would have moved the bug rather than fixed it.
    """
    page = _scanned_page()
    with patch(
        "socr.tables.source_evidence.classical_ocr_with_state",
        _witness(WITNESS_PACKAGE_MISSING),
    ):
        silent = build_scanned_evidence(page, ocr_image_fn=lambda pix: "", include_text_layer=False)
        speaking = build_scanned_evidence(
            page, ocr_image_fn=lambda pix: "Bundesbank 62.5", include_text_layer=False
        )
        raising = build_scanned_evidence(
            page,
            ocr_image_fn=MagicMock(side_effect=RuntimeError("reader died")),
            include_text_layer=False,
        )

    assert silent.witness_state == WITNESS_EMPTY_READING
    assert silent.no_reading is False
    assert speaking.has_content_evidence
    assert speaking.witness_state == WITNESS_READING
    assert raising.witness_state == WITNESS_EXEC_ERROR
    assert raising.no_reading is True


def test_public_text_only_reader_seam_is_unchanged() -> None:
    """``classical_ocr_pixmap`` is the documented ``OcrImageFn`` shape and other
    callers pass it around; the split must keep it returning a bare string.
    """
    with patch(
        "socr.tables.source_evidence.classical_ocr_with_state",
        _witness(WITNESS_READING, text="hello"),
    ):
        assert classical_ocr_pixmap(object()) == "hello"


def test_floor_latch_predicate_fires_for_the_no_witness_reason(absent_backend) -> None:
    """The latch predicate itself, copied from ``_phase_agentic``, run against
    the reason this judge actually produced -- so a future prefix edit fails
    here rather than in production.
    """
    reason = absent_backend[0].reason
    assert "source_evidence_table" in (reason or "")


def test_present_backend_really_read_the_page(contradicting_backend) -> None:
    """The control case must be a real contradiction, not a second empty
    bundle -- otherwise it would be testing the same branch twice.
    """
    _decision, _output, events = contradicting_backend
    detail = " ".join(e.detail for e in events)
    assert "numeric tokens unsupported" in detail


# --------------------------------------------------------------------------
# 1. Page level: the DIFFERENCE between the two endings.
# --------------------------------------------------------------------------


def test_both_endings_stay_fail_closed_identically(absent_backend, contradicting_backend) -> None:
    """This slice changes the REASON only. The page's disposition -- refused,
    ERROR, audit_passed False -- must be identical in both runs.
    """
    for decision, output, _events in (absent_backend, contradicting_backend):
        assert decision.accept is False
        assert decision.confidence == 0.0
        assert output.status is PageStatus.ERROR
        assert output.audit_passed is False


def test_absent_backend_is_not_reported_as_hallucination(
    absent_backend, contradicting_backend
) -> None:
    """The ticket's core assertion, as a difference: same candidate, same page,
    same judge; only the witness environment changes, and the failure mode and
    the decision reason must diverge with it.
    """
    absent_decision, absent_output, _ = absent_backend
    contra_decision, contra_output, _ = contradicting_backend

    assert absent_output.failure_mode is FailureMode.NO_WITNESS_BACKEND
    assert contra_output.failure_mode is FailureMode.HALLUCINATION
    assert absent_output.failure_mode is not contra_output.failure_mode

    assert CAUSE_NO_WITNESS_BACKEND in absent_decision.reason
    assert CAUSE_NO_WITNESS_BACKEND not in contra_decision.reason
    # The reason must name the missing tool, so an operator reading the page
    # record alone knows what to install.
    assert "tesseract" in absent_decision.reason


def test_floor_latch_marker_survives_in_both_reasons(absent_backend, contradicting_backend) -> None:
    """Reviewer item 3. ``_phase_agentic`` latches the scanned-table fail-closed
    floor on ``"source_evidence_table" in att.reason``. An earlier draft of this
    fix renamed the prefix for the no-witness ending, which silently stopped the
    floor from applying to exactly the pages the ticket is about -- fail-open.
    The marker must be present in BOTH reasons; only an extra token separates
    them.
    """
    assert "source_evidence_table" in absent_backend[0].reason
    assert "source_evidence_table" in contradicting_backend[0].reason


def test_absent_backend_gets_its_own_audit_event_kind(
    absent_backend, contradicting_backend
) -> None:
    """The no-witness run emits an ADDITIONAL kind, and keeps the generic one.

    Keeping ``source_evidence_table_reject`` matters: it is what
    ``TABLE_DISTRUST_KINDS`` watches, and a page that stopped emitting it would
    read as trusted in ``tables_trust.json`` -- the split must not buy an
    honest reason at the price of a silent trust gain.
    """
    absent_kinds = {e.kind for e in absent_backend[2]}
    contra_kinds = {e.kind for e in contradicting_backend[2]}

    assert "source_evidence_table_reject" in absent_kinds
    assert "source_evidence_table_reject" in contra_kinds
    assert NO_WITNESS_BACKEND_KIND in absent_kinds
    assert NO_WITNESS_BACKEND_KIND not in contra_kinds

    absent_event = next(e for e in absent_backend[2] if e.kind == NO_WITNESS_BACKEND_KIND)
    assert absent_event.data["cause"] == CAUSE_NO_WITNESS_BACKEND
    assert absent_event.page_num == 3
    assert "tesseract" in absent_event.detail
    contra_event = next(
        e for e in contradicting_backend[2] if e.kind == "source_evidence_table_reject"
    )
    assert contra_event.data["cause"] == ""


def test_new_kind_keeps_the_page_untrusted_in_the_sidecar(absent_backend) -> None:
    """The split must not move these pages from untrusted to trusted: they
    carried ``source_evidence_table_reject`` before it existed.
    """
    assert NO_WITNESS_BACKEND_KIND in TABLE_DISTRUST_KINDS
    trust = build_tables_trust("doc.pdf", absent_backend[2])
    assert 3 in trust.untrusted_pages


def test_new_mode_still_counts_as_an_escalation() -> None:
    """These attempts were recorded as escalations under HALLUCINATION; the
    rename must not shrink the escalation record.
    """
    assert FailureMode.NO_WITNESS_BACKEND in _ESCALATION_MODES


# --------------------------------------------------------------------------
# 2. Document + CLI level, through the real ``_phase_assemble``.
# --------------------------------------------------------------------------


def _pipeline() -> UnifiedPipeline:
    return UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.DEEPSEEK,
            enabled_engines=list(EngineType),
            agentic=False,
            quiet=False,
            native_first=True,
        )
    )


def _state(tmp_path: Path, page_count: int = 2) -> DocumentState:
    pdf = tmp_path / "doc.pdf"
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=pdf, page_count=page_count)
    state = DocumentState(handle=handle)
    for pn in range(1, page_count + 1):
        ps = state.pages[pn]
        ps.is_born_digital = False
        ps.native_text = f"page {pn} prose"
    return state


def _records(mode: FailureMode) -> list:
    """One affected page (1) plus one clean control page (2)."""
    from socr.core.manifest import (
        FinalizedPageRecord,
        PageDisposition,
        PageEnding,
        PagePrimaryReason,
        SelectionProvenance,
    )

    affected = PageOutput(
        page_num=1,
        text=CANDIDATE_TABLE,
        status=PageStatus.ERROR,
        engine="qwen",
        audit_passed=False,
        failure_mode=mode,
    )
    clean = PageOutput(
        page_num=2,
        text="page 2 clean text",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    disposition = PageDisposition(PageEnding.NATIVE_PROSE, PagePrimaryReason.CLEAN_NATIVE_PROSE)
    return [
        FinalizedPageRecord(
            output=affected,
            disposition=disposition,
            selection_provenance=SelectionProvenance.NATIVE_CLEAN,
        ),
        FinalizedPageRecord(
            output=clean,
            disposition=disposition,
            selection_provenance=SelectionProvenance.NATIVE_CLEAN,
        ),
    ]


def _assemble(tmp_path: Path, mode: FailureMode, witness_state: str = WITNESS_BINARY_MISSING):
    pipeline = _pipeline()
    state = _state(tmp_path)
    if mode is FailureMode.NO_WITNESS_BACKEND:
        state.events.append(
            AuditEvent(
                page_num=1,
                kind=NO_WITNESS_BACKEND_KIND,
                engine="qwen",
                detail="no classical OCR witness read this page",
                data={"cause": CAUSE_NO_WITNESS_BACKEND, "witness_state": witness_state},
            )
        )
    with patch("socr.core.manifest.finalized_page_records", return_value=_records(mode)):
        return pipeline._phase_assemble(state, tmp_path)


def test_document_metadata_names_the_missing_backend_only_for_the_no_witness_run(
    tmp_path: Path,
) -> None:
    """``metadata.json``'s free-text field must carry the remedy for the
    no-witness run and must NOT for the identical hallucination run.
    """
    absent = _assemble(tmp_path / "a", FailureMode.NO_WITNESS_BACKEND)
    contra = _assemble(tmp_path / "b", FailureMode.HALLUCINATION)

    absent_error = absent.error or ""
    contra_error = contra.error or ""
    assert "NO local OCR witness" in absent_error
    assert "tesseract" in absent_error
    assert "page(s) 1" in absent_error
    assert "NO local OCR witness" not in contra_error
    assert "tesseract" not in contra_error


def test_a_crashed_reader_is_never_told_to_install_the_reader(tmp_path: Path) -> None:
    """Reviewer follow-up: ``exec_error`` and ``render_error`` are unwitnessed
    too, but the reader is installed and working. Advising an install there
    sends the operator to fix a component that is not broken.
    """
    crashed = _assemble(tmp_path / "a", FailureMode.NO_WITNESS_BACKEND, WITNESS_EXEC_ERROR)
    unrenderable = _assemble(tmp_path / "b", FailureMode.NO_WITNESS_BACKEND, WITNESS_RENDER_ERROR)
    absent = _assemble(tmp_path / "c", FailureMode.NO_WITNESS_BACKEND, WITNESS_BINARY_MISSING)

    # An INSTRUCTION to install, not the word "install" -- the crashed-reader
    # message legitimately says the reader "is installed".
    commands = ("brew install", "apt install", "uv pip install", "pip install")
    for result in (crashed, unrenderable):
        error = result.error or ""
        # Still reported, still fail-closed, still not a hallucination.
        assert "NO local OCR witness" in error
        assert not any(cmd in error for cmd in commands)
    # The control: a genuinely absent binary DOES get the install instruction.
    assert "brew install tesseract" in (absent.error or "")


def test_cli_run_report_line_names_the_pages_and_the_remedy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _assemble(tmp_path / "a", FailureMode.NO_WITNESS_BACKEND)
    # Rich hard-wraps the console to the terminal width, so a phrase can be
    # split across lines. Compare on collapsed whitespace, not raw output.
    absent_out = " ".join(capsys.readouterr().out.split())
    _assemble(tmp_path / "b", FailureMode.HALLUCINATION)
    contra_out = " ".join(capsys.readouterr().out.split())

    assert "NO local OCR witness" in absent_out
    assert "brew install tesseract" in absent_out
    assert "[1]" in absent_out
    assert "OCR witness" not in contra_out


def test_a_historical_event_alone_is_not_a_terminal_diagnosis(tmp_path: Path) -> None:
    """Reviewer item 3. An earlier draft OR-ed every no-witness event into the
    reported set, so a page whose later reader succeeded still shipped the
    sentence "neither corroborated nor contradicted" -- false about the page
    that actually shipped.

    An event with no matching terminal outcome must produce no note at all.
    """
    state = _state(tmp_path)
    state.events.append(
        AuditEvent(
            page_num=1,
            kind=NO_WITNESS_BACKEND_KIND,
            engine="qwen",
            detail="no classical OCR witness",
            data={"cause": CAUSE_NO_WITNESS_BACKEND, "witness_state": WITNESS_BINARY_MISSING},
        )
    )
    assert UnifiedPipeline._no_witness_backend_pages(state) == []
    assert UnifiedPipeline._no_witness_backend_note(state) is None


def test_a_later_successful_read_retires_the_finding(tmp_path: Path) -> None:
    """The same page, twice: floored for want of a witness, then rescued by a
    reread. The historical event stays in the log; the terminal diagnosis must
    follow the page that shipped.
    """
    pdf = tmp_path / "doc.pdf"
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=pdf, page_count=1)
    state = DocumentState(handle=handle)
    _floored_scanned_page(state, no_witness=True)
    state.events.append(
        AuditEvent(
            page_num=1,
            kind=NO_WITNESS_BACKEND_KIND,
            engine="qwen",
            detail="no classical OCR witness",
            data={"cause": CAUSE_NO_WITNESS_BACKEND, "witness_state": WITNESS_BINARY_MISSING},
        )
    )
    assert UnifiedPipeline._no_witness_backend_pages(state) == [1]

    # The reread lands: the floor is released and a witnessed reading wins.
    ps = state.pages[1]
    ps.scanned_table_evidence_failed = False
    ps.scanned_table_no_witness = False
    ps.best_output = PageOutput(
        page_num=1,
        text=CANDIDATE_TABLE,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    assert UnifiedPipeline._no_witness_backend_pages(state) == []
    assert UnifiedPipeline._no_witness_backend_note(state) is None


def test_a_later_contradiction_is_not_reported_as_no_witness(tmp_path: Path) -> None:
    """The other half of item 3: the later witness READ the page and refuted the
    table. That page is a hallucination ending, and the note must not claim it
    was never contradicted.
    """
    pipeline = _pipeline()
    state = _state(tmp_path)
    state.events.append(
        AuditEvent(
            page_num=1,
            kind=NO_WITNESS_BACKEND_KIND,
            engine="qwen",
            detail="no classical OCR witness",
            data={"cause": CAUSE_NO_WITNESS_BACKEND, "witness_state": WITNESS_BINARY_MISSING},
        )
    )
    with patch(
        "socr.core.manifest.finalized_page_records",
        return_value=_records(FailureMode.HALLUCINATION),
    ):
        result = pipeline._phase_assemble(state, tmp_path)
    assert "neither corroborated nor contradicted" not in (result.error or "")
    assert (
        UnifiedPipeline._no_witness_backend_pages(state, _records(FailureMode.HALLUCINATION)) == []
    )


def test_resume_replays_the_event_so_the_remedy_survives(tmp_path: Path) -> None:
    """The witness state lives only on the audit event, and the note reads it to
    pick a remedy. ``resume_restore_kinds`` must replay it, or a resumed run
    degrades to "cause not recorded" on a page it can still fully explain.
    """
    assert NO_WITNESS_BACKEND_KIND in UnifiedPipeline.resume_restore_kinds()


# --------------------------------------------------------------------------
# 3. The FINAL selector, the sidecar it writes, and the resumed rerun.
#    Reviewer items 1 and 4: everything above reads the attempt. The shipped
#    page is REBUILT by ``_select_page_output_tagged``'s GH-90 branch, which
#    hard-coded ``FailureMode.HALLUCINATION`` -- so the honest reason died one
#    selection later and the artifact the corpus reads still accused the model.
# --------------------------------------------------------------------------


def _floored_scanned_page(state: DocumentState, *, no_witness: bool) -> None:
    """A scanned page sitting under the GH-90 fail-closed floor."""
    ps = state.pages[1]
    ps.is_born_digital = False
    ps.has_tables = True
    ps.native_text = ""
    ps.scanned_table_evidence_failed = True
    ps.scanned_table_no_witness = no_witness
    ps.d3_floor_png_ref = "![p1](figures/p001.png)"
    attempt = PageOutput(
        page_num=1,
        text=CANDIDATE_TABLE,
        status=PageStatus.ERROR,
        engine="qwen",
        audit_passed=False,
        failure_mode=(FailureMode.NO_WITNESS_BACKEND if no_witness else FailureMode.HALLUCINATION),
    )
    ps.attempts.append(attempt)
    ps.best_output = attempt


def _select(tmp_path: Path, *, no_witness: bool):
    from socr.core.manifest import _select_page_output_tagged

    pdf = tmp_path / "doc.pdf"
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=pdf, page_count=1)
    state = DocumentState(handle=handle)
    _floored_scanned_page(state, no_witness=no_witness)
    return _select_page_output_tagged(state, 1)


def test_final_selector_carries_the_reason_and_changes_nothing_else(tmp_path: Path) -> None:
    """Pin the DIFFERENCE at the selection seam: the two floored pages ship
    the same text, the same status and the same provenance, and differ ONLY in
    the recorded failure mode.
    """
    from socr.core.manifest import SelectionProvenance

    witness_out, witness_prov = _select(tmp_path, no_witness=True)
    halluc_out, halluc_prov = _select(tmp_path, no_witness=False)

    # Precondition: both fixtures really reach the GH-90 scanned-floor branch.
    # Without this the equality assertions below would pass on any two pages.
    assert witness_prov is SelectionProvenance.UNVERIFIABLE_TABLE_SCANNED

    assert witness_out.failure_mode is FailureMode.NO_WITNESS_BACKEND
    assert halluc_out.failure_mode is FailureMode.HALLUCINATION

    assert witness_out.text == halluc_out.text
    assert witness_out.status is halluc_out.status is PageStatus.ERROR
    assert witness_out.audit_passed is halluc_out.audit_passed is False
    assert witness_prov == halluc_prov


def test_floor_applier_sets_the_flag_and_the_mode_together(tmp_path: Path) -> None:
    """``_apply_scanned_table_floor`` is the single writer of both. A page
    floored WITHOUT the no-witness cause must keep reading as HALLUCINATION.
    """
    pdf = tmp_path / "doc.pdf"
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=pdf, page_count=1)
    state = DocumentState(handle=handle)
    pipeline = _pipeline()

    for no_witness, expected in (
        (True, FailureMode.NO_WITNESS_BACKEND),
        (False, FailureMode.HALLUCINATION),
    ):
        ps = state.pages[1]
        ps.best_output = PageOutput(
            page_num=1, text=CANDIDATE_TABLE, status=PageStatus.SUCCESS, engine="qwen"
        )
        pipeline._apply_scanned_table_floor(ps, pdf, 1, None, no_witness=no_witness)
        assert ps.scanned_table_evidence_failed is True
        assert ps.scanned_table_no_witness is no_witness
        assert ps.best_output.failure_mode is expected
        assert ps.best_output.status is PageStatus.ERROR


def test_reason_survives_the_sidecar_round_trip_and_the_resumed_rerun(tmp_path: Path) -> None:
    """Reviewer item 4. Run 1 floors the page for want of a witness; run 2
    resumes from run 1's sidecar with no events and no attempts in memory.

    The resumed run must reach the SAME reason. Before this, the restored page
    fell back to the constructor default and the second run reported the model
    as a fabricator on a page the first run had correctly called unwitnessed --
    a reason that degrades on rerun is not surfaced.

    The flag is DERIVED on restore from the winning output's persisted
    ``failure_mode`` rather than stored under a key of its own, so this test
    also pins that the sidecar carries enough to rebuild it.
    """
    import json

    pdf = tmp_path / "doc.pdf"
    pdf.touch()
    out_dir = tmp_path / "out"
    pipeline = _pipeline()

    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=pdf, page_count=1)
    run1 = DocumentState(handle=handle)
    _floored_scanned_page(run1, no_witness=True)

    sidecar = pipeline._flush_page_sidecar(run1, 1, out_dir, terminal=True)
    assert sidecar is not None and sidecar.exists()
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    assert meta["scanned_table_evidence_failed"] is True
    assert meta["failure_mode"] == FailureMode.NO_WITNESS_BACKEND.value
    # No key of its own: the P6 sidecar key set stays frozen.
    assert "scanned_table_no_witness" not in meta

    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle2 = DocumentHandle(path=pdf, page_count=1)
    run2 = DocumentState(handle=handle2)
    restored = PageOutput(
        page_num=1,
        text=CANDIDATE_TABLE,
        status=PageStatus(meta["status"]),
        engine="qwen",
        audit_passed=False,
        failure_mode=FailureMode(meta["failure_mode"]),
    )
    pipeline._restore_terminal_page_state(run2, 1, restored, out_dir)

    ps = run2.pages[1]
    assert ps.scanned_table_evidence_failed is True
    assert ps.scanned_table_no_witness is True
    # No events survive a resume; the restored flag alone must still name the
    # page for the document note and the CLI line.
    assert not run2.events
    assert UnifiedPipeline._no_witness_backend_pages(run2) == [1]
    note = UnifiedPipeline._no_witness_backend_note(run2)
    assert note is not None
    assert "page(s) 1" in note
    # No event survived this restore path, so the note says so instead of
    # guessing a remedy it cannot support.
    assert "audit_log.json" in note


def test_pre_ticket_sidecar_restores_the_reading_it_was_written_with(tmp_path: Path) -> None:
    """A sidecar written before this ticket records the floor with
    ``failure_mode=hallucination``. Resuming it must keep saying exactly that,
    never upgrade it to a no-witness claim this run cannot support.

    The control for the resume test above: same floor, same restore path, only
    the persisted mode differs.
    """
    pdf = tmp_path / "doc.pdf"
    pdf.touch()
    out_dir = tmp_path / "out"
    pipeline = _pipeline()

    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=pdf, page_count=1)
    run1 = DocumentState(handle=handle)
    _floored_scanned_page(run1, no_witness=False)
    sidecar = pipeline._flush_page_sidecar(run1, 1, out_dir, terminal=True)
    assert sidecar is not None

    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle2 = DocumentHandle(path=pdf, page_count=1)
    run2 = DocumentState(handle=handle2)
    pipeline._restore_terminal_page_state(
        run2,
        1,
        PageOutput(
            page_num=1,
            text=CANDIDATE_TABLE,
            status=PageStatus.ERROR,
            engine="qwen",
            failure_mode=FailureMode.HALLUCINATION,
        ),
        out_dir,
    )
    assert run2.pages[1].scanned_table_evidence_failed is True
    assert run2.pages[1].scanned_table_no_witness is False
    assert UnifiedPipeline._no_witness_backend_pages(run2) == []


def test_absent_backend_survives_judge_floor_finalization_and_resume(tmp_path: Path) -> None:
    """Reviewer item 1's falsifier, as ONE chain rather than four seams.

    Absent backend -> the real judge -> the real latch predicate copied from
    ``_phase_agentic`` -> the real ``_apply_scanned_table_floor`` -> the real
    ``_select_page_output_tagged`` -> the real sidecar -> the real restore.
    The no-witness cause must arrive at the far end, and the floor's shipped
    bytes and status must be exactly what a hallucination floor ships.

    Nothing here stubs a finalized record, which is what made the earlier
    document-level tests unable to prove this: they asserted on a record they
    had built themselves.
    """
    import json

    from socr.core.manifest import _select_page_output_tagged

    pdf = tmp_path / "doc.pdf"
    pdf.touch()
    out_dir = tmp_path / "out"
    pipeline = _pipeline()

    # 1. The judge, with no OCR backend anywhere.
    decision, attempt, events = _assess(witness=_witness(WITNESS_BINARY_MISSING))
    assert attempt.failure_mode is FailureMode.NO_WITNESS_BACKEND

    # 2. The latch predicate and the no-witness derivation, verbatim from
    #    ``_phase_agentic`` -- so a change to either fails here.
    assert "source_evidence_table" in (decision.reason or "")
    no_witness = attempt.failure_mode is FailureMode.NO_WITNESS_BACKEND

    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=pdf, page_count=1)
    state = DocumentState(handle=handle)
    ps = state.pages[1]
    ps.is_born_digital = False
    ps.has_tables = True
    ps.native_text = ""
    attempt.page_num = 1
    ps.attempts.append(attempt)
    ps.best_output = attempt
    state.events.extend(events)

    # 3. The real floor.
    pipeline._apply_scanned_table_floor(ps, pdf, 1, None, no_witness=no_witness)
    assert ps.scanned_table_evidence_failed is True
    assert ps.scanned_table_no_witness is True

    # 4. The real finalization.
    shipped, provenance = _select_page_output_tagged(state, 1)
    from socr.core.manifest import SelectionProvenance

    assert provenance is SelectionProvenance.UNVERIFIABLE_TABLE_SCANNED
    assert shipped.failure_mode is FailureMode.NO_WITNESS_BACKEND
    assert shipped.status is PageStatus.ERROR
    assert shipped.audit_passed is False

    # The floor's CONTENT is unchanged: same bytes a hallucination floor ships.
    control_state = DocumentState(handle=handle)
    _floored_scanned_page(control_state, no_witness=False)
    control_state.pages[1].d3_floor_png_ref = ps.d3_floor_png_ref
    control_shipped, control_prov = _select_page_output_tagged(control_state, 1)
    assert shipped.text == control_shipped.text
    assert provenance == control_prov

    # 5. The real sidecar, and the real restore.
    ps.best_output = shipped
    sidecar = pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)
    assert sidecar is not None
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    assert meta["failure_mode"] == FailureMode.NO_WITNESS_BACKEND.value

    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle2 = DocumentHandle(path=pdf, page_count=1)
    resumed = DocumentState(handle=handle2)
    pipeline._restore_terminal_page_state(
        resumed,
        1,
        PageOutput(
            page_num=1,
            text=shipped.text,
            status=PageStatus(meta["status"]),
            engine="qwen",
            audit_passed=False,
            failure_mode=FailureMode(meta["failure_mode"]),
        ),
        out_dir,
    )
    assert resumed.pages[1].scanned_table_no_witness is True
    assert UnifiedPipeline._no_witness_backend_pages(resumed) == [1]


# --------------------------------------------------------------------------
# 4. The PRODUCER of the terminal mode, through real routing.
#    Everything above either stubs the finalized records or hands the floor a
#    cause already decided. This section drives ``_phase_agentic``'s own
#    reduction over a two-attempt ladder, which is where the cause is chosen.
# --------------------------------------------------------------------------


def _gate_attempt(*, no_witness: bool, engine_name: str, reason_suffix: str):
    """A ProviderAttempt shaped exactly as the source-evidence judge leaves one."""
    from socr.core.config import EngineType as _EngineType
    from socr.pipeline.agentic import ProviderAttempt

    marker = f"[{CAUSE_NO_WITNESS_BACKEND}] " if no_witness else ""
    out = PageOutput(
        page_num=1,
        text=CANDIDATE_TABLE,
        status=PageStatus.ERROR,
        engine=engine_name,
        audit_passed=False,
        failure_mode=(FailureMode.NO_WITNESS_BACKEND if no_witness else FailureMode.HALLUCINATION),
    )
    return ProviderAttempt(
        engine=_EngineType.QWEN,
        output=out,
        cost_usd=0.0,
        accepted=False,
        reason=f"source_evidence_table: {marker}{reason_suffix}",
        provider_id="qwen-local",
        model="qwen3-vl",
        backend="ollama",
    )


def _unrelated_attempt(engine_name: str):
    """An attempt the source-evidence gate never adjudicated: a timeout. It
    carries no verdict about the table and must not act as one.
    """
    from socr.core.config import EngineType as _EngineType
    from socr.pipeline.agentic import ProviderAttempt

    out = PageOutput(
        page_num=1,
        text="",
        status=PageStatus.ERROR,
        engine=engine_name,
        audit_passed=False,
        failure_mode=FailureMode.TIMEOUT,
    )
    return ProviderAttempt(
        engine=_EngineType.QWEN,
        output=out,
        cost_usd=0.0,
        accepted=False,
        reason="provider timed out after 600s",
        provider_id="qwen-local",
        model="qwen3-vl",
        backend="ollama",
    )


def _run_ladder(tmp_path: Path, attempts: list):
    """Drive the REAL ``_phase_agentic`` over one scanned page with this ladder,
    then finalize it, and return (page state, shipped output).
    """
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.manifest import _select_page_output_tagged
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline.agentic import PageDecision

    pdf = tmp_path / "doc.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    doc.new_page(width=500, height=700)
    doc.save(str(pdf))
    doc.close()

    config = PipelineConfig(
        primary_engine=EngineType.QWEN,
        agentic=True,
        judge_backend="heuristic",
        enabled_engines=[EngineType.QWEN],
        quiet=True,
        save_figures=False,
        write_manifest=False,
        native_first=False,
        dual_pass_tables=False,
        escalate_ambiguous_tables=False,
        table_judge_ladder=False,
    )
    pipeline = UnifiedPipeline(config)
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    ps = state.pages[1]
    ps.is_born_digital = False
    ps.has_tables = True
    ps.native_text = ""

    def _route(page_num, *a, **kw):
        return PageDecision(
            page_num=page_num,
            final_output=attempts[-1].output,
            attempts=list(attempts),
            accepted=False,
        )

    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch("socr.pipeline.orchestrator.route_page", side_effect=_route),
        patch("socr.pipeline.orchestrator.probe_ollama_idle", return_value=True),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
    ):
        pipeline._phase_agentic(state, tmp_path / "out")

    shipped, _prov = _select_page_output_tagged(state, 1)
    return state.pages[1], shipped


def test_a_later_contradiction_beats_an_earlier_missing_witness(tmp_path: Path) -> None:
    """The producer's own reduction, through real routing.

    First rung: no witness. Second rung: a witness READ the page and refuted
    the table. The floor must record the contradiction -- ``any(no_witness)``
    over every attempt let the FIRST reading decide, and the terminal rollup
    then reported "neither corroborated nor contradicted" about a table a
    witness had contradicted.
    """
    ps, shipped = _run_ladder(
        tmp_path / "a",
        [
            _gate_attempt(no_witness=True, engine_name="qwen", reason_suffix="no OCR witness"),
            _gate_attempt(
                no_witness=False,
                engine_name="gemini",
                reason_suffix="numeric tokens unsupported by page evidence: ['62.5']",
            ),
        ],
    )

    assert ps.scanned_table_evidence_failed is True, "the floor must still apply"
    assert ps.scanned_table_no_witness is False
    assert shipped.failure_mode is FailureMode.HALLUCINATION
    assert UnifiedPipeline._no_witness_backend_pages(state_of(ps)) == []


def state_of(ps):
    """A minimal DocumentState carrying just this page, for the rollup helper."""
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=Path("/tmp/x.pdf"), page_count=1)
    st = DocumentState(handle=handle)
    st.pages[1] = ps
    return st


def test_every_gate_attempt_unwitnessed_still_reports_no_witness(tmp_path: Path) -> None:
    """The control. Two rungs, neither ever read the page: the cause survives.

    Without this the fix above could satisfy its sibling by simply never
    reporting a no-witness page again.
    """
    ps, shipped = _run_ladder(
        tmp_path / "b",
        [
            _gate_attempt(no_witness=True, engine_name="qwen", reason_suffix="no OCR witness"),
            _gate_attempt(no_witness=True, engine_name="gemini", reason_suffix="no OCR witness"),
        ],
    )

    assert ps.scanned_table_evidence_failed is True
    assert ps.scanned_table_no_witness is True
    assert shipped.failure_mode is FailureMode.NO_WITNESS_BACKEND
    assert UnifiedPipeline._no_witness_backend_pages(state_of(ps)) == [1]


def test_an_unrelated_later_failure_is_not_a_witnessed_contradiction(tmp_path: Path) -> None:
    """A timeout after a no-witness read never reached the source-evidence gate,
    so it holds no verdict about the table. Reading "not no-witness" off it
    would silently convert an unwitnessed page back into an accusation.
    """
    ps, shipped = _run_ladder(
        tmp_path / "c",
        [
            _gate_attempt(no_witness=True, engine_name="qwen", reason_suffix="no OCR witness"),
            _unrelated_attempt("gemini"),
        ],
    )

    assert ps.scanned_table_evidence_failed is True
    assert ps.scanned_table_no_witness is True
    assert shipped.failure_mode is FailureMode.NO_WITNESS_BACKEND
