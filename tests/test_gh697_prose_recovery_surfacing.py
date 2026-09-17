"""#697: recovered scanned prose must surface at the document level and the
CLI, not only on the page.

#649 already ships the prose -- losing three paragraphs of an FOMC policy
directive was the defect that closed, and this ticket does not touch that.
The gap is PROVENANCE: ``PageOutput.scanned_prose_recovered`` reaches the
page status, the document status (``PageState.needs_repair`` reads
``best_output.audit_passed``, which this ending always sets to ``False``, so
``pages_needing_repair`` is non-empty and the document is never a clean
SUCCESS) and the page's ``audit_notes`` -- but never the document metadata
note or the CLI run report. A consumer reading only ``metadata.json`` or the
CLI output cannot tell a recovered-scan page from any other AUDIT_FAILED
ending without opening the page sidecar.

``UnifiedPipeline._scanned_prose_recovered_pages`` / ``_scanned_prose_
recovered_note`` read the FINAL winning output (``rec.output.scanned_prose_
recovered``), the same precedence principle as ``_label_unverified_pages``
(#659) and ``_ditto_unresolved_pages`` (#625): a page whose earlier attempt
was a recovery but whose later, fully-verified candidate won instead reports
nothing, so the note retires on its own.

Hermetic: real ``_phase_assemble`` driven off a hand-built ``FinalizedPage
Record`` list (``finalized_page_records`` patched), no ollama, no provider
ladder, no real tesseract.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.manifest import (
    FinalizedPageRecord,
    PageDisposition,
    PageEnding,
    PagePrimaryReason,
    SelectionProvenance,
)
from socr.core.result import DocumentStatus, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline

BANNER = (
    "[page 1: unverified scan — the paragraphs below are this page's own "
    "text layer; every numeric row is withheld]"
)


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


def _state(tmp_path: Path, page_count: int = 1) -> DocumentState:
    pdf = tmp_path / "doc.pdf"
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=pdf, page_count=page_count)
    state = DocumentState(handle=handle)
    for pn in range(1, page_count + 1):
        ps = state.pages[pn]
        ps.is_born_digital = False
        ps.native_text = f"page {pn} prose"
    return state


def _disposition() -> PageDisposition:
    return PageDisposition(PageEnding.NATIVE_PROSE, PagePrimaryReason.CLEAN_NATIVE_PROSE)


def _recovered_output(page_num: int = 1) -> PageOutput:
    return PageOutput(
        page_num=page_num,
        text=f"{BANNER}\n\nSome recovered prose paragraph text.",
        status=PageStatus.ERROR,
        engine="qwen",
        audit_passed=False,
        audit_notes=[
            "scanned_prose_recovered: no OCR attempt could be spliced around "
            "the withheld table; the page's own trusted prose bands ship "
            "flagged instead"
        ],
        scanned_prose_recovered=True,
    )


def _clean_output(page_num: int = 2) -> PageOutput:
    return PageOutput(
        page_num=page_num,
        text=f"page {page_num} clean text",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )


def _record(output: PageOutput) -> FinalizedPageRecord:
    return FinalizedPageRecord(
        output=output,
        disposition=_disposition(),
        selection_provenance=SelectionProvenance.NATIVE_CLEAN,
    )


def _assemble(tmp_path: Path, records: list[FinalizedPageRecord], page_count: int = 1):
    pipeline = _pipeline()
    state = _state(tmp_path, page_count=page_count)
    for r in records:
        ps = state.pages[r.output.page_num]
        ps.best_output = r.output
        ps.attempts.append(r.output)
    with patch("socr.core.manifest.finalized_page_records", return_value=records):
        return pipeline._phase_assemble(state, tmp_path)


# --------------------------------------------------------------------------
# 1. The bucket reader itself.
# --------------------------------------------------------------------------


def test_scanned_prose_recovered_pages_reads_the_final_output_field() -> None:
    records = [_record(_recovered_output(page_num=1)), _record(_clean_output(page_num=2))]
    assert UnifiedPipeline._scanned_prose_recovered_pages(records) == [1]


def test_a_later_clean_candidate_retires_the_page_from_the_bucket() -> None:
    """#649 round 2's own precedence principle: only the FINAL winner counts."""
    records = [_record(_clean_output(page_num=1))]
    assert UnifiedPipeline._scanned_prose_recovered_pages(records) == []


def test_scanned_prose_recovered_note_names_the_page() -> None:
    records = [_record(_recovered_output(page_num=1))]
    note = UnifiedPipeline._scanned_prose_recovered_note(records)
    assert note is not None
    assert "page(s) 1" in note
    assert "withheld table" in note


def test_scanned_prose_recovered_note_is_none_on_a_clean_run() -> None:
    records = [_record(_clean_output(page_num=1))]
    assert UnifiedPipeline._scanned_prose_recovered_note(records) is None


# --------------------------------------------------------------------------
# 2. End-to-end through the real ``_phase_assemble``: document metadata + CLI.
# --------------------------------------------------------------------------


def test_document_metadata_names_the_recovered_page(tmp_path: Path) -> None:
    records = [_record(_recovered_output(page_num=1)), _record(_clean_output(page_num=2))]
    result = _assemble(tmp_path, records, page_count=2)
    error = result.error or ""
    assert "unverified" in error
    assert "page(s) 1" in error


def test_cli_report_line_names_the_page(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    records = [_record(_recovered_output(page_num=1)), _record(_clean_output(page_num=2))]
    _assemble(tmp_path, records, page_count=2)
    out = capsys.readouterr().out
    assert "unverified recovered prose" in out
    assert "[1]" in out


def test_clean_run_names_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Falsifier control: a run with no recovery must not gain a note or a
    CLI line just because the machinery now exists.
    """
    records = [_record(_clean_output(page_num=1)), _record(_clean_output(page_num=2))]
    result = _assemble(tmp_path, records, page_count=2)
    error = result.error or ""
    assert "recovered prose" not in error
    out = capsys.readouterr().out
    assert "recovered prose" not in out
    assert result.status is DocumentStatus.SUCCESS


# --------------------------------------------------------------------------
# 3. The reachability test: an otherwise-clean document whose ONLY signal is
#    the recovered page. This is exactly what #659 got wrong (Astra P2a) --
#    the CLI line lived inside a shared defect-bucket ``if`` a recovery-only
#    document would otherwise never reach.
# --------------------------------------------------------------------------


def test_recovery_only_document_is_not_reported_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Deliberately only ONE page, ONE defect -- every other bucket this
    document could fall into stays empty. If the CLI line depended on some
    OTHER bucket also firing, this document's line would never print.
    """
    records = [_record(_recovered_output(page_num=1))]
    result = _assemble(tmp_path, records, page_count=1)

    # Content-retained, non-clean -- the recovered prose ships (that is
    # #649's fix), but the page is still an ERROR ending.
    assert result.status is DocumentStatus.AUDIT_FAILED
    assert result.markdown and "recovered prose paragraph" in result.markdown
    error = result.error or ""
    assert "page(s) 1" in error
    out = capsys.readouterr().out
    assert "unverified recovered prose" in out
    assert "[1]" in out


# --------------------------------------------------------------------------
# 4. Retirement: a later run whose winner ships the page clean must remove
#    both the note and the CLI line. A note that never retires is a new
#    defect (per the ticket).
# --------------------------------------------------------------------------


def test_a_later_clean_ship_retires_the_note_and_the_cli_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recovered_records = [_record(_recovered_output(page_num=1))]
    recovered_result = _assemble(tmp_path, recovered_records, page_count=1)
    assert "page(s) 1" in (recovered_result.error or "")
    capsys.readouterr()  # drain the first run's console output

    clean_records = [_record(_clean_output(page_num=1))]
    clean_result = _assemble(tmp_path, clean_records, page_count=1)

    assert clean_result.status is DocumentStatus.SUCCESS
    assert "recovered prose" not in (clean_result.error or "")
    out = capsys.readouterr().out
    assert "recovered prose" not in out
