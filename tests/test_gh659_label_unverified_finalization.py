"""#659 round 2 (Astra REQUEST_CHANGES, P1): the label-unverified event had no
consumer.

The judge emitted ``source_evidence_table_label_unverified`` and left the page
SUCCESS / ``audit_passed=True`` / ``audit_notes=[]`` -- a fabricated row label
with a correct number shipped indistinguishable from a page nobody ever
doubted. Neither ``tables_trust.json``, page finalization, document metadata,
the CLI report, nor resume ever read the event.

Fix: ``SourceEvidenceTableJudge`` now marks the winning ``PageOutput`` itself
(``table_label_unverified``, ``status=WARNING``, an ``audit_notes`` entry)
without flipping ``audit_passed`` (that field selects the winner, per the
review). ``TABLE_DISTRUST_KINDS`` carries the event kind so
``tables_trust.json`` shows the page; ``resume_restore_kinds`` replays the
event; ``UnifiedPipeline._label_unverified_pages`` / ``_label_unverified_note``
read the FINAL winning output (not raw events), so a later resume that ships a
different, fully-corroborated candidate retires the warning on its own --
exactly the same terminal-diagnosis shape as ``_no_witness_backend_pages``
from #658.

Hermetic: real ``finalized_page_records`` / ``build_tables_trust`` /
``PageOutput.to_dict``+``from_dict`` round-trip, no ollama, no provider
ladder, no real tesseract.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from socr.core.audit_log import AuditEvent
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.manifest import (
    FinalizedPageRecord,
    PageDisposition,
    PageEnding,
    PagePrimaryReason,
    SelectionProvenance,
)
from socr.core.result import PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.core.tables_trust import TABLE_DISTRUST_KINDS, build_tables_trust
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.source_evidence import LABEL_UNVERIFIED_KIND

CANDIDATE_TABLE = "| Country | Amount |\n| --- | --- |\n| Germany | 10 |\n"
LABEL_DETAIL = "content labels unverified by page evidence: ['germany']"


def _flagged_output(page_num: int = 1) -> PageOutput:
    return PageOutput(
        page_num=page_num,
        text=CANDIDATE_TABLE,
        status=PageStatus.WARNING,
        engine="qwen",
        audit_passed=True,  # #659: the winner selector must NOT be flipped
        audit_notes=[f"table label unverified by page evidence: {LABEL_DETAIL}"],
        table_label_unverified=LABEL_DETAIL,
    )


def _clean_output(page_num: int = 2) -> PageOutput:
    return PageOutput(
        page_num=page_num,
        text="page 2 clean text",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )


def _disposition() -> PageDisposition:
    return PageDisposition(PageEnding.NATIVE_PROSE, PagePrimaryReason.CLEAN_NATIVE_PROSE)


def _records(flagged_output: PageOutput, clean_output: PageOutput) -> list[FinalizedPageRecord]:
    return [
        FinalizedPageRecord(
            output=flagged_output,
            disposition=_disposition(),
            selection_provenance=SelectionProvenance.NATIVE_CLEAN,
        ),
        FinalizedPageRecord(
            output=clean_output,
            disposition=_disposition(),
            selection_provenance=SelectionProvenance.NATIVE_CLEAN,
        ),
    ]


# --------------------------------------------------------------------------
# 1. The field survives ``PageOutput`` serialization (resume round-trip) and
#    does not disturb the content-addressed fingerprint of a clean page.
# --------------------------------------------------------------------------


def test_field_round_trips_through_to_dict_from_dict() -> None:
    out = _flagged_output()
    restored = PageOutput.from_dict(out.to_dict())
    assert restored.table_label_unverified == LABEL_DETAIL
    assert restored.status is PageStatus.WARNING
    assert restored.audit_passed is True


def test_empty_field_is_omitted_from_the_dict_not_emitted_as_empty_string() -> None:
    """A page that never touched the scanned-table gate must serialize
    byte-identically to before this ticket, or every already-terminal page's
    content-addressed fingerprint changes and every resume reprocesses it.
    """
    out = _clean_output()
    d = out.to_dict()
    assert "table_label_unverified" not in d


# --------------------------------------------------------------------------
# 2. ``tables_trust.json``: the kind is a durable distrust entry.
# --------------------------------------------------------------------------


def test_label_unverified_kind_is_a_table_distrust_kind() -> None:
    assert LABEL_UNVERIFIED_KIND in TABLE_DISTRUST_KINDS


def test_flagged_page_shows_up_in_tables_trust_json() -> None:
    events = [
        AuditEvent(
            page_num=1,
            kind=LABEL_UNVERIFIED_KIND,
            engine="qwen",
            detail=LABEL_DETAIL,
            data={"cause": ""},
        )
    ]
    trust = build_tables_trust("doc.pdf", events)
    assert 1 in trust.untrusted_pages


# --------------------------------------------------------------------------
# 3. Terminal-diagnosis helpers read the FINAL winning output, not history --
#    this is what makes the warning retire when a different candidate wins.
# --------------------------------------------------------------------------


def test_label_unverified_pages_reads_the_final_output_field() -> None:
    records = _records(_flagged_output(page_num=1), _clean_output(page_num=2))
    assert UnifiedPipeline._label_unverified_pages(records) == [1]


def test_a_later_clean_candidate_retires_the_warning() -> None:
    """The falsifier's second half: unsupported numbers still reject, but here
    the case that matters is the POSITIVE one -- a page that once shipped a
    flagged candidate but whose FINAL record (e.g. after a resume that
    escalated to a fully-supported reading) carries no doubt must not still
    report one. Simulates that final state directly: the record passed to the
    helper is the one the page ships NOW, not a log of what it ever shipped.
    """
    now_clean = _clean_output(page_num=1)  # same page, later, no doubt
    records = _records(now_clean, _clean_output(page_num=2))
    assert UnifiedPipeline._label_unverified_pages(records) == []


def test_label_unverified_note_names_the_page() -> None:
    records = _records(_flagged_output(page_num=1), _clean_output(page_num=2))
    note = UnifiedPipeline._label_unverified_note(records)
    assert note is not None
    assert "page(s) 1" in note
    assert "unverified" in note


def test_label_unverified_note_is_none_on_a_clean_run() -> None:
    records = _records(_clean_output(page_num=1), _clean_output(page_num=2))
    assert UnifiedPipeline._label_unverified_note(records) is None


# --------------------------------------------------------------------------
# 4. End-to-end through the real ``_phase_assemble``: document metadata + CLI.
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


def _assemble(tmp_path: Path):
    pipeline = _pipeline()
    state = _state(tmp_path)
    records = _records(_flagged_output(page_num=1), _clean_output(page_num=2))
    with patch("socr.core.manifest.finalized_page_records", return_value=records):
        return pipeline._phase_assemble(state, tmp_path)


def test_document_metadata_names_the_unverified_label_page(tmp_path: Path) -> None:
    result = _assemble(tmp_path)
    error = result.error or ""
    assert "unverified" in error
    assert "page(s) 1" in error


def test_cli_report_line_names_the_page(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _assemble(tmp_path)
    out = capsys.readouterr().out
    assert "unverified row/column label" in out
    assert "[1]" in out


def test_clean_run_names_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Falsifier control: a run with no unverified label must not gain a note
    or a CLI line just because the machinery now exists.
    """
    pipeline = _pipeline()
    state = _state(tmp_path)
    clean_records = _records(_clean_output(page_num=1), _clean_output(page_num=2))
    with patch("socr.core.manifest.finalized_page_records", return_value=clean_records):
        result = pipeline._phase_assemble(state, tmp_path)
    error = result.error or ""
    assert "unverified row/column label" not in error
    out = capsys.readouterr().out
    assert "unverified row/column label" not in out


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
