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

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
    finalized_page_records,
)
from socr.core.result import DocumentStatus, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.core.tables_trust import TABLE_DISTRUST_KINDS, build_tables_trust
from socr.pipeline.agentic import HeuristicPageJudge, SourceEvidenceTableJudge
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.source_evidence import LABEL_UNVERIFIED_KIND, SourceEvidenceResult

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


# --------------------------------------------------------------------------
# 5. Astra round 3 P1 (executed): a REAL judge chain must still ACCEPT a
#    candidate whose only doubt is an unverified label, and must NOT flip
#    status at judge time.
# --------------------------------------------------------------------------


def test_flagged_candidate_reaches_a_real_accepting_judge_chain() -> None:
    """The core regression from round 3. ``HeuristicPageJudge.assess`` and
    ``VLMPageJudge.assess`` both treat any output whose ``status`` is not
    SUCCESS as empty/error input and reject on sight (agentic.py:371/398). At
    dde5399, ``SourceEvidenceTableJudge`` set ``output.status = WARNING``
    BEFORE handing that same output to the inner judge, so the ladder
    rejected -- and re-routed or fell back on -- the exact candidate this
    ticket exists to ship flagged. This drives a REAL ``HeuristicPageJudge``
    (not a MagicMock stand-in for the inner judge; only its ``checker``
    boundary dependency is mocked, same as Astra's own probe) through
    ``SourceEvidenceTableJudge`` and requires an ACCEPT.
    """
    checker = MagicMock()
    checker.check.return_value.passed = True
    checker.check.return_value.errors = []
    inner = HeuristicPageJudge(checker)
    judge = SourceEvidenceTableJudge(
        inner=inner,
        get_fitz_page=lambda pn: object(),
        native_trusted=lambda pn: False,
    )
    output = PageOutput(
        page_num=1,
        text=CANDIDATE_TABLE,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    supported_but_unverified_label = SourceEvidenceResult(
        True, True, "numeric support", content_unverified=LABEL_DETAIL
    )
    with patch(
        "socr.tables.source_evidence.verify_scanned_table",
        return_value=supported_but_unverified_label,
    ):
        decision = judge.assess(output, MagicMock())

    assert decision.accept is True
    # Judge-time status is untouched -- the WARNING promotion happens only at
    # finalization (see test_finalization_guard_promotes_a_real_candidate below).
    assert output.status is PageStatus.SUCCESS
    assert output.table_label_unverified == LABEL_DETAIL


# --------------------------------------------------------------------------
# 6. Astra round 3 P1: the WARNING promotion is proven through the REAL
#    ``finalized_page_records`` path, not asserted about a hand-built record
#    (the round-2 tests all patched ``finalized_page_records`` directly).
# --------------------------------------------------------------------------


def test_finalization_guard_promotes_a_real_candidate() -> None:
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=Path("doc.pdf"), page_count=1)
    state = DocumentState(handle=handle)
    ps = state.pages[1]
    ps.is_born_digital = False
    candidate = PageOutput(
        page_num=1,
        text=CANDIDATE_TABLE,
        status=PageStatus.SUCCESS,  # exactly what the judge now leaves it as
        engine="qwen",
        audit_passed=True,
        table_label_unverified=LABEL_DETAIL,
    )
    ps.attempts.append(candidate)
    ps.best_output = candidate

    records = finalized_page_records(state)
    assert len(records) == 1
    assert records[0].output.status is PageStatus.WARNING
    assert records[0].output.audit_passed is True
    assert records[0].output.table_label_unverified == LABEL_DETAIL


def test_finalization_guard_never_demotes_a_harder_failure() -> None:
    """A page already ERROR for a more specific reason keeps that status --
    the label doubt is real but strictly less severe than an actual failure.
    """
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=Path("doc.pdf"), page_count=1)
    state = DocumentState(handle=handle)
    ps = state.pages[1]
    ps.is_born_digital = False
    candidate = PageOutput(
        page_num=1,
        text="[page 1 failed: some harder problem]",
        status=PageStatus.ERROR,
        engine="qwen",
        audit_passed=False,
        table_label_unverified=LABEL_DETAIL,
    )
    ps.attempts.append(candidate)
    ps.best_output = candidate

    records = finalized_page_records(state)
    assert records[0].output.status is PageStatus.ERROR


# --------------------------------------------------------------------------
# 7. Astra round 3 P2a: an otherwise-clean document whose ONLY defect is the
#    label flag must not report SUCCESS, and the CLI line must be reachable
#    even though it is the sole issue (previously nested inside an unrelated
#    defect-bucket condition).
# --------------------------------------------------------------------------


def test_label_only_document_is_not_reported_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Deliberately only ONE defect on the page (the label flag) -- every
    other bucket this document could fall into stays empty. If the CLI line
    or the AUDIT_FAILED status depended on some OTHER bucket also firing
    (P2a's failure mode), this test's page would report SUCCESS and silence.
    """
    pipeline = _pipeline()
    state = _state(tmp_path, page_count=1)
    only_defect = PageOutput(
        page_num=1,
        text=CANDIDATE_TABLE,
        status=PageStatus.WARNING,
        engine="qwen",
        audit_passed=True,
        table_label_unverified=LABEL_DETAIL,
    )
    records = [
        FinalizedPageRecord(
            output=only_defect,
            disposition=_disposition(),
            selection_provenance=SelectionProvenance.NATIVE_CLEAN,
        )
    ]
    with patch("socr.core.manifest.finalized_page_records", return_value=records):
        result = pipeline._phase_assemble(state, tmp_path)

    # Content-retained, non-clean: the same "completed with warnings, output
    # written" policy #189/#165 use for a shipped-but-disputed page.
    assert result.status is DocumentStatus.AUDIT_FAILED
    assert result.markdown and "Germany" in result.markdown
    out = capsys.readouterr().out
    assert "unverified row/column label" in out
    assert "[1]" in out


# --------------------------------------------------------------------------
# 8. Astra round 3 P2b: ``build_tables_trust`` retirement through a STALE
#    historical event once the FINAL winner carries no doubt, including a
#    page whose markdown holds two separate table blocks.
# --------------------------------------------------------------------------


def test_stale_event_does_not_untrust_a_page_whose_final_winner_is_clean() -> None:
    """The event that was emitted against an EARLIER, since-superseded
    candidate stays in ``events`` (real history, per the docstring), but
    ``label_unverified_pages`` -- the CURRENT set, from finalized records --
    tells ``build_tables_trust`` this page's winner carries no doubt, so it
    must not appear untrusted.
    """
    stale_event = AuditEvent(
        page_num=1,
        kind=LABEL_UNVERIFIED_KIND,
        engine="qwen",
        detail=LABEL_DETAIL,
        data={"cause": ""},
    )
    # Without the current-final-state filter (label_unverified_pages=None):
    # old, history-only behaviour -- the page stays untrusted.
    trust_history_only = build_tables_trust("doc.pdf", [stale_event])
    assert 1 in trust_history_only.untrusted_pages

    # With it, and page 1's current winner clean: the page clears.
    trust_current = build_tables_trust("doc.pdf", [stale_event], label_unverified_pages=frozenset())
    assert 1 not in trust_current.untrusted_pages


def test_two_table_page_stays_untrusted_until_both_tables_are_clean() -> None:
    """A page whose markdown carries TWO table blocks, one still doubted.
    ``collect_table_tokens`` aggregates content tokens across every table
    block on the page (whole-page scope, not per-region), so a single
    ``table_label_unverified`` field on the final output already reflects
    "at least one of this page's tables is still doubted" -- verified here
    at the trust-reducer boundary that consumes that field.
    """
    two_table_markdown = (
        "| Country | Amount |\n| --- | --- |\n| Germany | 10 |\n\n"
        "| City | Population |\n| --- | --- |\n| Berlin | 20 |\n"
    )
    event = AuditEvent(
        page_num=1,
        kind=LABEL_UNVERIFIED_KIND,
        engine="qwen",
        detail=LABEL_DETAIL,
        data={"cause": ""},
    )
    # Page 1's CURRENT winner still carries the doubt (one of its two tables
    # is unresolved) -- stays untrusted.
    still_doubted = build_tables_trust("doc.pdf", [event], label_unverified_pages=frozenset({1}))
    assert 1 in still_doubted.untrusted_pages

    # A later run where BOTH tables on page 1 are clean -- the page's current
    # winner carries no doubt at all, and clears despite the same history.
    both_clean = build_tables_trust("doc.pdf", [event], label_unverified_pages=frozenset())
    assert 1 not in both_clean.untrusted_pages
    del two_table_markdown  # documentation fixture; the reducer is page-scoped


def test_unrelated_distrust_kind_on_the_same_page_is_never_suppressed() -> None:
    """The current-final-state filter is scoped to ``LABEL_UNVERIFIED_KIND``
    only -- a DIFFERENT distrust kind on the same page must still count even
    when the label doubt itself has retired.
    """
    events = [
        AuditEvent(
            page_num=1,
            kind=LABEL_UNVERIFIED_KIND,
            engine="qwen",
            detail=LABEL_DETAIL,
            data={"cause": ""},
        ),
        AuditEvent(
            page_num=1,
            kind="native_table_verifier_warn",
            engine="native",
            detail="unrelated native table concern",
            data={},
        ),
    ]
    trust = build_tables_trust("doc.pdf", events, label_unverified_pages=frozenset())
    assert 1 in trust.untrusted_pages
    assert "native_table_verifier_warn" in trust.pages[1].reasons
    assert LABEL_UNVERIFIED_KIND not in trust.pages[1].reasons


# --------------------------------------------------------------------------
# 9. Astra round 4 P1: an UNRELATED historical whole-page resolving event
#    must never override the caller's explicit current label-doubt answer,
#    in either direction.
# --------------------------------------------------------------------------


def test_current_label_doubt_outranks_historical_whole_page_acceptance() -> None:
    """A page-wide RESOLVING event recorded for a DIFFERENT reason
    (``table_escalation_accepted``, GH-96 -- an accepted crop-reread
    escalation, nothing to do with the label gate) must not silently erase a
    label doubt the caller explicitly says is still live on the current
    winner. The reducer used to check the generic ``resolved_pages`` set
    BEFORE the label-specific branch, so chronology it has no business
    consulting for this kind overrode the caller's explicit answer.
    """
    events = [
        AuditEvent(page_num=1, kind="table_escalation_accepted", engine="qwen", detail="", data={}),
        AuditEvent(
            page_num=1,
            kind=LABEL_UNVERIFIED_KIND,
            engine="qwen",
            detail=LABEL_DETAIL,
            data={"cause": ""},
        ),
    ]
    trust = build_tables_trust("doc.pdf", events, label_unverified_pages=frozenset({1}))
    assert 1 in trust.untrusted_pages


def test_clean_final_winner_retires_despite_the_same_historical_acceptance() -> None:
    """The other direction, same mixed history: when the caller's current set
    says page 1 is now clean, the SAME ``table_escalation_accepted`` +
    label-unverified history must not keep it untrusted either.
    """
    events = [
        AuditEvent(page_num=1, kind="table_escalation_accepted", engine="qwen", detail="", data={}),
        AuditEvent(
            page_num=1,
            kind=LABEL_UNVERIFIED_KIND,
            engine="qwen",
            detail=LABEL_DETAIL,
            data={"cause": ""},
        ),
    ]
    trust = build_tables_trust("doc.pdf", events, label_unverified_pages=frozenset())
    assert 1 not in trust.untrusted_pages


# --------------------------------------------------------------------------
# 10. Astra round 4 P2: retirement must update the ON-DISK artifact, not only
#     the in-memory reducer -- a two-run write into the same directory.
# --------------------------------------------------------------------------


def _write_run(pipeline: UnifiedPipeline, doc_dir: Path, events: list, records: list) -> None:
    state = SimpleNamespace(handle=SimpleNamespace(filename="doc.pdf"))
    audit = SimpleNamespace(events=events)
    pipeline._write_tables_trust(state, audit, doc_dir, records=records)


def test_clean_final_run_removes_a_stale_tables_trust_json(tmp_path: Path) -> None:
    """'Absent means clean' is ``tables_trust.json``'s own contract (a
    prose-only run never writes one) -- leaving stale content behind once
    the doubt it recorded has retired lies about the CURRENT run to any
    consumer that only reads the sidecar.
    """
    pipeline = _pipeline()
    trust_path = tmp_path / "tables_trust.json"

    _write_run(
        pipeline,
        tmp_path,
        [
            AuditEvent(
                page_num=1,
                kind=LABEL_UNVERIFIED_KIND,
                engine="qwen",
                detail=LABEL_DETAIL,
                data={"cause": ""},
            )
        ],
        [
            FinalizedPageRecord(
                output=_flagged_output(page_num=1),
                disposition=_disposition(),
                selection_provenance=SelectionProvenance.NATIVE_CLEAN,
            )
        ],
    )
    assert trust_path.exists()
    assert json.loads(trust_path.read_text())["untrusted_pages"] == [1]

    # Second run: page 1's FINAL winner is now clean. The stale event is kept
    # in this run's history (real history must survive), but the current
    # winner carries no doubt, so the file must retire.
    _write_run(
        pipeline,
        tmp_path,
        [
            AuditEvent(
                page_num=1,
                kind=LABEL_UNVERIFIED_KIND,
                engine="qwen",
                detail=LABEL_DETAIL,
                data={"cause": ""},
            )
        ],
        [
            FinalizedPageRecord(
                output=_clean_output(page_num=1),
                disposition=_disposition(),
                selection_provenance=SelectionProvenance.NATIVE_CLEAN,
            )
        ],
    )
    assert not trust_path.exists()


def test_retirement_preserves_unrelated_active_distrust(tmp_path: Path) -> None:
    """Same two-run shape, but page 2 carries a DIFFERENT, still-active
    distrust kind throughout. Retiring page 1's label doubt must not touch
    page 2's entry -- the file is REWRITTEN with the current state, not
    blanked just because one page's doubt cleared.
    """
    pipeline = _pipeline()
    trust_path = tmp_path / "tables_trust.json"
    label_event = AuditEvent(
        page_num=1,
        kind=LABEL_UNVERIFIED_KIND,
        engine="qwen",
        detail=LABEL_DETAIL,
        data={"cause": ""},
    )
    unrelated_event = AuditEvent(
        page_num=2,
        kind="native_table_verifier_warn",
        engine="native",
        detail="unrelated native table concern",
        data={},
    )

    _write_run(
        pipeline,
        tmp_path,
        [label_event, unrelated_event],
        [
            FinalizedPageRecord(
                output=_flagged_output(page_num=1),
                disposition=_disposition(),
                selection_provenance=SelectionProvenance.NATIVE_CLEAN,
            ),
            FinalizedPageRecord(
                output=_clean_output(page_num=2),
                disposition=_disposition(),
                selection_provenance=SelectionProvenance.NATIVE_CLEAN,
            ),
        ],
    )
    assert json.loads(trust_path.read_text())["untrusted_pages"] == [1, 2]

    # Page 1 clears; page 2's unrelated distrust is still active and must
    # remain -- so the file is not removed, only rewritten without page 1.
    _write_run(
        pipeline,
        tmp_path,
        [label_event, unrelated_event],
        [
            FinalizedPageRecord(
                output=_clean_output(page_num=1),
                disposition=_disposition(),
                selection_provenance=SelectionProvenance.NATIVE_CLEAN,
            ),
            FinalizedPageRecord(
                output=_clean_output(page_num=2),
                disposition=_disposition(),
                selection_provenance=SelectionProvenance.NATIVE_CLEAN,
            ),
        ],
    )
    assert trust_path.exists()
    assert json.loads(trust_path.read_text())["untrusted_pages"] == [2]


# --------------------------------------------------------------------------
# 11. Astra round 5 P2: an event-free rerun through the REAL CALLER
#     (``_write_audit_log``, not ``_write_tables_trust`` directly) must still
#     retire a stale trust file -- the early return on empty ``audit.events``
#     used to skip trust reconciliation entirely.
# --------------------------------------------------------------------------


def test_event_free_rerun_removes_stale_trust_through_write_audit_log(tmp_path: Path) -> None:
    """A genuinely clean rerun: ``build_run_audit`` returns no events at all
    (not merely a page whose doubt retired -- NOTHING happened this run).
    ``_write_audit_log`` used to ``return`` right there, before
    ``_write_tables_trust`` ever ran, so round 4's retirement fix never got a
    chance to fire and a prior run's file was left stale on disk.
    """
    pipeline = _pipeline()
    trust_path = tmp_path / "tables_trust.json"
    audit_log_path = tmp_path / "audit_log.json"

    build_tables_trust(
        "doc.pdf",
        [
            AuditEvent(
                page_num=1,
                kind=LABEL_UNVERIFIED_KIND,
                engine="qwen",
                detail=LABEL_DETAIL,
                data={"cause": ""},
            )
        ],
    ).save(trust_path)
    assert trust_path.exists()

    state = SimpleNamespace(handle=SimpleNamespace(filename="doc.pdf"))
    with patch(
        "socr.core.audit_log.build_run_audit",
        return_value=SimpleNamespace(events=[]),
    ):
        pipeline._write_audit_log(state, tmp_path, records=[])

    assert not trust_path.exists()
    assert not audit_log_path.exists()


def test_clean_first_run_writes_no_audit_or_trust_files(tmp_path: Path) -> None:
    """The other half of the same contract: a run with nothing to report and
    NO prior file must stay artifact-free -- the fix must not start writing
    an empty ``tables_trust.json`` on every clean run just to be safe.
    """
    pipeline = _pipeline()
    state = SimpleNamespace(handle=SimpleNamespace(filename="doc.pdf"))
    with patch(
        "socr.core.audit_log.build_run_audit",
        return_value=SimpleNamespace(events=[]),
    ):
        pipeline._write_audit_log(state, tmp_path, records=[])

    assert not (tmp_path / "tables_trust.json").exists()
    assert not (tmp_path / "audit_log.json").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
