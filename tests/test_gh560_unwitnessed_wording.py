"""GH-560: a terminal that cannot be retried must not say it is retryable.

The 2026-09-03 live two-rung smoke found table ``p4-t0`` ending
``TABLE_UNVERIFIED`` with an EMPTY ``rung_trail`` and ``witness_scope: "none"``
-- no rung ever ran, because no witness could be prepared. The audit event, the
document note and the CLI summary all described it as "infra problem, retryable
on resume". Nothing retries it: the P1 latch correctly does not fire for a
no-witness terminal (there is no unavailable rung to wait for), so the document
is skipped on resume and the promise is empty.

Pinned as the DIFFERENCE #560 asks for -- a no-witness terminal against a
genuine rung outage -- because the two are only worth separating if they still
read differently.

One correction to the ticket's own pin. It says "empty ``rung_trail`` => that
wording". An empty trail is not sufficient: a witness that WAS located and then
found no reachable rung has an empty trail too, and that one genuinely is
retryable. The witness is what separates them, so the condition is the empty
trail AND ``witness_scope == "none"``. ``test_a_located_witness_with_no_rung_
keeps_the_retryable_wording`` is that case, and it fails if the condition is
loosened to the trail alone.

Latch semantics are untouched, as the ticket requires, and so are the assemble
buckets and the page disposition -- this is wording, and a ``retryable: False``
marker so consumers need not re-derive it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.result import FailureMode, PageOutput, PageStatus  # noqa: E402
from socr.core.state import DocumentState  # noqa: E402
from socr.judge.table_verdict import TABLE_LADDER_UNVERIFIED_KIND  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402

_TABLE_MD = "| a | b |\n| --- | --- |\n| 1 | 2 |\n"


def _pipeline() -> UnifiedPipeline:
    return UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            table_judge_ladder=True,
        )
    )


def _borderless_pdf(tmp_path: Path) -> Path:
    """No ruled boxes, so witness location fails and no crop is produced."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = tmp_path / "borderless.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for i in range(6):
        page.insert_text((72, 100 + i * 16), "a b 1 2 plain prose with no rules", fontsize=10)
    doc.save(str(pdf))
    doc.close()
    return pdf


def _ruled_pdf(tmp_path: Path) -> Path:
    """One ruled box, so a witness IS located and gets a crop."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = tmp_path / "ruled.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.draw_rect(fitz.Rect(72, 100, 500, 260), color=(0, 0, 0), width=1.2)
    page.draw_line(fitz.Point(72, 140), fitz.Point(500, 140))
    page.draw_line(fitz.Point(280, 100), fitz.Point(280, 260))
    page.insert_text((90, 130), "a", fontsize=10)
    page.insert_text((300, 130), "b", fontsize=10)
    page.insert_text((90, 200), "1", fontsize=10)
    page.insert_text((300, 200), "2", fontsize=10)
    doc.save(str(pdf))
    doc.close()
    return pdf


def _state(pdf: Path) -> tuple[DocumentState, PageOutput]:
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    out = PageOutput(
        page_num=1,
        text=_TABLE_MD,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    state.pages[1].attempts.append(out)
    state.pages[1].best_output = out
    return state, out


def _run(pdf: Path, rungs: list) -> DocumentState:
    pipeline = _pipeline()
    state, out = _state(pdf)
    pipeline._run_table_judge_gate(state, 1, state.pages[1], out, rungs)
    return state


def _unverified_event(state: DocumentState):
    events = [e for e in state.events if getattr(e, "kind", "") == TABLE_LADDER_UNVERIFIED_KIND]
    assert len(events) == 1, f"expected one unverified terminal, got {events}"
    return events[0]


def test_a_witness_with_no_crop_is_not_called_retryable(tmp_path: Path) -> None:
    """The bug: no rung ran, no witness existed, and the record promised a retry."""
    state = _run(_borderless_pdf(tmp_path / "b"), [])

    event = _unverified_event(state)
    assert event.data["rung_trail"] == [], "a rung ran; this is not the no-witness case"
    assert event.data["witness_scope"] == "none"
    assert "retryable on resume" not in event.detail, (
        f"the no-witness terminal still promises a retry: {event.detail}"
    )
    assert "no table witness could be prepared" in event.detail
    assert event.data.get("retryable") is False
    assert state.pages[1].table_unverified_unwitnessed is True


def test_a_located_witness_with_no_rung_keeps_the_retryable_wording(tmp_path: Path) -> None:
    """The difference control, and the correction to #560's stated pin.

    An empty ``rung_trail`` alone does NOT mean unretryable. Here the witness is
    located and gets a crop, and the ladder is handed no rungs -- an outage, and
    a rung coming back genuinely repairs it. If the condition were loosened to
    the empty trail alone, this page would be mislabelled as permanently
    unjudgeable.
    """
    state = _run(_ruled_pdf(tmp_path / "r"), [])

    event = _unverified_event(state)
    assert event.data["rung_trail"] == [], "a rung ran; this is not the outage case"
    assert event.data["witness_scope"] != "none", (
        "the witness was not located, so this test is measuring the no-witness "
        "case again rather than the outage it claims to"
    )
    assert "retryable on resume" in event.detail, (
        f"a genuine rung outage lost its retryable wording: {event.detail}"
    )
    assert "retryable" not in event.data, "an outage must not be marked unretryable"
    assert state.pages[1].table_unverified_unwitnessed is False


def test_the_document_note_separates_the_two(tmp_path: Path) -> None:
    """The note is what the CLI prints verbatim as ``result.error``, so an
    operator reads this sentence and decides whether to re-run."""
    pipeline = _pipeline()

    unwitnessed = _run(_borderless_pdf(tmp_path / "nb"), [])
    note = pipeline._table_judge_ladder_note(unwitnessed)
    assert note is not None, "a TABLE_UNVERIFIED page produced no document note"
    assert "no table witness could be prepared" in note
    assert "retryable on resume" not in note, f"the note still promises a retry: {note}"

    outage = _run(_ruled_pdf(tmp_path / "nr"), [])
    outage_note = pipeline._table_judge_ladder_note(outage)
    assert outage_note is not None
    assert "retryable on resume" in outage_note, (
        f"a genuine outage lost the retryable wording in the note: {outage_note}"
    )


def test_the_disposition_and_the_latch_are_untouched(tmp_path: Path) -> None:
    """#560 asks for wording only. If this fix moved the page out of
    TABLE_UNVERIFIED, or started latching a no-witness terminal, it would be
    changing document status and resume behaviour under cover of a rewording."""
    unwitnessed = _run(_borderless_pdf(tmp_path / "db"), [])
    ps = unwitnessed.pages[1]

    assert ps.table_ladder_disposition is FailureMode.TABLE_UNVERIFIED, (
        "the no-witness terminal changed disposition; document status moved"
    )
    assert not getattr(ps, "table_judge_retry_pending", False), (
        "a no-witness terminal now latches for retry -- the exact semantics "
        "#560 asked to keep unchanged"
    )
