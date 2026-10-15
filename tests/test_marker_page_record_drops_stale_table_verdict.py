"""A page that ships the fail-closed marker does not record a table verdict for it.

The scope3 harvest (commit 89d799ba in scope-of-scope-3) found page records saying
``table_ladder_disposition: table_unverified`` on dozens of pages whose Markdown is only
``[page N failed: unverifiable table — see image]``, and a ``table_ladder_accepted``
event with no disposition (read as "table checked") on others. The verdict was about
a reading the page never shipped.

Pinned as a DIFFERENCE (CLAUDE.md, #257): the same page state written twice through the
real sidecar writer, changing only the page's ending. Hermetic: no engine, no provider.
"""

from __future__ import annotations

import json
import pathlib

import pytest

fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")

from socr.core.audit_log import AuditEvent  # noqa: E402
from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.manifest import (  # noqa: E402
    FinalizedPageRecord,
    PageDisposition,
    PageEnding,
    PagePrimaryReason,
    SelectionProvenance,
)
from socr.core.result import FailureMode, PageOutput, PageStatus  # noqa: E402
from socr.core.state import DocumentState, PageState  # noqa: E402
from socr.judge.table_verdict import TABLE_LADDER_ACCEPTED_KIND  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402

MARKER = "[page 1 failed: unverifiable table — see image]"
TABLE = "| Scope | 2018 |\n| --- | --- |\n| 3 | 412 |"


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch) -> None:
    """The sidecar's run fingerprint resolves the judge model; never reach a daemon."""
    monkeypatch.setattr(UnifiedPipeline, "_resolve_judge_model", lambda self, *a, **kw: "")


def _pipeline() -> UnifiedPipeline:
    return UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
        )
    )


def _pdf(path: pathlib.Path) -> pathlib.Path:
    doc = fitz.open()
    doc.new_page().insert_text((60, 80), "Scope 3", fontsize=9)
    doc.save(str(path))
    doc.close()
    return path


def _record(ending: PageEnding) -> FinalizedPageRecord:
    if ending is PageEnding.FAIL_CLOSED_MARKER:
        out = PageOutput(
            page_num=1,
            text=MARKER,
            status=PageStatus.ERROR,
            engine="native",
            failure_mode=FailureMode.STRUCTURE_CLASS_LADDER_EXHAUSTED,
        )
        return FinalizedPageRecord(
            out,
            PageDisposition(ending, PagePrimaryReason.STRUCTURE_CLASS),
            SelectionProvenance.NO_TEXT_MARKER,
        )
    out = PageOutput(page_num=1, text=TABLE, status=PageStatus.WARNING, engine="qwen")
    return FinalizedPageRecord(
        out,
        PageDisposition(ending, PagePrimaryReason.ACCEPTED_OUTPUT),
        SelectionProvenance.PASSING_BEST_OUTPUT,
    )


def _sidecar(
    tmp_path, leg: str, ending: PageEnding, disposition: FailureMode | None, events=()
) -> dict:
    pdf = _pdf(tmp_path / f"{leg}.pdf")
    state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
    state.pages[1] = PageState(page_num=1, table_ladder_disposition=disposition)
    state.events.extend(AuditEvent(page_num=1, kind=k) for k in events)
    pipeline = _pipeline()
    path = pipeline._flush_page_sidecar(state, 1, tmp_path / f"out-{leg}", record=_record(ending))
    return json.loads(path.read_text(encoding="utf-8"))


def test_unverified_verdict_is_moved_off_a_marker_page(tmp_path) -> None:
    shipped = _sidecar(tmp_path, "a", PageEnding.MODEL_OUTPUT, FailureMode.TABLE_UNVERIFIED)
    marker = _sidecar(tmp_path, "b", PageEnding.FAIL_CLOSED_MARKER, FailureMode.TABLE_UNVERIFIED)

    assert shipped["table_ladder_disposition"] == "table_unverified"
    assert shipped["table_ladder_superseded"] is None
    assert marker["table_ladder_disposition"] is None
    assert marker["table_ladder_superseded"] == "table_unverified"


def test_accepted_event_on_a_marker_page_is_named_superseded(tmp_path) -> None:
    events = (TABLE_LADDER_ACCEPTED_KIND,)
    shipped = _sidecar(tmp_path, "a", PageEnding.MODEL_OUTPUT, None, events)
    marker = _sidecar(tmp_path, "b", PageEnding.FAIL_CLOSED_MARKER, None, events)

    assert shipped["table_ladder_superseded"] is None
    assert marker["table_ladder_disposition"] is None
    assert marker["table_ladder_superseded"] == "accepted"


@pytest.mark.parametrize("terminal", [FailureMode.TABLE_REJECTED, FailureMode.TABLE_WITHHELD])
def test_content_terminals_the_resume_gate_reads_are_kept(tmp_path, terminal) -> None:
    marker = _sidecar(tmp_path, "m", PageEnding.FAIL_CLOSED_MARKER, terminal)
    assert marker["table_ladder_disposition"] == terminal.value
    assert marker["table_ladder_superseded"] is None


def test_in_memory_disposition_is_untouched(tmp_path) -> None:
    pdf = _pdf(tmp_path / "s.pdf")
    state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
    state.pages[1] = PageState(page_num=1, table_ladder_disposition=FailureMode.TABLE_UNVERIFIED)
    _pipeline()._flush_page_sidecar(
        state, 1, tmp_path / "out", record=_record(PageEnding.FAIL_CLOSED_MARKER)
    )
    assert state.pages[1].table_ladder_disposition is FailureMode.TABLE_UNVERIFIED
