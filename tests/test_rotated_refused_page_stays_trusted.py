"""PR #907 regression: a lane-refused rotated born-digital page stays trusted.

a7dce6c made ``native_trusted`` return False for every page with
``native_table_lane_refused``. That routed every rotated born-digital table page
into the scanned source-evidence gate, which excludes the stored layer from the
evidence; on real rotated pages whose stored words DO support the model table
(the A/B in the #907 comments) this ended in ``source_evidence_table_reject``.

Pinned as a DIFFERENCE, not a value (repo CLAUDE.md): the same page, the same
table, the same judge, changing only whether the distrust is re-applied.

Hermetic: heuristic judge backend, no provider, no ollama, no OCR engine
(``ocr_image_fn`` is stubbed empty on the source-evidence path).
"""

from __future__ import annotations

from unittest.mock import patch

import fitz

from socr.core.born_digital import DocumentAssessment, PageAssessment
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline

_TABLE = (
    "| Counterparty | Amount | Drawn |\n"
    "| --- | --- | --- |\n"
    "| Bundesbank | 62.5 | 12.5 |\n"
    "| Bank of Japan | 67.0 | 15.0 |\n"
)

_SOURCE_EVIDENCE_KINDS = {
    "source_evidence_table_reject",
    "source_evidence_table_label_unverified",
}


def _judge_for_refused_page(tmp_path):
    """The real ``_build_page_judge`` wiring, on a born-digital, lane-refused page
    whose stored words carry exactly the model table's rows."""
    pdf = tmp_path / "rotated.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "Bundesbank 62.5 12.5", fontsize=10)
    page.insert_text((72, 120), "Bank of Japan 67.0 15.0", fontsize=10)
    doc.save(pdf)
    doc.close()

    pipeline = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            judge_backend="heuristic",
        )
    )
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    state.pages[1].is_born_digital = True
    state.pages[1].has_tables = True
    pipeline._last_assessment = DocumentAssessment(
        path=pdf,
        pages=[
            PageAssessment(
                page_num=1,
                is_born_digital=True,
                native_text="",
                confidence=1.0,
                has_tables=True,
                native_table_lane_refused=True,
            )
        ],
    )
    with patch.object(pipeline, "_resolve_judge_model", return_value=""):
        judge = pipeline._build_page_judge(state)
    return judge, state


def _assess(judge, state):
    output = PageOutput(
        page_num=1,
        text=_TABLE,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    before = len(state.events)
    judge._ocr_image_fn = lambda _pix: ""
    decision = judge.assess(output, PROFILE_QWEN_LOCAL)
    kinds = {e.kind for e in state.events[before:]}
    return decision, kinds


def test_lane_refused_born_digital_page_is_trusted(tmp_path) -> None:
    judge, _state = _judge_for_refused_page(tmp_path)
    assert judge._native_trusted(1) is True


def test_refused_page_does_not_enter_source_evidence_gate(tmp_path) -> None:
    """Difference-pin: re-applying the a7dce6c distrust changes the route."""
    judge, state = _judge_for_refused_page(tmp_path)
    decision_kept, kinds_kept = _assess(judge, state)

    # Re-apply the reverted behaviour: lane-refused pages are distrusted.
    original = judge._native_trusted
    judge._native_trusted = lambda pn: False if pn == 1 else original(pn)
    _decision_distrusted, kinds_distrusted = _assess(judge, state)

    assert not (kinds_kept & _SOURCE_EVIDENCE_KINDS), kinds_kept
    assert decision_kept.accept is True, decision_kept.reason
    assert kinds_distrusted & _SOURCE_EVIDENCE_KINDS, (
        "with the distrust re-applied the page must enter the source-evidence "
        "gate; otherwise this test cannot tell the two behaviours apart"
    )
