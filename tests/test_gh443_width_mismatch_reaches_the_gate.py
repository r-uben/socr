"""GH-443: the two width-mismatch spellings must reach the SHIPPING gate.

#439 closed #301 in production -- ``table_content_defect`` no longer requires a
matching delimiter width, so both leftover spellings return
``table_content_empty``. But its tests only ever called that helper. Nothing
pinned ``table_output_defect``, the judge's structural gate, or a
``PageStatus``, so unplugging the helper from the shipping path would have left
that suite green while empty tables shipped again. GH-190's own ``process()``
fixture is equal-width and does not cover these two spellings.

The fixtures here are the two spellings only; the equal-width case stays with
GH-190.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from socr.core.audit_log import AuditEvent
from socr.core.born_digital import BornDigitalDetector
from socr.core.result import PageOutput, PageStatus
from socr.pipeline.agentic import AcceptDecision, NativeTableVerifierJudge
from socr.tables.structure_check import DEFECT_TABLE_CONTENT_EMPTY, table_output_defect

# blank header, narrower delimiter, empty body
BLANK_HEADER = "|  |  |  |\n| --- | --- |\n|  |  |  |\n"
# populated header, narrower delimiter, empty body matching the delimiter
POPULATED_HEADER = "| A | B | C |\n| --- | --- |\n|  |  |\n"
# the control: same shape, one body cell carrying a value
POPULATED_BODY = "| A | B | C |\n| --- | --- |\n| 1 | 2 |\n"

SPELLINGS = [("blank header", BLANK_HEADER), ("populated header", POPULATED_HEADER)]


class _StubAcceptingInnerJudge:
    def accept(self, *_args, **_kwargs) -> AcceptDecision:
        return AcceptDecision(accept=True, reason="inner stub accepted", confidence=1.0)


@pytest.mark.parametrize(("name", "markdown"), SPELLINGS)
def test_the_shipping_gate_sees_the_defect(name: str, markdown: str) -> None:
    """``table_output_defect`` is the predicate the shipping path actually calls."""
    assert table_output_defect(markdown, None) == DEFECT_TABLE_CONTENT_EMPTY, (
        f"{name}: the width mismatch hid the empty table from the shipping gate"
    )


def test_a_width_mismatch_with_content_is_a_SHAPE_defect_not_an_empty_one() -> None:
    """Control, and the thing that keeps #301's fix from becoming a width rule.

    The gate does fire on the populated fixture -- but as ``grid_shape``, GH-151's
    ragged-widths term, which is the existing owner of a width mismatch. What
    must never happen is a table with values being reported as CONTENT-empty.
    Asserting ``== ""`` here would have been wrong, and would have pinned the
    content term as the owner of shape.
    """
    verdict = table_output_defect(POPULATED_BODY, None)
    assert verdict != DEFECT_TABLE_CONTENT_EMPTY, (
        f"a table carrying values was reported as content-empty: {verdict!r}"
    )
    assert verdict == "grid_shape", (
        f"the width mismatch should be reported by its own owner, got {verdict!r}"
    )


@pytest.mark.parametrize(("name", "markdown"), SPELLINGS)
def test_the_judge_refuses_the_page_and_says_why(name: str, markdown: str) -> None:
    """An accepting inner judge must be overturned, with an audit event."""
    events: list[AuditEvent] = []
    judge = NativeTableVerifierJudge(
        inner=_StubAcceptingInnerJudge(),
        get_fitz_page=lambda pn: None,
        is_table_page=lambda pn: True,
        record_event=events.append,
    )
    output = PageOutput(
        page_num=1,
        text=markdown,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    decision = judge._apply_structural_gate(
        AcceptDecision(accept=True, reason="inner stub accepted", confidence=1.0),
        output,
        page_num=1,
        words=None,
        rules=None,
    )

    assert decision.accept is False, f"{name}: the judge shipped an empty table"
    assert decision.reason == f"table_structure_failed: {DEFECT_TABLE_CONTENT_EMPTY}"
    assert events == [
        AuditEvent(
            page_num=1,
            kind="table_structure_failed",
            engine="qwen",
            detail=DEFECT_TABLE_CONTENT_EMPTY,
            data={"defect": DEFECT_TABLE_CONTENT_EMPTY},
        )
    ], f"{name}: the failure did not surface as an audit event"


@pytest.mark.parametrize(("name", "markdown"), SPELLINGS)
def test_the_detector_stamps_the_defect_off_extract_structured(
    name: str, markdown: str, tmp_path
) -> None:
    """Pinned as a DIFFERENCE at ``extract_structured``.

    Only the emitted markdown changes between the two runs, so a detector that
    stopped consulting the content term at all would fail the empty half while
    the populated half still passes.
    """
    fitz = pytest.importorskip("fitz")

    pdf_path = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text(
        (54, 72),
        "This is clean born-digital text that is definitely longer than fifty characters.",
    )
    doc.save(str(pdf_path))
    doc.close()

    def _assess(emitted: str):
        detector = BornDigitalDetector()
        with (
            patch.object(BornDigitalDetector, "_detect_tables", return_value=True),
            patch.object(BornDigitalDetector, "extract_structured", return_value=emitted),
        ):
            return detector.detect(pdf_path)

    empty = _assess(markdown).pages[0]
    populated = _assess(POPULATED_BODY).pages[0]

    assert empty.native_table_content_defect == DEFECT_TABLE_CONTENT_EMPTY, (
        f"{name}: the detector did not stamp the content defect off extract_structured"
    )
    assert populated.native_table_content_defect == "", (
        "control: a table carrying values must not be stamped CONTENT-empty, or the "
        "difference measures nothing"
    )
    assert empty.native_table_structure_defective is True
