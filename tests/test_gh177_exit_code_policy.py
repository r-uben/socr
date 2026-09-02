"""GH-177: `process` and `batch` must agree on the exit code for one status.

`ocr_output_contract.RunOutcome` documents ONE policy -- "nonzero if ANY file or
page failed, including partial documents" -- but only batch went through it.
Single-file `socr process` exited 0 on `AUDIT_FAILED` unless the error carried
`LOST_CONTENT_NOTE`, while batch mapped the same status to PARTIAL and exited 1.
A script wrapping `process` and one wrapping `batch` therefore saw OPPOSITE
signals for the same document.

Both paths now derive the status from `contract_status_for`, and this file
compares them directly rather than asserting each in isolation -- the defect was
a DISAGREEMENT, so the pin has to be a comparison.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from socr.cli import cli
from socr.core.result import DocumentStatus, EngineResult


def _pdf(tmp_path: Path) -> Path:
    fitz = pytest.importorskip("fitz")
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "d.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "born digital text long enough to be a text layer.")
    doc.save(str(path))
    doc.close()
    return path


def _result(pdf: Path, status: DocumentStatus, error: str | None) -> EngineResult:
    return EngineResult(document_path=pdf, engine="qwen", status=status, error=error)


def _exit_codes(tmp_path: Path, status: DocumentStatus, error: str | None) -> tuple[int, int]:
    """Exit code from `process` and from `batch`, for the SAME result."""
    from unittest.mock import patch

    pdf = _pdf(tmp_path / "in")
    result = _result(pdf, status, error)

    class _Stub:
        def __init__(self, config):
            from ocr_output_contract import RunOutcome, Status

            from socr.core.result import contract_status_for

            self.config = config
            self.last_outcome = RunOutcome()
            doc_status = contract_status_for(result)
            if doc_status is Status.COMPLETED:
                self.last_outcome.add(Status.COMPLETED, output_path=str(pdf))
            else:
                self.last_outcome.add(doc_status, detail=str(pdf))

        def process(self, *_a, **_k):
            return result

        def process_batch(self, *_a, **_k):
            return [result]

    with patch("socr.pipeline.orchestrator.UnifiedPipeline", _Stub):
        single = CliRunner().invoke(cli, ["process", str(pdf), "-o", str(tmp_path / "o1"), "-q"])
        batch = CliRunner().invoke(
            cli, ["batch", str(pdf.parent), "-o", str(tmp_path / "o2"), "-q"]
        )
    return single.exit_code, batch.exit_code


@pytest.mark.parametrize(
    ("status", "error", "expect_zero"),
    [
        (DocumentStatus.SUCCESS, None, True),
        (DocumentStatus.AUDIT_FAILED, "some pages failed audit", False),
        (DocumentStatus.AUDIT_FAILED, "page(s) 10 lost content", False),
        (DocumentStatus.ERROR, "nothing extracted", False),
    ],
)
def test_process_and_batch_agree(
    tmp_path: Path, status: DocumentStatus, error: str | None, expect_zero: bool
) -> None:
    single, batch = _exit_codes(tmp_path, status, error)

    assert (single == 0) is expect_zero, (
        f"process exited {single} for {status.value}; expected "
        f"{'zero' if expect_zero else 'nonzero'}"
    )
    assert (single == 0) == (batch == 0), (
        f"process and batch disagree for {status.value}: process={single} batch={batch}"
    )


def test_the_status_mapping_is_the_shared_one() -> None:
    """The policy itself, so the two callers cannot re-derive it separately."""
    from ocr_output_contract import Status

    from socr.core.result import contract_status_for

    pdf = Path("x.pdf")
    assert contract_status_for(_result(pdf, DocumentStatus.SUCCESS, None)) is Status.COMPLETED
    assert (
        contract_status_for(_result(pdf, DocumentStatus.AUDIT_FAILED, "pages failed"))
        is Status.PARTIAL
    )
    assert contract_status_for(_result(pdf, DocumentStatus.ERROR, "boom")) is Status.FAILED
