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


def _exit_codes(tmp_path: Path, status: DocumentStatus, error: str | None) -> tuple[int, int, str]:
    """Exit code from `process` and from `batch`, for the SAME result.

    GH-472: the batch half used to stub `process_batch` itself and invent
    `last_outcome` in the stub's `__init__` -- so reverting the production
    recording loop in `orchestrator.process_batch` left this suite green and the
    "cannot drift" claim was unpinned at the batch caller.

    Only `UnifiedPipeline.process` is stubbed now. The REAL `process_batch`
    runs, walks the directory, and records the outcome through
    `contract_status_for` exactly as production does.
    """
    from unittest.mock import patch

    from socr.pipeline.orchestrator import UnifiedPipeline

    pdf = _pdf(tmp_path / "in")
    result = _result(pdf, status, error)

    def _fake_process(self, pdf_path, output_dir=None, scan_root=None):
        return _result(Path(pdf_path), status, error)

    with patch.object(UnifiedPipeline, "process", _fake_process):
        single = CliRunner().invoke(cli, ["process", str(pdf), "-o", str(tmp_path / "o1"), "-q"])
        batch = CliRunner().invoke(
            cli, ["batch", str(pdf.parent), "-o", str(tmp_path / "o2"), "-q"]
        )
    return single.exit_code, batch.exit_code, single.output


@pytest.mark.parametrize(
    ("status", "error", "expect_zero"),
    [
        (DocumentStatus.SUCCESS, None, True),
        (DocumentStatus.AUDIT_FAILED, "some pages failed audit", False),
        # GH-177 review: this string must CONTAIN `LOST_CONTENT_NOTE`
        # ("produced no usable output") or it takes the same plain-warning
        # branch as the case above and the parametrisation is cosmetic. My
        # first version said "page(s) 10 lost content", which does not.
        (DocumentStatus.AUDIT_FAILED, "page(s) 10 produced no usable output", False),
        (DocumentStatus.ERROR, "nothing extracted", False),
    ],
)
def test_process_and_batch_agree(
    tmp_path: Path, status: DocumentStatus, error: str | None, expect_zero: bool
) -> None:
    single, batch, output = _exit_codes(tmp_path, status, error)

    assert (single == 0) is expect_zero, (
        f"process exited {single} for {status.value}; expected "
        f"{'zero' if expect_zero else 'nonzero'}"
    )
    assert (single == 0) == (batch == 0), (
        f"process and batch disagree for {status.value}: process={single} batch={batch}"
    )

    # GH-177 review: the two AUDIT_FAILED cases share an exit code by design --
    # lost content is not a separate code -- so the parametrisation would be
    # cosmetic if it stopped there. What differs is the MESSAGE, and the
    # lost-content string must actually contain LOST_CONTENT_NOTE to reach that
    # branch at all (my first version's did not).
    if status is DocumentStatus.AUDIT_FAILED:
        from socr.core.result import LOST_CONTENT_NOTE

        lost = bool(error and LOST_CONTENT_NOTE in error)
        assert ("lost content" in output.lower()) is lost, (
            f"the {'lost-content' if lost else 'plain'} run took the wrong "
            f"message branch: {output!r}"
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
