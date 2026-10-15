"""#1145 slice 2: a document socr could not read says so where a reader looks.

Design §4 and §7, test T10. ``_refuse_unreadable_input`` (#871) records the failure in
``metadata.json`` and the root index, which is what that ticket fixed -- but it wrote no
``.md`` at all and left ``output_path`` empty, so a reader or a downstream tool that opens
the document's markdown found nothing. "No file" is indistinguishable from "not processed
yet". This is the document-level analogue of the page rule: the whole artifact is the mark.

Hermeticity is NOT by construction here, and the first draft of this file claimed it was.
``process`` reaches ``_resolve_judge_model`` through ``_run_fingerprint`` before it ever
looks at the input, so it probes the configured Ollama host even for a file ``fitz`` cannot
open -- the exact trap CLAUDE.md names, caught by the repo's own GH-984 connection guard
rather than by review. So ``_resolve_judge_model`` returns "" and the engines are pinned,
and no page loop runs because the refusal returns first.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fitz")

from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.marks import DOCUMENT_FAILED_MARK_RE  # noqa: E402
from socr.core.result import DocumentStatus, FailureMode  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402


def _pipeline() -> UnifiedPipeline:
    pipeline = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            agentic=True,
            quiet=True,
            write_manifest=False,
        )
    )
    # ``_run_fingerprint`` resolves the judge model, which probes the host. Pinned to "" so
    # the run cannot depend on whether this machine has a judge model pulled -- and so it
    # cannot reach a daemon at all, which CI does not have.
    pipeline._resolve_judge_model = lambda *a, **k: ""
    return pipeline


def _unreadable_pdf(path: Path) -> Path:
    """A file with a PDF name and no loadable page. The refusal's own precondition."""
    path.write_bytes(b"%PDF-1.4\nnot a page tree at all\n%%EOF\n")
    return path


def _doc_metadata(out_dir: Path) -> dict:
    found = [p for p in out_dir.rglob("metadata.json") if p.parent != out_dir]
    assert len(found) == 1, found
    return json.loads(found[0].read_text())


def test_an_unreadable_document_ships_a_document_failed_mark(tmp_path: Path) -> None:
    """T10. The file exists, it is the mark, and ``output_path`` names it.

    Three assertions, because any two can hold while the thing is still broken: a file
    that exists but says nothing, a mark that is there but unreachable because
    ``output_path`` is empty, or a path recorded for a file that was never written.
    """
    pdf = _unreadable_pdf(tmp_path / "broken.pdf")
    out_dir = tmp_path / "out"
    pipeline = _pipeline()
    pipeline._scan_root = pdf.parent

    result = pipeline.process(pdf, output_dir=out_dir)

    assert result.status is DocumentStatus.ERROR
    assert result.failure_mode is FailureMode.UNREADABLE_INPUT

    meta = _doc_metadata(out_dir)
    recorded = meta.get("output_path") or ""
    assert recorded, "the document's markdown path is still empty, so the mark is unreachable"

    md = Path(recorded)
    assert md.is_file(), f"output_path names a file that was not written: {recorded}"
    body = md.read_text()
    assert DOCUMENT_FAILED_MARK_RE.search(body), f"the document mark is not in {body!r}"
    assert FailureMode.UNREADABLE_INPUT.value in body, (
        "the mark does not say what went wrong, so the reader learns nothing from it"
    )


def test_the_refusal_still_records_a_FAILED_document(tmp_path: Path) -> None:
    """The mark must not buy itself a softer status.

    Writing a markdown file is exactly the kind of change that quietly turns FAILED into
    PARTIAL -- ``_write_metadata`` derives both the status and ``output_path`` from
    ``has_text``, and #1145 had to separate them. If the status softened, the resume gate
    would stop refusing the entry and the next run would never retry the file: a document
    socr cannot read would be recorded as done, for ever.
    """
    pdf = _unreadable_pdf(tmp_path / "broken.pdf")
    out_dir = tmp_path / "out"
    pipeline = _pipeline()
    pipeline._scan_root = pdf.parent
    pipeline.process(pdf, output_dir=out_dir)

    meta = _doc_metadata(out_dir)
    assert meta["status"] == "failed", meta["status"]
    assert meta.get("pages") == 0 or isinstance(meta.get("pages"), int)
    assert meta.get("error"), "the recorded error went missing"


def test_the_mark_is_one_line_and_cannot_be_ended_early_by_the_error_text(
    tmp_path: Path,
) -> None:
    """The grammar's ``]`` terminator, on text socr does not author.

    The detail interpolates a PyMuPDF error string. A ``]`` in it would end the mark early
    and the remainder would read as prose, so a reader -- and the recogniser -- would see a
    truncated mark followed by noise. Driven through the real writer with a detail that
    contains one, rather than asserting on the regex alone.
    """
    pdf = _unreadable_pdf(tmp_path / "broken.pdf")
    out_dir = tmp_path / "out"
    pipeline = _pipeline()
    pipeline._scan_root = pdf.parent

    nasty = "boom ] and [ more"
    assert pipeline._write_document_failed_mark(pdf, out_dir, nasty) is True

    meta_md = [p for p in out_dir.rglob("*.md")]
    assert len(meta_md) == 1, meta_md
    body = meta_md[0].read_text()
    assert body.count("\n") == 1, f"the mark is not one line: {body!r}"
    found = DOCUMENT_FAILED_MARK_RE.findall(body)
    assert len(found) == 1, f"expected exactly one mark, got {found!r}"
    assert found[0].endswith("more]"), f"the mark was ended early: {found[0]!r}"
    assert "]" not in found[0][:-1], f"an unescaped terminator survived: {found[0]!r}"


def test_a_write_failure_leaves_output_path_empty_rather_than_lying(tmp_path: Path) -> None:
    """Fail closed: name the file only when the file is there.

    The refusal must survive a disk it cannot write to -- a document that cannot be written
    must still be REFUSED and recorded rather than crashing the batch -- but it must not
    then record a path to a file that does not exist, which would send every downstream
    reader to a missing file and look like corruption rather than a failure.
    """
    pdf = _unreadable_pdf(tmp_path / "broken.pdf")
    out_dir = tmp_path / "out"
    pipeline = _pipeline()
    pipeline._scan_root = pdf.parent

    import socr.pipeline.orchestrator as orch

    original = orch.UnifiedPipeline._write_document_failed_mark
    try:
        orch.UnifiedPipeline._write_document_failed_mark = lambda self, *a, **k: False
        result = pipeline.process(pdf, output_dir=out_dir)
    finally:
        orch.UnifiedPipeline._write_document_failed_mark = original

    assert result.status is DocumentStatus.ERROR, "the refusal itself stopped working"
    meta = _doc_metadata(out_dir)
    assert (meta.get("output_path") or "") == "", (
        "output_path names a markdown file that was never written"
    )


def test_every_other_caller_of_write_metadata_is_unchanged(tmp_path: Path) -> None:
    """``markdown_written`` defaults to ``has_text``, so nothing else moved.

    The parameter exists for one caller. A default that changed any other call site's
    ``output_path`` would be a silent behaviour change across every document socr writes,
    which is exactly the kind of thing a new keyword argument does by accident.
    """
    import inspect

    sig = inspect.signature(UnifiedPipeline._write_metadata)
    assert sig.parameters["markdown_written"].default is None, sig

    pdf = _unreadable_pdf(tmp_path / "broken.pdf")
    pipeline = _pipeline()
    pipeline._scan_root = pdf.parent

    # ``has_text`` alone still decides, both ways, when the new argument is not passed.
    from socr.core.document import DocumentHandle
    from socr.core.result import EngineResult
    from socr.core.state import DocumentState

    for has_text in (True, False):
        out_dir = tmp_path / f"o-{has_text}"
        state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1, page_count_known=True))
        pipeline._write_metadata(
            state,
            EngineResult(
                document_path=pdf,
                engine="qwen",
                status=DocumentStatus.SUCCESS if has_text else DocumentStatus.ERROR,
            ),
            out_dir,
            has_text=has_text,
        )
        recorded = (_doc_metadata(out_dir).get("output_path") or "") != ""
        assert recorded is has_text, (has_text, recorded)
