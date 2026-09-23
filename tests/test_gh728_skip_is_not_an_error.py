"""GH-728: re-invoking `socr process` on an already-processed document must not
exit 1 with "Processing failed: None".

``UnifiedPipeline._resume_skip`` returns a ``DocumentStatus.SKIPPED`` result with
no error text. The CLI treated every non-success result as a failure, so every
re-invocation of a finished document raised ``Processing failed: None``. A skip
reports a previous run; nothing was attempted, so nothing failed. ``socr batch``
already treats skipped files as neither completed nor failed.

Hermetic: ``UnifiedPipeline.process`` is replaced, so no engine, judge or PDF work
runs. The difference pinned is SKIPPED vs a genuine failure, same CLI path.
"""

from __future__ import annotations

from pathlib import Path

import fitz
from click.testing import CliRunner

from socr.cli import cli
from socr.core.result import DocumentStatus, EngineResult
from socr.pipeline.orchestrator import UnifiedPipeline


def _pdf(tmp_path: Path) -> Path:
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "text")
    path = tmp_path / "doc.pdf"
    doc.save(path)
    doc.close()
    return path


def _invoke(monkeypatch, tmp_path, status, error=None):
    def _fake_process(self, pdf_path, output_dir=None, scan_root=None):
        return EngineResult(document_path=Path(pdf_path), engine="qwen", status=status, error=error)

    monkeypatch.setattr(UnifiedPipeline, "process", _fake_process)
    return CliRunner().invoke(
        cli, ["process", str(_pdf(tmp_path)), "-o", str(tmp_path / "out"), "--primary", "qwen"]
    )


def test_a_skipped_document_exits_zero_without_a_failure(monkeypatch, tmp_path):
    result = _invoke(monkeypatch, tmp_path, DocumentStatus.SKIPPED)
    assert result.exit_code == 0, result.output
    assert "Processing failed" not in result.output


def test_a_real_failure_still_fails(monkeypatch, tmp_path):
    """The control: only the skip changed. A genuine error still exits nonzero
    with its reason."""
    result = _invoke(monkeypatch, tmp_path, DocumentStatus.ERROR, error="something broke")
    assert result.exit_code != 0
    assert "something broke" in result.output


def test_the_skip_message_names_the_flag():
    """The advice on a failed page ('re-run the page') is unreachable without
    --reprocess, because the document gate decides first; both messages must name it."""
    import inspect

    from socr.pipeline import orchestrator

    src = inspect.getsource(orchestrator.UnifiedPipeline._resume_skip)
    assert "--reprocess" in src
    assert "re-run the page with --reprocess" in inspect.getsource(orchestrator)
