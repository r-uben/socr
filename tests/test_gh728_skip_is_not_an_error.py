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
    tmp_path.mkdir(parents=True, exist_ok=True)
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


def test_a_skipped_partial_document_keeps_a_nonzero_exit_with_a_real_reason(monkeypatch, tmp_path):
    """#728 review: ``_resume_skippable`` also skips a document whose last run was
    PARTIAL, and GH-177's policy is that a partial document exits nonzero. The skip
    must not launder that into a clean exit -- but it must say why, not 'None'."""
    reason = "already processed and recorded as partial: pass --reprocess to retry"
    result = _invoke(monkeypatch, tmp_path, DocumentStatus.SKIPPED, error=reason)
    assert result.exit_code != 0
    assert "recorded as partial" in result.output
    assert "Processing failed: None" not in result.output


def _record(out_dir: Path, pdf: Path, status_name: str) -> None:
    from ocr_output_contract import (
        DocMetadata,
        RootIndex,
        Status,
        relative_key,
        safe_checksum,
        utc_timestamp,
    )

    index = RootIndex(out_dir)
    index.record(
        relative_key(pdf, pdf.parent),
        DocMetadata(
            status=Status[status_name],
            checksum=safe_checksum(pdf),
            model="qwen",
            backend="socr",
            processing_time=0.0,
            timestamp=utc_timestamp(),
            output_path=str(out_dir / "doc" / "doc.md"),
            pages=1,
        ),
    )


def test_the_skip_carries_the_recorded_outcome(monkeypatch, tmp_path):
    """The difference the fix depends on, at its source: the same skip, only the
    recorded status changing. Completed -> no error; partial -> an error naming it."""
    from socr.core.config import EngineType, PipelineConfig
    from socr.pipeline import orchestrator

    monkeypatch.setattr(orchestrator, "_resume_skippable", lambda *a, **k: True)
    outcomes = {}
    for name in ("COMPLETED", "PARTIAL"):
        pdf = _pdf(tmp_path / name)
        out = tmp_path / name / "out"
        out.mkdir(parents=True)
        _record(out, pdf, name)
        pipe = UnifiedPipeline(
            PipelineConfig(
                quiet=True,
                judge_backend="heuristic",
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN],
            )
        )
        pipe._scan_root = pdf.parent
        outcomes[name] = pipe._resume_skip(pdf, out)

    assert outcomes["COMPLETED"].status is DocumentStatus.SKIPPED
    assert outcomes["COMPLETED"].error is None
    assert outcomes["PARTIAL"].status is DocumentStatus.SKIPPED
    assert "partial" in (outcomes["PARTIAL"].error or "")


def test_the_skip_message_the_user_sees_names_the_flag(monkeypatch, tmp_path):
    """cubic on #896: pin the text actually printed, not the source (which a comment
    could satisfy). A completed record is skipped and the console says how to redo it."""
    import io

    from rich.console import Console

    from socr.core.config import EngineType, PipelineConfig
    from socr.pipeline import orchestrator

    buf = io.StringIO()
    monkeypatch.setattr(orchestrator, "console", Console(file=buf, width=200))
    monkeypatch.setattr(orchestrator, "_resume_skippable", lambda *a, **k: True)
    pdf = _pdf(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    _record(out, pdf, "COMPLETED")
    pipe = UnifiedPipeline(
        PipelineConfig(
            judge_backend="heuristic",
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
        )
    )
    pipe._scan_root = pdf.parent
    pipe._resume_skip(pdf, out)
    assert "--reprocess" in buf.getvalue()


def test_a_real_partial_record_exits_nonzero_end_to_end(monkeypatch, tmp_path):
    """cubic on #896: the two halves above never meet. Drive the real CLI and the
    real ``process()`` against a real recorded PARTIAL entry -- only the resume
    gate's fingerprint comparison is forced to match -- so a ``process()`` that
    stripped ``_resume_skip``'s reason would fail here."""
    from socr.pipeline import orchestrator

    monkeypatch.setattr(orchestrator, "_resume_skippable", lambda *a, **k: True)
    pdf = _pdf(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    _record(out, pdf, "PARTIAL")
    result = CliRunner().invoke(
        cli,
        ["process", str(pdf), "-o", str(out), "--primary", "qwen", "--judge-backend", "heuristic"],
    )
    assert result.exit_code != 0, result.output
    assert "recorded as partial" in result.output
    assert "Processing failed: None" not in result.output


def test_a_real_completed_record_exits_zero_end_to_end(monkeypatch, tmp_path):
    """The control for the test above: same CLI, same process(), only the recorded
    status differs."""
    from socr.pipeline import orchestrator

    monkeypatch.setattr(orchestrator, "_resume_skippable", lambda *a, **k: True)
    pdf = _pdf(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    _record(out, pdf, "COMPLETED")
    result = CliRunner().invoke(
        cli,
        ["process", str(pdf), "-o", str(out), "--primary", "qwen", "--judge-backend", "heuristic"],
    )
    assert result.exit_code == 0, result.output
    assert "Processing failed" not in result.output
