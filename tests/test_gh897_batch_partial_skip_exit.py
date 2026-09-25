"""GH-897: `socr batch` must not exit 0 when a skipped file was recorded PARTIAL.

`process_batch` filtered resume-skippable files out before anything reached the
`RunOutcome`, so a batch re-run over documents whose last run was partial printed
"All files already processed" and exited 0 -- while the same files through
`socr process` each exit 1 (#896). That laundered GH-177's nonzero-on-partial at
the batch caller.

Driven through the real CLI (`CliRunner` -> `socr batch`) and the real
`process_batch`, per the issue. Only the resume gate's fingerprint comparison is
forced to match (`_resume_skippable` -> True), so no file is processed and no
provider is touched; the recorded root-index entries are real.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest
from click.testing import CliRunner

from socr.cli import cli


def _pdf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), path.stem)
    doc.save(path)
    doc.close()
    return path


def _record(out_dir: Path, pdf: Path, status_name: str) -> None:
    from ocr_output_contract import (
        DocMetadata,
        RootIndex,
        Status,
        relative_key,
        safe_checksum,
        utc_timestamp,
    )

    RootIndex(out_dir).record(
        relative_key(pdf, pdf.parent),
        DocMetadata(
            status=Status[status_name],
            checksum=safe_checksum(pdf),
            model="qwen",
            backend="socr",
            processing_time=0.0,
            timestamp=utc_timestamp(),
            output_path=str(out_dir / pdf.stem / f"{pdf.stem}.md"),
            pages=1,
        ),
    )


def _batch(monkeypatch, tmp_path, statuses):
    from socr.pipeline import orchestrator

    monkeypatch.setattr(orchestrator, "_resume_skippable", lambda *a, **k: True)
    src = tmp_path / "in"
    out = tmp_path / "out"
    out.mkdir(parents=True)
    for i, status in enumerate(statuses):
        _record(out, _pdf(src / f"doc{i}.pdf"), status)
    return CliRunner().invoke(
        cli,
        ["batch", str(src), "-o", str(out), "--primary", "qwen", "--judge-backend", "heuristic"],
    )


def test_an_all_skipped_batch_with_a_partial_file_exits_nonzero(monkeypatch, tmp_path):
    result = _batch(monkeypatch, tmp_path, ["COMPLETED", "PARTIAL"])
    assert result.exit_code != 0, result.output
    assert "partial" in result.output


def test_an_all_skipped_batch_of_completed_files_exits_zero(monkeypatch, tmp_path):
    """The control: same CLI, same real process_batch, only the recorded status differs."""
    result = _batch(monkeypatch, tmp_path, ["COMPLETED", "COMPLETED"])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("status", ["PARTIAL"])
def test_the_partial_skip_matches_single_file(monkeypatch, tmp_path, status):
    """Batch and `socr process` must give the same exit code for the same file (#896)."""
    from socr.pipeline import orchestrator

    monkeypatch.setattr(orchestrator, "_resume_skippable", lambda *a, **k: True)
    out = tmp_path / "out"
    out.mkdir(parents=True)
    pdf = _pdf(tmp_path / "in" / "doc0.pdf")
    _record(out, pdf, status)
    single = CliRunner().invoke(
        cli,
        ["process", str(pdf), "-o", str(out), "--primary", "qwen", "--judge-backend", "heuristic"],
    )
    batch = CliRunner().invoke(
        cli,
        [
            "batch",
            str(pdf.parent),
            "-o",
            str(out),
            "--primary",
            "qwen",
            "--judge-backend",
            "heuristic",
        ],
    )
    assert (single.exit_code != 0) == (batch.exit_code != 0), (single.output, batch.output)


def test_a_dry_run_names_skipped_partials_but_keeps_exit_zero(monkeypatch, tmp_path):
    """PR #900 review: --dry-run previews and must not change the exit code (GH-368),
    but it should still tell the user which skipped files are partial."""
    from socr.pipeline import orchestrator

    monkeypatch.setattr(orchestrator, "_resume_skippable", lambda *a, **k: True)
    src, out = tmp_path / "in", tmp_path / "out"
    out.mkdir(parents=True)
    _record(out, _pdf(src / "doc0.pdf"), "PARTIAL")
    result = CliRunner().invoke(
        cli,
        [
            "batch",
            str(src),
            "-o",
            str(out),
            "--dry-run",
            "--primary",
            "qwen",
            "--judge-backend",
            "heuristic",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "partial" in result.output
