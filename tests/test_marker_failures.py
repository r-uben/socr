"""Marker rung failure modes (#1243)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from socr.core.config import PipelineConfig
from socr.core.result import FailureMode, PageStatus
from socr.engines.marker import MarkerEngine


def test_marker_exit_zero_without_output_is_cli_error(tmp_path: Path) -> None:
    engine = MarkerEngine()
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    with patch("socr.engines.marker.subprocess.run") as run:
        run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        with patch("socr.core.pdf.open_pdf") as open_pdf:
            doc = MagicMock()
            open_pdf.return_value.__enter__.return_value = doc
            with patch("fitz.open") as fitz_open:
                single = MagicMock()
                fitz_open.return_value = single
                outs = engine.process_pages(pdf, [1], PipelineConfig(), dpi=200)

    assert len(outs) == 1
    assert outs[0].status is PageStatus.ERROR
    assert outs[0].failure_mode is FailureMode.CLI_ERROR
    assert (
        "exit 0" in (outs[0].error or "").lower() or "no markdown" in (outs[0].error or "").lower()
    )


def test_marker_missing_cli_logs_and_is_unavailable(caplog: pytest.LogCaptureFixture) -> None:
    engine = MarkerEngine()
    with patch.object(MarkerEngine, "_warned_missing_cli", False):
        with patch(
            "socr.engines.base.subprocess.run",
            side_effect=FileNotFoundError("marker-ocr"),
        ):
            assert engine.is_available() is False
    assert any("marker-ocr CLI not found" in r.message for r in caplog.records)
