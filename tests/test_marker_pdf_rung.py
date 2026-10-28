"""Marker rung must target PDF input with --force-ocr (#1243)."""

from __future__ import annotations

from pathlib import Path

from socr.core.config import PipelineConfig
from socr.engines.marker import MarkerEngine


def test_marker_build_command_includes_force_ocr(tmp_path: Path) -> None:
    engine = MarkerEngine()
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    cmd = engine._build_command(pdf, tmp_path / "out", PipelineConfig())
    assert "--force-ocr" in cmd
    assert str(pdf) in cmd
    assert "page_" not in " ".join(cmd)
