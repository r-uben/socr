"""GH-1086: ``--gemini-model`` must reach the GeminiAPI figure engine."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import fitz
from click.testing import CliRunner

from socr.cli import cli
from socr.core.config import EngineType
from socr.core.result import DocumentStatus, EngineResult
from socr.engines.gemini_api import GeminiAPIEngine, LocalFirstFigureEngine, OllamaFigureEngine

_CUSTOM_MODEL = "gemini-custom-gh1086-fixture"


def _one_page_pdf(tmp_path: Path) -> Path:
    pdf = tmp_path / "one.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "hello")
    doc.save(pdf)
    doc.close()
    return pdf


def test_process_gemini_model_flag_wires_figure_engine_model(tmp_path, monkeypatch):
    """``socr process --gemini-model`` must configure the figure Gemini fallback."""
    pdf = _one_page_pdf(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    captured_models: list[str] = []

    def _recording_init(self, config=None):
        from socr.engines.gemini_api import GeminiAPIConfig

        self.config = config or GeminiAPIConfig()
        self._client = None
        self._initialized = False
        captured_models.append(self.config.model)

    def _stub_process(self, pdf_path, output_dir):
        engine = self._get_vision_engine()
        assert engine is not None
        if isinstance(engine, LocalFirstFigureEngine):
            assert engine._gemini is not None
            assert engine._gemini.config.model == _CUSTOM_MODEL
        else:
            assert isinstance(engine, GeminiAPIEngine)
            assert engine.config.model == _CUSTOM_MODEL
        return EngineResult(
            document_path=pdf_path,
            engine="test",
            status=DocumentStatus.SKIPPED,
        )

    with (
        patch("socr.cli._preflight_gate"),
        patch(
            "socr.engines.registry.resolve_auto_engine",
            return_value=EngineType.QWEN,
        ),
        patch.object(GeminiAPIEngine, "__init__", _recording_init),
        patch.object(GeminiAPIEngine, "initialize", return_value=True),
        patch.object(OllamaFigureEngine, "is_available", return_value=True),
        patch(
            "socr.pipeline.orchestrator.UnifiedPipeline.process",
            autospec=True,
            side_effect=_stub_process,
        ),
    ):
        result = CliRunner().invoke(
            cli,
            [
                "process",
                str(pdf),
                "-o",
                str(out),
                "--describe-figures",
                "--gemini-model",
                _CUSTOM_MODEL,
                "--quiet",
                "--no-table-judge-ladder",
                "--primary",
                "qwen",
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured_models == [_CUSTOM_MODEL]
