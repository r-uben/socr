"""Tests for GeminiAPIEngine (figure description via Gemini vision API).

Note: This module is NOT used for OCR (that goes through gemini-ocr-cli).
It's only used for describing extracted figures.
"""

import base64
import io
from unittest.mock import MagicMock, patch

import httpx
import pytest
from PIL import Image

from socr.engines.gemini_api import (
    GeminiAPIConfig,
    GeminiAPIEngine,
    LocalFirstFigureEngine,
    OllamaFigureEngine,
    _extract_text,
    _image_to_base64,
)


def _make_image(width: int = 100, height: int = 100) -> Image.Image:
    return Image.new("RGB", (width, height), color=(128, 128, 128))


def _make_engine(api_key: str = "test-key", **kwargs) -> GeminiAPIEngine:
    config = GeminiAPIConfig(api_key=api_key, **kwargs)
    return GeminiAPIEngine(config)


def _gemini_response(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


class TestImageToBase64:
    def test_rgb_image(self):
        img = _make_image()
        result = _image_to_base64(img)
        decoded = base64.b64decode(result)
        assert len(decoded) > 0
        reloaded = Image.open(io.BytesIO(decoded))
        assert reloaded.format == "JPEG"

    def test_rgba_converted_to_rgb(self):
        img = Image.new("RGBA", (50, 50), color=(128, 128, 128, 200))
        result = _image_to_base64(img)
        decoded = base64.b64decode(result)
        reloaded = Image.open(io.BytesIO(decoded))
        assert reloaded.mode == "RGB"


class TestExtractText:
    def test_normal_response(self):
        assert _extract_text(_gemini_response("Hello world")) == "Hello world"

    def test_empty_candidates(self):
        assert _extract_text({"candidates": []}) == ""

    def test_no_candidates_key(self):
        assert _extract_text({}) == ""

    def test_whitespace_stripped(self):
        assert _extract_text(_gemini_response("  text  \n")) == "text"


class TestInitialize:
    def test_no_api_key_returns_false(self):
        with patch.dict("os.environ", {}, clear=True):
            config = GeminiAPIConfig()
        engine = GeminiAPIEngine(config)
        assert engine.initialize() is False

    def test_successful_init(self):
        engine = _make_engine()
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(engine, "_get_client") as mock_client:
            mock_client.return_value.get.return_value = mock_response
            assert engine.initialize() is True

    def test_already_initialized_skips(self):
        engine = _make_engine()
        engine._initialized = True
        assert engine.initialize() is True


class TestDescribeFigure:
    def test_success(self):
        engine = _make_engine()
        engine._initialized = True

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _gemini_response(
            "This bar chart shows revenue over time."
        )

        with patch.object(engine, "_get_client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            result = engine.describe_figure(_make_image(), figure_type="chart")

        assert "bar chart" in result.description
        assert result.figure_type == "chart"

    def test_not_available(self):
        with patch.dict("os.environ", {}, clear=True):
            config = GeminiAPIConfig()
        engine = GeminiAPIEngine(config)
        result = engine.describe_figure(_make_image())
        assert "not available" in result.description


class TestGeminiAPIConfig:
    def test_default_config(self):
        with patch.dict("os.environ", {}, clear=True):
            config = GeminiAPIConfig(api_key="explicit")
            assert config.api_key == "explicit"
            assert config.model == "gemini-3-flash-preview"

    def test_env_gemini_api_key(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "from-env"}, clear=True):
            config = GeminiAPIConfig()
            assert config.api_key == "from-env"

    def test_env_google_api_key_fallback(self):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "google-key"}, clear=True):
            config = GeminiAPIConfig()
            assert config.api_key == "google-key"


class TestClose:
    def test_close(self):
        engine = _make_engine()
        engine._client = MagicMock()
        engine.close()
        assert engine._client is None


# ---------------------------------------------------------------------------
# OllamaFigureEngine
# ---------------------------------------------------------------------------


def _ollama_tags_response(models: list[str]) -> dict:
    """Build a mock /api/tags response containing the given model names."""
    return {"models": [{"name": m} for m in models]}


def _ollama_chat_response(text: str) -> dict:
    """Build a mock /api/chat response."""
    return {"message": {"content": text}}


class TestOllamaFigureEngineIsAvailable:
    def test_model_present_returns_true(self):
        engine = OllamaFigureEngine()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _ollama_tags_response(
            ["qwen3-vl:30b-a3b-instruct", "other:latest"]
        )
        with patch("httpx.get", return_value=mock_resp):
            assert engine.is_available() is True

    def test_model_absent_returns_false(self):
        engine = OllamaFigureEngine()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _ollama_tags_response(["other:latest"])
        with patch("httpx.get", return_value=mock_resp):
            assert engine.is_available() is False

    def test_non_200_returns_false(self):
        engine = OllamaFigureEngine()
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        with patch("httpx.get", return_value=mock_resp):
            assert engine.is_available() is False

    def test_connection_error_returns_false(self):
        engine = OllamaFigureEngine()
        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            assert engine.is_available() is False


class TestOllamaFigureEngineDescribeFigure:
    def test_describe_returns_figureinfo(self):
        """Stub the httpx call; assert FigureInfo is populated and engine is ollama-figure."""
        engine = OllamaFigureEngine()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = _ollama_chat_response(
            "This bar chart shows quarterly revenue growth."
        )
        with patch("httpx.post", return_value=mock_resp):
            result = engine.describe_figure(_make_image(), figure_type="chart")

        assert result.description == "This bar chart shows quarterly revenue growth."
        assert result.engine == "ollama-figure"
        # _detect_figure_type should detect "chart" from the description
        assert result.figure_type == "chart"

    def test_describe_rgba_image_converted(self):
        """RGBA images must be converted to RGB before encoding."""
        engine = OllamaFigureEngine()
        img = Image.new("RGBA", (50, 50), color=(128, 0, 0, 200))
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = _ollama_chat_response("A red square.")
        with patch("httpx.post", return_value=mock_resp) as mock_post:
            result = engine.describe_figure(img)
        assert result.engine == "ollama-figure"
        assert result.description == "A red square."
        # Verify image was sent as base64-encoded JPEG
        call_payload = mock_post.call_args[1]["json"]
        assert len(call_payload["messages"][0]["images"]) == 1

    def test_describe_error_returns_error_figureinfo(self):
        """On httpx failure, describe_figure returns a FigureInfo with error text."""
        engine = OllamaFigureEngine()
        with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
            result = engine.describe_figure(_make_image())
        assert "Ollama figure error" in result.description
        assert result.engine == "ollama-figure"

    def test_name_property(self):
        engine = OllamaFigureEngine()
        assert engine.name == "ollama-figure"


class TestLocalFirstFigureEngine:
    """Tests for LocalFirstFigureEngine per-call fallback logic."""

    def _make_ollama(self, description: str) -> OllamaFigureEngine:
        """Return a stubbed OllamaFigureEngine with a fixed describe_figure result."""
        from socr.core.result import FigureInfo

        eng = OllamaFigureEngine()
        eng.describe_figure = MagicMock(
            return_value=FigureInfo(
                figure_num=0,
                page_num=0,
                figure_type="chart",
                description=description,
                engine="ollama-figure",
            )
        )
        return eng

    def _make_gemini(self, description: str) -> GeminiAPIEngine:
        from socr.core.result import FigureInfo

        eng = GeminiAPIEngine(GeminiAPIConfig(api_key="test-key"))
        eng.describe_figure = MagicMock(
            return_value=FigureInfo(
                figure_num=0,
                page_num=0,
                figure_type="chart",
                description=description,
                engine="gemini-api",
            )
        )
        return eng

    def test_ollama_success_does_not_call_gemini(self):
        """When Ollama returns a non-empty description, Gemini must not be called."""
        ollama = self._make_ollama("A clear bar chart of Q3 results.")
        gemini = self._make_gemini("Gemini description")
        engine = LocalFirstFigureEngine(ollama, gemini)

        result = engine.describe_figure(_make_image())

        assert result.description == "A clear bar chart of Q3 results."
        assert result.engine == "ollama-figure"
        gemini.describe_figure.assert_not_called()

    def test_ollama_empty_falls_back_to_gemini(self):
        """When Ollama returns an empty description, Gemini must be called."""
        ollama = self._make_ollama("")
        gemini = self._make_gemini("Gemini fallback description.")
        engine = LocalFirstFigureEngine(ollama, gemini)

        result = engine.describe_figure(_make_image())

        assert result.description == "Gemini fallback description."
        assert result.engine == "gemini-api"
        gemini.describe_figure.assert_called_once()

    def test_ollama_error_falls_back_to_gemini(self):
        """When Ollama returns an error prefix, Gemini must be called."""
        ollama = self._make_ollama("Ollama figure error: ConnectError: refused")
        gemini = self._make_gemini("Gemini fallback after ollama error.")
        engine = LocalFirstFigureEngine(ollama, gemini)

        result = engine.describe_figure(_make_image())

        assert result.description == "Gemini fallback after ollama error."
        assert result.engine == "gemini-api"
        gemini.describe_figure.assert_called_once()

    def test_ollama_error_no_gemini_returns_ollama_result(self):
        """When Ollama errors and no Gemini is configured, return the Ollama result."""
        ollama = self._make_ollama("Ollama figure error: timeout")
        engine = LocalFirstFigureEngine(ollama, gemini=None)

        result = engine.describe_figure(_make_image())

        assert "Ollama figure error" in result.description
        assert result.engine == "ollama-figure"

    def test_name_property(self):
        engine = LocalFirstFigureEngine(OllamaFigureEngine(), gemini=None)
        assert engine.name == "local-first-figure"

    def test_close_calls_both(self):
        """close() must delegate to both sub-engines."""
        ollama = MagicMock()
        gemini = MagicMock()
        engine = LocalFirstFigureEngine(ollama, gemini)
        engine.close()
        ollama.close.assert_called_once()
        gemini.close.assert_called_once()


class TestGetVisionEngineFallback:
    """Test that the pipeline's _get_vision_engine uses correct local-first routing."""

    def _make_orch(self):
        from socr.pipeline.orchestrator import UnifiedPipeline

        class _FakeConfig:
            quiet = True
            gemini_model = "gemini-3-flash-preview"

        orch = object.__new__(UnifiedPipeline)
        orch.config = _FakeConfig()
        return orch

    def test_falls_back_to_gemini_when_ollama_unavailable(self):
        """When is_available() returns False, _get_vision_engine must return a
        GeminiAPIEngine (not None and not LocalFirstFigureEngine)."""
        orch = self._make_orch()

        with patch.object(OllamaFigureEngine, "is_available", return_value=False):
            with patch.object(GeminiAPIEngine, "initialize", return_value=True):
                with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=False):
                    result = orch._get_vision_engine()

        assert result is not None
        assert isinstance(result, GeminiAPIEngine)

    def test_returns_local_first_engine_when_ollama_available(self):
        """When Ollama is available, _get_vision_engine returns LocalFirstFigureEngine."""
        orch = self._make_orch()

        with patch.object(OllamaFigureEngine, "is_available", return_value=True):
            with patch.dict("os.environ", {}, clear=False):
                result = orch._get_vision_engine()

        assert result is not None
        assert isinstance(result, LocalFirstFigureEngine)

    def test_local_first_engine_includes_gemini_when_key_present(self):
        """When Ollama is available AND GEMINI_API_KEY is set, the LocalFirstFigureEngine
        should have a Gemini fallback engine attached."""
        orch = self._make_orch()

        with patch.object(OllamaFigureEngine, "is_available", return_value=True):
            with patch.object(GeminiAPIEngine, "initialize", return_value=True):
                with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=False):
                    result = orch._get_vision_engine()

        assert isinstance(result, LocalFirstFigureEngine)
        assert result._gemini is not None
        assert isinstance(result._gemini, GeminiAPIEngine)

    def test_returns_none_when_both_unavailable(self):
        """Returns None when Ollama is down and no API key is set."""
        orch = self._make_orch()

        with patch.object(OllamaFigureEngine, "is_available", return_value=False):
            with patch.dict("os.environ", {}, clear=True):
                result = orch._get_vision_engine()

        assert result is None
