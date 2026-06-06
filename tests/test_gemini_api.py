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
