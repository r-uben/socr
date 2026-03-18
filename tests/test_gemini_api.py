"""Tests for GeminiAPIEngine (per-page Gemini HTTP adapter)."""

import base64
import io
from unittest.mock import MagicMock, patch

import httpx
import pytest
from PIL import Image

from socr.core.result import FailureMode, PageStatus
from socr.engines.gemini_api import (
    GeminiAPIConfig,
    GeminiAPIEngine,
    _extract_text,
    image_to_base64,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image(width: int = 100, height: int = 100) -> Image.Image:
    """Create a simple RGB test image."""
    return Image.new("RGB", (width, height), color=(128, 128, 128))


def _make_rgba_image(width: int = 100, height: int = 100) -> Image.Image:
    """Create an RGBA test image."""
    return Image.new("RGBA", (width, height), color=(128, 128, 128, 200))


def _make_engine(api_key: str = "test-key", **kwargs) -> GeminiAPIEngine:
    """Create a GeminiAPIEngine with a test API key."""
    config = GeminiAPIConfig(api_key=api_key, **kwargs)
    return GeminiAPIEngine(config)


def _gemini_response(text: str) -> dict:
    """Build a Gemini generateContent response body."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": text}]
                }
            }
        ]
    }


# ---------------------------------------------------------------------------
# image_to_base64
# ---------------------------------------------------------------------------


class TestImageToBase64:
    def test_rgb_image(self):
        img = _make_image()
        result = image_to_base64(img)
        # Should be a valid base64 string
        decoded = base64.b64decode(result)
        assert len(decoded) > 0
        # Should be JPEG
        reloaded = Image.open(io.BytesIO(decoded))
        assert reloaded.format == "JPEG"

    def test_rgba_converted_to_rgb(self):
        img = _make_rgba_image()
        result = image_to_base64(img)
        decoded = base64.b64decode(result)
        reloaded = Image.open(io.BytesIO(decoded))
        assert reloaded.mode == "RGB"

    def test_palette_image_converted(self):
        img = Image.new("P", (50, 50))
        result = image_to_base64(img)
        decoded = base64.b64decode(result)
        reloaded = Image.open(io.BytesIO(decoded))
        assert reloaded.mode == "RGB"


# ---------------------------------------------------------------------------
# _extract_text
# ---------------------------------------------------------------------------


class TestExtractText:
    def test_normal_response(self):
        data = _gemini_response("Hello world")
        assert _extract_text(data) == "Hello world"

    def test_empty_candidates(self):
        assert _extract_text({"candidates": []}) == ""

    def test_no_candidates_key(self):
        assert _extract_text({}) == ""

    def test_empty_parts(self):
        data = {"candidates": [{"content": {"parts": []}}]}
        assert _extract_text(data) == ""

    def test_missing_content(self):
        data = {"candidates": [{}]}
        assert _extract_text(data) == ""

    def test_whitespace_stripped(self):
        data = _gemini_response("  some text  \n")
        assert _extract_text(data) == "some text"


# ---------------------------------------------------------------------------
# GeminiAPIEngine.initialize
# ---------------------------------------------------------------------------


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
        mock_response.json.return_value = {"models": []}

        with patch.object(engine, "_get_client") as mock_client:
            mock_client.return_value.get.return_value = mock_response
            assert engine.initialize() is True
            assert engine._initialized is True

    def test_already_initialized_skips(self):
        engine = _make_engine()
        engine._initialized = True
        # Should return True without making any HTTP calls
        assert engine.initialize() is True

    def test_api_error_returns_false(self):
        engine = _make_engine()
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch.object(engine, "_get_client") as mock_client:
            mock_client.return_value.get.return_value = mock_response
            assert engine.initialize() is False

    def test_connection_error_returns_false(self):
        engine = _make_engine()

        with patch.object(engine, "_get_client") as mock_client:
            mock_client.return_value.get.side_effect = httpx.ConnectError(
                "Connection refused"
            )
            assert engine.initialize() is False


# ---------------------------------------------------------------------------
# GeminiAPIEngine.process_image
# ---------------------------------------------------------------------------


class TestProcessImage:
    def test_success(self):
        engine = _make_engine()
        engine._initialized = True

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _gemini_response(
            "This is the extracted text from the page with enough words."
        )

        with patch.object(engine, "_get_client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            result = engine.process_image(_make_image(), page_num=3)

        assert result.status == PageStatus.SUCCESS
        assert result.page_num == 3
        assert "extracted text" in result.text
        assert result.engine == "gemini-api"
        assert result.processing_time > 0
        assert result.confidence == 0.85

    def test_not_initialized_auto_inits(self):
        engine = _make_engine()

        # Mock both initialize and the actual API call
        mock_init_response = MagicMock()
        mock_init_response.status_code = 200
        mock_init_response.json.return_value = {"models": []}

        mock_ocr_response = MagicMock()
        mock_ocr_response.status_code = 200
        mock_ocr_response.json.return_value = _gemini_response(
            "OCR text from auto-init with enough words to pass."
        )

        with patch.object(engine, "_get_client") as mock_client:
            mock_client.return_value.get.return_value = mock_init_response
            mock_client.return_value.post.return_value = mock_ocr_response
            result = engine.process_image(_make_image(), page_num=1)

        assert result.status == PageStatus.SUCCESS

    def test_not_available_returns_error(self):
        with patch.dict("os.environ", {}, clear=True):
            config = GeminiAPIConfig()
        engine = GeminiAPIEngine(config)
        result = engine.process_image(_make_image(), page_num=1)

        assert result.status == PageStatus.ERROR
        assert result.failure_mode == FailureMode.MODEL_UNAVAILABLE
        assert "not available" in result.error

    def test_api_error_status(self):
        engine = _make_engine()
        engine._initialized = True

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch.object(engine, "_get_client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            result = engine.process_image(_make_image(), page_num=1)

        assert result.status == PageStatus.ERROR
        assert result.failure_mode == FailureMode.API_ERROR
        assert "500" in result.error

    def test_empty_response(self):
        engine = _make_engine()
        engine._initialized = True

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _gemini_response("")

        with patch.object(engine, "_get_client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            result = engine.process_image(_make_image(), page_num=1)

        assert result.status == PageStatus.ERROR
        assert result.failure_mode == FailureMode.EMPTY_OUTPUT

    def test_minimal_response_below_threshold(self):
        engine = _make_engine()
        engine._initialized = True

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _gemini_response("Short")

        with patch.object(engine, "_get_client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            result = engine.process_image(_make_image(), page_num=1)

        assert result.status == PageStatus.ERROR
        assert result.failure_mode == FailureMode.EMPTY_OUTPUT

    def test_timeout_handling(self):
        engine = _make_engine()
        engine._initialized = True

        with patch.object(engine, "_get_client") as mock_client:
            mock_client.return_value.post.side_effect = httpx.ReadTimeout(
                "Read timed out"
            )
            result = engine.process_image(_make_image(), page_num=1)

        assert result.status == PageStatus.ERROR
        assert result.failure_mode == FailureMode.TIMEOUT

    def test_generic_exception(self):
        engine = _make_engine()
        engine._initialized = True

        with patch.object(engine, "_get_client") as mock_client:
            mock_client.return_value.post.side_effect = RuntimeError("boom")
            result = engine.process_image(_make_image(), page_num=1)

        assert result.status == PageStatus.ERROR
        assert result.failure_mode == FailureMode.API_ERROR
        assert "RuntimeError" in result.error

    def test_request_payload_structure(self):
        """Verify the API request body has the correct structure."""
        engine = _make_engine()
        engine._initialized = True

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _gemini_response(
            "Extracted text with enough content for validation."
        )

        with patch.object(engine, "_get_client") as mock_client:
            mock_client.return_value.post.return_value = mock_response
            engine.process_image(_make_image(), page_num=1)

            call_args = mock_client.return_value.post.call_args
            url = call_args[0][0]
            payload = call_args[1]["json"]

        # URL should contain model and API key
        assert "gemini-3-flash-preview" in url
        assert "key=test-key" in url

        # Payload structure
        assert "contents" in payload
        parts = payload["contents"][0]["parts"]
        assert len(parts) == 2
        assert "inline_data" in parts[0]
        assert parts[0]["inline_data"]["mime_type"] == "image/jpeg"
        assert "text" in parts[1]

        # Generation config
        assert "generationConfig" in payload
        assert payload["generationConfig"]["maxOutputTokens"] == 8192


# ---------------------------------------------------------------------------
# GeminiAPIEngine properties
# ---------------------------------------------------------------------------


class TestEngineProperties:
    def test_name(self):
        engine = _make_engine()
        assert engine.name == "gemini-api"

    def test_model_version(self):
        engine = _make_engine(model="gemini-2.0-flash")
        assert engine.model_version == "gemini-2.0-flash"

    def test_is_available_delegates_to_initialize(self):
        engine = _make_engine()
        engine._initialized = True
        assert engine.is_available() is True

    def test_close(self):
        engine = _make_engine()
        engine._client = MagicMock()
        engine.close()
        assert engine._client is None


# ---------------------------------------------------------------------------
# GeminiAPIConfig
# ---------------------------------------------------------------------------


class TestGeminiAPIConfig:
    def test_default_config(self):
        with patch.dict("os.environ", {}, clear=True):
            config = GeminiAPIConfig(api_key="explicit")
            assert config.api_key == "explicit"
            assert config.model == "gemini-3-flash-preview"
            assert config.timeout == 120.0

    def test_env_gemini_api_key(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "from-env"}, clear=True):
            config = GeminiAPIConfig()
            assert config.api_key == "from-env"

    def test_env_google_api_key_fallback(self):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "google-key"}, clear=True):
            config = GeminiAPIConfig()
            assert config.api_key == "google-key"

    def test_gemini_key_takes_precedence(self):
        with patch.dict(
            "os.environ",
            {"GEMINI_API_KEY": "gemini", "GOOGLE_API_KEY": "google"},
            clear=True,
        ):
            config = GeminiAPIConfig()
            assert config.api_key == "gemini"


# ---------------------------------------------------------------------------
# Per-page PageOutput construction
# ---------------------------------------------------------------------------


class TestPerPageOutput:
    def test_multiple_pages(self):
        """Simulate processing multiple pages and verify PageOutput list."""
        engine = _make_engine()
        engine._initialized = True

        pages = []
        for page_num in range(1, 4):
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = _gemini_response(
                f"Content for page {page_num} with enough words to pass the minimum threshold."
            )

            with patch.object(engine, "_get_client") as mock_client:
                mock_client.return_value.post.return_value = mock_response
                result = engine.process_image(_make_image(), page_num=page_num)
                pages.append(result)

        assert len(pages) == 3
        for i, page in enumerate(pages, start=1):
            assert page.page_num == i
            assert page.status == PageStatus.SUCCESS
            assert f"page {i}" in page.text
