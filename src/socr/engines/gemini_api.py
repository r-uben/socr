"""Gemini vision API for figure descriptions.

NOT used for OCR — OCR goes through gemini-ocr-cli.
This module is only used by the orchestrator's figure pass to describe
extracted figure images via the Gemini generateContent API.
"""

import base64
import io
import os
from dataclasses import dataclass

import httpx
from PIL import Image

from socr.core.result import FigureInfo
from socr.engines._figure_prompt import (
    CAPTION_MARKER as _CAPTION_MARKER,  # noqa: F401  re-exported for tests
)
from socr.engines._figure_prompt import (
    build_figure_prompt as _build_figure_prompt,
)
from socr.engines._figure_prompt import (
    wrap_caption as _wrap_caption,
)


@dataclass
class GeminiAPIConfig:
    """Configuration for the Gemini vision API."""

    api_key: str = ""
    model: str = "gemini-3-flash-preview"
    timeout: float = 120.0
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get(
                "GOOGLE_API_KEY", ""
            )


class GeminiAPIEngine:
    """Gemini vision API for figure descriptions only.

    Used by the orchestrator's figure pass. NOT an OCR engine.
    """

    def __init__(self, config: GeminiAPIConfig | None = None) -> None:
        self.config = config or GeminiAPIConfig()
        self._client: httpx.Client | None = None
        self._initialized: bool = False

    @property
    def name(self) -> str:
        return "gemini-api"

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=httpx.Timeout(self.config.timeout, connect=30.0),
            )
        return self._client

    def _build_url(self) -> str:
        return (
            f"{self.config.base_url}/models/{self.config.model}"
            f":generateContent?key={self.config.api_key}"
        )

    def initialize(self) -> bool:
        """Check API key availability and basic connectivity."""
        if self._initialized:
            return True
        if not self.config.api_key:
            return False
        try:
            client = self._get_client()
            url = f"{self.config.base_url}/models?key={self.config.api_key}"
            response = client.get(url, timeout=15.0)
            if response.status_code == 200:
                self._initialized = True
                return True
            return False
        except Exception:
            return False

    def describe_figure(
        self,
        image: Image.Image,
        figure_type: str = "unknown",
        context: str = "",
    ) -> FigureInfo:
        """Describe a figure image using the Gemini vision API."""
        if not self._initialized and not self.initialize():
            return FigureInfo(
                figure_num=0,
                page_num=0,
                figure_type=figure_type,
                description="Gemini API not available",
            )

        try:
            img_base64 = _image_to_base64(image)
            prompt = _build_figure_prompt(figure_type, context)

            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": img_base64,
                                }
                            },
                            {"text": prompt},
                        ]
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": 1024,
                    "temperature": 0.1,
                },
            }

            client = self._get_client()
            response = client.post(self._build_url(), json=payload)

            if response.status_code != 200:
                return FigureInfo(
                    figure_num=0,
                    page_num=0,
                    figure_type=figure_type,
                    description=f"Gemini API error ({response.status_code})",
                    engine=self.name,
                )

            raw = _extract_text(response.json()) or "Unable to generate description"
            detected_type = _detect_figure_type(raw, figure_type)
            # Only wrap genuine model output — not the fallback error string.
            description = _wrap_caption(raw) if raw != "Unable to generate description" else raw

            return FigureInfo(
                figure_num=0,
                page_num=0,
                figure_type=detected_type,
                description=description,
                engine=self.name,
            )

        except Exception as e:
            return FigureInfo(
                figure_num=0,
                page_num=0,
                figure_type=figure_type,
                description=f"Gemini API error: {type(e).__name__}: {e}",
                engine=self.name,
            )

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def __del__(self) -> None:
        self.close()


class OllamaFigureEngine:
    """Local-first figure description engine using Ollama.

    Uses qwen3-vl:30b-a3b-instruct (non-thinking MoE) via Ollama's /api/chat
    endpoint. Intended as the primary figure-description tier, with GeminiAPIEngine
    as fallback on empty result or error.
    """

    def __init__(
        self,
        model: str = "qwen3-vl:30b-a3b-instruct",
        host: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")

    @property
    def name(self) -> str:
        return "ollama-figure"

    def is_available(self) -> bool:
        """Return True if Ollama is reachable and this model is in its tag list."""
        try:
            resp = httpx.get(f"{self.host}/api/tags", timeout=3.0)
            if resp.status_code != 200:
                return False
            data = resp.json()
            models = data.get("models", [])
            return any(m.get("name", "") == self.model for m in models)
        except Exception:
            return False

    def describe_figure(
        self,
        image: Image.Image,
        figure_type: str = "unknown",
        context: str = "",
    ) -> FigureInfo:
        """Describe a figure image using Ollama's /api/chat endpoint."""
        try:
            buf = io.BytesIO()
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            image.save(buf, format="JPEG", quality=90)
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            prompt = _build_figure_prompt(figure_type, context)
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [img_b64],
                    }
                ],
                "stream": False,
            }
            resp = httpx.post(f"{self.host}/api/chat", json=payload, timeout=120.0)
            resp.raise_for_status()
            raw = resp.json()["message"]["content"].strip()

            detected_type = _detect_figure_type(raw, figure_type)
            return FigureInfo(
                figure_num=0,
                page_num=0,
                figure_type=detected_type,
                description=_wrap_caption(raw),
                engine=self.name,
            )
        except Exception as e:
            return FigureInfo(
                figure_num=0,
                page_num=0,
                figure_type=figure_type,
                description=f"Ollama figure error: {type(e).__name__}: {e}",
                engine=self.name,
            )

    def close(self) -> None:
        pass  # stateless HTTP — nothing to close


class LocalFirstFigureEngine:
    """Composite engine: try Ollama first, fall back to Gemini per-call.

    Used by the orchestrator's _get_vision_engine() when Ollama is available.
    On each describe_figure call:
    1. Call OllamaFigureEngine.
    2. If the result has a non-empty description that is not an error prefix,
       return it.
    3. Otherwise fall back to GeminiAPIEngine (if provided).
    """

    _OLLAMA_ERROR_PREFIX = "Ollama figure error:"

    def __init__(
        self,
        ollama: "OllamaFigureEngine",
        gemini: "GeminiAPIEngine | None" = None,
    ) -> None:
        self._ollama = ollama
        self._gemini = gemini

    @property
    def name(self) -> str:
        return "local-first-figure"

    def describe_figure(
        self,
        image: Image.Image,
        figure_type: str = "unknown",
        context: str = "",
    ) -> FigureInfo:
        """Describe via Ollama; fall back to Gemini on empty or error."""
        info = self._ollama.describe_figure(image, figure_type=figure_type, context=context)
        if info.description and not info.description.startswith(self._OLLAMA_ERROR_PREFIX):
            return info

        # Ollama returned empty or an error — try Gemini
        if self._gemini is not None:
            return self._gemini.describe_figure(image, figure_type=figure_type, context=context)

        # No Gemini fallback available — return what Ollama gave us
        return info

    def close(self) -> None:
        self._ollama.close()
        if self._gemini is not None:
            self._gemini.close()


def _detect_figure_type(description: str, default: str) -> str:
    desc_lower = description.lower()
    for fig_type, keywords in {
        "chart": ["bar chart", "pie chart", "chart"],
        "graph": ["line graph", "scatter plot", "graph", "plot"],
        "table": ["table", "tabular"],
        "diagram": ["diagram", "flowchart", "schematic", "architecture"],
        "map": ["map", "geographic", "spatial"],
        "equation": ["equation", "formula", "mathematical"],
    }.items():
        if any(kw in desc_lower for kw in keywords):
            return fig_type
    return default


def _image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _extract_text(data: dict) -> str:
    candidates = data.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        return ""
    return parts[0].get("text", "").strip()
