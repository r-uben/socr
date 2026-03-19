"""Per-page Gemini OCR engine via HTTP API.

Calls the Gemini generateContent API directly per page (render page as image,
send to API). Eliminates truncation entirely since each page is an independent
API call. Same pattern as DeepSeekVLLMEngine.
"""

import base64
import io
import os
import time
from dataclasses import dataclass

import httpx
from PIL import Image

from socr.core.result import FailureMode, FigureInfo, PageOutput
from socr.engines.base import BaseHTTPEngine


@dataclass
class GeminiAPIConfig:
    """Configuration for the Gemini API engine."""

    api_key: str = ""
    model: str = "gemini-3-flash-preview"
    timeout: float = 120.0
    max_tokens: int = 8192
    temperature: float = 0.1
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get(
                "GOOGLE_API_KEY", ""
            )


_OCR_PROMPT = (
    "Extract all text from this page. Preserve formatting, equations, "
    "tables, and structure. Output as clean markdown."
)


class GeminiAPIEngine(BaseHTTPEngine):
    """Per-page OCR via Gemini API directly (no CLI wrapper).

    Renders each PDF page as an image, sends to Gemini's
    generateContent API with an OCR prompt. Eliminates truncation
    since each page is processed independently.
    """

    def __init__(self, config: GeminiAPIConfig | None = None) -> None:
        super().__init__()
        self.config = config or GeminiAPIConfig()
        self._client: httpx.Client | None = None

    @property
    def name(self) -> str:
        return "gemini-api"

    @property
    def model_version(self) -> str:
        return self.config.model

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=httpx.Timeout(self.config.timeout, connect=30.0),
            )
        return self._client

    def _build_url(self) -> str:
        """Build the generateContent endpoint URL with API key."""
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
            # List models to verify the API key works
            url = f"{self.config.base_url}/models?key={self.config.api_key}"
            response = client.get(url, timeout=15.0)
            if response.status_code == 200:
                self._initialized = True
                return True
            return False
        except Exception:
            return False

    def process_image(self, image: Image.Image, page_num: int = 1) -> PageOutput:
        """Send a single page image to Gemini API for OCR."""
        if not self._initialized and not self.initialize():
            return self._create_error_result(
                page_num,
                "Gemini API not available (missing or invalid API key)",
                failure_mode=FailureMode.MODEL_UNAVAILABLE,
            )

        start_time = time.time()
        try:
            img_base64 = image_to_base64(image)

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
                            {"text": _OCR_PROMPT},
                        ]
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                },
            }

            client = self._get_client()
            response = client.post(self._build_url(), json=payload)

            processing_time = time.time() - start_time

            if response.status_code != 200:
                return self._create_error_result(
                    page_num,
                    f"Gemini API error ({response.status_code}): "
                    f"{response.text[:200]}",
                    failure_mode=FailureMode.API_ERROR,
                )

            text = _extract_text(response.json())
            if not text or len(text) < 10:
                return self._create_error_result(
                    page_num,
                    "OCR produced empty or minimal output",
                    failure_mode=FailureMode.EMPTY_OUTPUT,
                )

            return self._create_success_result(
                page_num=page_num,
                text=text,
                engine=self.name,
                confidence=0.85,
                processing_time=processing_time,
            )

        except httpx.TimeoutException:
            return self._create_error_result(
                page_num,
                f"Timeout after {self.config.timeout}s",
                failure_mode=FailureMode.TIMEOUT,
            )
        except Exception as e:
            return self._create_error_result(
                page_num,
                f"Gemini API error: {type(e).__name__}: {e}",
                failure_mode=FailureMode.API_ERROR,
            )

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
                description="Gemini API not available (missing or invalid API key)",
            )

        try:
            img_base64 = image_to_base64(image)
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

            description = (
                _extract_text(response.json())
                or "Unable to generate description"
            )
            detected_type = _detect_figure_type(description, figure_type)

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


def _build_figure_prompt(figure_type: str, context: str) -> str:
    """Build a prompt for describing a figure image."""
    base = (
        "Describe this figure in detail. What does the chart, graph, table, "
        "or diagram show? Explain the axes, data, key findings, and any "
        "notable patterns or trends. Be specific about numbers, labels, "
        "and relationships shown."
    )
    if figure_type and figure_type != "unknown":
        base = f"This appears to be a {figure_type}. {base}"
    if context:
        base += f"\n\nContext from surrounding text: {context[:500]}"
    return base


def _detect_figure_type(description: str, default: str) -> str:
    """Infer figure type from the description text."""
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


def image_to_base64(image: Image.Image) -> str:
    """Convert a PIL Image to a base64-encoded JPEG string."""
    buffered = io.BytesIO()
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _extract_text(data: dict) -> str:
    """Extract text from a Gemini generateContent response."""
    candidates = data.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        return ""
    return parts[0].get("text", "").strip()
