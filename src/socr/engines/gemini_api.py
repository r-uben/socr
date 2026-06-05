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

            description = _extract_text(response.json()) or "Unable to generate description"
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
