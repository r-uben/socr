"""vLLM vision engine adapter for figure description (HPC mode).

Uses vLLM's OpenAI-compatible API to run vision models like Qwen2-VL or InternVL2
for figure/chart description. This is a figures-only engine (no OCR).
"""

import base64
import io
import os
import time
from dataclasses import dataclass

import httpx
from PIL import Image

from socr.core.result import FigureInfo, PageResult, PageStatus
from socr.engines.base import BaseHTTPEngine


@dataclass
class VLLMConfig:
    """Configuration for vLLM vision engine."""

    base_url: str = ""
    api_key: str = ""
    model: str = "Qwen/Qwen2-VL-7B-Instruct"
    timeout: float = 120.0
    max_tokens: int = 1024
    temperature: float = 0.1

    def __post_init__(self) -> None:
        if not self.base_url:
            self.base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        if not self.api_key:
            self.api_key = os.environ.get("VLLM_API_KEY", "token-abc123")


class VLLMEngine(BaseHTTPEngine):
    """Adapter for vLLM vision models via OpenAI-compatible API.

    Specialized for figure description — does not support OCR.
    """

    def __init__(self, config: VLLMConfig | None = None) -> None:
        super().__init__()
        self.config = config or VLLMConfig()
        self._client: httpx.Client | None = None

    @property
    def name(self) -> str:
        return "vllm"

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.config.base_url,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(self.config.timeout, connect=30.0),
            )
        return self._client

    def initialize(self) -> bool:
        if self._initialized:
            return True
        try:
            client = self._get_client()
            response = client.get("/models")
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                model_ids = [m.get("id", "") for m in models]
                if self.config.model in model_ids or any(
                    self.config.model.lower() in m.lower() for m in model_ids
                ):
                    self._initialized = True
                    return True
                if models:
                    self._initialized = True
                    return True
            return False
        except Exception:
            return False

    def process_image(self, image: Image.Image, page_num: int = 1) -> PageResult:
        """Not supported — vLLM engine is for figure description only."""
        return self._create_error_result(
            page_num,
            "vLLM engine is for figure description only. Use deepseek/nougat/gemini for OCR.",
        )

    def describe_figure(
        self, image: Image.Image, figure_type: str = "unknown", context: str = ""
    ) -> FigureInfo:
        if not self._initialized and not self.initialize():
            return FigureInfo(
                figure_num=0, page_num=0, figure_type=figure_type,
                description=f"vLLM server not available at {self.config.base_url}",
            )

        start_time = time.time()
        try:
            buffered = io.BytesIO()
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            image.save(buffered, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            prompt = self._build_figure_prompt(figure_type, context)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            client = self._get_client()
            response = client.post(
                "/chat/completions",
                json={
                    "model": self.config.model,
                    "messages": messages,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                },
            )

            if response.status_code != 200:
                return FigureInfo(
                    figure_num=0, page_num=0, figure_type=figure_type,
                    description=f"vLLM API error ({response.status_code})", engine=self.name,
                )

            data = response.json()
            choices = data.get("choices", [])
            description = choices[0].get("message", {}).get("content", "").strip() if choices else ""
            if not description or len(description) < 10:
                description = "Unable to generate meaningful description"

            detected_type = self._detect_figure_type(description, figure_type)

            return FigureInfo(
                figure_num=0, page_num=0, figure_type=detected_type,
                description=description, engine=self.name,
            )

        except httpx.TimeoutException:
            return FigureInfo(
                figure_num=0, page_num=0, figure_type=figure_type,
                description=f"vLLM request timed out after {self.config.timeout}s", engine=self.name,
            )
        except Exception as e:
            return FigureInfo(
                figure_num=0, page_num=0, figure_type=figure_type,
                description=f"vLLM error: {type(e).__name__}: {e}", engine=self.name,
            )

    @staticmethod
    def _build_figure_prompt(figure_type: str, context: str) -> str:
        base = (
            "Describe this figure in detail. What does the chart, graph, table, or diagram show? "
            "Explain the axes, data, key findings, and any notable patterns or trends. "
            "Be specific about numbers, labels, and relationships shown."
        )
        if figure_type and figure_type != "unknown":
            base = f"This appears to be a {figure_type}. {base}"
        if context:
            base += f"\n\nContext from surrounding text: {context[:500]}"
        return base

    @staticmethod
    def _detect_figure_type(description: str, default: str) -> str:
        desc_lower = description.lower()
        for fig_type, keywords in {
            "chart": ["bar chart", "pie chart", "chart"],
            "graph": ["line graph", "scatter plot", "graph", "plot"],
            "table": ["table", "tabular"],
            "diagram": ["diagram", "flowchart", "schematic", "architecture"],
            "map": ["map", "geographic", "spatial"],
        }.items():
            if any(kw in desc_lower for kw in keywords):
                return fig_type
        return default

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def __del__(self) -> None:
        self.close()
