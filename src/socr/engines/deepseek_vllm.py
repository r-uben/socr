"""DeepSeek-OCR via vLLM engine adapter for HPC mode.

Uses vLLM's OpenAI-compatible API to run DeepSeek-OCR model for text extraction.
This is a full OCR engine (unlike VLLMEngine which is figures-only).
"""

import base64
import io
import os
import re
import time
from dataclasses import dataclass

import httpx
from PIL import Image

from socr.core.result import FigureInfo, PageResult, PageStatus
from socr.engines.base import BaseHTTPEngine


@dataclass
class DeepSeekVLLMConfig:
    """Configuration for DeepSeek-vLLM engine."""

    base_url: str = ""
    api_key: str = ""
    model: str = "deepseek-ai/DeepSeek-OCR"
    timeout: float = 120.0
    max_tokens: int = 4096
    temperature: float = 0.1
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.05

    def __post_init__(self) -> None:
        if not self.base_url:
            self.base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        if not self.api_key:
            self.api_key = os.environ.get("VLLM_API_KEY", "")


class DeepSeekVLLMEngine(BaseHTTPEngine):
    """Adapter for DeepSeek-OCR via vLLM OpenAI-compatible API.

    Performs full OCR using the DeepSeek-OCR model served via vLLM.
    Also supports figure description.
    """

    def __init__(self, config: DeepSeekVLLMConfig | None = None) -> None:
        super().__init__()
        self.config = config or DeepSeekVLLMConfig()
        self._client: httpx.Client | None = None

    @property
    def name(self) -> str:
        return "deepseek-vllm"

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            self._client = httpx.Client(
                base_url=self.config.base_url,
                headers=headers,
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
        if not self._initialized and not self.initialize():
            return self._create_error_result(
                page_num, f"vLLM server not available at {self.config.base_url}"
            )

        start_time = time.time()
        try:
            img_base64 = self._image_to_base64(image)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                        {"type": "text", "text": self._build_ocr_prompt()},
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
                    "frequency_penalty": self.config.frequency_penalty,
                    "repetition_penalty": self.config.repetition_penalty,
                },
            )

            processing_time = time.time() - start_time

            if response.status_code != 200:
                return self._create_error_result(
                    page_num, f"vLLM API error ({response.status_code}): {response.text[:200]}"
                )

            text = self._extract_text(response.json())
            if not text or len(text) < 10:
                return self._create_error_result(page_num, "OCR produced empty or minimal output")

            return self._create_success_result(
                page_num=page_num, text=text, confidence=0.85, processing_time=processing_time
            )

        except httpx.TimeoutException:
            return self._create_error_result(page_num, f"Timeout after {self.config.timeout}s")
        except Exception as e:
            return self._create_error_result(page_num, f"vLLM error: {type(e).__name__}: {e}")

    def describe_figure(
        self, image: Image.Image, figure_type: str = "unknown", context: str = ""
    ) -> FigureInfo:
        if not self._initialized and not self.initialize():
            return FigureInfo(
                figure_num=0, page_num=0, figure_type=figure_type,
                description=f"vLLM server not available at {self.config.base_url}",
            )

        try:
            img_base64 = self._image_to_base64(image)
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
                json={"model": self.config.model, "messages": messages, "max_tokens": 1024, "temperature": 0.1},
            )

            if response.status_code != 200:
                return FigureInfo(
                    figure_num=0, page_num=0, figure_type=figure_type,
                    description=f"vLLM API error ({response.status_code})", engine=self.name,
                )

            description = self._extract_text(response.json()) or "Unable to generate description"
            detected_type = self._detect_figure_type(description, figure_type)

            return FigureInfo(
                figure_num=0, page_num=0, figure_type=detected_type,
                description=description, engine=self.name,
            )

        except Exception as e:
            return FigureInfo(
                figure_num=0, page_num=0, figure_type=figure_type,
                description=f"vLLM error: {type(e).__name__}: {e}", engine=self.name,
            )

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        buffered = io.BytesIO()
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.save(buffered, format="JPEG", quality=90)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @staticmethod
    def _extract_text(data: dict) -> str:
        choices = data.get("choices", [])
        if not choices:
            return ""
        raw = choices[0].get("message", {}).get("content", "").strip()
        return DeepSeekVLLMEngine._clean_ocr_output(raw)

    @staticmethod
    def _build_ocr_prompt() -> str:
        # DeepSeek-OCR was fine-tuned on specific short prompts.
        # "Free OCR." produces clean text without bounding box coordinates.
        # Long generic prompts cause massive hallucinations (fake formatting
        # instructions, fabricated abstracts, etc.).
        return "Free OCR."

    @staticmethod
    def _clean_ocr_output(text: str) -> str:
        """Strip grounding tags and bounding boxes from DeepSeek-OCR output."""
        # Remove <|ref|>...<|/ref|> inline references
        text = re.sub(r'<\|ref\|>.*?<\|/ref\|>', '', text)
        # Remove <|det|>[[x,y,w,h]]<|/det|> detection boxes
        text = re.sub(r'<\|det\|>\[\[.*?\]\]<\|/det\|>', '', text)
        # Remove any remaining special tokens
        text = re.sub(r'<\|[^|]+\|>', '', text)
        # Remove bare bounding box coordinates
        text = re.sub(r'\[\[\d+,\s*\d+,\s*\d+,\s*\d+\]\]', '', text)
        # Normalize HTML line breaks
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        # Strip remaining HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Collapse excessive blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

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
            "equation": ["equation", "formula", "mathematical"],
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
